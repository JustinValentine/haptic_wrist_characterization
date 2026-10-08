"""Analyze the locked-output transmission test.

Phase 1 (quasi-static torque cycles), torque tau vs motor-side joint angle th:
  * stiffness k: slope of a linear fit to all samples with |tau| > 20% of the
    cycle torque (loading and unloading pooled, so friction averages out),
    plus per-branch slopes and low/high-torque secant stiffness (nonlinearity);
  * lost motion: zero-torque intercept of the positive-torque line minus that
    of the negative-torque line (backlash; ~0 for a preloaded cable);
  * loop width at zero torque: angle gap between the rising and falling
    branches where the torque crosses zero (backlash + friction hysteresis);
  * hysteresis torque: half the torque gap between loading and unloading at
    the same angle (the friction the motor side must overcome; it also shows
    up as a width of 2 f / k in angle), and the energy lost per cycle.
Phase 2 (torque multisine about a preload): least-squares sinusoid fits per
  period give the compliance H = th/tau at each excited frequency (mean and
  standard error over periods). H is fitted with 1 / (Jm s^2 + c s + k):
  dynamic stiffness k, reflected motor-side inertia Jm, damping c, and the
  transmission resonance f_r = sqrt(k / Jm) / 2 pi.
The motor's cogging acts on the motor side like an extra, position-dependent
spring (its local stiffness is a sizable fraction of a cable's), so when a
friction-sweep analysis of the same kind has identified a cogging model, its
torque at the measured motor position is subtracted from the command before
both fits.

Run:  python transmission_stiffness_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from analysis_common import (cogging_torque, export_figure, fit_sinusoids, latest_cogging_model,
                             load_run, make_parser, print_header, print_truth_comparison,
                             robust_sigma, save_outputs, sim_truth)
from ieee_plot import (INK, SERIES, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       direct_label, figure_size, log_ticks_hz, reference_line, style_axis,
                       zero_line)

EXPERIMENT = 'transmission'
STIFFNESS_BASENAME = 'fig_transmission_stiffness'
FRF_BASENAME = 'fig_transmission_frf'
DIAGNOSTIC_BASENAME = 'diag_transmission_timeseries'
FIT_FRACTION = 0.2        # Use |tau| above this fraction of the cycle torque for stiffness
HIGH_FRACTION = 0.5       # Split between low- and high-torque secant stiffness


# --- phase 1: quasi-static loop ---------------------------------------------------------
def cycle_frame(df, period, test_torque):
    c = df[df['phase'] == 'cycle'].copy()
    s = (c['t_local'] / period) % 1.0
    c['cycle_index'] = np.floor(c['t_local'] / period).astype(int)
    rising = (s < 0.25) | (s >= 0.75)
    c['rising'] = rising
    c['tau'] = c['cmd_ff_torque_nm'] - c['cogging_nm']
    c['theta'] = np.radians(c['joint_pos_deg'])
    c['loading'] = np.where(rising, c['tau'] >= 0, c['tau'] <= 0)
    # Skip the virgin quarter cycle from the unloaded state.
    return c[c['t_local'] >= 0.25 * period]


def linear_fit(x, y):
    if len(x) < 3:
        return math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def side_fit(th, tau, sel):
    """tau = k th + a + d sign(tau): one stiffness, separate zero-torque
    intercepts for the positive and negative sides (lost motion = -2 d / k)."""
    if sel.sum() < 4:
        return math.nan, math.nan, math.nan
    side = np.sign(tau[sel])
    if len(np.unique(side)) < 2:
        k, a = linear_fit(th[sel], tau[sel])
        return k, a, 0.0
    A = np.column_stack([th[sel], np.ones(sel.sum()), side])
    (k, a, d), *_ = np.linalg.lstsq(A, tau[sel], rcond=None)
    return float(k), float(a), float(d)


def zero_crossing_angles(c):
    """Angles where the commanded torque crosses zero, rising and falling."""
    tau = c['tau'].to_numpy()
    th = c['theta'].to_numpy()
    rising, falling = [], []
    for i in np.where(np.sign(tau[:-1]) * np.sign(tau[1:]) < 0)[0]:
        frac = -tau[i] / (tau[i + 1] - tau[i])
        angle = th[i] + frac * (th[i + 1] - th[i])
        (rising if tau[i + 1] > tau[i] else falling).append(angle)
    return np.array(rising), np.array(falling)


def hysteresis_energy(c, period):
    energies = []
    for cycle, g in c.groupby('cycle_index'):
        if g['t_local'].iloc[-1] - g['t_local'].iloc[0] < 0.95 * period:
            continue
        tau = g['tau'].to_numpy()
        th = g['theta'].to_numpy()
        energies.append(abs(float(np.sum(0.5 * (tau[1:] + tau[:-1]) * np.diff(th)))))
    return float(np.median(energies)) if energies else math.nan


def analyze_loop(df, period, test_torque):
    c = cycle_frame(df, period, test_torque)
    tau, th = c['tau'].to_numpy(), c['theta'].to_numpy()
    big = np.abs(tau) > FIT_FRACTION * test_torque
    k, a, d = side_fit(th, tau, big)

    branches = {}
    for name, sign, loading in (('pos_loading', 1, True), ('pos_unloading', 1, False),
                                ('neg_loading', -1, True), ('neg_unloading', -1, False)):
        sel = big & (np.sign(tau) == sign) & (c['loading'].to_numpy() == loading)
        branches[name] = linear_fit(th[sel], tau[sel])[0]

    low = big & (np.abs(tau) <= HIGH_FRACTION * test_torque)
    high = np.abs(tau) > HIGH_FRACTION * test_torque
    k_low = side_fit(th, tau, low)[0]
    k_high = side_fit(th, tau, high)[0]
    lost_motion = -2.0 * d / k if k else math.nan

    # Friction hysteresis: common-slope fit with a loading/unloading offset per side.
    pos = big & (tau > 0)
    neg = big & (tau < 0)
    gaps = []
    for sign, sel in ((1, pos), (-1, neg)):
        load = c['loading'].to_numpy()[sel].astype(float)
        A = np.column_stack([th[sel], np.ones(sel.sum()), load])
        if sel.sum() > 5 and 0 < load.mean() < 1:
            coef, *_ = np.linalg.lstsq(A, tau[sel], rcond=None)
            gaps.append(sign * coef[2])
    hysteresis_torque = 0.5 * float(np.mean(gaps)) if gaps else math.nan

    up, down = zero_crossing_angles(c)
    width = (float(np.median(down) - np.median(up))
             if len(up) and len(down) else math.nan)

    residual = tau[big] - (k * th[big] + a + d * np.sign(tau[big]))
    return c, {
        'stiffness_nm_rad': k,
        'stiffness_pos_loading_nm_rad': branches['pos_loading'],
        'stiffness_pos_unloading_nm_rad': branches['pos_unloading'],
        'stiffness_neg_loading_nm_rad': branches['neg_loading'],
        'stiffness_neg_unloading_nm_rad': branches['neg_unloading'],
        'stiffness_low_torque_nm_rad': k_low,
        'stiffness_high_torque_nm_rad': k_high,
        'stiffening_ratio': k_high / k_low if k_low else math.nan,
        'lost_motion_deg': math.degrees(lost_motion),
        'loop_width_at_zero_deg': math.degrees(width),
        'hysteresis_torque_nm': hysteresis_torque,
        'friction_width_deg': (math.degrees(2.0 * hysteresis_torque / k)
                               if k and np.isfinite(hysteresis_torque) else math.nan),
        'hysteresis_energy_mj_per_cycle': 1e3 * hysteresis_energy(c, period),
        'max_deflection_deg': float(np.degrees(np.max(np.abs(th)))),
        'stiffness_fit_rms_nm': float(np.sqrt(np.mean(residual ** 2))) if len(residual) else math.nan,
        'test_torque_nm': test_torque,
        'fit_intercept_nm': a,
        'fit_side_offset_nm': d,
    }


# --- phase 2: compliance FRF -------------------------------------------------------------
def zoh_fourier(starts, values, freqs, t_start, t_end):
    """Fourier coefficients (2/T) int u(t) exp(-j w t) dt over [t_start, t_end) of a
    piecewise-constant signal u = values[i] on [starts[i], starts[i+1])."""
    a = np.clip(starts[:-1], t_start, t_end)
    b = np.clip(starts[1:], t_start, t_end)
    w = 2.0 * np.pi * np.asarray(freqs)
    E = (np.exp(-1j * np.outer(a, w)) - np.exp(-1j * np.outer(b, w))) / (1j * w)
    return (2.0 / (t_end - t_start)) * (values[:-1] @ E)


def compliance_frf(df, design):
    """Compliance theta/tau at the excited lines, one estimate per period.

    moteus applies each torque command when it arrives and holds it until the
    next one, and samples position at that same instant. Arrival is taken as
    the midpoint of the command/reply round trip. The torque is therefore a
    piecewise-constant signal (Fourier coefficients integrated exactly) and
    the position is sampled at the arrival times (least-squares sinusoid fit).
    """
    m = df[df['phase'] == 'multisine']
    if m.empty or not design:
        return None
    freqs = np.asarray(design['freqs_hz'])
    period = design['period_s']
    arrive = 0.5 * (m['t_send'].to_numpy() + m['time'].to_numpy())
    arrive = arrive - arrive[0]
    torque = m['cmd_ff_torque_nm'].to_numpy()
    cogging = m['cogging_nm'].to_numpy()
    theta = np.radians(m['joint_pos_deg'].to_numpy())
    per_period = []
    for p in range(1, int(arrive[-1] // period)):      # Period 0 is the transient
        t0, t1 = p * period, (p + 1) * period
        sel = (arrive >= t0) & (arrive < t1)
        if sel.sum() < 10:
            continue
        T = zoh_fourier(arrive, torque, freqs, t0, t1)
        if np.any(cogging[sel]):
            T = T - fit_sinusoids(arrive[sel], cogging[sel], freqs)[0]
        X, _ = fit_sinusoids(arrive[sel], theta[sel], freqs)
        per_period.append(X / T)
    if len(per_period) < 2:
        return None
    H = np.array(per_period)
    mean = H.mean(axis=0)
    se = np.sqrt(np.sum(np.abs(H - mean) ** 2, axis=0) / (len(H) - 1)) / math.sqrt(len(H))
    return pd.DataFrame({'frequency_hz': freqs, 'H_real': mean.real, 'H_imag': mean.imag,
                         'magnitude_rad_per_nm': np.abs(mean),
                         'magnitude_db': 20 * np.log10(np.abs(mean)),
                         'phase_deg': np.degrees(np.unwrap(np.angle(mean))),
                         'se_rad_per_nm': se, 'num_periods': len(H)})


def fit_compliance(frf):
    """Fit H = 1 / (Jm s^2 + c s + k): linear WLS on 1/H, then refine on H."""
    f = frf['frequency_hz'].to_numpy()
    w = 2 * np.pi * f
    H = frf['H_real'].to_numpy() + 1j * frf['H_imag'].to_numpy()
    se = np.maximum(frf['se_rad_per_nm'].to_numpy(), 1e-3 * np.abs(H))
    Z = 1.0 / H
    sz = se / np.abs(H) ** 2
    A = np.vstack([np.column_stack([np.ones_like(w), np.zeros_like(w), -w ** 2]) / sz[:, None],
                   np.column_stack([np.zeros_like(w), w, np.zeros_like(w)]) / sz[:, None]])
    b = np.concatenate([Z.real / sz, Z.imag / sz])
    (k, c, J), *_ = np.linalg.lstsq(A, b, rcond=None)

    def residual(p):
        model = 1.0 / (p[0] + 1j * w * p[1] - p[2] * w ** 2)
        r = (H - model) / se
        return np.concatenate([r.real, r.imag])

    x0 = np.array([k, max(c, 1e-6), max(J, 1e-9)])
    sol = optimize.least_squares(residual, x0, x_scale=np.abs(x0),
                                 bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
    k, c, J = sol.x
    fr = math.sqrt(k / J) / (2 * math.pi) if J > 0 else math.nan
    return {'dynamic_stiffness_nm_rad': float(k), 'motor_side_inertia_kg_m2': float(J),
            'transmission_damping_nms_rad': float(c), 'resonance_hz': fr,
            'resonance_damping_ratio': float(c / (2 * math.sqrt(k * J))) if J > 0 else math.nan,
            'frf_fit_chi2_per_dof': float(np.sum(sol.fun ** 2) / max(1, 2 * len(w) - 3))}


def compliance_model(params, f):
    s = 2j * np.pi * f
    return 1.0 / (params['motor_side_inertia_kg_m2'] * s ** 2
                  + params['transmission_damping_nms_rad'] * s
                  + params['dynamic_stiffness_nm_rad'])


# --- figures -------------------------------------------------------------------------------
def plot_loop(c, est):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.5))
    for _, g in c.groupby('cycle_index'):
        ax.plot(1e3 * g['theta'], g['tau'], color=INK, linewidth=0.7)
    ax.plot([], [], color=INK, linewidth=0.7, label=f"Measured ({c['cycle_index'].nunique()} cycles)")
    test = est['test_torque_nm']
    k = est['stiffness_nm_rad']
    for side in (1, -1):
        sel = (np.sign(c['tau']) == side) & (np.abs(c['tau']) > FIT_FRACTION * test)
        if not sel.any():
            continue
        th = np.linspace(c['theta'][sel].min(), c['theta'][sel].max(), 20)
        tau_fit = k * th + est['fit_intercept_nm'] + side * est['fit_side_offset_nm']
        ax.plot(1e3 * th, tau_fit, color=SERIES[0], linestyle='--', linewidth=1.1,
                label=f'Linear fit, $k$ = {k:,.0f} {TORQUE_UNIT}/rad' if side == 1 else None)
    zero_line(ax)
    zero_line(ax, orientation='v')
    ax.set_xlabel('Deflection, joint side (mrad)')
    ax.set_ylabel(f'Torque ({TORQUE_UNIT})')
    style_axis(ax)
    ax.legend(loc='upper left')
    if abs(est['lost_motion_deg']) > 0.02:
        direct_label(ax, 0.0, -0.85 * test,
                     f"lost motion {est['lost_motion_deg']:.2f}°", dx=4)
    fig.tight_layout(pad=0.3)
    return fig


def plot_frf(frf, fit):
    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=figure_size('single', 3.4), sharex=True,
                                     gridspec_kw={'hspace': 0.12})
    f = frf['frequency_hz'].to_numpy()
    mag_db = frf['magnitude_db'].to_numpy()
    rel = frf['se_rad_per_nm'].to_numpy() / frf['magnitude_rad_per_nm'].to_numpy()
    err_db = 20 * np.log10(1 + 2 * np.minimum(rel, 0.9))
    ax_m.errorbar(f, mag_db, yerr=err_db, linestyle='None', marker='o', markersize=3.0,
                  markerfacecolor='white', markeredgecolor=INK, ecolor=INK, elinewidth=0.6,
                  label='Measured (±2 SE)')
    ff = np.logspace(math.log10(f.min()), math.log10(f.max()), 400)
    model = compliance_model(fit, ff)
    ax_m.semilogx(ff, 20 * np.log10(np.abs(model)), color=SERIES[0], linestyle='--',
                  linewidth=1.1, label='Fit $1/(J_m s^2 + c s + k)$')
    phase = frf['phase_deg'].to_numpy()
    phase = phase - 360.0 * np.round(phase[0] / 360.0)
    ax_p.semilogx(f, phase, linestyle='None', marker='o', markersize=3.0,
                  markerfacecolor='white', markeredgecolor=INK)
    ax_p.semilogx(ff, np.degrees(np.unwrap(np.angle(model))), color=SERIES[0], linestyle='--',
                  linewidth=1.1)
    fr = fit['resonance_hz']
    if np.isfinite(fr) and f.min() <= fr <= f.max():
        for ax in (ax_m, ax_p):
            reference_line(ax, fr, orientation='v')
        ax_m.annotate(f'$f_r$ = {fr:.0f} Hz', (fr, 0.08), xycoords=('data', 'axes fraction'),
                      xytext=(-4, 0), textcoords='offset points', ha='right', va='bottom',
                      fontsize=7, color=INK)
    ax_m.set_ylabel('Compliance (dB re rad/N·m)')
    ax_p.set_ylabel('Phase (deg)')
    ax_p.set_xlabel('Frequency (Hz)')
    ax_p.set_yticks([0, -45, -90, -135, -180])
    for ax in (ax_m, ax_p):
        style_axis(ax)
        log_ticks_hz(ax)
    ax_m.legend(loc='lower left')
    add_panel_label(ax_m, '(a)')
    add_panel_label(ax_p, '(b)')
    fig.align_ylabels([ax_m, ax_p])
    return fig


def plot_timeseries(df):
    fig, (ax_t, ax_d) = plt.subplots(2, 1, figsize=figure_size('double', 3.0), sharex=True)
    ax_t.plot(df['time'], df['cmd_ff_torque_nm'], color=INK, linewidth=0.6, label='Commanded')
    if 'joint_torque_nm' in df:
        ax_t.plot(df['time'], df['joint_torque_nm'], color=SERIES[1], linewidth=0.4,
                  label='moteus reported')
    ax_d.plot(df['time'], 1e3 * np.radians(df['joint_pos_deg']), color=INK, linewidth=0.6)
    if 'output_joint_raw_deg' in df:
        out = np.radians(df['output_joint_raw_deg'] - df['output_joint_raw_deg'].iloc[0])
        ax_d.plot(df['time'], 1e3 * out, color=SERIES[0], linewidth=0.6, label='Output encoder')
        ax_d.legend(loc='upper right')
    ax_t.set_ylabel(f'Torque ({TORQUE_UNIT})')
    ax_d.set_ylabel('Deflection (mrad)')
    ax_d.set_xlabel('Time (s)')
    ax_t.legend(loc='upper right')
    for ax in (ax_t, ax_d):
        style_axis(ax, minor=False)
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------------
def output_lock_check(df):
    """With an output encoder: how much of the measured deflection happens at
    the output (clamp/frame compliance). ~0 for a rigid lock."""
    if 'output_joint_raw_deg' not in df:
        return {}
    c = df[df['phase'] == 'cycle']
    tau = c['cmd_ff_torque_nm'].to_numpy()
    out = np.radians(c['output_joint_raw_deg'] - c['output_joint_raw_deg'].iloc[0])
    motor = np.radians(c['joint_pos_deg'].to_numpy())
    slope_out = np.polyfit(tau, out.to_numpy(), 1)[0]
    slope_motor = np.polyfit(tau, motor, 1)[0]
    smooth = out.rolling(51, center=True, min_periods=1).mean()
    return {'output_motion_pp_deg': float(np.degrees(smooth.max() - smooth.min())),
            'output_deflection_fraction': float(slope_out / slope_motor) if slope_motor else math.nan}


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    df = run.df
    test_torque = float(run.meta_value('test_torque_nm', column='test_torque_nm'))
    period = float(run.meta_value('cycle_period_s', column='cycle_period_s'))
    design = run.meta.get('multisine')
    cogging = latest_cogging_model(run.simulated)
    df = df.copy()
    if cogging is not None and 'motor_pos_rev' in df:
        df['cogging_nm'] = cogging_torque(cogging, df['motor_pos_rev'])
    else:
        df['cogging_nm'] = 0.0

    c, loop = analyze_loop(df, period, test_torque)
    frf = compliance_frf(df, design)
    fit = fit_compliance(frf) if frf is not None else {}
    noise = robust_sigma(np.diff(np.radians(df[df['phase'] == 'dwell']['joint_pos_deg']))) / math.sqrt(2)

    summary = {**loop, **fit, **output_lock_check(df),
               'position_noise_mrad': 1e3 * noise if np.isfinite(noise) else math.nan,
               'torque_command_vs_reported_rms_nm': float(np.sqrt(np.nanmean(
                   (c['cmd_ff_torque_nm'] - c['joint_torque_nm']) ** 2))),
               'cogging_removed': cogging is not None,
               'cogging_model_source': cogging['source'] if cogging else ''}
    if frf is not None and len(frf):
        low = frf.nsmallest(3, 'frequency_hz')
        summary['low_frequency_stiffness_nm_rad'] = float(1.0 / low['magnitude_rad_per_nm'].mean())
    metadata = {
        'analysis': 'transmission_stiffness_experiment_analysis.py',
        'source_data': str(run.source),
        'fit_fraction': FIT_FRACTION,
        'high_fraction': HIGH_FRACTION,
        'method': ('quasi-static: common-slope fit with per-side intercepts for |tau| > 20% '
                   'of cycle torque; dynamic: per-period Fourier coefficients of the commanded '
                   'torque (piecewise constant from each command arrival) and LS sinusoid fits '
                   'of motor angle at the arrival times, weighted fit of 1/(Jm s^2 + c s + k)'),
        'note': ('Angles are motor-encoder angles divided by GEAR_RATIO with the output '
                 'clamped, i.e. cable stretch plus lost motion, joint-referred. Torques are '
                 'in the torque scale moteus reports.'),
    }
    tables = {'loop': c[['time', 't_local', 'cycle_index', 'rising', 'loading', 'tau', 'theta']]}
    if frf is not None:
        tables['compliance_frf'] = frf
    save_outputs(run, tables, summary, metadata)

    print_header('TRANSMISSION (LOCKED OUTPUT)')
    print(f"  Static stiffness k         {loop['stiffness_nm_rad']:.4g} N m/rad "
          f"(pos/neg loading {loop['stiffness_pos_loading_nm_rad']:.4g} / "
          f"{loop['stiffness_neg_loading_nm_rad']:.4g})")
    print(f"  Low / high torque secant   {loop['stiffness_low_torque_nm_rad']:.4g} / "
          f"{loop['stiffness_high_torque_nm_rad']:.4g} N m/rad "
          f"(ratio {loop['stiffening_ratio']:.3f})")
    print(f"  Lost motion                {loop['lost_motion_deg']:.4f} deg")
    print(f"  Loop width at zero torque  {loop['loop_width_at_zero_deg']:.4f} deg")
    print(f"  Hysteresis torque          {loop['hysteresis_torque_nm']:.4g} N m "
          f"(= {loop['friction_width_deg']:.4f} deg of width)")
    print(f"  Hysteresis energy          {loop['hysteresis_energy_mj_per_cycle']:.4g} mJ/cycle")
    if fit:
        print(f"  Dynamic stiffness          {fit['dynamic_stiffness_nm_rad']:.4g} N m/rad")
        print(f"  Motor-side inertia Jm      {fit['motor_side_inertia_kg_m2']:.4g} kg m^2")
        print(f"  Equivalent damping c       {fit['transmission_damping_nms_rad']:.4g} N m s/rad")
        print(f"  Resonance                  {fit['resonance_hz']:.1f} Hz "
              f"(zeta {fit['resonance_damping_ratio']:.3f})")
    if 'output_motion_pp_deg' in summary:
        print(f"  Output encoder (lock check) {summary['output_motion_pp_deg']:.4f} deg p-p smoothed; "
              f"{100 * summary['output_deflection_fraction']:.1f}% of the deflection is at the output")

    truth = sim_truth(run)
    if truth:
        rows = [('Static stiffness k', loop['stiffness_nm_rad'],
                 truth['transmission_stiffness'], 'N m/rad')]
        if truth['backlash_deg']:
            rows.append(('Lost motion', loop['lost_motion_deg'], truth['backlash_deg'], 'deg'))
        if fit:
            rows += [('Dynamic stiffness', fit['dynamic_stiffness_nm_rad'],
                      truth['transmission_stiffness'], 'N m/rad'),
                     ('Motor-side inertia Jm', fit['motor_side_inertia_kg_m2'],
                      truth['motor_inertia'], 'kg m^2')]
        print_truth_comparison(rows)
        if not truth['backlash_deg']:
            print(f"  Lost motion {loop['lost_motion_deg']:.4f} deg (truth: no backlash)")
        if fit:
            print(f"  Equivalent damping c {fit['transmission_damping_nms_rad']:.3g} N m s/rad vs "
                  f"viscous-only truth {truth['transmission_damping'] + truth['viscous']:.3g} "
                  '(Coulomb friction adds equivalent damping)')

    fig = plot_loop(c, loop)
    paths = export_figure(fig, run, STIFFNESS_BASENAME, {**metadata, **summary}, args.paper_dir)
    if frf is not None and fit:
        fig = plot_frf(frf, fit)
        export_figure(fig, run, FRF_BASENAME, {**metadata, **summary}, args.paper_dir)
    export_figure(plot_timeseries(df), run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
