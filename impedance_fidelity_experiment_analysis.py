"""Rendered impedance spectrum: displayed stiffness, damping and loop delay per target.

For every measured target the first period is discarded; in each remaining
period the disturbance torque (rebuilt from the stored design and treated as
piecewise constant from each command's arrival, so its Fourier coefficients
are exact) and the position (least-squares sinusoid fit at the arrival times)
give the compliance C = Theta/T_d at the excited lines, and the displayed
impedance Z = 1/C (mean and standard error over periods).

Rendered impedance. The near-free target (weak centering spring K_c rendered
by the 500 Hz loop, B = 0) measures the device itself, Z_free = -J w^2 + j w b +
K_c e^{-j w T_loop}, including friction-equivalent damping and transmission
dynamics; T_loop is the loop's logged delay (2 SYNC periods: the CST ramp).
Subtracting it line by line,

    Z_r(w) = Z_target(w) - Z_free(w) + K_c e^{-j w T_loop},

cancels inertia, damping and friction without a model of the device, leaving
what the loop renders. Rendered stiffness K_r = Re Z_r, rendered damping
B_r = Im Z_r / w. Fits per target: (K_eff, T_eff) of (K + j w B) e^{-j w T}
with the commanded B, and (K_eff, B_eff) with T = 0. A delayed loop shows up as
B_eff = B - K T_eff: stiffness turned into negative damping. Lines where the
rendered part is small against the total impedance (inertia-dominated) or
noisy are excluded.

Stability prediction. With the viscous damping b (step response, else the
free target at high frequency) and T_eff, the rendered spring loses damping
at K_crit solving b + B cos(w T) - K sin(w T)/w = 0, w = sqrt(K/J) (~ b/T for
B = 0). Compared with the Z-width experiment when a same-kind run exists:
Coulomb friction quenches small delay-induced oscillations, so a pulse-based
Z-width exceeds the linear limit, most at low damping.

Free space. J_app(w) = (K_c - Re Z_free)/w^2 and b_app(w) = Im Z_free / w:
the inertia and damping an operator feels across frequency.

Run:  python impedance_fidelity_experiment_analysis.py [run_dir] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import device_config
from analysis_common import (applied_torque_times, cogging_torque, loop_timing, pwl_fourier,
                             run_joint_index, export_figure, fit_sinusoids, latest_cogging_model,
                             load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, sim_truth)
from experiment_common import latest_summary
from ieee_plot import (INK, MUTED, SERIES, add_panel_label, configure_ieee_style, figure_size,
                       log_ticks_hz, pad_top_tick, reference_line, series_style, style_axis,
                       zero_line)
from impedance_fidelity_experiment import EXPERIMENT
from schroeder_bandwidth_experiment import Multisine
from transmission_stiffness_experiment_analysis import zoh_fourier

SPECTRUM_BASENAME = 'fig_impedance_spectrum'
FREE_BASENAME = 'fig_impedance_free'
DIAGNOSTIC_BASENAME = 'diag_impedance_bode'
FREE_K_MAX_NM_RAD = 8.0      # A B = 0 target this soft serves as the near-free reference
DELAY_MIN_K_NM_RAD = 8.0     # Targets at least this stiff give a usable delay estimate
RESOLVE_RATIO = 10.0         # Use lines where |Z_target| <= this x |Z_desired|
MAX_REL_SE = 0.2             # ... and SE(Z_r) / |Z_r| below this
FIDELITY_MAG = 0.10          # Fidelity: |Z_r/Z_d - 1| within 10% ...
FIDELITY_PHASE_DEG = 10.0    # ... and phase within 10 deg
RESIDUAL_FLAG = 0.25         # Unexplained motion fraction that flags a target
DAMPING_BAND_TOL = 0.5       # Free-space damping within +/-50% of its high-frequency value


# --- estimation -----------------------------------------------------------------------
def position_column(df):
    """Handle-side motion when an output encoder is logged, else the motor encoder."""
    if 'output_joint_raw_deg' in df.columns and df['output_joint_raw_deg'].notna().any():
        return 'output_joint_raw_deg'
    return 'joint_pos_deg'


def estimate_target(frame, design, column, cogging=None, loop=None, joint=None, renderer=None):
    """Displayed impedance per period at the excited lines; mean and SE.

    With a cogging model, the cogging torque at the measured motor position is
    taken out of the input torque: it is an internal, position-dependent torque
    whose apparent stiffness and damping depend on the motion amplitude, so it
    would not cancel in the line-by-line subtraction of the near-free target.

    Timing. moteus rows are one command/reply each: torque and position both
    belong to the command's arrival. p4wrist rows are loop cycles (loop =
    {'delay_s', 'period_s'} from the run): the disturbance is the design value
    at the host's send time, and the loop's torque reaches it by a linear ramp
    ending loop_delay after the cycle that used it (piecewise linear, Fourier
    coefficients exact); positions are sampled at the cycle's SYNC.
    """
    freqs = np.asarray(design['freqs_hz'])
    period = float(design['period_s'])
    signal = Multisine(design['freqs_hz'], design['amplitudes_nm'], design['phases_rad'],
                       design['t_offset_s'])
    t_local = frame['t_local'].to_numpy()
    lag = frame['time'].to_numpy() - frame['t_send'].to_numpy()
    zoh = loop is None
    if loop and renderer == 'drive':
        # The drive loop adds the disturbance as its torque offset, latched at the
        # next SYNC and held for the period (no ramp).
        torque = np.array([signal(t)[0] for t in t_local - lag])
        t_torque = t_local + loop['period_s']
        t_theta = t_local
        zoh = True
    elif loop:
        torque = np.array([signal(t)[0] for t in t_local - lag])
        t_torque = applied_torque_times(t_local, loop)
        t_theta = t_local
    else:
        torque = np.array([signal(t)[0] for t in t_local])
        t_torque = t_theta = t_local + 0.5 * lag
    theta = np.radians(frame[column].to_numpy())
    cog = (cogging_torque(cogging, frame, joint)
           if cogging is not None and 'motor_pos_rev' in frame else None)
    Z, residual, motion = [], [], []
    for p in range(1, int(design['periods'])):
        t0, t1 = p * period, (p + 1) * period
        sel = (t_theta >= t0) & (t_theta < t1)
        if sel.sum() < 4 * len(freqs) or ((t_torque >= t0) & (t_torque < t1)).sum() < 4 * len(freqs):
            continue
        T = (zoh_fourier if zoh else pwl_fourier)(t_torque, torque, freqs, t0, t1)
        if cog is not None:
            T = T - fit_sinusoids(t_theta[sel], cog[sel], freqs)[0]
        X, res = fit_sinusoids(t_theta[sel], theta[sel], freqs)
        Z.append(T / X)
        residual.append(res)
        motion.append(float(np.std(theta[sel])))
    if len(Z) < 1:
        return None
    Z = np.array(Z)
    mean = Z.mean(axis=0)
    se = (np.sqrt(Z.real.var(axis=0, ddof=1) + Z.imag.var(axis=0, ddof=1)) / math.sqrt(len(Z))
          if len(Z) > 1 else np.full(len(freqs), np.nan))
    return {'freqs': freqs, 'Z': mean, 'se': se, 'num_periods': len(Z),
            'residual_fraction': float(np.mean(residual) / max(np.mean(motion), 1e-12)),
            'motion_rms_deg': float(np.degrees(np.mean(motion)))}


def fit_free(freqs, Z, se, k_center):
    """Rigid fit of the near-free impedance: Re Z = K_c - J w^2, Im Z = b w."""
    w = 2 * np.pi * freqs
    sigma = np.sqrt(np.where(np.isfinite(se), se, 0.0) ** 2 + (0.02 * np.abs(Z)) ** 2)
    A = np.column_stack([np.ones_like(w), -w ** 2]) / sigma[:, None]
    (kc, J), *_ = np.linalg.lstsq(A, Z.real / sigma, rcond=None)
    b = float(np.sum(Z.imag * w / sigma ** 2) / np.sum(w ** 2 / sigma ** 2))
    top = freqs >= np.quantile(freqs, 2.0 / 3.0)
    b_high = float(np.median(Z.imag[top] / w[top]))
    return {'J_kg_m2': float(J), 'b_nms_rad': b, 'b_high_freq_nms_rad': b_high,
            'k_center_fit_nm_rad': float(kc), 'k_center_nm_rad': float(k_center)}


def damping_band_min_hz(free):
    """Lowest frequency above which damping is resolvable: there the near-free
    device's apparent damping stays within +/-50% of its high-frequency value
    (below, Coulomb friction dominates the imaginary part of the impedance)."""
    f = free['freqs']
    b_app = free['Z'].imag / (2 * np.pi * f)
    b_hf = float(np.median(b_app[f >= np.median(f)]))
    bad = np.flatnonzero(np.abs(b_app - b_hf) > DAMPING_BAND_TOL * abs(b_hf))
    if len(bad) == 0:
        return float(f[0])
    return float(f[bad[-1] + 1]) if bad[-1] + 1 < len(f) else float(f[-1])


def fit_rendered(freqs, Zr, se, K, B):
    """(K_eff, T_eff) with the commanded B, and (K_eff, B_eff) with T = 0."""
    w = 2 * np.pi * freqs
    sigma = np.sqrt(se ** 2 + (0.02 * np.abs(Zr)) ** 2)

    def residual(x):
        model = (x[0] + 1j * w * B) * np.exp(-1j * w * x[1])
        err = (model - Zr) / sigma
        return np.concatenate([err.real, err.imag])

    sol = optimize.least_squares(residual, [max(K, 1e-3), 1e-3],
                                 bounds=([0.0, -0.005], [np.inf, 0.05]))
    k_eff, delay = sol.x
    # T = 0 equivalents: K from the real part, B from the imaginary part.
    wr = 1.0 / sigma ** 2
    k_static = float(np.sum(Zr.real * wr) / np.sum(wr))
    b_eff = float(np.sum(Zr.imag * w * wr) / np.sum(w ** 2 * wr))
    return {'k_eff_nm_rad': float(k_eff), 'delay_ms': 1000.0 * float(delay),
            'k_static_nm_rad': k_static, 'b_eff_nms_rad': b_eff}


def fit_given_delay(freqs, Zr, se, delay_s):
    """(K, B) of (K + j w B) e^{-j w T} for a known delay T (linear least squares).

    B then absorbs any frequency-independent damping offset (e.g. friction
    mismatch against the near-free reference) while T sets the delay-induced
    frequency dependence."""
    w = 2 * np.pi * freqs
    sigma = np.sqrt(se ** 2 + (0.02 * np.abs(Zr)) ** 2)
    rotated = Zr * np.exp(1j * w * delay_s)
    weights = 1.0 / sigma ** 2
    K = float(np.sum(rotated.real * weights) / np.sum(weights))
    B = float(np.sum(rotated.imag * w * weights) / np.sum(w ** 2 * weights))
    return K, B


def fidelity_bandwidth(freqs, Zr, K, B):
    """Highest line up to which Z_r stays within the fidelity tolerances
    (3-line running median of the errors so one noisy line does not end it)."""
    Zd = K + 1j * 2 * np.pi * freqs * B
    ratio = Zr / Zd
    mag = np.abs(np.abs(ratio) - 1.0)
    phase = np.abs(np.degrees(np.angle(ratio)))
    ok = (pd.Series(mag).rolling(3, center=True, min_periods=1).median().to_numpy() <= FIDELITY_MAG) & \
         (pd.Series(phase).rolling(3, center=True, min_periods=1).median().to_numpy() <= FIDELITY_PHASE_DEG)
    if not ok[0]:
        return math.nan, False
    fail = np.flatnonzero(~ok)
    if len(fail) == 0:
        return float(freqs[-1]), True        # Lower bound: fine to the last usable line
    return float(freqs[fail[0] - 1]), False


def model_fidelity_bandwidth(fit, K, B, f_max):
    """Highest frequency where the fitted delayed-loop model stays within the
    fidelity tolerances of Z_d; capped at f_max (then a lower bound)."""
    f = np.logspace(-1, math.log10(f_max), 600)
    w = 2 * np.pi * f
    model = (fit['k_eff_nm_rad'] + 1j * w * B) * np.exp(-1j * w * fit['delay_ms'] / 1000.0)
    ratio = model / (K + 1j * w * B)
    ok = (np.abs(np.abs(ratio) - 1.0) <= FIDELITY_MAG) & \
         (np.abs(np.degrees(np.angle(ratio))) <= FIDELITY_PHASE_DEG)
    if not ok[0]:
        return math.nan, False
    fail = np.flatnonzero(~ok)
    if len(fail) == 0:
        return float(f_max), True
    return float(f[fail[0] - 1]), False


def predicted_k_crit(b, B, delay_s, J):
    """Smallest K where b + B cos(wT) - K sin(wT)/w = 0, w = sqrt(K/J)."""
    if not (delay_s > 0 and J > 0 and b + B > 0):
        return math.nan

    def margin(K):
        w = math.sqrt(K / J)
        return b + B * math.cos(w * delay_s) - K * math.sin(w * delay_s) / w

    hi = 1e-3
    while margin(hi) > 0 and hi < 1e7:
        hi *= 2.0
    if hi >= 1e7:
        return math.inf
    return float(optimize.brentq(margin, hi / 2.0 if hi > 1e-3 else 1e-6, hi))


def linear_damping(run):
    """Viscous damping for the stability prediction. The free-space damping
    measured here includes friction-equivalent damping, so it is not used."""
    found = latest_summary('step_response', run.simulated)
    if found and isinstance(found[1].get('b_nms_rad'), (int, float)):
        return float(found[1]['b_nms_rad']), f'step_response analysis ({found[0].name})'
    return device_config.NOMINAL_JOINT_DAMPING_NMS_RAD, 'device_config nominal'


def zwidth_reference(run):
    found = latest_summary('zwidth', run.simulated)
    if found is None:
        return None, {}
    run_dir, summary = found
    ref = {}
    for key, value in summary.items():
        for tag in ('b0', 'bmin'):
            for kind in ('k_max', 'k_first_unstable'):
                prefix, suffix = f'zwidth_{kind}_', f'_{tag}_nm_rad'
                if key.startswith(prefix) and key.endswith(suffix):
                    ref.setdefault(key[len(prefix):-len(suffix)], {})[kind] = value
    return run_dir, ref


def expected_delay_s(run, target):
    """Nominal rendering delay of a target's loop (position term; a damping term,
    filtered and differenced, lags more).

    moteus host loop: 1.5 command periods. p4wrist: the 500 Hz loop reaches the
    motor 2 SYNC periods after its sample (the CST ramp); a host loop adds the
    sample's age (one loop period) and half its own period (zero-order hold)."""
    loop = loop_timing(run)
    if loop is None:
        return 1.5 / target['rate_hz']
    if target['renderer'] == 'drive':
        return 0.0   # the drive's own loop: its current loop and velocity filter only
    if target['renderer'] == 'rt':
        return loop['delay_s']
    return loop['delay_s'] + loop['period_s'] + 0.5 / target['rate_hz']


def frame_modes(df, target_id):
    """Logged mode column for a target's excitation (numeric)."""
    frame = df[(df['phase'] == 'excite') & (df['target_id'] == target_id)]
    return pd.to_numeric(frame['mode'], errors='coerce') if 'mode' in frame else pd.Series(dtype=float)


def target_label(t):
    mode = t['mode']
    loop = ('500 Hz loop' if mode == 'rt' else 'drive loop' if mode == 'drive'
            else f"host {float(mode.split('_')[1]):g} Hz")
    return f"{loop}, $K$={t['k_nm_rad']:g}, $B$={t['b_nms_rad']:g}"


# --- figures ---------------------------------------------------------------------------
def choose_shown(results):
    """Up to three stiff targets: a loop/host matched pair, then a host B = 0 target."""
    stiff = [r for r in results if r['k_nm_rad'] >= DELAY_MIN_K_NM_RAD or r['renderer'] == 'host']
    stiff = [r for r in stiff if r.get('fit')]
    shown = []
    for r in stiff:
        for other in stiff:
            if (other is not r and other['renderer'] != r['renderer']
                    and other['k_nm_rad'] == r['k_nm_rad'] and other['b_nms_rad'] == r['b_nms_rad']):
                shown = sorted([r, other], key=lambda x: x['renderer'] != 'rt')
                break
        if shown:
            break
    for r in sorted(stiff, key=lambda x: (x['renderer'] != 'host', x['b_nms_rad'])):
        if len(shown) >= 3:
            break
        if r not in shown:
            shown.append(r)
    return shown[:3]


def plot_spectrum(shown, damping_min_hz, loop_delays):
    fig, (ax_k, ax_b) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 3.6),
                                     gridspec_kw={'hspace': 0.12})
    f_all = []
    for i, r in enumerate(shown):
        ok = r['usable']
        f, Zr, se = r['freqs'][ok], r['Zr'][ok], r['se_r'][ok]
        w = 2 * np.pi * f
        style = series_style(i, line=False)
        K, B = r['k_nm_rad'], r['b_nms_rad']
        ax_k.errorbar(f, Zr.real / K, yerr=2 * se / K, elinewidth=0.6, label=target_label(r),
                      **style)
        band = f >= damping_min_hz
        ax_b.errorbar(f[band], Zr.imag[band] / w[band] - B, yerr=2 * se[band] / w[band],
                      elinewidth=0.6, **style)
        # Model at the loop's measured delay, (K, B) refitted so the line follows the
        # data level; the delay sets the frequency dependence.
        delay_s = max(loop_delays.get(r['mode'], 0.0), 0.0) / 1000.0
        k_fit, b_fit = fit_given_delay(f, Zr, se, delay_s)
        ff = np.logspace(math.log10(f.min()), math.log10(f.max()), 200)
        ww = 2 * np.pi * ff
        model = (k_fit + 1j * ww * b_fit) * np.exp(-1j * ww * delay_s)
        ax_k.plot(ff, model.real / K, color=SERIES[i], linestyle='--', linewidth=0.9)
        fb = ff >= damping_min_hz
        ax_b.plot(ff[fb], model.imag[fb] / ww[fb] - B, color=SERIES[i], linestyle='--',
                  linewidth=0.9)
        f_all.extend(f)
    reference_line(ax_k, 1.0)
    zero_line(ax_b)
    ax_k.set_ylabel('$K_r / K$')
    ax_b.set_ylabel('$B_r - B$ (N$\\cdot$m$\\cdot$s/rad)')
    ax_b.set_xlabel('Frequency (Hz)')
    ax_b.set_xscale('log')
    if f_all:
        ax_b.set_xlim(min(f_all) / 1.2, max(f_all) * 1.2)
    log_ticks_hz(ax_b)
    for ax in (ax_k, ax_b):
        style_axis(ax)
    lo, hi = ax_k.get_ylim()
    ax_k.set_ylim(lo, hi + 0.55 * (hi - lo))      # Room for the legend above the data
    pad_top_tick(ax_k)
    fit_handle, = ax_k.plot([], [], color=INK, linestyle='--', linewidth=0.9,
                            label='Fit at the loop delay')
    handles, labels = ax_k.get_legend_handles_labels()
    order = sorted(range(len(handles)), key=lambda i: handles[i] is fit_handle)
    ax_k.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left',
                fontsize=6.5)

    add_panel_label(ax_k, '(a)', x=-0.14)
    add_panel_label(ax_b, '(b)', x=-0.14)
    fig.align_ylabels([ax_k, ax_b])
    return fig


def plot_free(free, fit, b_linear, b_source, damping_min_hz):
    fig, (ax_j, ax_b) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 3.3),
                                     gridspec_kw={'hspace': 0.12})
    f, Z, se = free['freqs'], free['Z'], free['se']
    w = 2 * np.pi * f
    kc = fit['k_center_nm_rad']
    inertial = fit['J_kg_m2'] * w ** 2 >= 2.0 * kc       # J resolvable above the spring
    j_app = (kc - Z.real) / w ** 2
    j_se = se / w ** 2
    marker = dict(linestyle='None', marker='o', markersize=3.0, markerfacecolor='white',
                  markeredgecolor=INK, ecolor=INK, elinewidth=0.6)
    ax_j.errorbar(f[inertial], 1e4 * j_app[inertial], yerr=2e4 * j_se[inertial],
                  label='Measured (±2 SE)', **marker)
    reference_line(ax_j, 1e4 * fit['J_kg_m2'], color=INK, linewidth=0.9,
                   label=f"Rigid fit, {1e4 * fit['J_kg_m2']:.1f} kg$\\cdot$cm$^2$")
    band = f >= damping_min_hz
    ax_b.errorbar(f[band], 1e3 * Z.imag[band] / w[band], yerr=2e3 * se[band] / w[band],
                  label='Measured', **marker)
    if (~band).any():
        ax_b.errorbar(f[~band], 1e3 * Z.imag[~band] / w[~band], yerr=2e3 * se[~band] / w[~band],
                      label='Friction-dominated', **{**marker, 'markeredgecolor': MUTED,
                                                     'ecolor': MUTED})
    reference_line(ax_b, 1e3 * b_linear, color=MUTED, linestyle=':', linewidth=0.9,
                   label=f'Viscous $b$ ({b_source.split(" (")[0]})')
    ax_j.set_ylabel('Apparent inertia (kg$\\cdot$cm$^2$)')
    ax_b.set_ylabel('Apparent damping (mN$\\cdot$m$\\cdot$s/rad)')
    ax_b.set_xlabel('Frequency (Hz)')
    ax_b.set_xscale('log')
    ax_b.set_xlim(f.min() / 1.2, f.max() * 1.2)
    log_ticks_hz(ax_b)
    for ax in (ax_j, ax_b):
        style_axis(ax)
    j_shown = 1e4 * j_app[inertial]
    j_ref = 1e4 * fit['J_kg_m2']
    lo = min(np.nanmin(j_shown), j_ref) if j_shown.size else 0.8 * j_ref
    hi = max(np.nanmax(j_shown), j_ref) if j_shown.size else 1.2 * j_ref
    span = max(hi - lo, 0.05 * j_ref)
    ax_j.set_ylim(lo - 0.9 * span, hi + 0.35 * span)
    ax_j.legend(loc='lower right', fontsize=6.5)
    lo, hi = ax_b.get_ylim()
    ax_b.set_ylim(0.0, hi * 1.35)
    ax_b.legend(loc='upper right', fontsize=6.5)
    add_panel_label(ax_j, '(a)', x=-0.14)
    add_panel_label(ax_b, '(b)', x=-0.14)
    fig.align_ylabels([ax_j, ax_b])
    return fig


def plot_bode(results, free):
    n = len(results)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(2 * rows, cols, figsize=figure_size('double', 2.2 * rows + 0.4),
                             squeeze=False, sharex=True)
    for k, r in enumerate(results):
        ax_m, ax_p = axes[2 * (k // cols), k % cols], axes[2 * (k // cols) + 1, k % cols]
        f = r['freqs']
        if free is not None and r is not free:
            ax_m.loglog(f, np.abs(free['Z']), color=MUTED, linewidth=0.8, label='Near-free')
        ax_m.errorbar(f, np.abs(r['Z']), yerr=2 * r['se'], linestyle='None', marker='o',
                      markersize=2.2, markerfacecolor='white', markeredgecolor=INK, ecolor=INK,
                      elinewidth=0.5, label='Target')
        ax_m.set_xscale('log')
        ax_m.set_yscale('log')
        ax_p.semilogx(f, np.degrees(np.angle(r['Z'])), linestyle='None', marker='o',
                      markersize=2.2, markerfacecolor='white', markeredgecolor=INK)
        ax_m.set_title(target_label(r).replace('$', ''), fontsize=6.5, pad=2)
        for ax in (ax_m, ax_p):
            style_axis(ax, minor=False)
        if k % cols == 0:
            ax_m.set_ylabel('|Z| (N$\\cdot$m/rad)')
            ax_p.set_ylabel('Phase (deg)')
    for k in range(n, rows * cols):
        axes[2 * (k // cols), k % cols].axis('off')
        axes[2 * (k // cols) + 1, k % cols].axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Frequency (Hz)')
    legend_index = next((k for k, r in enumerate(results) if free is not None and r is not free), 0)
    axes[2 * (legend_index // cols), legend_index % cols].legend(loc='upper left', fontsize=6)
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    df = run.df
    targets = run.meta.get('targets') or []
    if not targets:
        raise SystemExit('No targets recorded in this run.')
    column = position_column(df)
    excite = df[df['phase'] == 'excite']

    cogging = latest_cogging_model(run.simulated)
    loop = loop_timing(run)
    results = []
    for t in targets:
        frame = excite[excite['target_id'] == t['target_id']] if 'target_id' in excite else excite.iloc[0:0]
        est = (estimate_target(frame, t['design'], column, cogging, loop, run_joint_index(run),
                               t.get('renderer'))
               if (not t.get('skipped') and len(frame)) else None)
        results.append({**t, **(est or {}), 'measured': est is not None})
    measured = [r for r in results if r['measured']]
    if not measured:
        raise SystemExit('No target was measured (all skipped or stopped).')

    # Near-free reference: the 500 Hz loop preferred, lowest K, B = 0.
    candidates = sorted([r for r in measured if r['b_nms_rad'] == 0],
                        key=lambda r: (r['renderer'] != 'rt', r['k_nm_rad']))
    free = candidates[0] if candidates and candidates[0]['k_nm_rad'] < FREE_K_MAX_NM_RAD else None
    # The reference's own spring reaches the joint through the loop delay (the
    # p4wrist CST ramp: 2 SYNC periods); K_c e^{-j w T} is what it renders, and
    # treating it as a delay-free K_c would bias every target by -K_c T of damping.
    ref_delay_s = (float(run.meta.get('loop_delay_s') or 0.0)
                   if free is not None and free['renderer'] == 'rt' else 0.0)
    if free is not None:
        free = dict(free)
        w_free = 2 * np.pi * free['freqs']
        k_c = free['k_nm_rad']
        # Device impedance plus a delay-free K_c, the form fit_free and the
        # line-by-line subtraction expect.
        free['Z'] = free['Z'] - k_c * np.exp(-1j * w_free * ref_delay_s) + k_c
    free_fit = fit_free(free['freqs'], free['Z'], free['se'], free['k_nm_rad']) if free else None
    if free_fit is None:
        found = latest_summary('step_response', run.simulated)
        J_ref = (float(found[1]['J_kg_m2']) if found and found[1].get('J_kg_m2')
                 else device_config.NOMINAL_JOINT_INERTIA_KG_M2)
        b_ref = (float(found[1]['b_nms_rad']) if found and found[1].get('b_nms_rad')
                 else device_config.NOMINAL_JOINT_DAMPING_NMS_RAD)
        reference_source = 'rigid model (step response or nominal): no near-free target'
    else:
        J_ref, b_ref = free_fit['J_kg_m2'], free_fit['b_nms_rad']
        reference_source = (f"near-free target {free['target_id']} (line-by-line subtraction"
                            + (f", its spring removed at the {1e3 * ref_delay_s:g} ms loop delay)"
                               if ref_delay_s else ')'))
    b_linear, b_source = linear_damping(run)
    damping_min = damping_band_min_hz(free) if free is not None else float(
        measured[0]['freqs'][0])

    rows = []
    for r in measured:
        f, Z, se = r['freqs'], r['Z'], r['se']
        w = 2 * np.pi * f
        se = np.where(np.isfinite(se), se, 0.0)
        if free is not None:
            Zr = Z - free['Z'] + free['k_nm_rad']
            se_r = np.sqrt(se ** 2 + np.where(np.isfinite(free['se']), free['se'], 0.0) ** 2)
        else:
            Zr = Z + J_ref * w ** 2 - 1j * w * b_ref
            se_r = se
        K, B = r['k_nm_rad'], r['b_nms_rad']
        Zd = K + 1j * w * B
        usable = (np.abs(Z) <= RESOLVE_RATIO * np.abs(Zd)) & (se_r <= MAX_REL_SE * np.abs(Zr))
        r.update({'Zr': Zr, 'se_r': se_r, 'usable': usable, 'fit': None})
        row = {'target_id': r['target_id'], 'mode': r['mode'], 'renderer': r['renderer'],
               'k_nm_rad': K, 'b_nms_rad': B, 'num_periods': r['num_periods'],
               'motion_rms_deg': r['motion_rms_deg'],
               'residual_fraction': r['residual_fraction'],
               'self_excited_flag': r['residual_fraction'] > RESIDUAL_FLAG,
               # Position mode (10) throughout: the controller never faulted or timed out.
               'mode_register_ok': bool((frame_modes(df, r['target_id']) == 10).all()),
               'usable_lines': int(usable.sum()),
               'resolvable_max_hz': float(f[usable].max()) if usable.any() else math.nan}
        if r is not free and usable.sum() >= 4:
            fit = fit_rendered(f[usable], Zr[usable], se_r[usable], K, B)
            if K < DELAY_MIN_K_NM_RAD:
                fit['delay_ms'] = 0.0
            measured_bw, measured_lower = fidelity_bandwidth(f[usable], Zr[usable], K, B)
            model_bw, model_lower = model_fidelity_bandwidth(fit, K, B, float(f[usable].max()))
            r['fit'] = fit
            row.update(fit)
            row.update({
                'active_rendering': bool(b_linear + fit['b_eff_nms_rad'] < 0.0),
                'stiffness_error_percent': 100.0 * (fit['k_eff_nm_rad'] - K) / K,
                'damping_shift_nms_rad': fit['b_eff_nms_rad'] - B,
                'fidelity_bandwidth_hz': model_bw,
                'fidelity_lower_bound': model_lower,
                'fidelity_bandwidth_measured_hz': measured_bw,
                'fidelity_measured_lower_bound': measured_lower,
                'delay_identifiable': K >= DELAY_MIN_K_NM_RAD,
            })
            if K < DELAY_MIN_K_NM_RAD:
                row['delay_ms'] = math.nan
        rows.append(row)
    table = pd.DataFrame(rows)

    # Effective delay and predicted stability limit per loop.
    summary = {'impedance_num_targets': len(targets),
               'impedance_num_measured': len(measured),
               'impedance_num_skipped': sum(1 for t in targets if t.get('skipped')),
               'impedance_position_source': column,
               'impedance_reference_source': reference_source,
               'impedance_b_linear_nms_rad': b_linear,
               'impedance_b_linear_source': b_source,
               'impedance_damping_band_min_hz': damping_min}
    if free_fit:
        summary.update({'impedance_free_target_id': free['target_id'],
                        'impedance_free_J_kg_m2': free_fit['J_kg_m2'],
                        'impedance_free_b_nms_rad': free_fit['b_nms_rad'],
                        'impedance_free_b_high_freq_nms_rad': free_fit['b_high_freq_nms_rad'],
                        'impedance_free_k_center_nm_rad': free_fit['k_center_nm_rad']})
    zw_run, zw = zwidth_reference(run)
    J_for_prediction = free_fit['J_kg_m2'] if free_fit else J_ref
    for mode, group in table.groupby('mode'):
        stiff = group[group['k_nm_rad'] >= DELAY_MIN_K_NM_RAD]
        if 'delay_ms' in stiff:
            stiff_delay = stiff.dropna(subset=['delay_ms'])
        else:
            stiff_delay = stiff.iloc[0:0]
        if len(stiff_delay):
            # The stiffest target resolves the delay best: K T dominates any friction
            # mismatch between its motion and the near-free reference.
            stiffest = stiff_delay.loc[stiff_delay['k_nm_rad'].idxmax()]
            delay = float(stiffest['delay_ms'])
            summary[f'impedance_delay_{mode}_ms'] = delay
            summary[f'impedance_delay_{mode}_target_id'] = int(stiffest['target_id'])
            k_crit = (predicted_k_crit(b_linear, 0.0, delay / 1000.0, J_for_prediction)
                      if delay > 0.05 else math.nan)   # No measurable delay: no linear limit
            if math.isfinite(k_crit):
                summary[f'impedance_predicted_k_crit_{mode}_b0_nm_rad'] = k_crit
            # Fidelity of the same (stiffest) target: the most demanding rendering.
            bw = stiffest.get('fidelity_bandwidth_hz', math.nan)
            if isinstance(bw, float) and math.isfinite(bw):
                summary[f'impedance_fidelity_bandwidth_{mode}_hz'] = float(bw)
                summary[f'impedance_fidelity_lower_bound_{mode}'] = bool(stiffest['fidelity_lower_bound'])
            if mode in zw:
                ref = zw[mode]
                for kind in ('k_max', 'k_first_unstable'):
                    value = ref.get(kind)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        summary[f'impedance_zwidth_{kind}_{mode}_b0_nm_rad'] = float(value)
                measured_limit = ref.get('k_first_unstable') or ref.get('k_max')
                if (isinstance(measured_limit, (int, float)) and math.isfinite(measured_limit)
                        and math.isfinite(k_crit) and k_crit > 0):
                    summary[f'impedance_zwidth_over_predicted_{mode}'] = measured_limit / k_crit
    for _, row in table.iterrows():
        i = int(row['target_id'])
        for key in ('mode', 'k_nm_rad', 'b_nms_rad', 'k_eff_nm_rad', 'k_static_nm_rad',
                    'b_eff_nms_rad', 'delay_ms', 'stiffness_error_percent',
                    'damping_shift_nms_rad', 'fidelity_bandwidth_hz', 'fidelity_lower_bound',
                    'fidelity_bandwidth_measured_hz', 'resolvable_max_hz', 'residual_fraction',
                    'active_rendering'):
            if key in row and not (isinstance(row[key], float) and math.isnan(row[key])):
                value = row[key]
                summary[f't{i}_{key}'] = value.item() if hasattr(value, 'item') else value
    for t in targets:
        if t.get('skipped'):
            summary[f"t{t['target_id']}_skipped"] = t['skipped']

    metadata = {
        'analysis': 'impedance_fidelity_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('per-period LS sinusoid fits of position at command-arrival times and exact '
                   'ZOH Fourier coefficients of the disturbance torque; Z = T/Theta; rendered '
                   'impedance by line-by-line subtraction of the near-free target'),
        'free_k_max_nm_rad': FREE_K_MAX_NM_RAD, 'delay_min_k_nm_rad': DELAY_MIN_K_NM_RAD,
        'resolve_ratio': RESOLVE_RATIO,
        'max_rel_se': MAX_REL_SE, 'fidelity_mag': FIDELITY_MAG,
        'fidelity_phase_deg': FIDELITY_PHASE_DEG,
        'zwidth_source': str(zw_run) if zw_run else None,
        'note': ('Torques are in the drive torque scale. Without an output encoder the motion '
                 'is motor-side; transmission compliance cancels in the subtraction but not in '
                 'the free-space inertia spectrum.'),
    }
    save_outputs(run, {'targets': table}, summary, metadata)
    spectra = []
    for r in measured:
        for k, f in enumerate(r['freqs']):
            spectra.append({'target_id': r['target_id'], 'frequency_hz': f,
                            'Z_real': r['Z'][k].real, 'Z_imag': r['Z'][k].imag,
                            'Z_se': r['se'][k], 'Zr_real': r['Zr'][k].real,
                            'Zr_imag': r['Zr'][k].imag, 'Zr_se': r['se_r'][k],
                            'usable': bool(r['usable'][k])})
    pd.DataFrame(spectra).to_csv(run.out_dir / 'impedance_spectra.csv', index=False)

    print_header('RENDERED IMPEDANCE SPECTRUM')
    print(f"Targets: {len(targets)} ({len(measured)} measured); motion from {column}; "
          f"reference: {reference_source}")
    if free_fit:
        print(f"  Free space (target {free['target_id']}): J = {free_fit['J_kg_m2']:.4g} kg m^2, "
              f"b = {free_fit['b_nms_rad']:.4g} N m s/rad (incl. friction-equivalent; "
              f"{free_fit['b_high_freq_nms_rad']:.4g} at high frequency)")
    for _, row in table.iterrows():
        if row.get('k_eff_nm_rad') is None or (isinstance(row.get('k_eff_nm_rad'), float)
                                               and math.isnan(row.get('k_eff_nm_rad'))):
            continue
        delay = row.get('delay_ms', math.nan)
        bw = row.get('fidelity_bandwidth_hz', math.nan)
        bw_text = ('--' if not isinstance(bw, float) or math.isnan(bw)
                   else (f'>= {bw:.1f}' if row.get('fidelity_lower_bound') else f'{bw:.1f}'))
        mbw = row.get('fidelity_bandwidth_measured_hz', math.nan)
        if isinstance(mbw, float) and not math.isnan(mbw):
            bw_text += f" (line-by-line {'>= ' if row.get('fidelity_measured_lower_bound') else ''}{mbw:.1f})"
        flag = '  [self-excited/nonlinear?]' if row['self_excited_flag'] else ''
        if row.get('active_rendering') is True:
            flag += ('  [ACTIVE: b + B_eff < 0, linearly unstable; only Coulomb friction kept '
                     'it from oscillating]')
        print(f"  [{int(row['target_id'])}] {row['mode']:<10} K={row['k_nm_rad']:<5g} "
              f"B={row['b_nms_rad']:<4g}: K_eff {row['k_static_nm_rad']:7.2f} "
              f"(fit {row['k_eff_nm_rad']:.2f}, {row['stiffness_error_percent']:+.1f}%), "
              f"B_eff {row['b_eff_nms_rad']:+.4f} "
              f"(shift {row['damping_shift_nms_rad']:+.4f}), delay "
              f"{'--' if isinstance(delay, float) and math.isnan(delay) else f'{delay:.2f} ms'}, "
              f"fidelity {bw_text} Hz{flag}")
    for t in targets:
        if t.get('skipped'):
            print(f"  [{t['target_id']}] {t['mode']} K={t['k_nm_rad']:g} B={t['b_nms_rad']:g}: "
                  f"skipped ({t['skipped']})")
    for key in sorted(summary):
        if key.startswith('impedance_predicted_k_crit_'):
            mode = key[len('impedance_predicted_k_crit_'):-len('_b0_nm_rad')]
            text = (f"  {mode}: delay {summary[f'impedance_delay_{mode}_ms']:.2f} ms -> linear "
                    f"stability limit K_crit ~ {summary[key]:.4g} N m/rad at B = 0 "
                    f"(b = {b_linear:.4g} from {b_source})")
            measured_limit = summary.get(f'impedance_zwidth_k_first_unstable_{mode}_b0_nm_rad')
            if measured_limit:
                text += (f"; Z-width first unstable {measured_limit:.4g} "
                         f"({summary[f'impedance_zwidth_over_predicted_{mode}']:.2f}x: Coulomb "
                         'friction quenches small oscillations)')
            print(text)
    for mode in table['mode'].unique():
        delay = summary.get(f'impedance_delay_{mode}_ms')
        if delay is not None and f'impedance_predicted_k_crit_{mode}_b0_nm_rad' not in summary:
            print(f'  {mode}: no measurable rendering delay ({delay:.2f} ms); no delay-induced '
                  'stability limit in the measured band')
    truth = sim_truth(run)
    if truth and free_fit:
        rows = [('Free-space inertia J', free_fit['J_kg_m2'],
                 truth['motor_inertia'] + truth['output_inertia'], 'kg m^2')]
        for t in targets:
            mode = t['mode']
            key = f'impedance_delay_{mode}_ms'
            delayed = (t['renderer'] == 'host' or loop_timing(run) is not None) \
                and t['renderer'] != 'drive'
            if delayed and key in summary and \
                    summary.get(f'impedance_delay_{mode}_target_id') == t['target_id']:
                expected = 1000.0 * (expected_delay_s(run, t) + truth['current_loop_time_constant'])
                rows.append((f'{mode} delay (approx. truth)', summary[key], expected, 'ms'))
        print_truth_comparison(rows)
        print(f"  (free-space damping {free_fit['b_nms_rad']:.4g} includes friction-equivalent "
              f"damping; viscous truth {truth['viscous'] + truth['output_viscous']:.4g})")

    paths = {}
    shown = choose_shown(measured)
    if shown:
        loop_delays = {mode: summary[f'impedance_delay_{mode}_ms'] for mode in table['mode'].unique()
                       if f'impedance_delay_{mode}_ms' in summary}
        fig = plot_spectrum(shown, damping_min, loop_delays)
        paths = export_figure(fig, run, SPECTRUM_BASENAME, {**metadata, **summary}, args.paper_dir)
    if free_fit:
        fig = plot_free(free, free_fit, b_linear, b_source, damping_min)
        export_figure(fig, run, FREE_BASENAME, {**metadata, **summary}, args.paper_dir)
    export_figure(plot_bode(measured, free), run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    if paths:
        print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
