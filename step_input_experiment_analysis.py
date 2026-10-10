"""Identify J, b, and fc from the P-only step responses.

Model (joint side, a = th - th_cmd):  J a'' + b a' + fc sign(a') = -Kp a + tau_bias

Between turning points (velocity zeros) the motion has one direction d_n, so
the model is linear and each half-cycle starting from rest at a_n ends at

    a_{n+1} = -r a_n - (1 + r) (fc/Kp) d_n + (1 + r) tau_bias/Kp,
    r = exp(-pi zeta / sqrt(1 - zeta^2)),

after exactly half a damped period pi/wd. A pooled linear regression over all
half-cycles of all steps gives r (viscous damping) and fc (Coulomb friction)
separately; the half-cycle durations give wd. Then

    wn = wd / sqrt(1 - zeta^2),  J = Kp / wn^2,  b = 2 J zeta wn.

Half-cycles are left out of the regression when the motor could not deliver the
commanded torque (back-EMF or current limiting, seen as reported torque falling
short of the command) or when they start inside the dead band where static
friction and cogging trap the joint, estimated from where the steps came to rest.

Plain logarithmic decrement attributes Coulomb decay to viscous damping, and
oscillation-center offsets are dominated by viscous decay, so both are biased
for real joints; this analysis avoids both. The identified model is replayed
on every step (piecewise-analytic solution) to report how well it fits.

Run:  python step_input_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

import device_config
from experiment_common import latest_summary
from analysis_common import (bootstrap_ci, correct_for_loop, export_figure, load_run,
                             loop_delays, make_parser, print_header, print_truth_comparison,
                             robust_sigma, save_outputs, sim_truth)
from ieee_plot import (INK, MUTED, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       figure_size, ordinal_colors, reference_line, style_axis)

EXPERIMENT = 'step_response'
LEGACY = [('step_response_data.csv', 'step_response_collection_metadata.json')]
FIGURE_BASENAME = 'fig_step_identification'
DIAGNOSTIC_BASENAME = 'diag_step_turning_points'
MIN_SWING_DEG = 0.02          # Turning-point hysteresis floor
NOISE_SWING_FACTOR = 8.0      # Hysteresis as a multiple of position noise
FIT_WINDOW_S = 1.5            # Window used for the model-fit metric and the figure
SHORTFALL_FRACTION = 0.2      # A half-cycle is torque-limited when the delivered torque
SHORTFALL_FLOOR_NM = 0.1      # falls short of the command by more than this fraction of
                              # its peak (and this floor): back-EMF or current limiting
DEADBAND_FACTOR = 4.0         # Half-cycles must start beyond this many median rest offsets


# --- data preparation ----------------------------------------------------------------
def step_segments(df):
    """Yield (segment_id, frame) for each step, newest data format or legacy."""
    if 'phase' in df.columns:
        steps = df[df['phase'] == 'step']
        yield from steps.groupby('segment', sort=True)
        return
    # Legacy: segments are command levels; a step is a segment whose command
    # differs from the previous one.
    previous = None
    for seg_id, frame in df.groupby('segment', sort=True):
        cmd = float(frame['cmd_joint_deg'].iloc[-1])
        if previous is not None and abs(cmd - previous) > 1.0:
            frame = frame.copy()
            frame['step_size_deg'] = abs(cmd - previous)
            frame['prev_target_deg'] = previous
            yield seg_id, frame
        previous = cmd


def uniform(frame):
    t = frame['time'].to_numpy()
    t = t - t[0]
    y = np.radians(frame['joint_pos_deg'].to_numpy() - frame['cmd_joint_deg'].to_numpy())
    fs = 1.0 / np.median(np.diff(t))
    tu = np.arange(0.0, t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, y), fs


def turning_points(x, delta):
    """Alternating extrema whose swing exceeds delta, including the first sample
    and a final resting point reached after the last reversal."""
    points = []
    imax = imin = 0
    looking = 0          # 0: unknown, +1: next is a max, -1: next is a min
    for i in range(1, len(x)):
        if x[i] > x[imax]:
            imax = i
        if x[i] < x[imin]:
            imin = i
        if looking >= 0 and x[imax] - x[i] > delta:
            points.append(imax)
            looking = -1
            imin = i
        elif looking <= 0 and x[i] - x[imin] > delta:
            points.append(imin)
            looking = 1
            imax = i
    if points:
        # Final half-cycle that ended at rest without another reversal: place
        # it where the motion stopped, not at the noise extreme of the tail.
        last = points[-1]
        if looking == -1 and x[last] - x[imin] > delta and imin > last:
            points.append(next(i for i in range(last, len(x)) if x[i] <= x[imin] + 0.25 * delta))
        elif looking == 1 and x[imax] - x[last] > delta and imax > last:
            points.append(next(i for i in range(last, len(x)) if x[i] >= x[imax] - 0.25 * delta))
    return points


def refine(t, x, i):
    """Parabolic interpolation of an extremum at sample i."""
    if i <= 0 or i >= len(x) - 1:
        return t[i], x[i]
    y0, y1, y2 = x[i - 1], x[i], x[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return t[i], x[i]
    shift = 0.5 * (y0 - y2) / denom
    shift = max(-1.0, min(1.0, shift))
    dt = t[1] - t[0]
    return t[i] + shift * dt, y1 - 0.25 * (y0 - y2) * shift


def extract_half_cycles(seg_id, frame, kp, nominal_period):
    tu, a, fs = uniform(frame)
    if len(a) < 20:
        return None
    window = max(5, int(round(0.03 * nominal_period * fs)) | 1)
    a_s = signal.savgol_filter(a, window, 2) if window < len(a) else a
    tail = a[int(0.8 * len(a)):]
    noise = robust_sigma(np.diff(tail)) / math.sqrt(2.0) if len(tail) > 10 else 0.0
    delta = max(math.radians(MIN_SWING_DEG), NOISE_SWING_FACTOR * noise)
    idx = turning_points(a_s, delta)
    if len(idx) < 3:
        return None
    times, values = zip(*(refine(tu, a_s, i) for i in idx))
    times, values = np.array(times), np.array(values)
    # The first point is the rest position before the step; use the raw mean
    # of the first few samples (the smoother distorts the corner at t = 0).
    values[0] = float(np.mean(a[:max(3, int(0.003 * fs))]))
    times[0] = 0.0
    cap = frame['max_torque_nm'].iloc[0] if 'max_torque_nm' in frame else math.inf
    t_raw = frame['time'].to_numpy() - frame['time'].iloc[0]
    has_torque = {'control_joint_torque_nm', 'joint_torque_nm'} <= set(frame.columns)
    if has_torque:
        commanded = frame['control_joint_torque_nm'].to_numpy(float)
        delivered = frame['joint_torque_nm'].to_numpy(float)
    rows = []
    for n in range(len(values) - 1):
        a_n, a_next = values[n], values[n + 1]
        shortfall, limited = math.nan, False
        window = (t_raw >= times[n]) & (t_raw <= times[n + 1])
        if has_torque and window.any():
            shortfall = float(np.nanmax(np.abs(commanded[window] - delivered[window])))
            peak = float(np.nanmax(np.abs(commanded[window])))
            limited = shortfall > max(SHORTFALL_FRACTION * peak, SHORTFALL_FLOOR_NM)
        rows.append({
            'segment': seg_id,
            'step_size_deg': float(frame['step_size_deg'].iloc[0]),
            'n': n,
            'a_n_rad': a_n,
            'a_next_rad': a_next,
            'direction': float(np.sign(a_next - a_n)),
            'half_period_s': times[n + 1] - times[n],
            'last': n == len(values) - 2,
            'saturated': kp * abs(a_n) > 0.98 * cap,
            'torque_shortfall_nm': shortfall,
            'torque_limited': limited,
            't_n': times[n],
        })
    return {'rows': rows, 'times': times, 'values': values, 'tu': tu, 'a': a,
            'noise_rad': noise, 'delta_rad': delta}


# --- identification ---------------------------------------------------------------
def identify(half_cycles, kp, kd=0.0, delays=None):
    """Pooled regression a_next = b0 + b1 a_n + b2 d_n and half-period median.

    The ring-down shows the loop (spring Kp, damping b + Kd); J and b are then
    corrected for Kd and the loop delays (analysis_common.correct_for_loop)."""
    excluded = half_cycles['saturated']
    for flag in ('torque_limited', 'in_deadband'):
        if flag in half_cycles:
            excluded = excluded | half_cycles[flag]
    hc = half_cycles[~excluded]
    if len(hc) < 4:
        return None
    A = np.column_stack([np.ones(len(hc)), hc['a_n_rad'], hc['direction']])
    coef, *_ = np.linalg.lstsq(A, hc['a_next_rad'].to_numpy(), rcond=None)
    b0, b1, b2 = coef
    r = -b1
    if not 0.0 < r < 1.0:
        return None
    full = hc[~hc['last']]
    half_period = float(np.median((full if len(full) else hc)['half_period_s']))
    wd = math.pi / half_period
    ln_r = -math.log(r)
    zeta = ln_r / math.sqrt(math.pi ** 2 + ln_r ** 2)
    wn = wd / math.sqrt(1.0 - zeta ** 2)
    J_app = kp / wn ** 2
    b_app = 2.0 * J_app * zeta * wn
    J, b = correct_for_loop(J_app, b_app, kp, kd, delays or {'position': 0.0, 'velocity': 0.0})
    residual = hc['a_next_rad'].to_numpy() - A @ coef
    return {
        'J_kg_m2': J,
        'b_nms_rad': b,
        'J_apparent_kg_m2': J_app,
        'b_apparent_nms_rad': b_app,
        'zeta': zeta,
        'wn_rad_s': wn,
        'fn_hz': wn / (2 * math.pi),
        'wd_rad_s': wd,
        'r': r,
        'fc_nm': -b2 * kp / (1.0 + r),
        'bias_torque_nm': b0 * kp / (1.0 + r),
        'regression_rms_deg': math.degrees(float(np.sqrt(np.mean(residual ** 2)))),
        'num_half_cycles': int(len(hc)),
        'num_steps': int(hc['segment'].nunique()),
    }


# --- model replay -------------------------------------------------------------------
def simulate(a0, t, est, kp, static_nm):
    """Piecewise-analytic response from rest at a0 (joint rad, relative to cmd)."""
    zeta, wn = est['zeta'], est['wn_rad_s']
    wd = wn * math.sqrt(1 - zeta ** 2)
    sigma = zeta * wn
    half = math.pi / wd
    fc, bias, fs = est['fc_nm'] / kp, est['bias_torque_nm'] / kp, static_nm / kp
    out = np.empty_like(t)
    t0, an = 0.0, a0
    while True:
        mask = t >= t0
        net = bias - an
        if abs(net) <= fs or not mask.any():
            out[mask] = an
            break
        d = math.copysign(1.0, net)
        a_eq = bias - fc * d
        seg = mask & (t < t0 + half)
        tau = t[seg] - t0
        out[seg] = a_eq + (an - a_eq) * np.exp(-sigma * tau) * (
            np.cos(wd * tau) + sigma / wd * np.sin(wd * tau))
        an = a_eq - est['r'] * (an - a_eq)
        t0 += half
        if t0 > t[-1]:
            break
    return out


def fit_percent(measured, model):
    denom = np.linalg.norm(measured - np.mean(measured))
    return 100.0 * (1.0 - np.linalg.norm(measured - model) / denom) if denom > 0 else math.nan


def latest_static_friction(simulated):
    """Breakaway torque from the newest ramp analysis of the same kind (real/sim)."""
    found = latest_summary('friction_ramp', simulated)
    if found is None:
        return None
    value = found[1].get('static_friction_nm')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


# --- figures ---------------------------------------------------------------------------
def plot_identification(steps, half_cycles, est, kp, static_nm, sizes):
    fig, (ax_t, ax_m) = plt.subplots(2, 1, figsize=figure_size('single', 3.9),
                                     gridspec_kw={'height_ratios': [1.15, 1.0], 'hspace': 0.42})
    colors = dict(zip(sizes, ordinal_colors(len(sizes))))
    largest = sizes[-1]

    # (a) largest steps, normalized, with the identified model for one of them.
    shown = [s for s in steps if s['size'] == largest]
    for s in shown:
        mask = s['tu'] <= FIT_WINDOW_S
        scale = -s['a'][0] if s['a'][0] != 0 else 1.0
        ax_t.plot(s['tu'][mask], 1.0 + s['a'][mask] / scale, color=MUTED, linewidth=0.6)
    example = shown[0]
    mask = example['tu'] <= FIT_WINDOW_S
    scale = -example['a'][0]
    model = simulate(example['a'][0], example['tu'][mask], est, kp, static_nm)
    ax_t.plot(example['tu'][mask], 1.0 + example['a'][mask] / scale, color=INK,
              linewidth=0.9, label='Measured')
    ax_t.plot(example['tu'][mask], 1.0 + model / scale, color='#2a78d6', linestyle='--',
              linewidth=1.0, label='Identified model')
    reference_line(ax_t, 1.0)
    ax_t.set_xlim(0, FIT_WINDOW_S)
    ax_t.set_xlabel('Time since step (s)')
    ax_t.set_ylabel('Normalized position')
    style_axis(ax_t)
    ax_t.legend(loc='upper right', ncol=2)
    add_panel_label(ax_t, '(a)')

    # (b) amplitude lost per half-cycle vs amplitude: the intercept is the
    # Coulomb contribution, the slope the viscous one.
    hc = half_cycles[~half_cycles['saturated']]
    crossing = hc[np.sign(hc['a_n_rad']) != np.sign(hc['a_next_rad'])]
    for i, size in enumerate(sizes):
        group = crossing[crossing['step_size_deg'] == size]
        amp = np.degrees(group['a_n_rad'].abs())
        loss = amp - np.degrees(group['a_next_rad'].abs())
        ax_m.plot(amp, loss, linestyle='None', marker='os^'[i % 3], markersize=3.2,
                  markerfacecolor=colors[size] if i % 2 else 'white',
                  markeredgecolor=colors[size], label=f'{size:g}\u00b0 steps')
    amax = np.degrees(crossing['a_n_rad'].abs().max()) * 1.05
    x = np.linspace(0, amax, 50)
    coulomb_deg = math.degrees((1 + est['r']) * est['fc_nm'] / kp)
    ax_m.plot(x, (1 - est['r']) * x + coulomb_deg, color=INK, linestyle='--', linewidth=1.0,
              label=f"Fit ($f_c$ = {est['fc_nm']:.3f} {TORQUE_UNIT})")
    ax_m.plot(x, (1 - est['r']) * x, color=MUTED, linestyle=':', linewidth=0.9,
              label='Viscous part of fit')
    ax_m.set_xlim(0, amax)
    # Headroom so the two-column legend sits above the data at every amplitude.
    ax_m.set_ylim(0, ((1 - est['r']) * amax + coulomb_deg) * 1.6)
    ax_m.set_xlabel('Turning-point amplitude $|a_n|$ (deg)')
    ax_m.set_ylabel('Loss per half-cycle (deg)')
    style_axis(ax_m)
    ax_m.legend(loc='upper left', ncol=2, columnspacing=1.0)
    add_panel_label(ax_m, '(b)')
    return fig


def plot_diagnostics(steps):
    n = len(steps)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figure_size('double', 1.3 * rows + 0.4),
                             squeeze=False, sharex=True)
    for ax, s in zip(axes.flat, steps):
        ax.plot(s['tu'], np.degrees(s['a']), color=INK, linewidth=0.6)
        ax.plot(s['times'], np.degrees(s['values']), linestyle='None', marker='o',
                markerfacecolor='none', markeredgecolor='#eb6834', markersize=3)
        ax.text(0.98, 0.9, f"seg {s['segment']}, {s['size']:g}°", transform=ax.transAxes,
                ha='right', va='top', fontsize=6)
        style_axis(ax, minor=False)
    for ax in axes.flat[n:]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Time since step (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Error (deg)')
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data, LEGACY)
    df = run.df
    kp = float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad'))
    kd = float(df.loc[df['phase'] == 'step', 'kd_joint_nms_rad'].median()) \
        if 'kd_joint_nms_rad' in df else 0.0
    delays = loop_delays(run)
    nominal_period = 2 * math.pi * math.sqrt(device_config.NOMINAL_JOINT_INERTIA_KG_M2 / kp)

    steps, rows = [], []
    for seg_id, frame in step_segments(df):
        result = extract_half_cycles(seg_id, frame, kp, nominal_period)
        if result is None:
            print(f'Segment {seg_id}: too few turning points; skipped.')
            continue
        rows.extend(result['rows'])
        steps.append({'segment': seg_id, 'size': float(frame['step_size_deg'].iloc[0]), **result})
    if not rows:
        raise SystemExit('No usable step responses (need an underdamped ring-down).')
    half_cycles = pd.DataFrame(rows)
    # Dead band: where static friction and cogging trap the joint. The steps come to
    # rest inside it, so the median rest offset is about half its width.
    rest = np.array([abs(s['values'][-1]) for s in steps])
    deadband = float(2.0 * np.median(rest)) if len(rest) else 0.0
    min_start = 0.5 * DEADBAND_FACTOR * deadband
    half_cycles['in_deadband'] = half_cycles['a_n_rad'].abs() < min_start
    est = identify(half_cycles, kp, kd, delays)
    if est is None:
        raise SystemExit('Identification failed: decay ratio outside (0, 1).')

    static_nm = latest_static_friction(run.simulated)
    static_for_model = static_nm if static_nm else est['fc_nm']

    # Bootstrap over whole steps (all parameters from the same resamples).
    by_step = [g for _, g in half_cycles.groupby('segment')]
    ci = bootstrap_ci(by_step, lambda gs: identify(pd.concat(gs), kp, kd, delays),
                      keys=('J_kg_m2', 'b_nms_rad', 'zeta', 'fc_nm', 'fn_hz'))

    # Per step size: amplitude dependence check.
    sizes = sorted(half_cycles['step_size_deg'].unique())
    per_size = []
    for size in sizes:
        e = identify(half_cycles[half_cycles['step_size_deg'] == size], kp, kd, delays)
        if e:
            per_size.append({'step_size_deg': size, **e})
    per_size = pd.DataFrame(per_size)

    # Replay the model on every step between the end of any torque-limited swing and the
    # entry into the dead band: the replay has neither back-EMF nor cogging. It starts
    # from rest at a turning point, so a later start is a valid initial condition.
    per_step = []
    for s in steps:
        rows_s = half_cycles[half_cycles['segment'] == s['segment']]
        trapped = rows_s.loc[rows_s['in_deadband'], 't_n']
        end = min(FIT_WINDOW_S, float(trapped.min())) if len(trapped) else FIT_WINDOW_S
        limited = rows_s[rows_s['torque_limited']]
        start_n = int(limited['n'].max()) + 1 if len(limited) else 0
        start = float(s['times'][start_n]) if start_n < len(s['times']) else end
        mask = (s['tu'] >= start) & (s['tu'] <= end)
        if mask.sum() < 10:
            continue
        a0 = float(s['values'][start_n]) if start_n else s['a'][0]
        model = simulate(a0, s['tu'][mask] - start, est, kp, static_for_model)
        per_step.append({'segment': s['segment'], 'step_size_deg': s['size'],
                         'num_turning_points': len(s['values']),
                         'fit_window_s': (start, end),
                         'fit_percent': fit_percent(s['a'][mask], model),
                         'noise_deg': math.degrees(s['noise_rad'])})
    per_step = pd.DataFrame(per_step)

    summary = {
        **est,
        **{f'{k}_ci95_low': v[0] for k, v in ci.items()},
        **{f'{k}_ci95_high': v[1] for k, v in ci.items()},
        'kp_joint_nm_rad': kp,
        'kd_joint_nms_rad': kd,
        'loop_delay_position_s': delays['position'],
        'loop_delay_velocity_s': delays['velocity'],
        'static_friction_used_in_replay_nm': static_for_model,
        'model_fit_percent_median': float(per_step['fit_percent'].median()),
        'model_fit_percent_min': float(per_step['fit_percent'].min()),
        'deadband_deg': math.degrees(deadband),
        'min_start_amplitude_deg': math.degrees(min_start),
        'num_half_cycles_torque_limited': int(half_cycles['torque_limited'].sum()),
        'num_half_cycles_in_deadband': int(half_cycles['in_deadband'].sum()),
    }
    metadata = {
        'analysis': 'step_input_experiment_analysis.py',
        'source_data': str(run.source),
        'legacy_input': run.legacy,
        'method': 'pooled turning-point regression (viscous + Coulomb), bootstrap over steps',
        'min_swing_deg': MIN_SWING_DEG,
        'noise_swing_factor': NOISE_SWING_FACTOR,
        'fit_window_s': FIT_WINDOW_S,
        'shortfall_fraction': SHORTFALL_FRACTION,
        'shortfall_floor_nm': SHORTFALL_FLOOR_NM,
        'deadband_factor': DEADBAND_FACTOR,
        'note': ('J, b and fc are in the torque scale the drives report; multiply by the '
                 'torque calibration ratio if one was measured.'),
    }
    save_outputs(run, {'half_cycles': half_cycles, 'per_step': per_step,
                       'per_size': per_size}, summary, metadata)

    print_header('STEP RESPONSE IDENTIFICATION')
    print(f"Steps used: {est['num_steps']}  half-cycles: {est['num_half_cycles']}  "
          f"Kp = {kp:g} N m/rad")
    print(f"  Left out: {summary['num_half_cycles_torque_limited']} torque-limited half-cycles "
          f"(delivered torque short of the command), "
          f"{summary['num_half_cycles_in_deadband']} starting inside "
          f"{summary['min_start_amplitude_deg']:.3g} deg (dead band ~{summary['deadband_deg']:.3g} deg)")

    def line(label, key, unit, fmt='.4g'):
        lo, hi = ci.get(key, (math.nan, math.nan))
        print(f'  {label:<26} {est[key]:{fmt}} {unit}   95% CI [{lo:{fmt}}, {hi:{fmt}}]')

    line('Inertia J', 'J_kg_m2', 'kg m^2')
    line('Viscous damping b', 'b_nms_rad', 'N m s/rad')
    line('Damping ratio', 'zeta', '')
    line('Natural frequency', 'fn_hz', 'Hz')
    line('Coulomb friction fc', 'fc_nm', 'N m')
    print(f"  {'Bias torque':<26} {est['bias_torque_nm']:.4g} N m")
    print(f"  Model fit (median over steps): {summary['model_fit_percent_median']:.1f}%")
    if len(per_size) > 1:
        print('  Per step size (amplitude dependence):')
        for _, r in per_size.iterrows():
            print(f"    {r['step_size_deg']:5.1f} deg: J {r['J_kg_m2']:.4g}  b {r['b_nms_rad']:.4g}"
                  f"  fc {r['fc_nm']:.4g}")
    truth = sim_truth(run)
    if truth:
        rigid = not math.isfinite(truth['transmission_stiffness'])
        print_truth_comparison([
            ('Inertia J', est['J_kg_m2'], truth['motor_inertia'] + truth['output_inertia'],
             'kg m^2'),
            ('Viscous b', est['b_nms_rad'], truth['viscous'] + truth['output_viscous'],
             'N m s/rad'),
            ('Coulomb fc', est['fc_nm'], truth['coulomb'], 'N m'),
        ])
        if not rigid:
            print('  (two-mass plant: J and b are the rigid-body equivalents)')

    fig = plot_identification(steps, half_cycles, est, kp, static_for_model, sizes)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    diag = plot_diagnostics(steps)
    export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
