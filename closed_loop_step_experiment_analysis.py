"""Closed-loop step metrics and dynamic capability (stiff PD).

Per step (normalized response s = (y - y0) / (target - y0)):
* 10-90% rise time, overshoot, peak time;
* settling time into a 2% band of the step around the final value (friction and
  cogging leave a steady-state error, reported separately) and into an absolute
  band around the target;
* steady-state error (mean of the last 20% of the hold);
* peak velocity (the loop's filtered estimate), peak acceleration (Savitzky-Golay
  derivative of velocity), peak torque, peak mechanical power;
* torque saturation: fraction of samples with the controller torque at the
  cap, and the acceleration the joint achieves per unit of torque over the
  first motoring interval at the cap (after a 2 ms torque rise). Position is
  fitted as theta0 + v0 t + c * double integral of the delivered (reported)
  torque, so back-EMF, which lets the delivered torque fall below the command
  as speed rises, does not masquerade as missing acceleration; position is used
  because the motor-side velocity rings with the cable for the first few
  milliseconds. Friction from the newest friction-sweep analysis (Coulomb plus
  viscous) is subtracted from the delivered torque first, because near the
  no-load speed it is a large part of what little torque remains. 1/c is a
  large-signal inertia, and cap * c, the acceleration at
  the cap, is compared with cap / J. The speed at which the delivered torque
  first drops below 95% of the cap marks the back-EMF knee.

Results are aggregated per step size (mean and standard deviation).

Run:  python closed_loop_step_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

import device_config
from analysis_common import (export_figure, load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, savgol_derivative, sim_truth)
from experiment_common import latest_summary
from ieee_plot import (GRID, INK, SERIES, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       direct_label, figure_size, log_ticks_hz, ordinal_colors, reference_line,
                       series_style, style_axis)

EXPERIMENT = 'closed_loop_step'
LEGACY = []
FIGURE_BASENAME = 'fig_closed_loop_steps'
DIAGNOSTIC_BASENAME = 'diag_closed_loop_step_kinematics'
SETTLING_FRACTION = 0.02         # Relative settling band
SETTLING_BAND_DEG = 0.1          # Absolute settling band
SATURATION_THRESHOLD = 0.98      # |controller torque| >= this * cap counts as saturated
MIN_SATURATED_S = 0.003          # Shorter saturation is not counted as a saturated step
ACCEL_WINDOW_S = 0.006           # Savitzky-Golay window for the acceleration estimate
ACCEL_FIT_SKIP_S = 0.002         # Skip the torque rise at saturation onset
ACCEL_FIT_MIN_S = 0.010          # Shorter motoring intervals are not fitted (5 samples at 500 Hz)
KNEE_FRACTION = 0.95             # Delivered torque below this fraction of the cap = derated


def first_crossing(t, s, level):
    above = np.flatnonzero(s >= level)
    if len(above) == 0:
        return math.nan
    i = above[0]
    if i == 0:
        return float(t[0])
    frac = (level - s[i - 1]) / (s[i] - s[i - 1]) if s[i] != s[i - 1] else 0.0
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def settling_time(t, error, band):
    outside = np.flatnonzero(np.abs(error) > band)
    if len(outside) == 0:
        return 0.0
    i = outside[-1]
    if i >= len(t) - 1:
        return math.nan        # Never settled within the hold
    return float(t[i + 1])


def saturated_runs(mask):
    """(start, stop) index pairs of contiguous True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fit_torque_acceleration(t, theta, torque):
    """Least squares theta = theta0 + v0 t + c * double integral of torque; returns c,
    the acceleration per unit of delivered torque (1 / large-signal inertia)."""
    t = t - t[0]
    velocity_like = cumulative_trapezoid(torque, t, initial=0.0)
    position_like = cumulative_trapezoid(velocity_like, t, initial=0.0)
    A = np.column_stack([np.ones_like(t), t, position_like])
    coef, *_ = np.linalg.lstsq(A, theta, rcond=None)
    return float(coef[2])


def sweep_friction(simulated):
    """(Coulomb N m, viscous N m s/rad, source) from the newest friction-sweep analysis."""
    found = latest_summary('friction_sweep', simulated)
    if found is not None:
        run_dir, summary = found
        fc, b = summary.get('coulomb_friction_nm'), summary.get('viscous_damping_nms_rad')
        if isinstance(fc, (int, float)) and isinstance(b, (int, float)):
            return float(fc), float(b), f'friction_sweep {run_dir.name}'
    return 0.0, 0.0, 'none (no friction-sweep analysis)'


def analyze_step(frame, cap, friction=(0.0, 0.0)):
    t = frame['t_local'].to_numpy()
    t = t - t[0]
    y = frame['joint_pos_deg'].to_numpy()
    target = float(frame['cmd_joint_deg'].iloc[0])
    y0 = float(y[0])
    step = target - y0
    if abs(step) < 1e-6:
        return None
    s = (y - y0) / step
    vel = frame['joint_vel_deg_s'].to_numpy()
    torque = frame['joint_torque_nm'].to_numpy()
    control = frame['control_joint_torque_nm'].to_numpy()
    w = np.radians(vel)
    accel = savgol_derivative(t, w, ACCEL_WINDOW_S, polyorder=2)
    dt = float(np.median(np.diff(t)))
    direction = math.copysign(1.0, step)

    saturated = np.abs(control) >= SATURATION_THRESHOLD * cap
    runs = [(a, b) for a, b in saturated_runs(saturated)
            if (b - a) * dt >= MIN_SATURATED_S]
    sat_accel = mean_sat_accel = per_torque = knee_speed = math.nan
    motoring = [(a, b) for a, b in runs if math.copysign(1.0, control[a]) == direction]
    if motoring:
        a, b = motoring[0]
        ts, ws = t[a:b], w[a:b] * direction
        if ts[-1] - ts[0] > 0:
            mean_sat_accel = float((ws[-1] - ws[0]) / (ts[-1] - ts[0]))
        delivered = torque[a:b] * direction
        derated = np.flatnonzero((ts >= ts[0] + ACCEL_FIT_SKIP_S)
                                 & (delivered < KNEE_FRACTION * cap))
        coulomb, viscous = friction
        net = delivered - (coulomb * np.sign(ws) + viscous * ws)
        if derated.size:
            knee_speed = float(abs(vel[a + derated[0]]))
        window = ts >= ts[0] + ACCEL_FIT_SKIP_S
        if window.sum() >= 5 and ts[window][-1] - ts[window][0] >= ACCEL_FIT_MIN_S:
            theta = np.radians(y[a:b]) * direction
            per_torque = fit_torque_acceleration(ts[window], theta[window], net[window])
            sat_accel = cap * per_torque

    i_peak = int(np.argmax(s))
    tail = t >= 0.8 * t[-1]
    s_final = float(np.mean(s[tail]))
    return {
        'step_size_deg': float(frame['step_size_deg'].iloc[0]),
        'direction': int(direction),
        'rise_time_ms': 1000.0 * (first_crossing(t, s, 0.9) - first_crossing(t, s, 0.1)),
        'overshoot_percent': 100.0 * max(0.0, float(s.max()) - 1.0),
        'peak_time_ms': 1000.0 * float(t[i_peak]),
        'settling_time_ms': 1000.0 * settling_time(t, s - s_final, SETTLING_FRACTION),
        'settling_time_band_ms': 1000.0 * settling_time(t, y - target, SETTLING_BAND_DEG),
        'steady_state_error_deg': float(np.mean(y[tail] - target)),
        'peak_velocity_deg_s': float(np.max(np.abs(vel))),
        'peak_acceleration_rad_s2': float(np.nanmax(np.abs(accel))),
        'saturated_acceleration_rad_s2': sat_accel,
        'large_signal_inertia_kg_m2': 1.0 / per_torque if per_torque > 0 else math.nan,
        'torque_knee_speed_deg_s': knee_speed,
        'mean_saturated_acceleration_rad_s2': mean_sat_accel,
        'peak_torque_nm': float(np.max(np.abs(torque))),
        'peak_power_w': float(np.max(np.abs(torque * w))),
        'saturation_fraction': float(saturated.mean()),
        'saturated_duration_ms': 1000.0 * sum((b - a) for a, b in runs) * dt,
        'saturated': bool(runs),
        '_t': t, '_s': s, '_vel': vel, '_accel': accel, '_torque': torque,
        '_control': control,
    }


def reference_inertia(simulated):
    """J from the newest step_response analysis of the same kind (sim or real)."""
    found = latest_summary('step_response', simulated)
    if found is not None:
        run_dir, summary = found
        if summary.get('J_kg_m2'):
            return float(summary['J_kg_m2']), f'step_response analysis ({run_dir.name})'
    return device_config.NOMINAL_JOINT_INERTIA_KG_M2, 'device_config nominal'


# --- figures -----------------------------------------------------------------------------
def plot_steps(steps, per_size, sizes):
    fig, (ax_s, ax_t) = plt.subplots(2, 1, figsize=figure_size('single', 3.9),
                                     gridspec_kw={'height_ratios': [1.25, 1.0], 'hspace': 0.45})
    colors = dict(zip(sizes, ordinal_colors(len(sizes))))
    settle = per_size['settling_time_ms_mean'].replace([np.inf], np.nan)
    t_show = 1.3 * float(np.nanmax(settle)) / 1000.0 if settle.notna().any() else 0.3
    t_show = min(max(t_show, 0.05), max(s['_t'][-1] for s in steps))
    for size in sizes:
        group = [s for s in steps if s['step_size_deg'] == size]
        # Representative step: the one with the median settling time.
        order = sorted(group, key=lambda s: (np.nan_to_num(s['settling_time_ms'], nan=1e9)))
        rep = order[len(order) // 2]
        mask = rep['_t'] <= t_show
        label = f"{size:g}°" + (' (saturated)' if rep['saturated'] else '')
        ax_s.plot(1000.0 * rep['_t'][mask], rep['_s'][mask], color=colors[size], linewidth=1.0,
                  label=label)
    reference_line(ax_s, 1.0)
    ax_s.axhspan(1 - SETTLING_FRACTION, 1 + SETTLING_FRACTION, facecolor=GRID, edgecolor='none',
                 zorder=0, label=f'±{100 * SETTLING_FRACTION:g}% band')
    ax_s.set_xlim(0, 1000.0 * t_show)
    ax_s.set_xlabel('Time since step (ms)')
    ax_s.set_ylabel('Normalized position')
    style_axis(ax_s)
    ax_s.legend(loc='lower right', ncol=2, fontsize=6.5)
    add_panel_label(ax_s, '(a)', x=-0.12)

    x = per_size['step_size_deg'].to_numpy()
    for i, (key, label) in enumerate([('rise_time_ms', '10–90% rise time'),
                                      ('settling_time_ms', f'{100 * SETTLING_FRACTION:g}% settling time')]):
        ax_t.errorbar(x, per_size[f'{key}_mean'], yerr=per_size[f'{key}_std'].fillna(0.0),
                      label=label, elinewidth=0.6, **series_style(i))
    ax_t.set_xscale('log')
    ax_t.set_xticks(x)
    ax_t.set_xticklabels([f'{v:g}' for v in x])
    ax_t.minorticks_off()
    ax_t.set_yscale('log')
    values = np.concatenate([per_size['rise_time_ms_mean'], per_size['settling_time_ms_mean']])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size:
        ax_t.set_ylim(values.min() / 1.8, values.max() * 5.0)   # Headroom for the legend
    log_ticks_hz(ax_t, axis='y')   # Plain-number labels on the log axis
    ax_t.set_xlabel('Step size (deg)')
    ax_t.set_ylabel('Time (ms)')
    style_axis(ax_t, minor=False)
    ax_t.legend(loc='upper left')
    add_panel_label(ax_t, '(b)', x=-0.12)
    return fig


def plot_kinematics(step, cap, accel_capability):
    t = 1000.0 * step['_t']
    mask = step['_t'] <= min(step['_t'][-1], 0.25)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size('single', 4.0),
                             gridspec_kw={'hspace': 0.12})
    axes[0].plot(t[mask], step['_vel'][mask], color=INK, linewidth=0.9)
    axes[0].set_ylabel('Velocity (deg/s)')
    axes[1].plot(t[mask], step['_accel'][mask], color=INK, linewidth=0.9)
    reference_line(axes[1], accel_capability)
    reference_line(axes[1], -accel_capability)
    direct_label(axes[1], t[mask][-1], accel_capability, 'cap / J', dx=0, dy=3, ha='right',
                 va='bottom')
    axes[1].set_ylabel('Accel. (rad/s$^2$)')
    lim = 1.35 * max(accel_capability, float(np.nanmax(np.abs(step['_accel'][mask]))))
    axes[1].set_ylim(-lim, lim)
    axes[2].plot(t[mask], step['_torque'][mask], color=INK, linewidth=0.9, label='Measured')
    axes[2].plot(t[mask], step['_control'][mask], color=SERIES[0], linestyle='--', linewidth=0.9,
                 label='Controller')
    reference_line(axes[2], cap)
    reference_line(axes[2], -cap)
    axes[2].set_ylabel(f'Torque ({TORQUE_UNIT})')
    axes[2].set_xlabel('Time since step (ms)')
    axes[2].legend(loc='upper right', fontsize=6.5)
    for ax, label in zip(axes, ('(a)', '(b)', '(c)')):
        style_axis(ax)
        add_panel_label(ax, label, x=-0.14)
    fig.align_ylabels(axes)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data, LEGACY)
    df = run.df
    kp = float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad'))
    kd = float(run.meta_value('kd_joint_nms_rad', column='kd_joint_nms_rad'))
    cap = float(run.meta.get('torque_cap_nm') or df['max_torque_nm'].dropna().iloc[0])

    coulomb, viscous, friction_source = sweep_friction(run.simulated)
    steps = []
    for _, frame in df[df['phase'] == 'step'].groupby('segment', sort=True):
        result = analyze_step(frame, cap, (coulomb, viscous))
        if result is not None:
            steps.append(result)
    if not steps:
        raise SystemExit('No step segments found.')
    per_step = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith('_')} for s in steps])

    metrics = ['rise_time_ms', 'overshoot_percent', 'peak_time_ms', 'settling_time_ms',
               'settling_time_band_ms', 'steady_state_error_deg', 'peak_velocity_deg_s',
               'peak_acceleration_rad_s2', 'saturated_acceleration_rad_s2',
               'peak_torque_nm', 'peak_power_w', 'saturation_fraction']
    grouped = per_step.groupby('step_size_deg')
    per_size = grouped[metrics].agg(['mean', 'std'])
    per_size.columns = [f'{m}_{stat}' for m, stat in per_size.columns]
    per_size = per_size.reset_index()
    per_size['num_steps'] = grouped.size().to_numpy()
    per_size['saturated_steps'] = grouped['saturated'].sum().to_numpy()
    sizes = sorted(per_step['step_size_deg'].unique())

    J_ref, J_source = reference_inertia(run.simulated)
    accel_capability = cap / J_ref
    linear = per_step[~per_step['saturated']]
    linear_size = float(linear['step_size_deg'].min()) if len(linear) else math.nan
    lin = per_step[per_step['step_size_deg'] == linear_size] if len(linear) else per_step.iloc[0:0]
    sat = per_step[per_step['saturated']]
    fitted = sat['saturated_acceleration_rad_s2'].dropna()
    sustained = float(fitted.median()) if len(fitted) else math.nan

    summary = {
        'kp_joint_nm_rad': kp,
        'kd_joint_nms_rad': kd,
        'torque_cap_nm': cap,
        'num_steps': int(len(per_step)),
        'linear_step_deg': linear_size,
        'rise_time_ms': float(lin['rise_time_ms'].mean()) if len(lin) else math.nan,
        'overshoot_percent': float(lin['overshoot_percent'].mean()) if len(lin) else math.nan,
        'settling_time_ms': float(lin['settling_time_ms'].mean()) if len(lin) else math.nan,
        'settling_time_band_ms': float(lin['settling_time_band_ms'].mean()) if len(lin) else math.nan,
        'steady_state_error_deg_rms': float(np.sqrt(np.mean(per_step['steady_state_error_deg'] ** 2))),
        'peak_velocity_deg_s': float(per_step['peak_velocity_deg_s'].max()),
        'peak_acceleration_rad_s2': float(per_step['peak_acceleration_rad_s2'].max()),
        'saturated_acceleration_rad_s2': sustained,
        'large_signal_inertia_kg_m2': float(sat['large_signal_inertia_kg_m2'].median())
        if len(sat) else math.nan,
        'torque_knee_speed_deg_s': float(sat['torque_knee_speed_deg_s'].median())
        if len(sat) and sat['torque_knee_speed_deg_s'].notna().any() else math.nan,
        'acceleration_capability_rad_s2': accel_capability,
        'acceleration_ratio': sustained / accel_capability if math.isfinite(sustained) else math.nan,
        'J_reference_kg_m2': J_ref,
        'J_reference_source': J_source,
        'peak_torque_nm': float(per_step['peak_torque_nm'].max()),
        'peak_power_w': float(per_step['peak_power_w'].max()),
        'saturated_step_sizes_deg': [float(v) for v in sorted(sat['step_size_deg'].unique())],
    }
    metadata = {
        'analysis': 'closed_loop_step_experiment_analysis.py',
        'source_data': str(run.source),
        'settling_fraction': SETTLING_FRACTION,
        'settling_band_deg': SETTLING_BAND_DEG,
        'saturation_threshold': SATURATION_THRESHOLD,
        'accel_window_s': ACCEL_WINDOW_S,
        'accel_fit_skip_s': ACCEL_FIT_SKIP_S,
        'accel_fit_min_s': ACCEL_FIT_MIN_S,
        'saturated_acceleration_method': ('position = theta0 + v0 t + c * double integral of '
                                          'the delivered torque over the first motoring '
                                          'interval at the cap (>= accel_fit_min_s); '
                                          'acceleration at the cap = cap * c'),
        'knee_fraction': KNEE_FRACTION,
        'friction_subtracted': {'coulomb_nm': coulomb, 'viscous_nms_rad': viscous,
                                'source': friction_source},
    }
    save_outputs(run, {'per_step': per_step, 'per_size': per_size}, summary, metadata)

    print_header('CLOSED-LOOP STEP PERFORMANCE')
    print(f'Kp = {kp:g} N m/rad, Kd = {kd:g} N m s/rad, torque cap {cap:g} N m, '
          f'{len(per_step)} steps')
    print(f"{'size':>6} {'rise':>8} {'overshoot':>10} {'settle':>8} {'peak vel':>10} "
          f"{'peak acc':>10} {'sat acc':>9} {'sat':>5}")
    for _, r in per_size.iterrows():
        print(f"{r['step_size_deg']:5.1f}° {r['rise_time_ms_mean']:6.1f}ms "
              f"{r['overshoot_percent_mean']:9.1f}% {r['settling_time_ms_mean']:6.0f}ms "
              f"{r['peak_velocity_deg_s_mean']:7.0f}d/s {r['peak_acceleration_rad_s2_mean']:8.0f}r/s2 "
              f"{r['saturated_acceleration_rad_s2_mean']:8.0f} "
              f"{int(r['saturated_steps'])}/{int(r['num_steps'])}")
    print(f"  Acceleration capability cap/J = {accel_capability:.0f} rad/s^2 "
          f"(J = {J_ref:.4g} kg m^2 from {J_source}); measured acceleration at the cap "
          f"{sustained:.0f} rad/s^2 (median of fitted steps, ratio "
          f"{summary['acceleration_ratio']:.2f}); large-signal J "
          f"{summary['large_signal_inertia_kg_m2']:.4g} kg m^2")
    if math.isfinite(summary['torque_knee_speed_deg_s']):
        print(f"  Delivered torque falls below {100 * KNEE_FRACTION:.0f}% of the cap above "
              f"{summary['torque_knee_speed_deg_s']:.0f} deg/s (back-EMF knee)")
    print(f"  Peak velocity {summary['peak_velocity_deg_s']:.0f} deg/s, peak torque "
          f"{summary['peak_torque_nm']:.2f} N m, peak power {summary['peak_power_w']:.1f} W")
    truth = sim_truth(run)
    if truth:
        J_true = truth['motor_inertia'] + truth['output_inertia']
        print_truth_comparison([
            ('Saturated acceleration', sustained, cap * truth['torque_constant_ratio'] / J_true,
             'rad/s^2'),
        ])

    fig = plot_steps(steps, per_size, sizes)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    largest = max(steps, key=lambda s: (s['step_size_deg'], s['saturation_fraction']))
    diag = plot_kinematics(largest, cap, accel_capability)
    export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
