"""Apparent impedance felt at the handle with zero torque vs friction compensation.

For each condition the interaction torque (external sensor between hand and
handle) is fit as

    tau_int = J_app * acc + b_app * omega + f_app * sign(omega) + offset

which gives the inertia, damping, and friction the operator feels. Velocity
and acceleration come from Savitzky-Golay derivatives of the encoder
position. The figure shows the resistive part (interaction torque minus the
fitted inertial term and offset) against velocity, and the RMS interaction
torque per speed. Without a continuous torque sensor the interaction torque
is a MODEL-BASED estimate from the identified J, b, fc and the motor torque;
the output is labeled accordingly and only predicts the effect of
compensation.

Run:  python transparency_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (export_figure, load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, savgol_derivative, sim_truth)
from ieee_plot import (INK, SERIES, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       figure_size, pad_top_tick, series_style, style_axis, zero_line)

EXPERIMENT = 'transparency'
FIGURE_BASENAME = 'fig_transparency'
DIAGNOSTIC_BASENAME = 'diag_transparency_trials'
TRIM_S = 1.0               # Drop the start and end of each trial (operator ramping in/out)
VELOCITY_WINDOW_S = 0.03   # Savitzky-Golay windows for the derivatives
ACCEL_WINDOW_S = 0.05
MIN_SPEED_RAD_S = 0.05     # Exclude near-reversal samples from the fit (sign ambiguity)
NUM_VELOCITY_BINS = 16
MIN_BIN_SAMPLES = 30
LABELS = {'zero_torque': 'Zero torque', 'compensated': 'Friction compensation'}


def trial_data(run):
    df = run.df
    estimates = run.meta.get('compensation', {})
    J, b, fc = (estimates.get('J_kg_m2', math.nan), estimates.get('b_nms_rad', math.nan),
                estimates.get('fc_nm', math.nan))
    trials = df[df['phase'] == 'trial']
    has_sensor = ('ext_torque_nm' in trials and trials['ext_torque_nm'].notna().mean() > 0.9
                  and run.meta.get('sensor_continuous', True))
    frames = []
    for trial_id, g in trials.groupby('trial_id', sort=True):
        t = g['time'].to_numpy()
        theta = np.radians(g['joint_pos_deg'].to_numpy())
        omega = savgol_derivative(t, theta, VELOCITY_WINDOW_S, 3, 1)
        acc = savgol_derivative(t, theta, ACCEL_WINDOW_S, 3, 2)
        tau_motor = g['joint_torque_nm'].to_numpy()
        model = J * acc + b * omega + fc * np.sign(omega) - tau_motor
        frame = pd.DataFrame({
            'trial_id': int(trial_id),
            'condition': g['condition'].iloc[0],
            'speed': g['speed'].iloc[0],
            'target_hz': float(g['target_hz'].iloc[0]),
            't_local': g['t_local'].to_numpy(),
            'theta_rad': theta,
            'omega_rad_s': omega,
            'acc_rad_s2': acc,
            'tau_motor_nm': tau_motor,
            'tau_model_nm': model,
            'tau_int_nm': g['ext_torque_nm'].to_numpy() if has_sensor else model,
        })
        duration = frame['t_local'].max()
        keep = (frame['t_local'] >= TRIM_S) & (frame['t_local'] <= duration - TRIM_S)
        frames.append(frame[keep])
    return pd.concat(frames, ignore_index=True), has_sensor


def fit_impedance(data):
    moving = data[data['omega_rad_s'].abs() > MIN_SPEED_RAD_S]
    A = np.column_stack([moving['acc_rad_s2'], moving['omega_rad_s'],
                         np.sign(moving['omega_rad_s']), np.ones(len(moving))])
    y = moving['tau_int_nm'].to_numpy()
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        'J_app_kg_m2': float(coef[0]),
        'b_app_nms_rad': float(coef[1]),
        'f_app_nm': float(coef[2]),
        'offset_nm': float(coef[3]),
        'fit_r_squared': 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else math.nan,
        'fit_rms_residual_nm': float(np.sqrt(np.mean(residual ** 2))),
        'num_samples': int(len(moving)),
    }


def motion_frequency(theta, t):
    centered = theta - theta.mean()
    crossings = np.where(np.diff(np.sign(centered)) != 0)[0]
    if len(crossings) < 3:
        return math.nan
    return (len(crossings) - 1) / (2.0 * (t[crossings[-1]] - t[crossings[0]]))


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else math.nan


def per_trial_table(data, fits):
    rows = []
    for trial_id, g in data.groupby('trial_id', sort=True):
        fit = fits[g['condition'].iloc[0]]
        resistive = g['tau_int_nm'] - fit['J_app_kg_m2'] * g['acc_rad_s2'] - fit['offset_nm']
        rows.append({
            'trial_id': trial_id,
            'condition': g['condition'].iloc[0],
            'speed': g['speed'].iloc[0],
            'target_hz': g['target_hz'].iloc[0],
            'measured_hz': motion_frequency(g['theta_rad'].to_numpy(), g['t_local'].to_numpy()),
            'rms_torque_nm': rms(g['tau_int_nm']),
            'p99_abs_torque_nm': float(np.percentile(np.abs(g['tau_int_nm']), 99)),
            'rms_resistive_nm': rms(resistive),
            'rms_speed_deg_s': math.degrees(rms(g['omega_rad_s'])),
            'p99_speed_deg_s': math.degrees(float(np.percentile(np.abs(g['omega_rad_s']), 99))),
            'model_vs_measured_rms_nm': rms(g['tau_int_nm'] - g['tau_model_nm']),
        })
    return pd.DataFrame(rows)


def velocity_bins(data, fits, v_limit_deg_s):
    edges = np.linspace(-v_limit_deg_s, v_limit_deg_s, NUM_VELOCITY_BINS + 1)
    rows = []
    for condition, g in data.groupby('condition'):
        fit = fits[condition]
        resistive = (g['tau_int_nm'] - fit['J_app_kg_m2'] * g['acc_rad_s2']
                     - fit['offset_nm']).to_numpy()
        velocity = np.degrees(g['omega_rad_s'].to_numpy())
        index = np.digitize(velocity, edges) - 1
        for i in range(NUM_VELOCITY_BINS):
            sel = index == i
            if sel.sum() < MIN_BIN_SAMPLES:
                continue
            rows.append({'condition': condition, 'velocity_deg_s': 0.5 * (edges[i] + edges[i + 1]),
                         'resistive_mean_nm': float(resistive[sel].mean()),
                         'resistive_sd_nm': float(resistive[sel].std(ddof=1)),
                         'num_samples': int(sel.sum())})
    return pd.DataFrame(rows)


def plot_transparency(bins, fits, trials, conditions, speeds, torque_label):
    fig, (ax_v, ax_r) = plt.subplots(2, 1, figsize=figure_size('single', 4.0),
                                     gridspec_kw={'height_ratios': [1.35, 1.0], 'hspace': 0.5})
    vmax = bins['velocity_deg_s'].abs().max() if len(bins) else 1.0
    handles = []
    for i, condition in enumerate(conditions):
        b = bins[bins['condition'] == condition]
        style = series_style(i, line=False, filled=i == 1)
        handle = ax_v.errorbar(b['velocity_deg_s'], b['resistive_mean_nm'],
                               yerr=b['resistive_sd_nm'], elinewidth=0.6, label=LABELS[condition],
                               **style)
        handles.append(handle)
        fit = fits[condition]
        for sign in (-1, 1):
            v = np.linspace(0.02 * vmax, 1.05 * vmax, 30) * sign
            ax_v.plot(v, fit['f_app_nm'] * sign + fit['b_app_nms_rad'] * np.radians(v),
                      color=SERIES[i], linestyle='--', linewidth=0.9)
    model_handle, = ax_v.plot([], [], color=INK, linestyle='--', linewidth=0.9,
                              label='Fitted $f\\,\\mathrm{sgn}(\\omega) + b\\,\\omega$')
    zero_line(ax_v)
    zero_line(ax_v, orientation='v')
    ax_v.set_xlabel('Velocity (deg/s)')
    ax_v.set_ylabel(f'Resistive torque ({TORQUE_UNIT})')
    style_axis(ax_v)
    ax_v.legend(handles=handles + [model_handle], loc='upper left')
    lo, hi = ax_v.get_ylim()
    ax_v.set_ylim(lo, hi + 0.45 * (hi - lo))   # Room for the legend above the data
    pad_top_tick(ax_v)
    add_panel_label(ax_v, '(a)')

    positions = np.arange(len(speeds))
    offsets = np.linspace(-0.1, 0.1, len(conditions)) if len(conditions) > 1 else [0.0]
    for i, condition in enumerate(conditions):
        values = [trials[(trials['condition'] == condition) & (trials['speed'] == s)]
                  ['rms_torque_nm'].mean() for s, _ in speeds]
        ax_r.plot(positions + offsets[i], values, label=LABELS[condition],
                  **{**series_style(i, filled=i == 1), 'linewidth': 0.7})
    ax_r.set_xticks(positions)
    ax_r.set_xticklabels([f'{s.capitalize()}\n({f:g} Hz)' for s, f in speeds])
    ax_r.set_xlim(-0.5, len(speeds) - 0.5)
    ax_r.set_ylim(0, None)
    lo, hi = ax_r.get_ylim()
    ax_r.set_ylim(0, hi * 1.35)
    pad_top_tick(ax_r)
    ax_r.set_ylabel(f'{torque_label} ({TORQUE_UNIT})')
    style_axis(ax_r, minor=False)
    ax_r.tick_params(axis='x', which='both', top=False)
    ax_r.legend(loc='upper left')
    add_panel_label(ax_r, '(b)')
    return fig


def plot_trials(data, has_sensor):
    trial_ids = sorted(data['trial_id'].unique())
    cols = 3
    rows = math.ceil(len(trial_ids) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figure_size('double', 1.5 * rows + 0.4),
                             squeeze=False, sharey=True)
    for ax, trial_id in zip(axes.flat, trial_ids):
        g = data[data['trial_id'] == trial_id]
        if has_sensor:
            ax.plot(g['t_local'], g['tau_model_nm'], color=SERIES[1], linewidth=0.8,
                    label='Identified model')
        ax.plot(g['t_local'], g['tau_int_nm'], color=INK, linewidth=0.5,
                label='Measured' if has_sensor else 'Model-based')
        ax.set_title(f"{LABELS[g['condition'].iloc[0]]}, {g['speed'].iloc[0]}", fontsize=7,
                     pad=2)
        style_axis(ax, minor=False)
    for ax in axes.flat[len(trial_ids):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Time in trial (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel(f'Interaction ({TORQUE_UNIT})')
    axes.flat[0].legend(loc='lower left', fontsize=6)
    fig.tight_layout(pad=0.3)
    return fig


def reduction(before, after):
    if not (math.isfinite(before) and math.isfinite(after)) or before == 0:
        return math.nan
    return 100.0 * (1.0 - after / before)


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    data, has_sensor = trial_data(run)
    if data.empty:
        raise SystemExit('No trial data found.')
    conditions = [c for c in ('zero_torque', 'compensated') if c in set(data['condition'])]
    speed_order = (run.meta.get('experiment_config') or {}).get('speeds')
    speeds = ([tuple(s) for s in speed_order] if speed_order else
              list(data.groupby('speed')['target_hz'].first().sort_values().items()))
    speeds = [(s, f) for s, f in speeds if s in set(data['speed'])]

    fits = {c: fit_impedance(data[data['condition'] == c]) for c in conditions}
    trials = per_trial_table(data, fits)
    v_limit = math.degrees(float(np.percentile(np.abs(data['omega_rad_s']), 95)))
    bins = velocity_bins(data, fits, v_limit)

    source = 'measured' if has_sensor else 'model_based'
    summary = {'torque_source': source}
    for condition in conditions:
        fit = fits[condition]
        g = data[data['condition'] == condition]
        resistive = g['tau_int_nm'] - fit['J_app_kg_m2'] * g['acc_rad_s2'] - fit['offset_nm']
        summary.update({f'{condition}_{k}': v for k, v in fit.items()})
        summary[f'{condition}_rms_torque_nm'] = rms(g['tau_int_nm'])
        summary[f'{condition}_rms_resistive_nm'] = rms(resistive)
        summary[f'{condition}_p99_abs_torque_nm'] = float(np.percentile(np.abs(g['tau_int_nm']), 99))
        if has_sensor:
            summary[f'{condition}_model_vs_measured_rms_nm'] = rms(g['tau_int_nm'] - g['tau_model_nm'])
    if {'zero_torque', 'compensated'} <= set(conditions):
        z, c = 'zero_torque', 'compensated'
        summary['friction_reduction_percent'] = reduction(summary[f'{z}_f_app_nm'],
                                                          summary[f'{c}_f_app_nm'])
        summary['damping_reduction_percent'] = reduction(summary[f'{z}_b_app_nms_rad'],
                                                         summary[f'{c}_b_app_nms_rad'])
        summary['rms_torque_reduction_percent'] = reduction(summary[f'{z}_rms_torque_nm'],
                                                            summary[f'{c}_rms_torque_nm'])
        summary['rms_resistive_reduction_percent'] = reduction(summary[f'{z}_rms_resistive_nm'],
                                                               summary[f'{c}_rms_resistive_nm'])
    compensation = run.meta.get('compensation', {})
    summary.update({'compensation_gain': compensation.get('gain', math.nan),
                    'compensation_fc_nm': compensation.get('fc_nm', math.nan),
                    'compensation_b_nms_rad': compensation.get('b_nms_rad', math.nan),
                    'compensation_source': compensation.get('source')})
    metadata = {
        'analysis': 'transparency_experiment_analysis.py',
        'source_data': str(run.source),
        'torque_source': source,
        'method': ('least squares tau_int = J acc + b omega + f sign(omega) + offset per '
                   'condition; derivatives by Savitzky-Golay'),
        'trim_s': TRIM_S, 'velocity_window_s': VELOCITY_WINDOW_S,
        'accel_window_s': ACCEL_WINDOW_S, 'min_speed_rad_s': MIN_SPEED_RAD_S,
        'note': ('Resistive torque = interaction torque minus fitted inertial term and offset. '
                 + ('' if has_sensor else
                    'MODEL-BASED: no continuous torque sensor; values follow from the identified '
                    'model and the commanded compensation, not a measurement.')),
    }
    save_outputs(run, {'trials': trials, 'velocity_bins': bins,
                       'impedance_fits': pd.DataFrame([{'condition': c, **fits[c]}
                                                       for c in conditions])},
                 summary, metadata)

    print_header('TRANSPARENCY (apparent impedance at the handle)')
    if not has_sensor:
        print('MODEL-BASED ESTIMATE: no continuous torque sensor in this run.')
    for condition in conditions:
        f = fits[condition]
        print(f"  {LABELS[condition]:<22} J {f['J_app_kg_m2']:.4g} kg m^2  "
              f"b {f['b_app_nms_rad']:.4g} N m s/rad  f {f['f_app_nm']:.4g} N m  "
              f"(R^2 {f['fit_r_squared']:.3f})")
    print('  RMS interaction torque per trial:')
    for _, r in trials.iterrows():
        print(f"    {LABELS[r['condition']]:<22} {r['speed']:<7} {r['rms_torque_nm']:.4f} N m  "
              f"(resistive {r['rms_resistive_nm']:.4f}, {r['measured_hz']:.2f} Hz, "
              f"{r['rms_speed_deg_s']:.0f} deg/s RMS)")
    if has_sensor and 'zero_torque' in fits and fits['zero_torque']['J_app_kg_m2'] < 0:
        print('WARNING: negative apparent inertia; the sensor sign is probably flipped. '
              'ext_torque_nm must be the torque the hand applies in the joint direction '
              "(device_config EXTERNAL_TORQUE_SENSOR 'sign').")
    if 'friction_reduction_percent' in summary:
        print(f"  Compensation reduces felt friction {summary['friction_reduction_percent']:.0f}%, "
              f"damping {summary['damping_reduction_percent']:.0f}%, RMS torque "
              f"{summary['rms_torque_reduction_percent']:.0f}%")
    truth = sim_truth(run)
    if truth and 'zero_torque' in fits and has_sensor:
        z = fits['zero_torque']
        print_truth_comparison([
            ('Zero-torque apparent J', z['J_app_kg_m2'],
             truth['motor_inertia'] + truth['output_inertia'], 'kg m^2'),
            ('Zero-torque apparent b', z['b_app_nms_rad'],
             truth['viscous'] + truth['output_viscous'], 'N m s/rad'),
            ('Zero-torque apparent f', z['f_app_nm'], truth['coulomb'], 'N m'),
        ])

    torque_label = 'RMS torque' if has_sensor else 'RMS model torque'
    fig = plot_transparency(bins, fits, trials, conditions, speeds, torque_label)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    diag = plot_trials(data, has_sensor)
    export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
