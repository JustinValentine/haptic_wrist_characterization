"""Large-signal dynamic envelope: largest linear amplitude per frequency.

Every trial is re-analyzed from the raw data with the experiment's own rule
(dynamic_envelope_experiment.trial_metrics / classify): least-squares fits at
f, 2f and 3f over the steady window give the gain and phase relative to the
small-signal reference at that frequency, the harmonic distortion, and the
fraction of the window with the controller torque at the cap or the measured
torque unable to follow it (voltage limit). A_max(f) is the measured joint
amplitude of the largest linear trial below the first non-linear one; the
first non-linear trial names the limiting mechanism. When no trial failed
before the safety caps, A_max is a lower bound.

The envelope is plotted against the torque limit tau_cap / (J w^2), the speed
limit v / w, and the range of motion, with J from the newest step-response
analysis of the same kind (real or simulated) when available.

Run:  python dynamic_envelope_experiment_analysis.py [run_dir] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import device_config
from analysis_common import (export_figure, load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, sim_truth)
from dynamic_envelope_experiment import (EXPERIMENT, EnvelopeConfig, classify, thresholds_from,
                                         trial_metrics)
from experiment_common import latest_summary
from ieee_plot import (INK, MUTED, SERIES, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       figure_size, log_ticks_hz, reference_line, style_axis)

FIGURE_BASENAME = 'fig_dynamic_envelope'
DIAGNOSTIC_BASENAME = 'diag_dynamic_envelope_trials'
REGION_FILL = '#e9eef5'   # Opaque light tint (EPS has no transparency)
CROSSOVER_LEVELS_DEG = (10.0, 1.0)


def frequency_tag(f):
    return f'{f:g}'.replace('.', 'p')


def reference_inertia(run):
    """J for the limit lines: newest same-kind step analysis, else the planning J."""
    found = latest_summary('step_response', run.simulated)
    if found is not None:
        run_dir, summary = found
        J = summary.get('J_kg_m2')
        if isinstance(J, (int, float)) and math.isfinite(J) and J > 0:
            return float(J), f'step_response analysis ({run_dir.name})'
    protocol = run.meta.get('envelope_protocol') or {}
    if protocol.get('J_planning_kg_m2'):
        return float(protocol['J_planning_kg_m2']), protocol.get('J_planning_source', 'planning')
    return device_config.NOMINAL_JOINT_INERTIA_KG_M2, 'device_config nominal'


def analyze_trials(run, thresholds):
    df = run.df
    trials = df[df['phase'] == 'trial']
    online = {int(t['trial_id']): t for t in run.meta.get('trials', [])}
    rows = []
    for trial_id, frame in trials.groupby('trial_id', sort=True):
        trial_id = int(trial_id)
        f = float(frame['frequency_hz'].iloc[0])
        ramp_s, steady_s = float(frame['ramp_s'].iloc[0]), float(frame['steady_s'].iloc[0])
        metrics = trial_metrics(frame['t_local'], frame['joint_pos_deg'], frame['cmd_joint_deg'],
                                frame['joint_vel_deg_s'], frame['joint_torque_nm'],
                                frame['control_joint_torque_nm'], frame['max_torque_nm'],
                                f, ramp_s, steady_s,
                                torque_delay_s=float(run.meta.get('loop_delay_s') or 0.0))
        rows.append({'trial_id': trial_id, 'frequency_hz': f,
                     'command_setpoint_deg': float(frame['command_amplitude_deg'].iloc[0]),
                     'role': frame['role'].iloc[0],
                     'ended_by': online.get(trial_id, {}).get('ended_by'),
                     'online_linear': online.get(trial_id, {}).get('linear'),
                     '_metrics': metrics, **(metrics or {})})
    table = pd.DataFrame(rows)
    # Classify against the reference trial at each frequency.
    verdicts = []
    for _, row in table.iterrows():
        ref_rows = table[(table['frequency_hz'] == row['frequency_hz'])
                         & (table['role'] == 'reference')]
        reference = ref_rows['_metrics'].iloc[0] if len(ref_rows) else None
        ended = row['ended_by'] if isinstance(row['ended_by'], str) else None
        verdicts.append(classify(row['_metrics'], reference, thresholds, ended))
    verdicts = pd.DataFrame(verdicts)
    table = pd.concat([table.drop(columns=['_metrics']), verdicts], axis=1)
    return table.sort_values(['frequency_hz', 'command_setpoint_deg']).reset_index(drop=True)


def actuator_limit_deg(f, J, cap, stall, no_load):
    """Largest sinusoid amplitude (deg) whose inertial torque J A w^2 |sin| stays within
    the cap and, while motoring, within the back-EMF line stall (1 - A w |cos| / no_load)."""
    w = 2.0 * math.pi * f
    phase = np.linspace(0.0, 0.5 * math.pi, 2001)

    def feasible(amplitude):
        torque = J * amplitude * w ** 2 * np.sin(phase)
        speed = amplitude * w * np.cos(phase)
        available = np.minimum(cap, stall * np.clip(1.0 - speed / no_load, 0.0, None))
        return bool(np.all(torque <= available + 1e-12))

    lo, hi = 0.0, cap / (J * w ** 2)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
    return math.degrees(lo)


def limits_at(f, J, cap, speed, rom):
    w = 2.0 * math.pi * f
    return math.degrees(cap / (J * w ** 2)), speed / w, rom


def build_envelope(trials, plans, J, cap, speed, rom):
    rows = []
    for f, group in trials.groupby('frequency_hz', sort=True):
        reference = group[group['role'] == 'reference']
        nonlinear = group[~group['linear'] & (group['role'] != 'reference')]
        first_fail = (nonlinear.sort_values('command_setpoint_deg').iloc[0]
                      if len(nonlinear) else None)
        below = group[group['linear']]
        if first_fail is not None:
            below = below[below['command_setpoint_deg'] < first_fail['command_setpoint_deg']]
        last = below.sort_values('command_setpoint_deg').iloc[-1] if len(below) else None
        plan = plans.get(round(f, 9), {})
        a_tau, a_v, a_rom = limits_at(f, J, cap, speed, rom)
        if first_fail is not None:
            limited_by = first_fail['reason']
        elif len(reference) and not bool(reference['linear'].iloc[0]):
            limited_by = 'reference not linear'
        else:
            limited_by = f"lower bound: safety cap ({plan.get('cap_binding', 'unknown')})"
        # Joint amplitude of a stopped trial is unknown; scale the command by G0.
        fail_joint = math.nan
        if first_fail is not None:
            fail_joint = first_fail.get('joint_amplitude_deg', math.nan)
            if not np.isfinite(fail_joint) and len(reference):
                fail_joint = first_fail['command_setpoint_deg'] * float(reference['gain'].iloc[0])
        rows.append({
            'frequency_hz': f,
            'amax_joint_deg': float(last['joint_amplitude_deg']) if last is not None else math.nan,
            'amax_command_deg': (float(last['command_setpoint_deg'])
                                 if last is not None else math.nan),
            'first_nonlinear_joint_deg': fail_joint,
            'first_nonlinear_command_deg': (float(first_fail['command_setpoint_deg'])
                                            if first_fail is not None else math.nan),
            'lower_bound': first_fail is None,
            'limited_by': limited_by,
            'all_reasons': first_fail['all_reasons'] if first_fail is not None else '',
            'torque_at_amax_nm': float(last['peak_torque_nm']) if last is not None else math.nan,
            'control_torque_at_amax_nm': (float(last['peak_control_torque_nm'])
                                          if last is not None else math.nan),
            'power_at_amax_w': float(last['peak_power_w']) if last is not None else math.nan,
            'velocity_at_amax_deg_s': (float(last['peak_velocity_deg_s'])
                                       if last is not None else math.nan),
            'torque_limit_deg': a_tau,
            'speed_limit_deg': a_v,
            'range_limit_deg': a_rom,
            'num_trials': int(len(group)),
        })
    envelope = pd.DataFrame(rows)
    envelope['torque_line_ratio'] = envelope['amax_joint_deg'] / envelope['torque_limit_deg']
    return envelope


def crossover(envelope, level):
    """Frequency where the envelope falls through `level` deg (log-log interpolation)."""
    env = envelope.dropna(subset=['amax_joint_deg']).sort_values('frequency_hz')
    f = env['frequency_hz'].to_numpy()
    a = env['amax_joint_deg'].to_numpy()
    for i in range(1, len(f)):
        if a[i - 1] >= level > a[i]:
            x0, x1 = math.log(f[i - 1]), math.log(f[i])
            y0, y1 = math.log(a[i - 1]), math.log(a[i])
            return float(math.exp(x0 + (math.log(level) - y0) / (y1 - y0) * (x1 - x0)))
    return math.nan


# --- figures -----------------------------------------------------------------------------
def plot_envelope(envelope, J, cap, speed, rom, regions):
    fig, (ax_a, ax_t) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 4.0),
                                     gridspec_kw={'height_ratios': [1.55, 1.0], 'hspace': 0.12})
    f_lo = envelope['frequency_hz'].min() / 1.4
    f_hi = envelope['frequency_hz'].max() * 1.4
    f = np.logspace(math.log10(f_lo), math.log10(f_hi), 200)
    w = 2.0 * np.pi * f
    for region in regions:
        ax_a.add_patch(Rectangle((region['f_min'], region['amp_min']),
                                 region['f_max'] - region['f_min'],
                                 region['amp_max'] - region['amp_min'],
                                 facecolor=REGION_FILL, edgecolor='none', zorder=0))
        ax_a.text(math.sqrt(region['f_min'] * region['f_max']),
                  math.sqrt(region['amp_min'] * region['amp_max']), region['label'],
                  ha='center', va='center', fontsize=6.5, color=INK, zorder=1)
    ax_a.plot(f, np.degrees(cap / (J * w ** 2)), color=MUTED, linestyle='--', linewidth=0.9,
              label=f'Torque limit ({cap:g} {TORQUE_UNIT})')
    ax_a.plot(f, speed / w, color=MUTED, linestyle='-.', linewidth=0.9,
              label=f'Speed cap ({speed:.0f} deg/s)')
    ax_a.plot(f, np.full_like(f, rom), color=MUTED, linestyle=':', linewidth=1.0,
              label=f'Amplitude cap ({rom:g} deg)')

    env = envelope.dropna(subset=['amax_joint_deg'])
    ax_a.plot(env['frequency_hz'], env['amax_joint_deg'], color=INK, linewidth=1.0, zorder=3)
    bracketed = env[~env['lower_bound']]
    bounds = env[env['lower_bound']]
    ax_a.plot(bracketed['frequency_hz'], bracketed['amax_joint_deg'], linestyle='None',
              marker='o', markersize=3.8, color=INK, zorder=4, label='Largest linear amplitude')
    if len(bounds):
        ax_a.plot(bounds['frequency_hz'], bounds['amax_joint_deg'], linestyle='None', marker='o',
                  markersize=3.8, markerfacecolor='white', markeredgecolor=INK, zorder=4,
                  label='Lower bound (safety cap)')
        for _, row in bounds.iterrows():
            ax_a.annotate('', xy=(row['frequency_hz'], row['amax_joint_deg'] * 1.9),
                          xytext=(row['frequency_hz'], row['amax_joint_deg'] * 1.12),
                          arrowprops=dict(arrowstyle='-|>', color=INK, linewidth=0.7,
                                          mutation_scale=5), zorder=4)
    fails = envelope.dropna(subset=['first_nonlinear_joint_deg'])
    if len(fails):
        ax_a.plot(fails['frequency_hz'], fails['first_nonlinear_joint_deg'], linestyle='None',
                  marker='x', markersize=3.8, markeredgewidth=0.8, color=SERIES[1], zorder=4,
                  label='First non-linear amplitude')
    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    values = pd.concat([env['amax_joint_deg'], fails['first_nonlinear_joint_deg']]).dropna()
    lo = max(min(values.min() / 3.0, rom / 50.0), 1e-3)
    ax_a.set_ylim(lo, rom * 5.0)
    ax_a.set_xlim(f_lo, f_hi)
    ax_a.set_ylabel('Amplitude (deg)')
    log_ticks_hz(ax_a, axis='y')
    style_axis(ax_a)
    # Above the axes: the limit lines sweep through every corner of the plot.
    ax_a.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=6.3,
                columnspacing=0.8, handlelength=2.0, borderaxespad=0.0)
    add_panel_label(ax_a, '(a)', x=-0.12)

    ax_t.plot(env['frequency_hz'], env['torque_at_amax_nm'], color=INK, linewidth=1.0,
              marker='o', markersize=3.5, label='Peak torque at the limit')
    reference_line(ax_t, cap, label='Torque cap')
    ax_t.set_ylim(0, cap * 1.45)
    ax_t.set_xlabel('Frequency (Hz)')
    ax_t.set_ylabel(f'Torque ({TORQUE_UNIT})')
    log_ticks_hz(ax_t)
    style_axis(ax_t)
    ax_t.legend(loc='upper left', fontsize=6.5)
    add_panel_label(ax_t, '(b)', x=-0.12)
    fig.align_ylabels([ax_a, ax_t])
    return fig


def plot_trials(trials, thresholds):
    freqs = sorted(trials['frequency_hz'].unique())
    cols = 4 if len(freqs) > 3 else len(freqs)
    rows = math.ceil(len(freqs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figure_size('double', 1.55 * rows + 0.5),
                             squeeze=False)
    tol = thresholds['gain_tolerance']
    for ax, f in zip(axes.flat, freqs):
        group = trials[trials['frequency_hz'] == f].sort_values('command_setpoint_deg')
        ok = group[group['linear']]
        bad = group[~group['linear']]
        ax.plot(ok['command_setpoint_deg'], ok['gain_ratio'], linestyle='None', marker='o',
                markersize=3, color=INK)
        ax.plot(bad['command_setpoint_deg'], bad['gain_ratio'].fillna(1.0), linestyle='None',
                marker='x', markersize=4, color=SERIES[1])
        for k, (_, row) in enumerate(bad.iterrows()):
            y = row['gain_ratio'] if np.isfinite(row['gain_ratio']) else 1.0
            ax.annotate(row['reason'], (row['command_setpoint_deg'], y),
                        xytext=(-4, -9 - 7 * k), textcoords='offset points', ha='right',
                        fontsize=5.5)
        for level in (1 - tol, 1 + tol):
            reference_line(ax, level, linewidth=0.6)
        ax.set_xscale('log')
        log_ticks_hz(ax)
        ax.set_title(f'{f:g} Hz', fontsize=7, pad=2)
        style_axis(ax, minor=False)
    for ax in axes.flat[len(freqs):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Command amplitude (deg)')
    for ax in axes[:, 0]:
        ax.set_ylabel('$|G/G_0|$')
    fig.tight_layout(pad=0.3)
    return fig


# --- main -------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    config = run.meta.get('experiment_config') or {}
    defaults = EnvelopeConfig()
    protocol = run.meta.get('envelope_protocol') or {}
    thresholds = thresholds_from(config)
    trials = analyze_trials(run, thresholds)
    if trials.empty:
        raise SystemExit('No envelope trials found in this run.')

    J, J_source = reference_inertia(run)
    cap = float(protocol.get('torque_cap_nm') or run.meta.get('max_torque_nm')
                or device_config.MAX_JOINT_TORQUE_NM)
    speed_cap = float(protocol.get('speed_cap_deg_s') or
                      defaults.speed_fraction * device_config.MAX_JOINT_VELOCITY_DEG_S)
    speed = float(config.get('speed_limit_deg_s') or speed_cap)
    rom = float(config.get('max_amplitude_deg', defaults.max_amplitude_deg))
    plans = {round(p['frequency_hz'], 9): p for p in protocol.get('plans', [])}
    envelope = build_envelope(trials, plans, J, cap, speed, rom)

    linear = trials[trials['linear']]
    actuator = envelope[envelope['limited_by'].isin(['torque saturation', 'voltage limit'])]
    online = trials['online_linear'].dropna()
    summary = {
        'kp_joint_nm_rad': float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad')),
        'kd_joint_nms_rad': float(run.meta_value('kd_joint_nms_rad', column='kd_joint_nms_rad')),
        'velocity_feedforward': bool(config.get('send_velocity_feedforward', True)),
        'torque_cap_nm': cap,
        'speed_limit_deg_s': speed,
        'rom_limit_deg': rom,
        'J_reference_kg_m2': J,
        'J_reference_source': J_source,
        'num_frequencies': int(len(envelope)),
        'num_trials': int(len(trials)),
        'num_lower_bounds': int(envelope['lower_bound'].sum()),
        'online_offline_agreement': (float((trials.loc[online.index, 'linear']
                                            == online.astype(bool)).mean())
                                     if len(online) else math.nan),
        'torque_line_ratio_median': (float(actuator['torque_line_ratio'].median())
                                     if len(actuator) else math.nan),
        'peak_velocity_deg_s': (float(linear['peak_velocity_deg_s'].max())
                                if len(linear) else math.nan),
        'peak_torque_nm': float(linear['peak_torque_nm'].max()) if len(linear) else math.nan,
        'peak_power_w': float(linear['peak_power_w'].max()) if len(linear) else math.nan,
        'peak_power_all_trials_w': float(trials['peak_power_w'].max()),
    }
    for level in CROSSOVER_LEVELS_DEG:
        summary[f'crossover_{level:g}deg_hz'] = crossover(envelope, level)
    for _, row in envelope.iterrows():
        tag = frequency_tag(row['frequency_hz'])
        summary[f'envelope_{tag}hz_deg'] = row['amax_joint_deg']
        summary[f'envelope_{tag}hz_lower_bound'] = bool(row['lower_bound'])
        summary[f'envelope_{tag}hz_limited_by'] = row['limited_by']

    metadata = {
        'analysis': 'dynamic_envelope_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('per trial, LS fits at f, 2f, 3f over the steady window; linear while '
                   '|G/G0| and phase stay near the small-signal reference, THD is low, and '
                   'the torque neither saturates nor falls short of the controller'),
        'thresholds': thresholds,
        'limit_lines': {'J_kg_m2': J, 'J_source': J_source, 'torque_cap_nm': cap,
                        'speed_deg_s': speed, 'range_deg': rom},
        'note': ('Amplitudes are joint-side (motor encoder / gear ratio). Torque is the '
                 'drive-reported torque; apply the torque calibration ratio if measured.'),
    }
    save_outputs(run, {'trials': trials, 'envelope': envelope}, summary, metadata)

    print_header('LARGE-SIGNAL DYNAMIC ENVELOPE')
    print(f'Kp = {summary["kp_joint_nm_rad"]:g} N m/rad, Kd = {summary["kd_joint_nms_rad"]:g} '
          f'N m s/rad, torque cap {cap:g} N m, J = {J:.4g} kg m^2 ({J_source})')
    for _, row in envelope.iterrows():
        bound = '>=' if row['lower_bound'] else '  '
        print(f"  {row['frequency_hz']:5.1f} Hz: A_max {bound}{row['amax_joint_deg']:7.2f} deg "
              f"(torque line {row['torque_limit_deg']:8.2f}, speed {row['speed_limit_deg']:7.1f}); "
              f"limited by {row['limited_by']}; peak torque {row['torque_at_amax_nm']:.2f} N m, "
              f"power {row['power_at_amax_w']:.0f} W")
    for level in CROSSOVER_LEVELS_DEG:
        value = summary[f'crossover_{level:g}deg_hz']
        text = f'{value:.1f} Hz' if math.isfinite(value) else 'not crossed in the tested range'
        print(f'  Envelope falls below {level:g} deg at {text}')
    print(f"  Actuator-limited frequencies reach {summary['torque_line_ratio_median']:.2f} x the "
          f"torque line (median); peak power {summary['peak_power_w']:.0f} W (linear trials)")
    if len(online):
        print(f"  Online/offline classification agreement: "
              f"{100 * summary['online_offline_agreement']:.0f}%")
    truth = sim_truth(run)
    if truth:
        J_true = truth['motor_inertia'] + truth['output_inertia']
        rows = [('Reference J', J, J_true, 'kg m^2')]
        tau_cap = cap * truth['torque_constant_ratio']
        back_emf = 'stall_torque' in truth and 'no_load_speed' in truth
        for _, row in actuator.iterrows():
            f = row['frequency_hz']
            if back_emf:
                line = actuator_limit_deg(f, J_true, tau_cap, truth['stall_torque'],
                                          truth['no_load_speed'])
                label = f"A_max at {f:g} Hz vs actuator limit"
            else:
                line = math.degrees(tau_cap / (J_true * (2.0 * math.pi * f) ** 2))
                label = f"A_max at {f:g} Hz vs torque line"
            rows.append((label, row['amax_joint_deg'], line, 'deg'))
        print_truth_comparison(rows)

    regions = [dict(r) for r in config.get('reference_regions') or []]
    fig = plot_envelope(envelope, J, cap, speed, rom, regions)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    export_figure(plot_trials(trials, thresholds), run, DIAGNOSTIC_BASENAME, metadata,
                  paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
