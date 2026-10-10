"""Z-width analysis: the stability boundary K_max(B) for each rendering mode.

Every trial is reclassified from the raw data with the same rule the
experiment used online (zwidth_experiment.classify_trial): a trial is stable
if the response to the torque pulse decays to the noise floor, and unstable
if it was stopped early (deviation, velocity, or torque chatter) or if the
position or torque oscillation in the last part of the observation is still
more than half of what it was just after the pulse. For each mode and virtual
damping B, K_max is the largest stable stiffness below the smallest unstable
one. When no instability occurred up to the largest tested K, K_max is a lower
bound and is reported as such.

With Coulomb friction the boundary depends on the disturbance size (friction
can quench small delay-driven oscillations), so the pulse is part of the
protocol and is recorded with the results.

Run:  python zwidth_experiment_analysis.py [run_dir] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis_common import (export_figure, load_run, make_parser, print_header,
                             save_outputs)
from ieee_plot import (INK, LINESTYLES, MARKERS, MUTED, SERIES, TORQUE_UNIT,
                       configure_ieee_style, figure_size, log_ticks_hz, reference_line,
                       style_axis)
from zwidth_experiment import EXPERIMENT, classify_trial, noise_floor

FIGURE_BASENAME = 'fig_zwidth'
DIAGNOSTIC_BASENAME = 'diag_zwidth_trials'


def mode_label(name):
    if name == 'rt':
        return '500 Hz loop'
    if name == 'drive':
        return 'drive loop'
    if name.startswith('host_'):
        rate = float(name.split('_', 1)[1])
        return f'Host loop, {rate / 1000:g} kHz' if rate >= 1000 else f'Host loop, {rate:g} Hz'
    return name


def protocol(run):
    """Pulse timing, trial length and torque cap (metadata, else the config)."""
    proto = run.meta.get('zwidth_protocol') or {}
    cfg = run.meta.get('experiment_config') or {}
    pulse_end = proto.get('pulse_end_s',
                          cfg.get('pulse_start_s', 0.2) + cfg.get('pulse_width_s', 0.005))
    planned = proto.get('planned_trial_s', pulse_end + cfg.get('observe_s', 1.5))
    cap = proto.get('trial_torque_cap_nm')
    if cap is None:
        cap = cfg.get('torque_cap_fraction', 0.5) * run.meta.get('max_torque_nm', 10.0)
    return pulse_end, planned, cap


def classify_all(run):
    df = run.df
    pulse_end, planned, cap = protocol(run)
    online = {int(t['trial_id']): t for t in run.meta.get('trials', [])}
    center = (run.meta.get('experiment_config') or {}).get('center_deg', 0.0)
    rows = []
    trial_rows = df[df['phase'] == 'trial']
    for trial_id, frame in trial_rows.groupby('trial_id', sort=True):
        trial_id = int(trial_id)
        settle = df[(df['phase'] == 'settle') & (df['trial_id'] == trial_id)]
        noise_deg, noise_nm = noise_floor(settle['t_local'], settle['joint_pos_deg'] - center,
                                          settle['joint_torque_nm'])
        ended_by = online.get(trial_id, {}).get('ended_by')
        result = classify_trial(frame['t_local'], frame['joint_pos_deg'] - center,
                                frame['joint_torque_nm'], pulse_end_s=pulse_end,
                                planned_end_s=planned, noise_deg=noise_deg, noise_nm=noise_nm,
                                cap_nm=cap, ended_by=ended_by)
        dt = np.diff(frame['time'].to_numpy())
        rows.append({
            'trial_id': trial_id,
            # 'render_mode' (older runs logged it as 'mode', which overwrote the
            # logged mode column).
            'mode': (frame['render_mode'] if 'render_mode' in frame else frame['mode']).iloc[0],
            'virtual_b_nms_rad': float(frame['virtual_b_nms_rad'].iloc[0]),
            'virtual_k_nm_rad': float(frame['virtual_k_nm_rad'].iloc[0]),
            'render_rate_hz': 1.0 / float(np.median(dt)) if len(dt) else math.nan,
            'ended_by': ended_by,
            'noise_deg': noise_deg,
            'noise_nm': noise_nm,
            'online_stable': online.get(trial_id, {}).get('stable'),
            **result,
        })
    return pd.DataFrame(rows)


def boundary_table(trials):
    rows = []
    for (mode, b), group in trials.groupby(['mode', 'virtual_b_nms_rad'], sort=False):
        unstable = group.loc[~group['stable'], 'virtual_k_nm_rad']
        first_unstable = float(unstable.min()) if len(unstable) else math.nan
        stable = group.loc[group['stable'], 'virtual_k_nm_rad']
        below = stable[stable < first_unstable] if len(unstable) else stable
        rows.append({
            'mode': mode,
            'virtual_b_nms_rad': b,
            'k_last_stable_nm_rad': float(below.max()) if len(below) else math.nan,
            'k_first_unstable_nm_rad': first_unstable,
            'limit_reached': bool(len(unstable)),
            'stable_above_instability': bool(len(unstable) and (stable > first_unstable).any()),
            'num_trials': int(len(group)),
        })
    return pd.DataFrame(rows)


def summarize(trials, boundary, modes):
    summary = {
        'zwidth_num_trials': int(len(trials)),
        'zwidth_num_unstable': int((~trials['stable']).sum()),
    }
    agree = trials['online_stable'].dropna()
    if len(agree):
        same = trials.loc[agree.index, 'stable'] == agree.astype(bool)
        summary['zwidth_online_offline_agreement'] = float(same.mean())
    for mode in modes:
        b = boundary[boundary['mode'] == mode].sort_values('virtual_b_nms_rad')
        if b.empty:
            continue
        lowest = b.iloc[0]
        tag = 'b0' if lowest['virtual_b_nms_rad'] == 0 else 'bmin'
        summary[f'zwidth_k_max_{mode}_{tag}_nm_rad'] = lowest['k_last_stable_nm_rad']
        summary[f'zwidth_k_first_unstable_{mode}_{tag}_nm_rad'] = lowest['k_first_unstable_nm_rad']
        best = b.loc[b['k_last_stable_nm_rad'].idxmax()] if b['k_last_stable_nm_rad'].notna().any() else None
        summary[f'zwidth_k_max_{mode}_nm_rad'] = (float(best['k_last_stable_nm_rad'])
                                                  if best is not None else math.nan)
        summary[f'zwidth_b_at_k_max_{mode}_nms_rad'] = (float(best['virtual_b_nms_rad'])
                                                        if best is not None else math.nan)
        summary[f'zwidth_limit_reached_fraction_{mode}'] = float(b['limit_reached'].mean())
        rates = trials.loc[trials['mode'] == mode, 'render_rate_hz']
        summary[f'zwidth_render_rate_{mode}_hz'] = float(rates.median())
    return summary


# --- figures ---------------------------------------------------------------------------
def plot_zwidth(trials, boundary, modes):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.9))
    handles = []
    for i, mode in enumerate(modes):
        color, marker, linestyle = SERIES[i], MARKERS[i], LINESTYLES[i]
        b = boundary[boundary['mode'] == mode].sort_values('virtual_b_nms_rad')
        x = b['virtual_b_nms_rad'].to_numpy()
        k_stable = b['k_last_stable_nm_rad'].to_numpy()
        k_unstable = b['k_first_unstable_nm_rad'].to_numpy()
        ax.plot(x, k_stable, color=color, linestyle=linestyle, linewidth=1.1, marker=marker,
                markersize=3.8, markerfacecolor=color, markeredgecolor=color, zorder=3)
        ax.plot(x, k_unstable, linestyle='None', marker='x', markersize=4.2,
                markeredgewidth=0.9, color=color, zorder=3)
        for xi, ks, ku in zip(x, k_stable, k_unstable):
            if np.isfinite(ks) and np.isfinite(ku):
                ax.plot([xi, xi], [ks, ku], color=color, linewidth=0.6, zorder=2)
            elif np.isfinite(ks):   # Lower bound: no instability up to the largest K.
                ax.annotate('', xy=(xi, ks * 2.0), xytext=(xi, ks),
                            arrowprops=dict(arrowstyle='-|>', color=color, linewidth=0.8,
                                            mutation_scale=6), zorder=2)
        handles.append(Line2D([], [], color=color, linestyle=linestyle, linewidth=1.1,
                              marker=marker, markersize=3.8, markerfacecolor=color,
                              label=mode_label(mode)))
    handles.append(Line2D([], [], linestyle='None', marker='x', markersize=4.2,
                          markeredgewidth=0.9, color=INK, label='First unstable $K$'))
    if (~boundary['limit_reached']).any():
        handles.append(Line2D([], [], linestyle='None', marker=r'$\uparrow$', markersize=6,
                              color=INK, label='Stable to largest $K$ tested'))
    ks = pd.concat([boundary['k_last_stable_nm_rad'], boundary['k_first_unstable_nm_rad']])
    ks = ks[np.isfinite(ks)]
    top = ks.max() * (4.0 if (~boundary['limit_reached']).any() else 2.0)
    ax.set_yscale('log')
    ax.set_ylim(ks.min() / 2.0, top)
    bs = boundary['virtual_b_nms_rad']
    span = bs.max() - bs.min() if bs.max() > bs.min() else 1.0
    ax.set_xlim(bs.min() - 0.06 * span, bs.max() + 0.06 * span)
    ax.set_xlabel(f'Virtual damping $B$ ({TORQUE_UNIT}·s/rad)')
    ax.set_ylabel(f'Virtual stiffness $K$ ({TORQUE_UNIT}/rad)')
    log_ticks_hz(ax, axis='y')
    style_axis(ax)
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2,
              borderaxespad=0.0, columnspacing=1.2)
    return fig


def plot_examples(run, trials, modes):
    """Pulse responses of the last stable and first unstable trial (lowest B)."""
    df = run.df
    center = (run.meta.get('experiment_config') or {}).get('center_deg', 0.0)
    pulse_end, _, _ = protocol(run)
    fig, axes = plt.subplots(2, len(modes), figsize=figure_size('double', 3.0),
                             squeeze=False, sharex=True)
    for col, mode in enumerate(modes):
        sub = trials[trials['mode'] == mode]
        b_min = sub['virtual_b_nms_rad'].min()
        sub = sub[sub['virtual_b_nms_rad'] == b_min]
        unstable = sub[~sub['stable']].sort_values('virtual_k_nm_rad')
        cutoff = unstable['virtual_k_nm_rad'].iloc[0] if len(unstable) else math.inf
        stable = sub[sub['stable'] & (sub['virtual_k_nm_rad'] < cutoff)].sort_values('virtual_k_nm_rad')
        picks = [('Last stable', stable.iloc[-1] if len(stable) else None),
                 ('First unstable', unstable.iloc[0] if len(unstable) else None)]
        for row, (title, trial) in enumerate(picks):
            ax = axes[row, col]
            if trial is None:
                ax.set_title(f'{mode_label(mode)}: {title.lower()}', fontsize=7)
                ax.text(0.5, 0.5, 'No unstable trial up to the largest K tested'
                        if row == 1 else 'No stable trial', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color=MUTED)
                ax.set_yticks([])
                continue
            frame = df[(df['phase'] == 'trial') & (df['trial_id'] == trial['trial_id'])]
            ax.plot(frame['t_local'], frame['joint_pos_deg'] - center, color=INK, linewidth=0.7)
            reference_line(ax, pulse_end, orientation='v')
            reason = '' if trial['stable'] else f" ({trial['reason']})"
            ax.set_title(f"{mode_label(mode)}: {title.lower()}, "
                         f"K = {trial['virtual_k_nm_rad']:.4g} {TORQUE_UNIT}/rad, "
                         f"B = {trial['virtual_b_nms_rad']:.3g}{reason}", fontsize=7)
            style_axis(ax, minor=False)
            if col == 0:
                ax.set_ylabel('Deviation (deg)')
            if row == 1:
                ax.set_xlabel('Time in trial (s)')
    fig.tight_layout(pad=0.3)
    return fig


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    trials = classify_all(run)
    if trials.empty:
        raise SystemExit('No Z-width trials found in this run.')
    modes = list(dict.fromkeys(trials['mode']))
    boundary = boundary_table(trials)
    summary = summarize(trials, boundary, modes)
    cfg = run.meta.get('experiment_config') or {}
    pulse_end, planned, cap = protocol(run)
    metadata = {
        'analysis': 'zwidth_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('pulse response per (mode, B, K); stable if the oscillation decays to the '
                   'noise floor; K_max = largest stable K below the first unstable K'),
        'pulse_torque_nm': cfg.get('pulse_torque_nm'),
        'pulse_width_s': cfg.get('pulse_width_s'),
        'trial_torque_cap_nm': cap,
        'observe_s': planned - pulse_end,
        'classification': (run.meta.get('zwidth_protocol') or {}).get('classification'),
        'note': ('With Coulomb friction the boundary depends on the disturbance size; '
                 'marginally stable trials count as unstable.'),
    }
    save_outputs(run, {'trials': trials, 'boundary': boundary}, summary, metadata)

    print_header('Z-WIDTH (LARGEST STABLE VIRTUAL STIFFNESS)')
    print(f"Trials: {summary['zwidth_num_trials']} ({summary['zwidth_num_unstable']} unstable); "
          f"pulse {cfg.get('pulse_torque_nm')} N m x {cfg.get('pulse_width_s')} s")
    for mode in modes:
        rate = summary.get(f'zwidth_render_rate_{mode}_hz', math.nan)
        print(f'  {mode_label(mode)} (measured loop rate {rate:.0f} Hz)')
        for _, b in boundary[boundary['mode'] == mode].sort_values('virtual_b_nms_rad').iterrows():
            ks = b['k_last_stable_nm_rad']
            ku = b['k_first_unstable_nm_rad']
            if not b['limit_reached']:
                text = f'>= {ks:.4g} (no instability up to the largest K tested)'
            elif math.isnan(ks):
                text = f'< {ku:.4g} (unstable at the first K tested)'
            else:
                text = f'{ks:.4g} (first unstable {ku:.4g})'
            flag = '  [non-monotonic]' if b['stable_above_instability'] else ''
            print(f"    B = {b['virtual_b_nms_rad']:<6.3g} N m s/rad: K_max {text} N m/rad{flag}")
    if 'zwidth_online_offline_agreement' in summary:
        print(f"  Online/offline classification agreement: "
              f"{100 * summary['zwidth_online_offline_agreement']:.0f}%")
    if run.simulated:
        print('Simulated run: no ground-truth Z-width; for host loops expect roughly '
              'K_max ~ (b + B) / T_delay, raised at B = 0 by friction.')

    fig = plot_zwidth(trials, boundary, modes)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    diag = plot_examples(run, trials, modes)
    export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
