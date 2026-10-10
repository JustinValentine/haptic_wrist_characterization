"""Static (breakaway) friction from the ramp-and-hold data.

Breakaway is detected from displacement, not from a differentiated velocity
(which is swamped by encoder noise at these speeds): in each ramp the stuck
position is the median of the first samples, and the joint has broken away
once its displacement along the ramp direction exceeds max(6 sigma, 0.005 deg),
with sigma the encoder noise measured during the pauses.

Any displacement threshold fires late: right after breakaway the net torque
on the joint is nearly zero and only grows with the ramp, so displacement
grows like (t - t0)^3 and the spring torque keeps rising while the joint
creeps up to the threshold. The onset t0 is therefore found by fitting
theta = theta0 + A max(0, t - t0)^p to the samples around the crossing, with
p = 3 for a pure torque ramp and p = 2 or 1 where cogging lets the joint creep
along a detent before it snaps, and
the static friction torque is the spring torque Kp (th_cmd(t0) - theta0).
If that fit fails, the onset falls back to the last sample whose displacement
was below a quarter of the threshold (biased high by a few percent).

On a cogging motor the breakaway torque is static friction plus the cogging
torque at the stuck position, which spreads individual breakaways by the
cogging amplitude. When a friction-sweep analysis of the same kind (real or
simulated) has identified a cogging model, its torque at each breakaway's
absolute motor position is subtracted; both values are reported.

Run:  python ramp_input_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import (cogging_torque, export_figure, latest_cogging_model, load_run,
                             run_joint_index,
                             make_parser, mean_std_ci, print_header, print_truth_comparison,
                             robust_sigma, save_outputs, sim_truth)
from ieee_plot import (INK, MUTED, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       figure_size, reference_line, series_style, style_axis)

EXPERIMENT = 'friction_ramp'
LEGACY = [('friction_ramp_data.csv', 'friction_ramp_collection_metadata.json')]
FIGURE_BASENAME = 'fig_static_friction'
EXAMPLE_BASENAME = 'fig_static_friction_example'
DIAGNOSTIC_BASENAME = 'diag_static_friction_ramps'
NOISE_FACTOR = 6.0            # Threshold = NOISE_FACTOR * encoder noise sigma ...
MIN_THRESHOLD_DEG = 0.005     # ... but never below this
ONSET_FRACTION = 0.25         # Back-date onset to displacement <= this * threshold
STUCK_WINDOW_S = 0.02         # Samples at ramp start that define the stuck position
THRESHOLD_FILL = '#e7e6e3'    # MUTED blended 20% over white
ONSET_FIT_PRE_S = 0.25        # Cubic onset fit: window before the threshold crossing ...
ONSET_FIT_POST = 4.0          # ... and up to this many thresholds of displacement
ONSET_POWERS = (1.0, 2.0, 3.0)   # Onset shapes tried: displacement ~ (t - t0)^p


# --- data preparation ----------------------------------------------------------------
def ramp_groups(df):
    """Yield (ramp_id, ramp_rows, pause_rows) in time order.

    New data carries ramp_id on both the ramp and the pause that follows it;
    legacy data used one segment number for a ramp and its pause.
    """
    key = 'ramp_id' if 'ramp_id' in df.columns else 'segment'
    data = df[df[key].notna()] if key in df.columns else df
    for ramp_id, frame in data.groupby(key, sort=True):
        frame = frame.sort_values('time')
        ramp = frame[frame['phase'] == 'ramp']
        pause = frame[frame['phase'] == 'pause']
        if len(ramp) > 10:
            yield int(ramp_id), ramp, pause


def noise_sigma_deg(df):
    """Encoder noise from the second half of every pause (difference-based)."""
    diffs = []
    for _, _, pause in ramp_groups(df):
        if len(pause) < 20:
            continue
        late = pause['joint_pos_deg'].to_numpy()[len(pause) // 2:]
        diffs.append(np.diff(late))
    if not diffs:
        return 0.0
    return robust_sigma(np.concatenate(diffs)) / math.sqrt(2.0)


def ramp_direction(ramp):
    if 'direction' in ramp.columns and ramp['direction'].notna().any():
        value = float(ramp['direction'].dropna().iloc[0])
        if value != 0:
            return int(np.sign(value))
    delta = ramp['cmd_joint_deg'].iloc[-1] - ramp['cmd_joint_deg'].iloc[0]
    return int(np.sign(delta))


def refine_onset(t, theta, direction, first, threshold_deg, lower_bound):
    """Fit theta = theta0 + A max(0, t - t0)^p around the threshold crossing.

    Each p in ONSET_POWERS is tried and the best least-squares fit kept. Returns
    (t0, theta0, A, p) with A in deg/s^p along the ramp direction, or None when no
    power describes the onset (no positive A, or t0 pinned to the window edge).
    """
    dt = float(np.median(np.diff(t)))
    i0 = max(lower_bound, first - int(round(ONSET_FIT_PRE_S / dt)))
    if first - i0 < 10:
        return None
    y_all = direction * theta
    base = np.median(y_all[i0:first])
    above = np.flatnonzero(y_all[first:] - base > ONSET_FIT_POST * threshold_deg)
    i1 = first + (int(above[0]) if above.size else int(round(0.1 / dt)))
    i1 = min(i1, len(t) - 1)
    tt, yy = t[i0:i1 + 1], y_all[i0:i1 + 1]
    candidates = tt[:first - i0 + 1]
    y_mean = yy.mean()
    best = None
    for power in ONSET_POWERS:
        x = np.clip(tt[None, :] - candidates[:, None], 0.0, None) ** power
        x_mean = x.mean(axis=1, keepdims=True)
        sxx = ((x - x_mean) ** 2).sum(axis=1)
        sxy = ((x - x_mean) * (yy - y_mean)[None, :]).sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            slope = sxy / sxx
        intercept = y_mean - slope * x_mean[:, 0]
        sse = (((intercept[:, None] + slope[:, None] * x) - yy[None, :]) ** 2).sum(axis=1)
        sse[~(slope > 0)] = np.inf
        k = int(np.argmin(sse))
        if np.isfinite(sse[k]) and k > 0 and (best is None or sse[k] < best[0][best[1]]):
            best = (sse, k, slope, intercept, power)
    if best is None:
        return None
    sse, k, slope, intercept, power = best
    t0 = float(candidates[k])
    if 0 < k < len(candidates) - 1 and np.isfinite(sse[k - 1]) and np.isfinite(sse[k + 1]):
        denom = sse[k - 1] - 2 * sse[k] + sse[k + 1]
        if denom > 0:
            shift = 0.5 * (sse[k - 1] - sse[k + 1]) / denom
            t0 += max(-1.0, min(1.0, shift)) * (candidates[1] - candidates[0])
    return t0, direction * float(intercept[k]), float(slope[k]), power


def detect_breakaway(ramp_id, ramp, kp, threshold_deg):
    t = ramp['time'].to_numpy()
    theta = ramp['joint_pos_deg'].to_numpy()
    cmd = ramp['cmd_joint_deg'].to_numpy()
    direction = ramp_direction(ramp)
    result = {
        'ramp_id': ramp_id,
        'cycle': int(ramp['cycle'].iloc[0]) if 'cycle' in ramp.columns else 0,
        'direction': direction,
        'ramp_start_deg': float(cmd[0]),
        'ramp_end_deg': float(cmd[-1]),
        'threshold_deg': threshold_deg,
        'breakaway': False,
        'reason': '',
    }
    if direction == 0:
        result['reason'] = 'no command change'
        return result
    n0 = max(5, int(round(STUCK_WINDOW_S / np.median(np.diff(t)))))
    stuck = float(np.median(theta[:n0]))
    displacement = direction * (theta - stuck)
    if np.ptp(theta[:n0]) > threshold_deg:
        result['reason'] = 'moving at ramp start'
        return result
    moved = np.flatnonzero(displacement > threshold_deg)
    if moved.size == 0:
        result['reason'] = 'no breakaway during ramp'
        return result
    first = moved[0]
    quiet = np.flatnonzero(displacement[:first] <= ONSET_FRACTION * threshold_deg)
    onset = int(quiet[-1]) if quiet.size else first
    if onset < n0:
        result['reason'] = 'breakaway inside the stuck window'
        return result
    threshold_torque = kp * math.radians(cmd[onset] - theta[onset])
    refined = refine_onset(t, theta, direction, int(first), threshold_deg, n0)
    if refined is not None:
        t0, theta0, onset_a, onset_power = refined
        cmd0 = float(np.interp(t0, t, cmd))
        onset_method = 'fit'
    else:
        t0, theta0, cmd0 = float(t[onset]), float(theta[onset]), float(cmd[onset])
        onset_a, onset_power = math.nan, math.nan
        onset_method = 'threshold'
    spring_torque = kp * math.radians(cmd0 - theta0)
    result.update({
        'breakaway': True,
        'onset_method': onset_method,
        'onset_fit_a': onset_a,
        'onset_power': onset_power,
        'onset_index': onset,
        'first_crossing_index': int(first),
        'time_s': t0,
        'time_in_ramp_s': t0 - float(t[0]),
        'pos_deg': theta0,
        'cmd_deg': cmd0,
        'lag_deg': cmd0 - theta0,
        'stuck_pos_deg': stuck,
        'torque_signed_nm': spring_torque,
        'torque_nm': direction * spring_torque,
        'torque_threshold_nm': direction * threshold_torque,
        'control_torque_nm': (float(ramp['control_joint_torque_nm'].iloc[onset])
                                     if 'control_joint_torque_nm' in ramp.columns else math.nan),
        'measured_torque_nm': (float(ramp['joint_torque_nm'].iloc[onset])
                                      if 'joint_torque_nm' in ramp.columns else math.nan),
        'motor_pos_rev': (float(np.interp(t0, t, ramp['motor_pos_rev'].to_numpy()))
                          if 'motor_pos_rev' in ramp.columns else math.nan),
        **{c: float(np.interp(t0, t, ramp[c].to_numpy())) for c in ramp.columns
           if c.startswith('motor_angle_')},
    })
    return result


def position_trend(events):
    """Linear trend of breakaway torque with position (pooled directions)."""
    if len(events) < 4 or events['pos_deg'].nunique() < 3:
        return {'slope_nm_per_deg': math.nan, 'r2': math.nan, 'p_value': math.nan}
    fit = stats.linregress(events['pos_deg'], events['torque_nm'])
    return {'slope_nm_per_deg': float(fit.slope), 'r2': float(fit.rvalue ** 2),
            'p_value': float(fit.pvalue)}


# --- figures ---------------------------------------------------------------------------
def plot_summary(events, stats_by_direction):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.35))
    top = 0.0
    for i, (direction, label) in enumerate([(1, 'Positive'), (-1, 'Negative')]):
        group = events[events['direction'] == direction]
        if group.empty:
            continue
        mean = stats_by_direction[direction]['mean']
        ax.plot(group['pos_deg'], group['torque_nm'],
                label=f'{label} (mean {mean:.3f} {TORQUE_UNIT})', **series_style(i, line=False))
        ax.axhline(mean, color=series_style(i)['color'], linestyle='--', linewidth=0.9,
                   zorder=1)
        top = max(top, group['torque_nm'].max())
    ax.set_ylim(0, top * 1.35)
    ax.set_xlabel('Joint position at breakaway (deg)')
    ax.set_ylabel(f'Breakaway torque ({TORQUE_UNIT})')
    style_axis(ax)
    ax.legend(loc='upper center', ncol=2, columnspacing=1.2)
    fig.tight_layout(pad=0.3)
    return fig


def example_ramp(events, groups):
    """Typical ramp for the example figure: positive direction, breakaway well
    after the ramp starts (so the torque rise is visible), torque near the median."""
    candidates = events[(events['direction'] == 1) & (events['time_in_ramp_s'] > 0.3)]
    if candidates.empty:
        candidates = events[events['time_in_ramp_s'] > 0.3]
    if candidates.empty:
        candidates = events
    median = candidates['torque_nm'].median()
    ramp_id = int(candidates.iloc[(candidates['torque_nm'] - median).abs().argmin()]['ramp_id'])
    return ramp_id, groups[ramp_id]


def plot_example(event, ramp, kp):
    t = ramp['time'].to_numpy()
    t_start = t[0]
    t = t - t_start
    theta = ramp['joint_pos_deg'].to_numpy()
    cmd = ramp['cmd_joint_deg'].to_numpy()
    direction = event['direction']
    torque = direction * kp * np.radians(cmd - theta)
    displacement = direction * (theta - event['stuck_pos_deg'])
    t0 = event['time_s'] - t_start
    threshold = event['threshold_deg']
    first = int(event['first_crossing_index'])
    end = min(len(t) - 1, first + max(int(0.5 * first), 100))
    window = slice(0, end + 1)

    fig, (ax_tq, ax_d) = plt.subplots(2, 1, figsize=figure_size('single', 3.3), sharex=True,
                                      gridspec_kw={'hspace': 0.32})
    ax_tq.plot(t[window], torque[window], color=INK, label='Spring torque')
    ax_tq.plot(t0, event['torque_nm'], linestyle='None', marker='D', markersize=4.5,
               markerfacecolor='white', markeredgecolor=INK, label='Breakaway')
    reference_line(ax_tq, event['torque_nm'])
    ax_tq.set_ylim(min(0.0, float(np.min(torque[window])) * 1.1), event['torque_nm'] * 1.3)
    ax_tq.set_ylabel(f'Torque ({TORQUE_UNIT})')
    style_axis(ax_tq)
    ax_tq.legend(loc='lower right')
    add_panel_label(ax_tq, '(a)', x=-0.17)

    # Zoomed on the onset: the slip itself runs off the top of the axis.
    scale = 1e3
    # Opaque light fill: EPS has no transparency.
    ax_d.axhspan(-threshold * scale, threshold * scale, color=THRESHOLD_FILL, linewidth=0,
                 label='Detection threshold')
    ax_d.plot(t[window], displacement[window] * scale, color=INK, linewidth=0.8,
              label='Displacement')
    if event.get('onset_method') == 'fit':
        tt = t[window]
        offset = direction * (event['pos_deg'] - event['stuck_pos_deg'])
        power = event['onset_power']
        model = offset + event['onset_fit_a'] * np.clip(tt - t0, 0.0, None) ** power
        ax_d.plot(tt, model * scale, color='#2a78d6', linestyle='--', linewidth=1.0,
                  label=f'Onset fit, (t - t0)^{power:g}')
    ax_d.plot(t0, direction * (event['pos_deg'] - event['stuck_pos_deg']) * scale,
              linestyle='None', marker='D', markersize=4.5, markerfacecolor='white',
              markeredgecolor=INK, label='Breakaway onset')
    ax_d.set_ylim(-3 * threshold * scale, 12 * threshold * scale)
    ax_d.set_xlabel('Time since ramp start (s)')
    ax_d.set_ylabel('Displacement (mdeg)')
    style_axis(ax_d)
    ax_d.legend(loc='upper left')
    add_panel_label(ax_d, '(b)', x=-0.17)
    fig.align_ylabels([ax_tq, ax_d])
    return fig


def plot_diagnostics(results, groups, kp):
    n = len(results)
    cols = 6
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figure_size('double', 0.95 * rows + 0.4),
                             squeeze=False)
    for ax, (_, r) in zip(axes.flat, results.iterrows()):
        ramp = groups[int(r['ramp_id'])]
        t = ramp['time'].to_numpy() - ramp['time'].iloc[0]
        theta = ramp['joint_pos_deg'].to_numpy()
        stuck = np.median(theta[:5])
        ax.plot(t, r['direction'] * (theta - stuck), color=INK, linewidth=0.6)
        ax.axhline(r['threshold_deg'], color=MUTED, linewidth=0.6)
        if r['breakaway']:
            onset = int(r['onset_index'])
            ax.plot(t[onset], r['direction'] * (theta[onset] - stuck), 'o',
                    markerfacecolor='none', markeredgecolor='#eb6834', markersize=3)
            label = f"#{int(r['ramp_id'])} {r['torque_nm']:.3f}"
        else:
            label = f"#{int(r['ramp_id'])} {r['reason']}"
        ax.text(0.03, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontsize=5)
        ax.tick_params(labelsize=5, direction='in')
    for ax in axes.flat[n:]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('Displacement (deg)', fontsize=6)
    fig.supxlabel('Time since ramp start (s)', fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data, LEGACY)
    df = run.df.sort_values('time').reset_index(drop=True)
    kp = float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad'))

    sigma = noise_sigma_deg(df)
    threshold = max(NOISE_FACTOR * sigma, MIN_THRESHOLD_DEG)
    groups = {}
    results = []
    for ramp_id, ramp, _ in ramp_groups(df):
        groups[ramp_id] = ramp
        results.append(detect_breakaway(ramp_id, ramp, kp, threshold))
    if not results:
        raise SystemExit('No ramps found in the data.')
    results = pd.DataFrame(results)
    events = results[results['breakaway']].copy()
    if events.empty:
        raise SystemExit('No breakaway detected in any ramp; check that the spring torque '
                         'exceeds static friction (increase kp_joint_nm_rad or ramp_step_deg).')

    # Breakaway = static friction + cogging at the stuck position: remove the cogging.
    events['breakaway_torque_nm'] = events['torque_nm']
    cogging = latest_cogging_model(run.simulated)
    if cogging is not None and events['motor_pos_rev'].notna().all():
        events['cogging_nm'] = cogging_torque(cogging, events, run_joint_index(run))
        events['torque_nm'] = events['direction'] * (events['torque_signed_nm']
                                                     - events['cogging_nm'])
    else:
        events['cogging_nm'] = math.nan
    for column in ('breakaway_torque_nm', 'cogging_nm', 'torque_nm'):
        results.loc[events.index, column] = events[column]
    raw = mean_std_ci(events['breakaway_torque_nm'])
    overall = mean_std_ci(events['torque_nm'])
    by_direction = {d: mean_std_ci(events.loc[events['direction'] == d, 'torque_nm'])
                    for d in (1, -1)}
    trend = position_trend(events)
    summary = {
        'static_friction_nm': overall['mean'],
        'static_friction_std_nm': overall['std'],
        'static_friction_ci95_nm': overall['ci95'],
        'static_friction_pos_nm': by_direction[1]['mean'],
        'static_friction_pos_std_nm': by_direction[1]['std'],
        'static_friction_pos_ci95_nm': by_direction[1]['ci95'],
        'static_friction_neg_nm': by_direction[-1]['mean'],
        'static_friction_neg_std_nm': by_direction[-1]['std'],
        'static_friction_neg_ci95_nm': by_direction[-1]['ci95'],
        'breakaway_torque_nm': raw['mean'],
        'breakaway_torque_std_nm': raw['std'],
        'cogging_removed': cogging is not None,
        'cogging_model_source': cogging['source'] if cogging else '',
        'static_friction_threshold_method_nm': float(events['torque_threshold_nm'].mean()),
        'num_fit_onsets': int((events['onset_method'] == 'fit').sum()),
        'onset_power_counts': {f'{p:g}': int((events['onset_power'] == p).sum())
                               for p in ONSET_POWERS},
        'static_friction_max_nm': float(events['torque_nm'].max()),
        'static_friction_min_nm': float(events['torque_nm'].min()),
        'direction_asymmetry_nm': by_direction[1]['mean'] - by_direction[-1]['mean'],
        'position_slope_nm_per_deg': trend['slope_nm_per_deg'],
        'position_trend_r2': trend['r2'],
        'position_trend_p_value': trend['p_value'],
        'num_ramps': int(len(results)),
        'num_breakaways': int(len(events)),
        'num_breakaways_pos': by_direction[1]['n'],
        'num_breakaways_neg': by_direction[-1]['n'],
        'num_without_breakaway': int((~results['breakaway']).sum()),
        'noise_sigma_deg': sigma,
        'threshold_deg': threshold,
        'kp_joint_nm_rad': kp,
    }
    metadata = {
        'analysis': 'ramp_input_experiment_analysis.py',
        'source_data': str(run.source),
        'legacy_input': run.legacy,
        'method': ('displacement-threshold breakaway detection; onset from a power-law fit '
                   'theta0 + A max(0, t - t0)^p, p in ONSET_POWERS (fallback: back-dated to '
                   f'{ONSET_FRACTION} x threshold); torque = Kp (cmd - theta) at onset, minus '
                   'the cogging model at the onset motor position when one exists'),
        'onset_powers': list(ONSET_POWERS),
        'onset_fit_pre_s': ONSET_FIT_PRE_S,
        'onset_fit_post_thresholds': ONSET_FIT_POST,
        'noise_factor': NOISE_FACTOR,
        'min_threshold_deg': MIN_THRESHOLD_DEG,
        'onset_fraction': ONSET_FRACTION,
        'stuck_window_s': STUCK_WINDOW_S,
        'note': ('Torques are in the torque scale the drives apply (Kp in N m/rad); multiply by '
                 'the torque calibration ratio if one was measured.'),
    }
    save_outputs(run, {'breakaway_events': results}, summary, metadata)

    print_header('STATIC FRICTION (RAMP-AND-HOLD)')
    print(f"Ramps: {summary['num_ramps']}  breakaways: {summary['num_breakaways']} "
          f"(+{summary['num_breakaways_pos']} / -{summary['num_breakaways_neg']})  "
          f"Kp = {kp:g} N m/rad")
    print(f'Encoder noise sigma {sigma * 1e3:.2f} mdeg -> threshold {threshold * 1e3:.2f} mdeg')
    if cogging is not None:
        print(f"  Cogging removed with the model from friction_sweep {cogging['source']}: "
              f"breakaway torque {raw['mean']:.4g} N m (SD {raw['std']:.3g}) includes it")
    else:
        print('  No cogging model found (run the friction sweep analysis first); '
              'breakaway torque includes any cogging')
    print(f"  Static friction (all)   {overall['mean']:.4g} N m  SD {overall['std']:.3g}  "
          f"95% CI +/-{overall['ci95']:.3g}")
    for d, label in ((1, 'positive'), (-1, 'negative')):
        s = by_direction[d]
        print(f"  Static friction ({label:<8}) {s['mean']:.4g} N m  SD {s['std']:.3g}  n {s['n']}")
    print(f"  Onsets from the power-law fit: {summary['num_fit_onsets']}/{len(events)} "
          f"(p = 1/2/3: {'/'.join(str(v) for v in summary['onset_power_counts'].values())}); "
          f"threshold-only method gives {summary['static_friction_threshold_method_nm']:.4g} N m")
    print(f"  Position trend: {trend['slope_nm_per_deg']:.3g} N m/deg "
          f"(R^2 {trend['r2']:.2f}, p {trend['p_value']:.2g})")
    missing = results[~results['breakaway']]
    if not missing.empty:
        print(f'  Ramps without a usable breakaway: {len(missing)} '
              f"({', '.join(sorted(set(missing['reason'])))})")
    truth = sim_truth(run)
    if truth:
        print_truth_comparison([('Static friction', overall['mean'], truth['static'], 'N m')])
        if cogging is None:
            print(f"  (cogging +/-{truth['cogging_amplitude']:.3g} N m spreads individual "
                  'breakaways)')

    fig = plot_summary(events, by_direction)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    ramp_id, ramp = example_ramp(events, groups)
    event = events[events['ramp_id'] == ramp_id].iloc[0]
    export_figure(plot_example(event, ramp, kp), run, EXAMPLE_BASENAME,
                  {**metadata, 'ramp_id': ramp_id}, args.paper_dir)
    export_figure(plot_diagnostics(results, groups, kp), run, DIAGNOSTIC_BASENAME, metadata,
                  paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
