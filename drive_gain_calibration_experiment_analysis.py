"""Fit the P4 drive loop's gain units from its own torque demand.

For every drive that moves with the joint under test and every gain set,
least squares on the drive-loop samples

    demand = A (target - position) - B velocity + D offset

(demand and offset in per mille of rated torque, the drive's frame; position
and target in counts; velocity in counts/s from the logged positions). The
target the drive acts on at a sample is the one sent L cycles earlier (the
measured loop delay: latched at the next SYNC, reached by interpolation over
the period after), and so is the torque offset. Then

    k_unit = A / (kv kp kout / 100),   b_unit = B / kv,   D ~ 1 if 0x60B2 works in CSP.

These are the drive firmware's units, the same for every puck. The spread
across gain sets and drives checks that the law is linear in the gains.

Run:  python drive_gain_calibration_experiment_analysis.py [run_dir] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (export_figure, load_run, make_parser, print_header,
                             print_truth_comparison, run_j2mp, run_joint_index, run_nodes,
                             save_outputs)
from ieee_plot import INK, configure_ieee_style, figure_size, series_style, style_axis

EXPERIMENT = 'drive_gain_calibration'
FIGURE_BASENAME = 'fig_drive_gain_calibration'
PHASES = ('sinusoid', 'settle', 'offset')
MAX_SPREAD = 0.10          # Gain-set/drive spread above this: the law is not linear in the gains
OFFSET_TOLERANCE = 0.2     # |D - 1| below this: the torque offset is honored


def drive_frame_demand(frame, run, i, node):
    """The drive's torque demand (per mille, drive frame): logged demand, else current."""
    info = run.meta['drive_info'][i]
    config = run.meta['p4wrist_config']
    sign = config['torque_sign'][i]
    rated = info['rated_torque_mnm']
    demand = frame.get(f'demand_{node}_nm')
    if demand is not None and np.isfinite(demand).all():
        return sign * demand.to_numpy(float) * 1e6 / rated
    current = frame[f'current_{node}_a'].to_numpy(float)
    return current * info['kt_mnm_per_a'] * config['current_torque_factor'] * 1000.0 / rated


def fit_law(frame, run, i, node, lag, gains):
    """A, B, D for one drive and gain set."""
    period = float(run.meta['p4wrist_config']['period_s'])
    counts = frame[f'counts_{node}'].to_numpy(float)
    target = frame[f'target_{node}'].to_numpy(float)
    demand = drive_frame_demand(frame, run, i, node)
    T = run_j2mp(run)
    inv_t = np.linalg.inv(np.array(T, dtype=float).T)
    ff_joint = np.zeros((len(frame), 3))
    ff_joint[:, run_joint_index(run)] = frame['cmd_ff_torque_nm'].to_numpy(float)
    info = run.meta['drive_info'][i]
    sign = run.meta['p4wrist_config']['torque_sign'][i]
    offset = sign * (ff_joint @ inv_t.T)[:, i] * 1e6 / info['rated_torque_mnm']
    e = target[:-lag] - counts[lag:] if lag else target - counts
    v = np.gradient(counts, period)[lag:] if lag else np.gradient(counts, period)
    y = demand[lag:] if lag else demand
    off = offset[:-lag] if lag else offset
    keep = np.isfinite(e) & np.isfinite(v) & np.isfinite(y)
    A_mat = np.column_stack([e[keep], -v[keep], off[keep]])
    coef, *_ = np.linalg.lstsq(A_mat, y[keep], rcond=None)
    A, B, D = (float(c) for c in coef)
    residual = y[keep] - A_mat @ coef
    g_p = gains['kp'] * gains['kout'] / 100.0
    return {'node': node, 'A_permille_per_count': A, 'B_permille_per_count_s': B,
            'offset_coefficient': D, 'kv': gains['kv'], 'g_p': g_p,
            'k_unit_permille_per_count': A / (gains['kv'] * g_p),
            'b_unit_permille_per_count_s': B / gains['kv'],
            'residual_rms_permille': float(np.sqrt(np.mean(residual ** 2))),
            'num_samples': int(keep.sum()),
            '_y': y[keep], '_model': A_mat @ coef}


def plot_fit(fits):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.6))
    for k, f in enumerate(fits[:3]):
        ax.plot(f['_model'], f['_y'], label=f"drive {f['node']}, set {f['gain_set']}",
                **series_style(k, line=False))
    lim = max(max(np.abs(f['_y']).max() for f in fits), 1e-9)
    ax.plot([-lim, lim], [-lim, lim], color=INK, linewidth=0.8, linestyle='--', label='1:1')
    ax.set_xlabel('Fitted drive law (per mille)')
    ax.set_ylabel('Drive torque demand (per mille)')
    style_axis(ax)
    ax.legend(loc='upper left')
    fig.tight_layout(pad=0.3)
    return fig


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    df = run.df
    if 'drive_loop' not in df:
        raise SystemExit('No drive-loop data in this run.')
    sets = {s['gain_set']: s['gains'] for s in run.meta.get('gain_sets', [])}
    T, joint, nodes = run_j2mp(run), run_joint_index(run), run_nodes(run)
    lag = int(round(float(run.meta.get('loop_delay_s', 0.004))
                    / float(run.meta['p4wrist_config']['period_s'])))
    data = df[(df['drive_loop'] == 1) & df['phase'].isin(PHASES)]
    fits = []
    for set_id, frame in data.groupby('gain_set'):
        frame = frame.sort_values('loop_cycle')
        for i, node in enumerate(nodes):
            if T[i][joint] == 0.0 or f'target_{node}' not in frame:
                continue
            fit = fit_law(frame, run, i, node, lag, sets[int(set_id)][i])
            fit['gain_set'] = int(set_id)
            fits.append(fit)
    if not fits:
        raise SystemExit('No drive moved with the joint under test.')
    table = pd.DataFrame([{k: v for k, v in f.items() if not k.startswith('_')} for f in fits])
    k_unit = float(table['k_unit_permille_per_count'].mean())
    b_unit = float(table['b_unit_permille_per_count_s'].mean())
    spread = max(float(table['k_unit_permille_per_count'].std(ddof=0)) / abs(k_unit),
                 float(table['b_unit_permille_per_count_s'].std(ddof=0)) / abs(b_unit))
    offset = float(table['offset_coefficient'].mean())
    offset_works = bool((table['offset_coefficient'] - 1.0).abs().max() < OFFSET_TOLERANCE)
    info = run.meta['drive_info']
    to_si = [d['counts_per_rev'] / (2.0 * math.pi) * d['rated_torque_mnm'] / 1e6 for d in info]
    summary = {
        'k_unit_permille_per_count': k_unit,
        'b_unit_permille_per_count_s': b_unit,
        'gain_linearity_spread': spread,
        'linear_in_gains': spread < MAX_SPREAD,
        'torque_offset_coefficient': offset,
        'torque_offset_works': offset_works,
        'motor_k_per_unit_nm_rad': [k_unit * f for f in to_si],
        'motor_b_per_unit_nms_rad': [b_unit * f for f in to_si],
        'loop_delay_cycles': lag,
        'num_fits': len(fits),
    }
    metadata = {
        'analysis': 'drive_gain_calibration_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('least squares of each drive\'s torque demand on (target - position), '
                   'velocity and torque offset, target and offset taken loop_delay earlier'),
        'units': 'demand and offset in per mille of rated torque; position in counts',
    }
    save_outputs(run, {'fits': table}, summary, metadata)

    print_header('DRIVE GAIN CALIBRATION')
    for f in fits:
        print(f"  set {f['gain_set']} drive {f['node']}: kv {f['kv']:.4g}, g_p {f['g_p']:.4g} -> "
              f"k_unit {f['k_unit_permille_per_count']:.4g}, b_unit "
              f"{f['b_unit_permille_per_count_s']:.4g}, offset x{f['offset_coefficient']:.3f}, "
              f"residual {f['residual_rms_permille']:.2f} per mille")
    print(f"  Units: k {k_unit:.5g} per mille/count per (kv g_p), b {b_unit:.5g} per mille/(count/s) "
          f"per kv; spread {100 * spread:.1f}% "
          f"({'linear' if summary['linear_in_gains'] else 'NOT linear: check the gain sets'})")
    print(f"  Torque offset in the drive loop: x{offset:.3f} "
          f"({'honored' if offset_works else 'NOT honored: no feedforward in the drive loop'})")
    if run.simulated:
        params = run.meta.get('sim_params') or {}
        truth_offset = 1.0 if params.get('torque_offset_in_csp', True) else 0.0
        print_truth_comparison([('k unit', k_unit, 1.0, ''), ('b unit', b_unit, 1.0, ''),
                                ('Torque offset coefficient', offset, truth_offset, '')])
    paths = export_figure(plot_fit(fits), run, FIGURE_BASENAME, {**metadata, **summary},
                          args.paper_dir)
    print(f"Saved results to {run.out_dir}; figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
