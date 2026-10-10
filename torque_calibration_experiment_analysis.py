"""Fit measured output torque against commanded torque (locked output).

For every held level the analysis averages the external sensor (or takes the
typed reading) and the drive-reported torque, then fits

    measured = k * commanded + offset

over both approach directions. k is the torque scale factor (actual /
commanded): multiply J, b, fc, and stiffness values identified from
drive-reported torque by k. It combines torque-constant error and static
transmission efficiency. Nonlinearity is the largest fit residual as a
percentage of full scale; hysteresis is the difference between levels
approached from below (up branch) and from above (down branch), which with a
locked output is mostly friction between the motor and the cable.

The motor's cogging torque at the clamped position also reaches the sensor.
When a friction-sweep analysis of the same kind has identified a cogging
model, its torque at each level's motor position is added back to the
measured torque before the fit.

Run:  python torque_calibration_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_common import (cogging_torque, export_figure, latest_cogging_model, load_run,
                             run_joint_index,
                             make_parser, print_header, print_truth_comparison, save_outputs,
                             sim_truth)
from ieee_plot import (INK, LINESTYLES, MUTED, SERIES, TORQUE_UNIT, add_panel_label,
                       configure_ieee_style, figure_size, pad_top_tick, series_style,
                       style_axis, zero_line)

EXPERIMENT = 'torque_calibration'
FIGURE_BASENAME = 'fig_torque_calibration'
DIAGNOSTIC_BASENAME = 'diag_torque_calibration_timeseries'
BRANCH_LABELS = {'up': 'Increasing torque', 'down': 'Decreasing torque'}


def level_table(run):
    df = run.df
    measure = df[df['phase'] == 'measure']
    manual = {int(r['level_id']): r for r in run.meta.get('manual_readings', [])}
    levels = {int(l['level_id']): l for l in run.meta.get('levels', [])}
    rows = []
    for level_id, group in measure.groupby('level_id'):
        level_id = int(level_id)
        level = levels.get(level_id, {})
        row = {
            'level_id': level_id,
            'commanded_nm': float(group['level_nm'].iloc[0]),
            'branch': str(group['branch'].iloc[0]),
            'reported_nm': float(group['joint_torque_nm'].mean()),
            'q_current_a': float(group['q_current_a'].mean()),
            'deflection_deg': float(group['joint_pos_deg'].mean()),
            'motor_pos_rev': (float(group['motor_pos_rev'].mean())
                              if 'motor_pos_rev' in group else math.nan),
            **{c: float(group[c].mean()) for c in group.columns if c.startswith('motor_angle_')},
            'num_samples': int(len(group)),
            'previous_nm': level.get('previous_nm', math.nan),
        }
        if level_id in manual:
            row['measured_nm'] = float(manual[level_id]['torque_nm'])
            row['measured_sd_nm'] = math.nan
        elif 'ext_torque_nm' in group and group['ext_torque_nm'].notna().any():
            row['measured_nm'] = float(group['ext_torque_nm'].mean())
            row['measured_sd_nm'] = float(group['ext_torque_nm'].std(ddof=1))
        else:
            row['measured_nm'] = math.nan
            row['measured_sd_nm'] = math.nan
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('level_id').reset_index(drop=True)
    return table[table['measured_nm'].notna()].reset_index(drop=True)


def linear_fit(x, y, extra=None):
    """Least squares y = k x + offset (+ extra columns); returns coefficients, R^2, residual."""
    columns = [x, np.ones_like(x)] + ([] if extra is None else [extra])
    A = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else math.nan
    return [float(c) for c in coef], r2, residual


def hysteresis_pairs(table):
    """Loop width (down minus up) at levels approached from both sides."""
    loaded = table[table['branch'].isin(['up', 'down'])]
    pairs = []
    for level, group in loaded.groupby(loaded['commanded_nm'].round(6)):
        up = group[group['branch'] == 'up']['measured_nm']
        down = group[group['branch'] == 'down']['measured_nm']
        if len(up) and len(down):
            pairs.append({'commanded_nm': float(level), 'up_nm': float(up.mean()),
                          'down_nm': float(down.mean()),
                          'hysteresis_nm': float(down.mean() - up.mean())})
    return pd.DataFrame(pairs)


def analyze(table):
    """Hysteresis-aware calibration: measured = k cmd + offset + h * branch.

    branch is -1 on the up branch and +1 on the down branch, so h is half the
    loop width. Fitting one line through both branches instead biases k,
    because the extreme levels are each visited on only one branch.
    """
    fit_rows = table[table['branch'].isin(['up', 'down'])]
    x = fit_rows['commanded_nm'].to_numpy()
    y = fit_rows['measured_nm'].to_numpy()
    branch = np.where(fit_rows['branch'] == 'up', -1.0, 1.0)
    (k, offset, half_width), r2, residual = linear_fit(x, y, branch)
    (k_naive, _), r2_naive, _ = linear_fit(x, y)
    full_scale = float(np.max(np.abs(x)))
    (k_rep, _), _, _ = linear_fit(x, fit_rows['reported_nm'].to_numpy())
    branch_k = {}
    for name in ('up', 'down'):
        sel = (fit_rows['branch'] == name).to_numpy()
        if sel.sum() >= 2:
            branch_k[name] = linear_fit(x[sel], y[sel])[0][0]
    pairs = hysteresis_pairs(table)
    zero = pairs[pairs['commanded_nm'].abs() < 1e-9] if len(pairs) else pairs
    start = table[table['branch'] == 'start']
    summary = {
        'torque_scale_factor': k,
        'torque_offset_nm': offset,
        'hysteresis_half_width_nm': half_width,
        'r_squared': r2,
        'single_line_scale_factor': k_naive,
        'full_scale_nm': full_scale,
        'nonlinearity_percent_fs': 100.0 * float(np.max(np.abs(residual))) / full_scale,
        'rms_residual_nm': float(np.sqrt(np.mean(residual ** 2))),
        'hysteresis_max_nm': float(pairs['hysteresis_nm'].abs().max()) if len(pairs) else math.nan,
        'hysteresis_mean_nm': float(pairs['hysteresis_nm'].mean()) if len(pairs) else math.nan,
        'hysteresis_max_percent_fs': (100.0 * float(pairs['hysteresis_nm'].abs().max()) / full_scale
                                      if len(pairs) else math.nan),
        'hysteresis_at_zero_nm': float(zero['hysteresis_nm'].iloc[0]) if len(zero) else math.nan,
        'up_branch_scale_factor': branch_k.get('up', math.nan),
        'down_branch_scale_factor': branch_k.get('down', math.nan),
        'reported_vs_commanded_scale': k_rep,
        'measured_at_start_nm': float(start['measured_nm'].iloc[0]) if len(start) else math.nan,
        'max_deflection_deg': float(table['deflection_deg'].abs().max()),
        'num_levels': int(len(fit_rows)),
    }
    fit_rows = fit_rows.assign(fit_nm=k * x + offset + half_width * branch, residual_nm=residual,
                               residual_percent_fs=100.0 * residual / full_scale)
    return summary, fit_rows, pairs


def plot_calibration(fit_rows, summary):
    fig, (ax_fit, ax_dev) = plt.subplots(
        2, 1, figsize=figure_size('single', 3.8), sharex=True,
        gridspec_kw={'height_ratios': [2.0, 1.0], 'hspace': 0.16})
    span = summary['full_scale_nm'] * 1.08
    line = np.array([-span, span])
    k, offset = summary['torque_scale_factor'], summary['torque_offset_nm']
    ax_fit.plot(line, line, color=MUTED, linestyle='--', linewidth=0.8, label='Ideal (1:1)')
    ax_fit.plot(line, k * line + offset, color=INK, linestyle='--', linewidth=1.0,
                label=f'Calibration ($k$ = {k:.3f})')
    full_scale = summary['full_scale_nm']
    for i, branch in enumerate(('up', 'down')):
        rows = fit_rows[fit_rows['branch'] == branch].sort_values('commanded_nm')
        style = series_style(i, line=False, filled=i == 1)
        ax_fit.plot(rows['commanded_nm'], rows['measured_nm'], label=BRANCH_LABELS[branch],
                    **style)
        deviation = 100.0 * (rows['measured_nm'] - (k * rows['commanded_nm'] + offset)) / full_scale
        ax_dev.plot(rows['commanded_nm'], deviation, **{**style, 'linestyle': LINESTYLES[i],
                                                        'linewidth': 0.7})
    ax_fit.set_xlim(-span, span)
    ax_fit.set_ylim(-span, span)
    ax_fit.set_ylabel(f'Measured torque ({TORQUE_UNIT})')
    style_axis(ax_fit)
    ax_fit.legend(loc='upper left')
    add_panel_label(ax_fit, '(a)')

    zero_line(ax_dev)
    deviation_all = 100.0 * (fit_rows['measured_nm'] - (k * fit_rows['commanded_nm'] + offset)) / full_scale
    limit = max(0.5, float(np.nanmax(np.abs(deviation_all))) * 1.35)
    ax_dev.set_ylim(-limit, limit)
    ax_dev.set_xlabel(f'Commanded torque ({TORQUE_UNIT})')
    ax_dev.set_ylabel('Deviation (% FS)')
    style_axis(ax_dev)
    pad_top_tick(ax_dev)
    add_panel_label(ax_dev, '(b)')
    return fig


def plot_timeseries(df):
    data = df[df['phase'].isin(['transition', 'settle', 'measure', 'lock_check'])]
    t = data['time'] - data['time'].iloc[0]
    fig, (ax_t, ax_d) = plt.subplots(2, 1, figsize=figure_size('double', 3.2), sharex=True)
    ax_t.plot(t, data['cmd_ff_torque_nm'], color=INK, linewidth=0.8, label='Commanded')
    if 'ext_torque_nm' in data and data['ext_torque_nm'].notna().any():
        ax_t.plot(t, data['ext_torque_nm'], color=SERIES[0], linewidth=0.6, label='External sensor')
    ax_t.plot(t, data['joint_torque_nm'], color=SERIES[1], linewidth=0.6, linestyle='--',
              label='drive reported')
    measure = data['phase'] == 'measure'
    ax_t.plot(t[measure], data['cmd_ff_torque_nm'][measure], linestyle='None', marker='.',
              markersize=1.0, color=MUTED, label='Averaging windows')
    ax_t.set_ylabel(f'Torque ({TORQUE_UNIT})')
    style_axis(ax_t)
    lo, hi = ax_t.get_ylim()
    ax_t.set_ylim(lo, hi + 0.3 * (hi - lo))
    ax_t.legend(loc='upper right', ncol=4)
    ax_d.plot(t, data['joint_pos_deg'], color=INK, linewidth=0.7)
    ax_d.set_xlabel('Time (s)')
    ax_d.set_ylabel('Motor-side deflection (deg)')
    style_axis(ax_d)
    fig.tight_layout(pad=0.3)
    return fig


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    table = level_table(run)
    if len(table[table['branch'] != 'start']) < 3:
        raise SystemExit('Too few levels with a sensor reading to fit a calibration.')
    # The cogging torque at the clamped motor position also reaches the sensor.
    cogging = latest_cogging_model(run.simulated)
    table['measured_raw_nm'] = table['measured_nm']
    if cogging is not None and table['motor_pos_rev'].notna().all():
        table['cogging_nm'] = cogging_torque(cogging, table, run_joint_index(run))
        table['measured_nm'] = table['measured_raw_nm'] + table['cogging_nm']
    else:
        table['cogging_nm'] = math.nan
    summary, fit_rows, pairs = analyze(table)
    summary['sensor_mode'] = run.meta.get('sensor_mode', 'unknown')
    summary['cogging_removed'] = cogging is not None
    summary['cogging_model_source'] = cogging['source'] if cogging else ''
    metadata = {
        'analysis': 'torque_calibration_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('hysteresis-aware least squares over held levels: measured = '
                   'k * commanded + offset + h * branch (branch -1 up, +1 down)'),
        'note': ('torque_scale_factor = measured / commanded torque. Multiply torque-derived '
                 'parameters (J, b, fc, stiffness) from other analyses by it.'),
    }
    save_outputs(run, {'levels': table, 'fit': fit_rows, 'hysteresis': pairs}, summary, metadata)

    print_header('TORQUE CALIBRATION')
    print(f"Levels: {summary['num_levels']} ({summary['sensor_mode']} sensor), "
          f"full scale {summary['full_scale_nm']:.3g} N m")
    print(f"  Torque scale factor k     {summary['torque_scale_factor']:.4f}  "
          f"(offset {summary['torque_offset_nm']:+.4f} N m, R^2 {summary['r_squared']:.5f})")
    print(f"  Up / down branch k        {summary['up_branch_scale_factor']:.4f} / "
          f"{summary['down_branch_scale_factor']:.4f}  "
          f"(single line through both: {summary['single_line_scale_factor']:.4f})")
    print(f"  Nonlinearity              {summary['nonlinearity_percent_fs']:.2f} % FS")
    print(f"  Hysteresis width (max / mean) {summary['hysteresis_max_nm']:.4f} / "
          f"{summary['hysteresis_mean_nm']:.4f} N m "
          f"({summary['hysteresis_max_percent_fs']:.2f} % FS)")
    print(f"  Hysteresis at zero        {summary['hysteresis_at_zero_nm']:.4f} N m")
    print(f"  drive reported / commanded {summary['reported_vs_commanded_scale']:.4f}")
    print(f"  Reading at start (0 N m)  {summary['measured_at_start_nm']:+.4f} N m")
    if summary['torque_scale_factor'] < 0:
        print('WARNING: negative scale factor; the sensor sign is probably flipped '
              "(device_config EXTERNAL_TORQUE_SENSOR 'sign').")
    truth = sim_truth(run)
    if truth:
        print_truth_comparison([
            ('Torque scale factor', summary['torque_scale_factor'],
             truth['torque_constant_ratio'], ''),
            ('Hysteresis at zero (~2 fs)', summary['hysteresis_at_zero_nm'],
             2 * truth['static'], 'N m'),
        ])

    fig = plot_calibration(fit_rows, summary)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    diag = plot_timeseries(run.df)
    export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
