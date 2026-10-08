"""Continuous torque rating from the locked-output thermal test.

For each constant-torque level the temperature is fitted with a first-order
response that allows an incompletely cooled start,

    T(t) = T_ss + (T_0 - T_ss) exp(-t / tau),

giving the steady-state rise dT_ss = T_ss - T_start (T_start from the
zero-torque baseline) and the thermal time constant tau. Copper loss is
I^2 R with I proportional to torque, so dT_ss = c * torque^2; a fit through
the origin gives c and the continuous torque rating

    tau_cont = sqrt((T_limit - T_start) / c),

the torque whose steady state just reaches the temperature limit at the
measured starting temperature. The mean q-axis current squared per level is
reported as the copper-loss proxy (dT_ss = c_I * I^2).

Run:  python thermal_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from analysis_common import (export_figure, load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, sim_truth)
from ieee_plot import (INK, MUTED, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       direct_label, figure_size, ordinal_colors, reference_line, style_axis)

EXPERIMENT = 'thermal'
FIGURE_BASENAME = 'fig_thermal'
DIAGNOSTIC_BASENAME = 'diag_thermal_timeseries'
MIN_COVERAGE = 1.0     # Duration / tau below this means dT_ss is extrapolated


def first_order(t, t_ss, t0, tau):
    return t_ss + (t0 - t_ss) * np.exp(-t / tau)


def fit_level(t, temp):
    rise = temp[-1] - temp[0]
    p0 = [temp[-1] + 0.5 * rise, temp[0], max((t[-1] - t[0]) / 3.0, 1.0)]
    try:
        popt, pcov = optimize.curve_fit(first_order, t, temp, p0=p0,
                                        bounds=([-50, -50, 0.5], [500, 500, 1e5]), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [math.nan] * 3
    residual = temp - first_order(t, *popt)
    return {'T_ss_c': float(popt[0]), 'T0_c': float(popt[1]), 'tau_s': float(popt[2]),
            'T_ss_sd_c': float(perr[0]), 'tau_sd_s': float(perr[2]),
            'fit_rms_c': float(np.sqrt(np.mean(residual ** 2)))}


def analyze(run):
    df = run.df
    channel = run.meta.get('temperature_channel', 'motor_temp_c')
    base = df[df['phase'] == 'baseline']
    start_c = float(run.meta.get('start_temperature_c', base[channel].median()))
    limit_c = float(run.meta.get('temperature_limit_c', 90.0))
    reasons = run.meta.get('level_end_reasons') or []
    levels = []
    fits = {}
    for level, g in df[df['phase'] == 'heat'].groupby('level', sort=True):
        level = int(level)
        t = g['time'].to_numpy() - g['time'].iloc[0]
        temp = g[channel].to_numpy()
        ok = np.isfinite(temp)
        fit = fit_level(t[ok], temp[ok]) if ok.sum() > 10 else None
        torque = float(g['joint_torque_nm'].abs().mean())
        row = {
            'level': level,
            'commanded_torque_nm': float(g['cmd_ff_torque_nm'].abs().mean()),
            'measured_torque_nm': torque,
            'mean_current_sq_a2': float(np.nanmean(g['q_current_a'] ** 2)),
            'duration_s': float(t[-1]),
            'temperature_start_c': float(temp[ok][0]) if ok.any() else math.nan,
            'temperature_end_c': float(temp[ok][-1]) if ok.any() else math.nan,
            'end_reason': reasons[level] if level < len(reasons) else '',
        }
        if fit:
            row.update(fit)
            row['dT_ss_c'] = fit['T_ss_c'] - start_c
            row['coverage'] = row['duration_s'] / fit['tau_s']
            fits[level] = (t, temp, fit)
        levels.append(row)
    table = pd.DataFrame(levels)
    good = table.dropna(subset=['dT_ss_c']) if 'dT_ss_c' in table else table.iloc[0:0]
    summary = {'temperature_channel': channel, 'start_temperature_c': start_c,
               'temperature_limit_c': limit_c, 'num_levels': int(len(good))}
    if len(good):
        x = good['measured_torque_nm'].to_numpy() ** 2
        y = good['dT_ss_c'].to_numpy()
        c = float(np.sum(x * y) / np.sum(x * x))
        xi = good['mean_current_sq_a2'].to_numpy()
        c_i = float(np.sum(xi * y) / np.sum(xi * xi))
        allowed = limit_c - start_c
        tau_cont = math.sqrt(allowed / c) if c > 0 and allowed > 0 else math.nan
        weights = np.clip(good['coverage'].to_numpy(), 0, 5)
        tau_th = float(np.sum(weights * good['tau_s']) / np.sum(weights))
        peak = float(run.meta.get('device_config', {}).get('MAX_JOINT_TORQUE_NM', math.nan))
        summary.update({
            'rise_per_torque_sq_c_per_nm2': c,
            'rise_per_current_sq_c_per_a2': c_i,
            'continuous_torque_nm': tau_cont,
            'continuous_current_a': math.sqrt(allowed / c_i) if c_i > 0 and allowed > 0 else math.nan,
            'peak_torque_cap_nm': peak,
            'peak_to_continuous_ratio': peak / tau_cont if tau_cont else math.nan,
            'thermal_time_constant_s': tau_th,
            'thermal_time_constant_spread_s': float(good['tau_s'].max() - good['tau_s'].min()),
            'fit_r2': float(1 - np.sum((y - c * x) ** 2) / np.sum((y - y.mean()) ** 2))
            if len(y) > 1 else math.nan,
            'min_coverage': float(good['coverage'].min()),
        })
    return table, summary, fits


# --- figures -------------------------------------------------------------------------------
def plot_thermal(table, summary, fits):
    fig, (ax_t, ax_s) = plt.subplots(2, 1, figsize=figure_size('single', 3.9),
                                     gridspec_kw={'height_ratios': [1.1, 1.0], 'hspace': 0.45})
    levels = list(fits)
    colors = dict(zip(levels, ordinal_colors(len(levels))))
    start = summary['start_temperature_c']
    minutes = max(t[-1] for t, _, _ in fits.values()) > 180
    scale = 60.0 if minutes else 1.0
    for level, (t, temp, fit) in fits.items():
        torque = table.loc[table['level'] == level, 'measured_torque_nm'].iloc[0]
        ax_t.plot(t / scale, temp - start, color=colors[level], linewidth=1.8,
                  label=f'{torque:.1f} {TORQUE_UNIT}')
        ax_t.plot(t / scale, first_order(t, fit['T_ss_c'], fit['T0_c'], fit['tau_s']) - start,
                  color=INK, linestyle=(0, (3, 3)), linewidth=0.6)
    ax_t.plot([], [], color=INK, linestyle=(0, (3, 3)), linewidth=0.6, label='First-order fit')
    ax_t.set_xlabel('Time at level (min)' if minutes else 'Time at level (s)')
    ax_t.set_ylabel('Temperature rise (°C)')
    style_axis(ax_t)
    ax_t.legend(loc='upper left', ncol=2)
    top = max(np.nanmax(temp - start) for _, temp, _ in fits.values())
    ax_t.set_ylim(min(0.0, ax_t.get_ylim()[0]), 1.45 * top)
    add_panel_label(ax_t, '(a)')

    good = table.dropna(subset=['dT_ss_c'])
    c = summary['rise_per_torque_sq_c_per_nm2']
    allowed = summary['temperature_limit_c'] - start
    tau_cont = summary['continuous_torque_nm']
    for _, row in good.iterrows():
        level = int(row['level'])
        ax_s.errorbar(row['measured_torque_nm'] ** 2, row['dT_ss_c'],
                      yerr=2 * row['T_ss_sd_c'] if np.isfinite(row['T_ss_sd_c']) else None,
                      linestyle='None', marker='o', markersize=4, markerfacecolor=colors[level],
                      markeredgecolor=colors[level], ecolor=colors[level], elinewidth=0.6)
    x_max = max(good['measured_torque_nm'].max() ** 2, tau_cont ** 2 if np.isfinite(tau_cont) else 0)
    x = np.linspace(0, 1.08 * x_max, 50)
    ax_s.plot(x, c * x, color=INK, linestyle='--', linewidth=1.0,
              label=f'Fit, {c:.2f} °C/({TORQUE_UNIT})$^2$')
    reference_line(ax_s, allowed, label=f"Limit ({summary['temperature_limit_c']:.0f} °C)")
    if np.isfinite(tau_cont):
        ax_s.plot([tau_cont ** 2], [allowed], linestyle='None', marker='D', markersize=4.5,
                  markerfacecolor='white', markeredgecolor=INK)
        direct_label(ax_s, tau_cont ** 2, allowed,
                     f'continuous {tau_cont:.1f} {TORQUE_UNIT}', dx=-6, dy=5, ha='right',
                     va='bottom')
    ax_s.set_xlim(0, 1.08 * x_max)
    ax_s.set_ylim(0, 1.2 * max(allowed, good['dT_ss_c'].max()))
    ax_s.set_xlabel(f'Torque$^2$ (({TORQUE_UNIT})$^2$)')
    ax_s.set_ylabel('Steady-state rise (°C)')
    style_axis(ax_s)
    ax_s.legend(loc='lower right')
    add_panel_label(ax_s, '(b)')
    return fig


def plot_timeseries(df, channel):
    fig, (ax_t, ax_q) = plt.subplots(2, 1, figsize=figure_size('double', 3.0), sharex=True)
    ax_t.plot(df['time'], df[channel], color=INK, linewidth=0.8, label=channel)
    other = 'board_temp_c' if channel == 'motor_temp_c' else 'motor_temp_c'
    if other in df and np.isfinite(df[other]).any():
        ax_t.plot(df['time'], df[other], color=MUTED, linewidth=0.8, label=other)
    ax_q.plot(df['time'], df['joint_torque_nm'], color=INK, linewidth=0.6)
    ax_t.set_ylabel('Temperature (°C)')
    ax_q.set_ylabel(f'Torque ({TORQUE_UNIT})')
    ax_q.set_xlabel('Time (s)')
    ax_t.legend(loc='upper left')
    for ax in (ax_t, ax_q):
        style_axis(ax, minor=False)
    fig.tight_layout(pad=0.3)
    return fig


def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data)
    table, summary, fits = analyze(run)
    if not fits:
        raise SystemExit('No heat levels could be fitted.')
    metadata = {
        'analysis': 'thermal_experiment_analysis.py',
        'source_data': str(run.source),
        'model': 'T(t) = T_ss + (T_0 - T_ss) exp(-t/tau); dT_ss = c torque^2',
        'min_coverage': MIN_COVERAGE,
        'note': ('Continuous torque is for the measured starting temperature; board '
                 'temperature (if used) rates the driver, not the winding.'),
    }
    save_outputs(run, {'levels': table}, summary, metadata)

    print_header('THERMAL / CONTINUOUS TORQUE')
    print(f"  Channel {summary['temperature_channel']}, start {summary['start_temperature_c']:.1f} C, "
          f"limit {summary['temperature_limit_c']:.0f} C")
    for _, r in table.iterrows():
        print(f"  Level {int(r['level'])}: {r['measured_torque_nm']:.2f} N m, "
              f"dT_ss {r.get('dT_ss_c', math.nan):.2f} C, tau {r.get('tau_s', math.nan):.1f} s, "
              f"coverage {r.get('coverage', math.nan):.1f}, ended by {r['end_reason']}")
        if r.get('coverage', 99) < MIN_COVERAGE:
            print('    WARNING: held for less than one time constant; dT_ss is extrapolated.')
    if 'continuous_torque_nm' in summary:
        print(f"  Rise per torque^2      {summary['rise_per_torque_sq_c_per_nm2']:.4g} C/(N m)^2 "
              f"(R^2 {summary['fit_r2']:.4f})")
        print(f"  Continuous torque      {summary['continuous_torque_nm']:.3g} N m "
              f"({summary['continuous_current_a']:.3g} A q-axis)")
        print(f"  Peak cap / continuous  {summary['peak_to_continuous_ratio']:.2f}")
        print(f"  Thermal time constant  {summary['thermal_time_constant_s']:.1f} s")
    if summary['temperature_channel'] != 'motor_temp_c':
        print('  WARNING: rated from the BOARD temperature (driver), not the motor winding.')
    observed_rise = (table['temperature_end_c'] - table['temperature_start_c']).max()
    if not np.isfinite(observed_rise) or observed_rise < 1.0:
        print('  WARNING: the temperature barely changed; check the thermistor and '
              'servo.enable_motor_temperature before trusting this rating.')
        summary['warning'] = 'temperature channel did not respond'

    truth = sim_truth(run)
    if truth and 'continuous_torque_nm' in summary:
        k = truth['thermal_resistance'] * truth['winding_resistance'] / (
            truth['gear_ratio'] * truth['motor_torque_constant']) ** 2
        allowed = summary['temperature_limit_c'] - summary['start_temperature_c']
        rows = [('Rise per torque^2', summary['rise_per_torque_sq_c_per_nm2'], k, 'C/(N m)^2'),
                ('Continuous torque', summary['continuous_torque_nm'], math.sqrt(allowed / k), 'N m'),
                ('Thermal time constant', summary['thermal_time_constant_s'],
                 truth['thermal_time_constant'], 's')]
        print_truth_comparison(rows)

    channel = summary['temperature_channel']
    fig = plot_thermal(table, summary, fits)
    paths = export_figure(fig, run, FIGURE_BASENAME, {**metadata, **summary}, args.paper_dir)
    export_figure(plot_timeseries(run.df, channel), run, DIAGNOSTIC_BASENAME, metadata,
                  paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
