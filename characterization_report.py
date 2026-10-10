"""Collect the latest analysis summaries into one characterization table.

Reads data/<experiment>/<run>/analysis/summary.json for each experiment (the
newest real run by default) and writes

    report/characterization_summary.csv   every metric with its source run
    report/characterization_table.tex     IEEE-style table for \\input{}
    report/characterization_table.md      the same table in Markdown

Torque-derived quantities (inertia, damping, friction, stiffness, torque) are
in the torque scale the drives report. If a torque calibration run exists, its
scale factor (actual / commanded torque) is applied in a separate
"calibrated" column, so both are visible. The report also cross-checks
quantities that several experiments measure independently.

Run:  python characterization_report.py [--include-sim] [--run EXPERIMENT=PATH ...]
"""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from experiment_common import DATA_ROOT, run_is_simulated

REPORT_DIR = Path('report')
DELAY_RESOLUTION_MS = 0.1   # Impedance-spectrum delay resolution at 40 Hz
NM = 'N$\\cdot$m'
MNM = 'mN$\\cdot$m'


def metric(group, experiment, key, label, unit, scale=1.0, decimals=2, torque_scaled=False,
           ci=None, display=None):
    """ci: None (try <key>_ci95_low/high), ('half', key) or ('bounds', low, high)."""
    return dict(group=group, experiment=experiment, key=key, label=label, unit=unit,
                scale=scale, decimals=decimals, torque_scaled=torque_scaled, ci=ci,
                display=display)


def bandwidth_display(summary, value, fmt):
    if summary.get('bandwidth_above_range') and summary.get('bandwidth_lower_bound_hz'):
        return f"$>${summary['bandwidth_lower_bound_hz']:.0f}"
    return fmt(value)


METRICS = [
    metric('Inertia and friction', 'step_response', 'J_kg_m2', 'Inertia $J$ (step response)',
           'kg$\\cdot$cm$^2$', 1e4, 1, True),
    metric('Inertia and friction', 'step_response', 'b_nms_rad',
           'Viscous damping $b$ (step response)', f'{MNM}$\\cdot$s/rad', 1e3, 1, True),
    metric('Inertia and friction', 'step_response', 'fc_nm',
           'Coulomb friction $f_c$ (step response)', MNM, 1e3, 1, True),
    metric('Inertia and friction', 'friction_sweep', 'coulomb_friction_nm',
           'Coulomb friction $f_c$ (velocity sweep)', MNM, 1e3, 1, True,
           ci=('half_se', 'coulomb_friction_se_nm')),
    metric('Inertia and friction', 'friction_sweep', 'viscous_damping_nms_rad',
           'Viscous damping $b$ (velocity sweep)', f'{MNM}$\\cdot$s/rad', 1e3, 1, True,
           ci=('half_se', 'viscous_damping_se_nms_rad')),
    metric('Inertia and friction', 'friction_sweep', 'stribeck_velocity_deg_s',
           'Stribeck velocity', 'deg/s', 1.0, 2),
    metric('Inertia and friction', 'friction_ramp', 'static_friction_nm',
           'Static (breakaway) friction $f_s$', MNM, 1e3, 1, True,
           ci=('half', 'static_friction_ci95_nm')),
    metric('Inertia and friction', 'friction_sweep', 'coulomb_percent_of_max_torque',
           'Coulomb friction / torque cap', '\\%', 1.0, 2),
    metric('Inertia and friction', 'friction_sweep', 'ripple_std_corrected_nm',
           'Position-dependent torque ripple (RMS)', MNM, 1e3, 1, True),
    metric('Closed-loop performance', 'bandwidth', 'bandwidth_hz',
           'Position bandwidth ($-3$ dB)', 'Hz', 1.0, 1, display=bandwidth_display),
    metric('Closed-loop performance', 'bandwidth', 'fit_bandwidth_hz',
           'Position bandwidth (fitted model)', 'Hz', 1.0, 1),
    metric('Closed-loop performance', 'bandwidth', 'fit_J_kg_m2', 'Inertia $J$ (FRF fit)',
           'kg$\\cdot$cm$^2$', 1e4, 1, True),
    metric('Closed-loop performance', 'closed_loop_step', 'rise_time_ms',
           'Rise time, 10--90\\% (small step)', 'ms', 1.0, 1),
    metric('Closed-loop performance', 'closed_loop_step', 'overshoot_percent',
           'Overshoot (small step)', '\\%', 1.0, 1),
    metric('Closed-loop performance', 'closed_loop_step', 'settling_time_ms',
           'Settling time, 2\\% (small step)', 'ms', 1.0, 0),
    metric('Closed-loop performance', 'closed_loop_step', 'peak_velocity_deg_s',
           'Peak velocity', 'deg/s', 1.0, 0),
    metric('Closed-loop performance', 'closed_loop_step', 'saturated_acceleration_rad_s2',
           'Acceleration at the torque cap', 'rad/s$^2$', 1.0, 0),
    metric('Closed-loop performance', 'closed_loop_step', 'peak_power_w',
           'Peak mechanical power (steps)', 'W', 1.0, 0, True),
    metric('Closed-loop performance', 'dynamic_envelope', 'crossover_10deg_hz',
           'Highest frequency tracking $\\pm$10$^\\circ$ linearly', 'Hz', 1.0, 1),
    metric('Closed-loop performance', 'dynamic_envelope', 'torque_line_ratio_median',
           'Large-signal envelope / torque-limited amplitude', '--', 1.0, 2),
    metric('Closed-loop performance', 'dynamic_envelope', 'peak_power_w',
           'Peak mechanical power (sinusoidal tracking)', 'W', 1.0, 0, True),
    metric('Haptic rendering', 'impedance', 'impedance_free_J_kg_m2',
           'Free-space apparent inertia (impedance spectrum)', 'kg$\\cdot$cm$^2$', 1e4, 1, True),
    # Per-rendering-loop rows (Z-width, impedance, virtual wall) come from PATTERN_METRICS.
    metric('Actuator and transmission', 'transmission', 'stiffness_nm_rad',
           'Transmission stiffness', f'{NM}/rad', 1.0, 0, True),
    metric('Actuator and transmission', 'transmission', 'lost_motion_deg',
           'Lost motion (backlash)', 'deg', 1.0, 3),
    metric('Actuator and transmission', 'transmission', 'resonance_hz',
           'Locked-output resonance', 'Hz', 1.0, 0),
    metric('Actuator and transmission', 'transmission', 'motor_side_inertia_kg_m2',
           'Reflected motor-side inertia', 'kg$\\cdot$cm$^2$', 1e4, 1, True),
    metric('Actuator and transmission', 'torque_calibration', 'torque_scale_factor',
           'Torque scale factor (measured / commanded)', '--', 1.0, 3),
    metric('Actuator and transmission', 'torque_calibration', 'nonlinearity_percent_fs',
           'Torque nonlinearity', '\\% FS', 1.0, 2),
    metric('Actuator and transmission', 'torque_calibration', 'hysteresis_at_zero_nm',
           'Torque hysteresis at zero', MNM, 1e3, 0),
    metric('Actuator and transmission', 'thermal', 'peak_torque_cap_nm',
           'Peak torque (configured cap)', NM, 1.0, 1, True),
    metric('Actuator and transmission', 'thermal', 'continuous_torque_nm',
           'Continuous torque', NM, 1.0, 1, True),
    metric('Actuator and transmission', 'thermal', 'thermal_time_constant_s',
           'Winding thermal time constant', 's', 1.0, 0),
    metric('Transparency (felt at the handle)', 'transparency', 'zero_torque_J_app_kg_m2',
           'Apparent inertia, zero torque', 'kg$\\cdot$cm$^2$', 1e4, 1),
    metric('Transparency (felt at the handle)', 'transparency', 'zero_torque_b_app_nms_rad',
           'Apparent damping, zero torque', f'{MNM}$\\cdot$s/rad', 1e3, 1),
    metric('Transparency (felt at the handle)', 'transparency', 'zero_torque_f_app_nm',
           'Apparent friction, zero torque', MNM, 1e3, 1),
    metric('Transparency (felt at the handle)', 'transparency', 'compensated_f_app_nm',
           'Apparent friction, with compensation', MNM, 1e3, 1),
    metric('Transparency (felt at the handle)', 'transparency', 'friction_reduction_percent',
           'Friction reduction by compensation', '\\%', 1.0, 0),
]

# Same physical quantity from different experiments: (label, [(experiment, key)]).
CROSS_CHECKS = [
    ('Inertia J (kg m^2)', [('step_response', 'J_kg_m2'), ('bandwidth', 'fit_J_kg_m2'),
                            ('transparency', 'zero_torque_J_app_kg_m2')]),
    ('Coulomb friction (N m)', [('step_response', 'fc_nm'),
                                ('friction_sweep', 'coulomb_friction_nm'),
                                ('transparency', 'zero_torque_f_app_nm')]),
    ('Viscous damping (N m s/rad)', [('step_response', 'b_nms_rad'),
                                     ('friction_sweep', 'viscous_damping_nms_rad'),
                                     ('transparency', 'zero_torque_b_app_nms_rad')]),
    ('Static friction (N m)', [('friction_ramp', 'static_friction_nm'),
                               ('friction_sweep', 'static_friction_stribeck_nm')]),
]



def latest_analysis(experiment, include_sim, override=None):
    """(summary.json path, run dir) of the newest analyzed real run (or sim)."""
    if override:
        path = Path(override)
        summary = path / 'analysis' / 'summary.json' if path.is_dir() else path
        run = path if path.is_dir() else summary.parent.parent
        return (summary, run) if summary.exists() else (None, None)
    root = DATA_ROOT / experiment
    if not root.is_dir():
        return None, None
    runs = sorted((d for d in root.iterdir() if (d / 'analysis' / 'summary.json').exists()),
                  key=lambda d: d.name)
    real = [d for d in runs if not run_is_simulated(d, experiment)]
    candidates = real if real else (runs if include_sim else [])
    if not candidates:
        return None, None
    return candidates[-1] / 'analysis' / 'summary.json', candidates[-1]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ci_bounds(summary, m):
    ci = m['ci']
    value = summary.get(m['key'])
    if ci is None:
        low, high = summary.get(f"{m['key']}_ci95_low"), summary.get(f"{m['key']}_ci95_high")
        return (low, high) if finite(low) and finite(high) else (None, None)
    kind, *keys = ci
    if kind == 'half' and finite(summary.get(keys[0])) and finite(value):
        return value - summary[keys[0]], value + summary[keys[0]]
    if kind == 'half_se' and finite(summary.get(keys[0])) and finite(value):
        return value - 1.96 * summary[keys[0]], value + 1.96 * summary[keys[0]]
    if kind == 'bounds' and finite(summary.get(keys[0])) and finite(summary.get(keys[1])):
        return summary[keys[0]], summary[keys[1]]
    return None, None


def mode_label(mode):
    if mode == 'rt':
        return '500 Hz loop'
    match = re.fullmatch(r'host_(\d+(?:\.\d+)?)', mode)
    if match:
        rate = float(match.group(1))
        return f'host loop, {rate / 1000:g}\\,kHz' if rate >= 1000 else f'host loop, {rate:g}\\,Hz'
    return mode.replace('_', ' ')


# Rows generated per rendering loop: (experiment, key regex with the loop name as group 1,
# label with {loop} and {kref}, unit, scale, decimals, torque-scaled, lower-bound rule).
# The lower-bound rule returns True when the value is only a lower bound (shown as >=).
PATTERN_METRICS = [
    ('zwidth', r'zwidth_k_max_(.+)_b0_nm_rad', 'Max.\\ stable virtual stiffness, $B=0$ ({loop})',
     f'{NM}/rad', 1.0, 0, True,
     lambda s, loop: not finite(s.get(f'zwidth_k_first_unstable_{loop}_b0_nm_rad'))),
    ('impedance', r'impedance_delay_(.+)_ms', 'Effective rendering delay ({loop})', 'ms', 1.0, 2,
     False, 'resolution'),
    ('impedance', r'impedance_predicted_k_crit_(.+)_b0_nm_rad',
     'Predicted linear stiffness limit, $B=0$ ({loop})', f'{NM}/rad', 1.0, 0, True, None),
    ('impedance', r'impedance_fidelity_bandwidth_(.+)_hz', 'Impedance fidelity bandwidth ({loop})',
     'Hz', 1.0, 1, False, lambda s, loop: bool(s.get(f'impedance_fidelity_lower_bound_{loop}'))),
    ('virtual_wall', r'restitution_(.+)_kref', 'Wall restitution, $K$={kref} ({loop})', '--', 1.0,
     2, False, None),
    ('virtual_wall', r'rate_hardness_ratio_(.+)_kref', 'Rate-hardness / $K$, $K$={kref} ({loop})',
     '--', 1.0, 2, False, None),
    ('virtual_wall', r'max_passive_k_(.+)_nm_rad', 'Largest passive wall stiffness ({loop})',
     f'{NM}/rad', 1.0, 0, True,
     lambda s, loop: bool(s.get(f'max_passive_k_{loop}_is_lower_bound'))),
]


def pattern_rows(summaries):
    rows = []
    for experiment, pattern, label, unit, scale, decimals, scaled, bound in PATTERN_METRICS:
        summary = summaries.get(experiment)
        if not summary:
            continue
        kref = summary.get('reference_k_nm_rad')
        for key in sorted(summary):
            match = re.fullmatch(pattern, key)
            if not match or isinstance(summary[key], bool):
                continue
            loop = match.group(1)
            if bound == 'resolution':
                # Delays within the method's resolution (~0.1 ms) read as "< 0.1".
                display = (lambda s, v, fmt: f'$<${DELAY_RESOLUTION_MS:g}'
                           if finite(v) and v < DELAY_RESOLUTION_MS else fmt(v))
            elif bound and bound(summary, loop):
                display = lambda s, v, fmt: f'$\\geq${fmt(v)}'
            else:
                display = None
            rows.append(metric('Haptic rendering', experiment, key,
                               label.format(loop=mode_label(loop),
                                            kref=f'{kref:g}' if finite(kref) else '--'),
                               unit, scale, decimals, scaled, display=display))
    return rows


def build_table(summaries, sources, k_torque):
    metrics = list(METRICS)
    insert = max(i for i, m in enumerate(metrics) if m['group'] == 'Haptic rendering') + 1
    metrics[insert:insert] = pattern_rows(summaries)
    rows = []
    for m in metrics:
        summary = summaries.get(m['experiment'])
        if summary is None or not finite(summary.get(m['key'])) and m['display'] is None:
            continue
        value = summary.get(m['key'])

        def fmt(v, m=m):
            if not finite(v):
                return '--'
            shown = v * m['scale']
            if abs(shown) < 0.5 * 10 ** -m['decimals']:
                shown = 0.0   # Avoid '-0.000'
            return f"{shown:.{m['decimals']}f}"

        display = m['display'](summary, value, fmt) if m['display'] else fmt(value)
        if display == '--':
            continue
        low, high = ci_bounds(summary, m)
        calibrated = value * k_torque if (m['torque_scaled'] and k_torque and finite(value)) else None
        label = m['label']
        if m['experiment'] == 'transparency' and summary.get('torque_source') == 'model_based':
            label += ' (model-based)'
        rows.append({
            'group': m['group'],
            'experiment': m['experiment'],
            'key': m['key'],
            'label': label,
            'unit': m['unit'],
            'value': value,
            'ci95_low': low,
            'ci95_high': high,
            'calibrated_value': calibrated,
            'display': display,
            'display_ci': f'[{fmt(low)}, {fmt(high)}]' if low is not None else '',
            'display_calibrated': ((m['display'](summary, calibrated, fmt) if m['display']
                                    else fmt(calibrated)) if calibrated is not None else ''),
            'source_run': str(sources[m['experiment']]),
        })
    return pd.DataFrame(rows)


def latex_table(table, k_torque):
    has_cal = k_torque is not None
    ncols = 5 if has_cal else 4
    header = 'Quantity & Unit & Value & 95\\% CI' + (' & Calibrated' if has_cal else '')
    lines = [
        '% Generated by characterization_report.py; edit labels and captions as needed.',
        '\\begin{table}[t]',
        '\\centering',
        '\\caption{Measured characteristics of the wrist joint (joint side).}',
        '\\label{tab:characterization}',
        '\\footnotesize',
        f"\\begin{{tabular}}{{l c c c{' c' if has_cal else ''}}}",
        '\\hline',
        header + ' \\\\',
        '\\hline',
    ]
    for group, rows in table.groupby('group', sort=False):
        lines.append(f'\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{group}}}}} \\\\')
        for _, row in rows.iterrows():
            cells = [row['label'], row['unit'], row['display'], row['display_ci'] or '--']
            if has_cal:
                cells.append(row['display_calibrated'] or '--')
            lines.append(' & '.join(cells) + ' \\\\')
    lines += ['\\hline', '\\end{tabular}']
    if has_cal:
        lines.append(f'\\par\\vspace{{2pt}}{{\\footnotesize Calibrated: torque-derived values '
                     f'multiplied by the measured torque scale factor {k_torque:.3f}.}}')
    lines += ['\\end{table}', '']
    return '\n'.join(lines)


def plain(text):
    for old, new in (('$\\cdot$', '·'), ('$^2$', '²'), ('$\\geq$', '≥'), ('$>$', '>'),
                     ('$<$', '<'),
                     ('$\\pm$', '±'), ('$^\\circ$', '°'),
                     ('$-3$', '−3'), ('\\%', '%'), ('\\,', ' '), ('\\ ', ' '), ('--', '–'),
                     ('$', '')):
        text = text.replace(old, new)
    return text


def markdown_table(table):
    lines = ['| Quantity | Unit | Value | 95% CI | Calibrated | Source |',
             '|---|---|---|---|---|---|']
    for group, rows in table.groupby('group', sort=False):
        lines.append(f'| **{group}** | | | | | |')
        for _, row in rows.iterrows():
            lines.append(f"| {plain(row['label'])} | {plain(row['unit'])} | {plain(row['display'])} "
                         f"| {plain(row['display_ci'])} | {plain(row['display_calibrated'])} | "
                         f"{Path(row['source_run']).name} |")
    return '\n'.join(lines) + '\n'


def cross_check_lines(summaries, k_torque):
    lines = []
    for label, entries in CROSS_CHECKS:
        values = [(e, summaries[e][k]) for e, k in entries
                  if e in summaries and finite(summaries[e].get(k))]
        if len(values) < 2:
            continue
        text = ', '.join(f'{e} {v:.4g}' for e, v in values)
        lo, hi = min(v for _, v in values), max(v for _, v in values)
        spread = 100.0 * (hi - lo) / abs(values[0][1]) if values[0][1] else math.nan
        lines.append(f'  {label}: {text} (spread {spread:.0f}%)')
    imp, wall = summaries.get('impedance', {}), summaries.get('virtual_wall', {})
    for key in sorted(imp):
        match = re.fullmatch(r'impedance_delay_(.+)_ms', key)
        if match and finite(imp[key]) and finite(wall.get(f'torque_delay_{match.group(1)}_median_ms')):
            loop = match.group(1)
            lines.append(f"  Rendering delay ({loop}, ms): impedance spectrum {imp[key]:.3g}, "
                         f"wall impacts {wall[f'torque_delay_{loop}_median_ms']:.3g}")
        match = re.fullmatch(r'impedance_predicted_k_crit_(.+)_b0_nm_rad', key)
        if match and finite(imp[key]):
            loop = match.group(1)
            zw = summaries.get('zwidth', {}).get(f'zwidth_k_first_unstable_{loop}_b0_nm_rad')
            if finite(zw):
                lines.append(f'  Stiffness limit ({loop}, B=0, N m/rad): linear prediction '
                             f'{imp[key]:.3g}, pulse Z-width {zw:.3g} (friction quenches small '
                             'oscillations, so the pulse test reads higher)')
    if lines and k_torque:
        lines.append(f'  (transparency values are sensor-measured; the others are in the drive torque '
                     f'scale: multiply them by {k_torque:.3f} before comparing)')
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--include-sim', action='store_true',
                        help='fall back to simulated runs when no real run exists')
    parser.add_argument('--run', action='append', default=[],
                        help='EXPERIMENT=PATH to a run directory or summary.json')
    parser.add_argument('--out', default=str(REPORT_DIR))
    args = parser.parse_args(argv)
    overrides = dict(item.split('=', 1) for item in args.run)

    summaries, sources = {}, {}
    experiments = {m['experiment'] for m in METRICS} | {p[0] for p in PATTERN_METRICS}
    for experiment in sorted(experiments | set(overrides)):
        summary_path, run = latest_analysis(experiment, args.include_sim, overrides.get(experiment))
        if summary_path is None:
            print(f'  (no analyzed run for {experiment})')
            continue
        summaries[experiment] = json.loads(summary_path.read_text())
        sources[experiment] = run
    if not summaries:
        raise SystemExit('No analyzed runs found. Run the experiments and their analyses first '
                         '(or pass --include-sim to use simulated runs).')

    cal = summaries.get('torque_calibration', {})
    k_torque = cal.get('torque_scale_factor') if finite(cal.get('torque_scale_factor')) else None
    table = build_table(summaries, sources, k_torque)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'characterization_summary.csv', index=False)
    (out / 'characterization_table.tex').write_text(latex_table(table, k_torque))
    (out / 'characterization_table.md').write_text(markdown_table(table))

    print(markdown_table(table))
    checks = cross_check_lines(summaries, k_torque)
    if checks:
        print('Cross-checks (same quantity from different experiments):')
        print('\n'.join(checks))
    simulated = [e for e, run in sources.items() if run_is_simulated(run, e)]
    if simulated:
        print(f"WARNING: simulated runs used for: {', '.join(simulated)}")
    print(f'Wrote {out}/characterization_summary.csv, characterization_table.tex, '
          'characterization_table.md')
    return table


if __name__ == '__main__':
    main()
