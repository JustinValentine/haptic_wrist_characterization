import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from ieee_plot import (
    configure_ieee_style,
    figure_size,
    save_ieee_figure,
    style_axis,
    write_json,
)


# --- CONFIGURATION ---
GEAR_RATIO = 10.0
KP_JOINT_NM_RAD = 15.0
DATA_FILE = 'step_response_data.csv'
RESULTS_FILE = 'step_response_results.csv'
SUMMARY_FILE = 'step_response_summary.csv'
METADATA_FILE = 'step_response_metadata.json'
FIGURE_BASENAME = 'fig_step_response'
MIN_PROMINENCE_DEG = 0.05
MIN_PEAK_DISTANCE_S = 0.05
SETTLING_FRACTION = 0.20
SHOW_FIGURES = True


def main():
    configure_ieee_style()

    df = pd.read_csv(DATA_FILE)
    gear_ratio = value_from_data_or_default(df, 'gear_ratio', GEAR_RATIO)
    kp_joint = value_from_data_or_default(df, 'kp_joint_nm_rad', KP_JOINT_NM_RAD)

    df = add_joint_columns(df, gear_ratio)
    df = add_segments(df)

    results = []
    marked_extrema = []

    for segment_id, segment_df in df.groupby('segment'):
        if len(segment_df) < 20:
            continue

        cmd_step_deg = float(segment_df['cmd_joint_deg'].iloc[-1])
        previous_cmd_deg = previous_command(df, segment_df)
        step_size_deg = cmd_step_deg - previous_cmd_deg
        if abs(step_size_deg) < 1.0:
            continue

        result, extrema = analyze_segment(segment_id, segment_df, kp_joint)
        if result is None:
            continue

        result['cmd_step_deg'] = cmd_step_deg
        result['step_size_deg'] = step_size_deg
        results.append(result)
        marked_extrema.extend(extrema)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No usable underdamped step responses found.")
        print("Check that the joint oscillated and that the CSV contains multiple peaks.")
        return

    results_df.to_csv(RESULTS_FILE, index=False)
    summary_df = summarize_results(results_df)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    metadata = {
        'analysis': 'step_input_experiment_analysis.py',
        'source_data': DATA_FILE,
        'results_file': RESULTS_FILE,
        'summary_file': SUMMARY_FILE,
        'gear_ratio': gear_ratio,
        'kp_joint_nm_rad': kp_joint,
        'min_prominence_deg': MIN_PROMINENCE_DEG,
        'min_peak_distance_s': MIN_PEAK_DISTANCE_S,
        'settling_fraction': SETTLING_FRACTION,
        'num_samples': int(len(df)),
        'num_segments_analyzed': int(len(results_df)),
    }
    write_json(METADATA_FILE, metadata)

    print("-" * 34)
    print("STEP RESPONSE RESULTS")
    print("-" * 34)
    print(f"Segments analyzed: {len(results_df)}")
    print(f"Mean inertia J:       {results_df['J_kg_m2'].mean():.6g} kg*m^2")
    print(f"Mean viscous damping: {results_df['b_nms_rad'].mean():.6g} Nm*s/rad")
    print(f"Mean damping ratio:   {results_df['zeta'].mean():.4f}")
    print(f"Mean natural freq:    {results_df['omega_n_rad_s'].mean():.4f} rad/s")
    print(f"Mean kinetic friction estimate: {results_df['fk_nm'].mean():.6g} Nm")
    print(f"Saved per-step results to '{RESULTS_FILE}'")
    print(f"Saved summary to '{SUMMARY_FILE}'")

    figure_paths = plot_step_response(df, marked_extrema, metadata)
    print(f"Saved IEEE figures: {', '.join(figure_paths.values())}")


def value_from_data_or_default(df, column, default):
    if column in df.columns and not df[column].dropna().empty:
        return float(df[column].dropna().iloc[0])
    return default


def add_joint_columns(df, gear_ratio):
    df = df.copy()

    if 'motor_pos_rel_rev' in df.columns:
        motor_pos_rel_rev = df['motor_pos_rel_rev']
    else:
        motor_pos_rel_rev = df['motor_pos_rev'] - df['motor_pos_rev'].iloc[0]

    if 'joint_pos_deg' not in df.columns:
        df['joint_pos_deg'] = (motor_pos_rel_rev / gear_ratio) * 360.0

    if 'cmd_joint_deg' not in df.columns:
        cmd_origin_rev = df['cmd_pos_rev'].iloc[0]
        df['cmd_joint_deg'] = ((df['cmd_pos_rev'] - cmd_origin_rev) / gear_ratio) * 360.0

    df['joint_pos_rad'] = np.radians(df['joint_pos_deg'])
    df['cmd_joint_rad'] = np.radians(df['cmd_joint_deg'])
    return df


def add_segments(df):
    df = df.copy()
    if 'segment' in df.columns:
        return df

    command_change = df['cmd_joint_deg'].diff().abs().fillna(0.0) > 1e-6
    df['segment'] = command_change.cumsum()
    return df


def previous_command(full_df, segment_df):
    start_index = segment_df.index[0]
    previous_rows = full_df.loc[:start_index - 1]
    if previous_rows.empty:
        return float(segment_df['cmd_joint_deg'].iloc[0])
    return float(previous_rows['cmd_joint_deg'].iloc[-1])


def analyze_segment(segment_id, segment_df, kp_joint):
    t = segment_df['time'].to_numpy()
    t = t - t[0]
    y = segment_df['joint_pos_rad'].to_numpy()
    cmd_rad = float(segment_df['cmd_joint_rad'].iloc[-1])

    tail_count = max(5, int(len(y) * SETTLING_FRACTION))
    equilibrium = float(np.median(y[-tail_count:]))
    y_centered = y - equilibrium

    max_amp = float(np.max(np.abs(y_centered)))
    if max_amp < math.radians(MIN_PROMINENCE_DEG):
        return None, []

    dt = float(np.median(np.diff(t)))
    min_distance = max(1, int(MIN_PEAK_DISTANCE_S / dt))
    prominence = max(math.radians(MIN_PROMINENCE_DEG), 0.03 * max_amp)

    peaks, _ = signal.find_peaks(
        y_centered,
        prominence=prominence,
        distance=min_distance
    )
    valleys, _ = signal.find_peaks(
        -y_centered,
        prominence=prominence,
        distance=min_distance
    )

    extrema = sorted(
        [(int(i), 1, y_centered[i]) for i in peaks]
        + [(int(i), -1, y_centered[i]) for i in valleys],
        key=lambda item: item[0]
    )

    extrema = [
        item for item in extrema
        if abs(item[2]) >= max(0.05 * max_amp, math.radians(MIN_PROMINENCE_DEG))
    ]

    if len(extrema) < 4:
        return None, []

    decrements = []
    periods = []
    for first, second in zip(extrema, extrema[2:]):
        i0, kind0, amp0 = first
        i1, kind1, amp1 = second
        if kind0 != kind1 or abs(amp1) <= 0:
            continue
        ratio = abs(amp0) / abs(amp1)
        period = t[i1] - t[i0]
        if ratio > 1.02 and period > 0:
            decrements.append(math.log(ratio))
            periods.append(period)

    if not decrements or not periods:
        return None, []

    log_decrement = float(np.median(decrements))
    damped_period = float(np.median(periods))
    zeta = log_decrement / math.sqrt((2.0 * math.pi) ** 2 + log_decrement ** 2)
    omega_d = (2.0 * math.pi) / damped_period
    omega_n = omega_d / math.sqrt(max(1.0 - zeta ** 2, 1e-12))
    inertia = kp_joint / (omega_n ** 2)
    damping = 2.0 * inertia * zeta * omega_n

    center_offsets = []
    for first, second in zip(extrema, extrema[1:]):
        i0, kind0, _ = first
        i1, kind1, _ = second
        if kind0 == kind1:
            continue
        midpoint = 0.5 * (y[i0] + y[i1])
        center_offsets.append(midpoint - cmd_rad)

    fk_nm = kp_joint * float(np.median(np.abs(center_offsets))) if center_offsets else np.nan

    result = {
        'segment': segment_id,
        'num_extrema': len(extrema),
        'log_decrement': log_decrement,
        'zeta': zeta,
        'omega_d_rad_s': omega_d,
        'omega_n_rad_s': omega_n,
        'J_kg_m2': inertia,
        'b_nms_rad': damping,
        'fk_nm': fk_nm,
        'equilibrium_deg': math.degrees(equilibrium),
        'cmd_deg': math.degrees(cmd_rad),
    }

    marked_extrema = [
        {
            'time': segment_df['time'].iloc[i],
            'joint_pos_deg': segment_df['joint_pos_deg'].iloc[i],
        }
        for i, _, _ in extrema
    ]
    return result, marked_extrema


def summarize_results(results_df):
    summary = {}
    for column in [
        'log_decrement',
        'zeta',
        'omega_d_rad_s',
        'omega_n_rad_s',
        'J_kg_m2',
        'b_nms_rad',
        'fk_nm',
    ]:
        summary[f'{column}_mean'] = results_df[column].mean()
        summary[f'{column}_std'] = results_df[column].std(ddof=1)

    summary['num_segments'] = len(results_df)
    return pd.DataFrame([summary])


def plot_step_response(df, extrema, metadata):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.35))
    ax.plot(df['time'], df['cmd_joint_deg'], color='0.0', linestyle='--', label='Command')
    ax.plot(df['time'], df['joint_pos_deg'], color='0.35', linestyle='-', label='Measured')

    if extrema:
        extrema_df = pd.DataFrame(extrema)
        ax.plot(
            extrema_df['time'],
            extrema_df['joint_pos_deg'],
            linestyle='None',
            marker='o',
            markerfacecolor='none',
            markeredgecolor='0.0',
            label='Used extrema'
        )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint position (deg)')
    style_axis(ax)
    ax.legend(loc='best')
    fig.tight_layout(pad=0.3)
    paths = save_ieee_figure(fig, FIGURE_BASENAME, metadata=metadata)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return paths


if __name__ == '__main__':
    main()
