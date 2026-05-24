import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ieee_plot import (
    configure_ieee_style,
    figure_size,
    save_ieee_figure,
    style_axis,
    write_json,
)


# --- CONFIGURATION ---
GEAR_RATIO = 10.0
KP_JOINT_NM_RAD = 3.0
DATA_FILE = 'friction_ramp_data.csv'
RESULTS_FILE = 'static_friction_results.csv'
SUMMARY_FILE = 'static_friction_summary.csv'
METADATA_FILE = 'static_friction_metadata.json'
FIGURE_BASENAME = 'fig_static_friction'
CHECK_FIGURE_BASENAME = 'fig_static_friction_breakaway_check'
VELOCITY_THRESHOLD_DEG_S = 0.1
SETTLED_VELOCITY_THRESHOLD_DEG_S = 0.05
REST_SAMPLES_BEFORE_BREAKAWAY = 5
SHOW_FIGURES = True


def main():
    configure_ieee_style()

    df = pd.read_csv(DATA_FILE)
    gear_ratio = value_from_data_or_default(df, 'gear_ratio', GEAR_RATIO)
    kp_joint = value_from_data_or_default(df, 'kp_joint_nm_rad', KP_JOINT_NM_RAD)

    df = add_joint_columns(df, gear_ratio, kp_joint)
    df = add_segments(df)
    results, breakaway_indices = detect_breakaway_events(df)

    if results.empty:
        print("No breakaway events detected.")
        print("Try lowering VELOCITY_THRESHOLD_DEG_S or checking that each ramp starts from rest.")
        return

    results.to_csv(RESULTS_FILE, index=False)
    summary = summarize_results(results)
    summary.to_csv(SUMMARY_FILE, index=False)

    metadata = {
        'analysis': 'ramp_input_experiment_analysis.py',
        'source_data': DATA_FILE,
        'results_file': RESULTS_FILE,
        'summary_file': SUMMARY_FILE,
        'gear_ratio': gear_ratio,
        'kp_joint_nm_rad': kp_joint,
        'velocity_threshold_deg_s': VELOCITY_THRESHOLD_DEG_S,
        'settled_velocity_threshold_deg_s': SETTLED_VELOCITY_THRESHOLD_DEG_S,
        'rest_samples_before_breakaway': REST_SAMPLES_BEFORE_BREAKAWAY,
        'num_samples': int(len(df)),
        'num_breakaway_points': int(len(results)),
    }
    write_json(METADATA_FILE, metadata)

    print("-" * 34)
    print("STATIC FRICTION RESULTS")
    print("-" * 34)
    print(f"Points detected: {len(results)}")
    print(f"Mean magnitude:  {results['friction_abs_nm'].mean():.6g} Nm")
    print(f"Max magnitude:   {results['friction_abs_nm'].max():.6g} Nm")

    for direction, group in results.groupby('direction'):
        label = 'positive' if direction > 0 else 'negative'
        print(f"{label.title()} mean: {group['friction_abs_nm'].mean():.6g} Nm")

    print(f"Saved breakaway points to '{RESULTS_FILE}'")
    print(f"Saved summary to '{SUMMARY_FILE}'")

    figure_paths = plot_results(df, results, breakaway_indices, metadata)
    print(f"Saved IEEE figures: {', '.join(figure_paths.values())}")


def value_from_data_or_default(df, column, default):
    if column in df.columns and not df[column].dropna().empty:
        return float(df[column].dropna().iloc[0])
    return default


def add_joint_columns(df, gear_ratio, kp_joint):
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
    df['joint_cmd_rad'] = np.radians(df['cmd_joint_deg'])
    df['calc_torque_joint_nm'] = kp_joint * (df['joint_cmd_rad'] - df['joint_pos_rad'])

    dt = df['time'].diff()
    dt = dt.mask(dt <= 0).fillna(dt[dt > 0].median())
    df['diff_vel_rad_s'] = df['joint_pos_rad'].diff().fillna(0.0) / dt
    df['diff_vel_deg_s'] = np.degrees(df['diff_vel_rad_s'])
    df['plot_vel_deg_s'] = (
        df['diff_vel_deg_s']
        .rolling(window=5, center=False, min_periods=1)
        .mean()
    )

    return df


def add_segments(df):
    df = df.copy()
    if 'segment' in df.columns:
        return df

    if 'phase' not in df.columns:
        command_change = df['cmd_joint_deg'].diff().abs().fillna(0.0) > 1e-6
        df['segment'] = command_change.cumsum()
        return df

    phase = df['phase'].astype(str)
    new_ramp = phase.eq('ramp') & ~phase.shift(fill_value='').eq('ramp')
    df['segment'] = new_ramp.cumsum() - 1
    df['segment'] = df['segment'].clip(lower=0)
    return df


def detect_breakaway_events(df):
    events = []
    breakaway_indices = []

    for segment_id, segment_df in df.groupby('segment'):
        ramp_df = segment_df[segment_df['phase'].eq('ramp')] if 'phase' in segment_df else segment_df
        if len(ramp_df) <= REST_SAMPLES_BEFORE_BREAKAWAY + 1:
            continue

        direction = command_direction(ramp_df)
        if direction == 0:
            continue

        velocity_along_ramp = ramp_df['diff_vel_deg_s'] * direction

        for local_i in range(REST_SAMPLES_BEFORE_BREAKAWAY, len(ramp_df)):
            prior_velocity = ramp_df['diff_vel_deg_s'].iloc[
                local_i - REST_SAMPLES_BEFORE_BREAKAWAY:local_i
            ].abs()

            was_settled = prior_velocity.max() <= SETTLED_VELOCITY_THRESHOLD_DEG_S
            moved_now = velocity_along_ramp.iloc[local_i] >= VELOCITY_THRESHOLD_DEG_S

            if not (was_settled and moved_now):
                continue

            idx_static = ramp_df.index[local_i - 1]
            row = df.loc[idx_static]
            events.append({
                'segment': segment_id,
                'time_s': row['time'],
                'pos_deg': row['joint_pos_deg'],
                'direction': direction,
                'friction_signed_nm': row['calc_torque_joint_nm'],
                'friction_abs_nm': abs(row['calc_torque_joint_nm']),
                'cmd_joint_deg': row['cmd_joint_deg'],
                'joint_pos_deg': row['joint_pos_deg'],
                'velocity_before_deg_s': row['diff_vel_deg_s'],
            })
            breakaway_indices.append(idx_static)
            break

    return pd.DataFrame(events), breakaway_indices


def command_direction(ramp_df):
    cmd_delta = ramp_df['cmd_joint_deg'].iloc[-1] - ramp_df['cmd_joint_deg'].iloc[0]
    if abs(cmd_delta) < 1e-9:
        return 0
    return int(np.sign(cmd_delta))


def summarize_results(results):
    rows = [{
        'direction': 0,
        'label': 'all',
        'num_points': len(results),
        'friction_abs_nm_mean': results['friction_abs_nm'].mean(),
        'friction_abs_nm_std': results['friction_abs_nm'].std(ddof=1),
        'friction_abs_nm_max': results['friction_abs_nm'].max(),
    }]

    for direction, group in results.groupby('direction'):
        label = 'positive' if direction > 0 else 'negative'
        rows.append({
            'direction': int(direction),
            'label': label,
            'num_points': len(group),
            'friction_abs_nm_mean': group['friction_abs_nm'].mean(),
            'friction_abs_nm_std': group['friction_abs_nm'].std(ddof=1),
            'friction_abs_nm_max': group['friction_abs_nm'].max(),
        })

    return pd.DataFrame(rows)


def plot_results(df, results, breakaway_indices, metadata):
    saved_paths = {}

    fig, ax = plt.subplots(figsize=figure_size('single', 2.35))
    positive = results[results['direction'] > 0]
    negative = results[results['direction'] < 0]
    ax.plot(
        positive['pos_deg'],
        positive['friction_abs_nm'],
        linestyle='None',
        marker='o',
        markerfacecolor='none',
        markeredgecolor='0.0',
        label='Positive'
    )
    ax.plot(
        negative['pos_deg'],
        negative['friction_abs_nm'],
        linestyle='None',
        marker='s',
        markerfacecolor='0.75',
        markeredgecolor='0.0',
        label='Negative'
    )
    ax.set_xlabel('Joint position (deg)')
    ax.set_ylabel('Static friction (N m)')
    style_axis(ax)
    ax.legend(loc='best')
    fig.tight_layout(pad=0.3)
    for fmt, path in save_ieee_figure(fig, FIGURE_BASENAME, metadata=metadata).items():
        saved_paths[f'friction_{fmt}'] = path

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=figure_size('single', 2.45))
    ax1.plot(
        df['time'],
        df['calc_torque_joint_nm'],
        color='0.0',
        linestyle='-',
        label='Commanded torque'
    )
    ax1.set_ylabel('Joint torque (N m)')
    style_axis(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        df['time'],
        df['plot_vel_deg_s'],
        color='0.45',
        linestyle='--',
        label='Velocity'
    )
    ax2.axhline(VELOCITY_THRESHOLD_DEG_S, color='0.2', linestyle=':', linewidth=0.8)
    ax2.axhline(-VELOCITY_THRESHOLD_DEG_S, color='0.2', linestyle=':', linewidth=0.8)
    ax2.set_ylabel('Joint velocity (deg/s)')
    ax2.tick_params(direction='in', top=True, right=True)

    ax1.plot(
        df.loc[breakaway_indices, 'time'],
        df.loc[breakaway_indices, 'calc_torque_joint_nm'],
        linestyle='None',
        marker='D',
        markerfacecolor='none',
        markeredgecolor='0.0',
        label='Breakaway'
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    ax1.set_xlabel('Time (s)')
    fig.tight_layout(pad=0.3)
    for fmt, path in save_ieee_figure(
        fig,
        CHECK_FIGURE_BASENAME,
        metadata=metadata
    ).items():
        saved_paths[f'breakaway_{fmt}'] = path

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths


if __name__ == '__main__':
    main()
