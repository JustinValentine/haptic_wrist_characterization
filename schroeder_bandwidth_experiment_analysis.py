import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from ieee_plot import (
    add_panel_label,
    configure_ieee_style,
    figure_size,
    save_ieee_figure,
    style_axis,
    write_json,
)


# --- CONFIGURATION ---
GEAR_RATIO = 10.0
FREQ_MIN = 0.1
FREQ_MAX = 20.0
LOW_GAIN_MAX_HZ = 0.5
BANDWIDTH_SEARCH_START_HZ = 0.5
COHERENCE_THRESHOLD = 0.60
MIN_INPUT_POWER_FRACTION = 1e-8
DISCARD_INITIAL_SECONDS = 0.0
DATA_FILE = 'bandwidth_data.csv'
SPECTRUM_FILE = 'bandwidth_response.csv'
SUMMARY_FILE = 'bandwidth_summary.csv'
METADATA_FILE = 'bandwidth_metadata.json'
TIME_FIGURE_BASENAME = 'fig_bandwidth_time_response'
FREQUENCY_FIGURE_BASENAME = 'fig_bandwidth_frequency_response'
SHOW_FIGURES = True


def main():
    configure_ieee_style()

    df = pd.read_csv(DATA_FILE)
    gear_ratio = value_from_data_or_default(df, 'gear_ratio', GEAR_RATIO)
    df = add_joint_columns(df, gear_ratio)

    if DISCARD_INITIAL_SECONDS > 0:
        df = df[df['time'] >= DISCARD_INITIAL_SECONDS].copy()

    t = df['time'].to_numpy()
    u = signal.detrend(df['cmd_pos_deg'].to_numpy(), type='constant')
    y = signal.detrend(df['joint_pos_deg'].to_numpy(), type='constant')

    fs, jitter_fraction = sampling_stats(t)
    print(f"Detected sampling frequency: {fs:.2f} Hz")
    print(f"Sample-period jitter, std/mean: {jitter_fraction:.3f}")

    response = estimate_frequency_response(u, y, fs)
    bandwidth_hz, reference_gain_db = find_bandwidth(response)

    response['magnitude_relative_db'] = response['magnitude_db'] - reference_gain_db
    response.to_csv(SPECTRUM_FILE, index=False)

    summary = pd.DataFrame([{
        'bandwidth_hz': bandwidth_hz,
        'reference_gain_db': reference_gain_db,
        'sample_frequency_hz': fs,
        'sample_jitter_fraction': jitter_fraction,
        'coherence_threshold': COHERENCE_THRESHOLD,
    }])
    summary.to_csv(SUMMARY_FILE, index=False)

    metadata = {
        'analysis': 'schroeder_bandwidth_experiment_analysis.py',
        'source_data': DATA_FILE,
        'spectrum_file': SPECTRUM_FILE,
        'summary_file': SUMMARY_FILE,
        'gear_ratio': gear_ratio,
        'freq_min_hz': FREQ_MIN,
        'freq_max_hz': FREQ_MAX,
        'low_gain_max_hz': LOW_GAIN_MAX_HZ,
        'bandwidth_search_start_hz': BANDWIDTH_SEARCH_START_HZ,
        'coherence_threshold': COHERENCE_THRESHOLD,
        'min_input_power_fraction': MIN_INPUT_POWER_FRACTION,
        'discard_initial_seconds': DISCARD_INITIAL_SECONDS,
        'num_samples': int(len(df)),
        'sample_frequency_hz': fs,
        'sample_jitter_fraction': jitter_fraction,
        'num_valid_frequency_bins': int(response['valid'].sum()),
        'bandwidth_hz': None if np.isnan(bandwidth_hz) else float(bandwidth_hz),
    }
    write_json(METADATA_FILE, metadata)

    print("-" * 34)
    print("BANDWIDTH RESULTS")
    print("-" * 34)
    print(f"Low-frequency reference gain: {reference_gain_db:.3f} dB")
    if np.isnan(bandwidth_hz):
        print("Closed-loop bandwidth (-3 dB): not found in the valid frequency range")
    else:
        print(f"Closed-loop bandwidth (-3 dB): {bandwidth_hz:.3f} Hz")
    print(f"Saved response to '{SPECTRUM_FILE}' and summary to '{SUMMARY_FILE}'")

    figure_paths = plot_results(df, response, bandwidth_hz, metadata)
    print(f"Saved IEEE figures: {', '.join(figure_paths.values())}")


def value_from_data_or_default(df, column, default):
    if column in df.columns and not df[column].dropna().empty:
        return float(df[column].dropna().iloc[0])
    return default


def add_joint_columns(df, gear_ratio):
    df = df.copy()

    if 'joint_pos_deg' in df.columns:
        return df

    if 'motor_pos_rel_rev' in df.columns:
        motor_pos_rel_rev = df['motor_pos_rel_rev']
    else:
        motor_pos_rel_rev = df['motor_pos_rev'] - df['motor_pos_rev'].iloc[0]

    df['joint_pos_deg'] = (motor_pos_rel_rev / gear_ratio) * 360.0
    return df


def sampling_stats(t):
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / np.mean(dt)
    jitter_fraction = float(np.std(dt) / np.mean(dt))
    return fs, jitter_fraction


def estimate_frequency_response(u, y, fs):
    nperseg = min(1024, len(u))
    noverlap = nperseg // 2 if nperseg >= 4 else 0

    freqs, Pxx = signal.welch(
        u,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant'
    )
    _, Pxy = signal.csd(
        u,
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant'
    )
    _, coherence = signal.coherence(
        u,
        y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend='constant'
    )

    H = np.divide(Pxy, Pxx, out=np.full_like(Pxy, np.nan), where=Pxx > 0)
    magnitude = np.abs(H)
    magnitude_db = 20.0 * np.log10(magnitude)
    phase_deg = np.degrees(np.unwrap(np.angle(H)))

    valid = (
        (freqs >= FREQ_MIN)
        & (freqs <= FREQ_MAX)
        & (Pxx >= np.nanmax(Pxx) * MIN_INPUT_POWER_FRACTION)
        & (coherence >= COHERENCE_THRESHOLD)
    )

    return pd.DataFrame({
        'frequency_hz': freqs,
        'magnitude_db': magnitude_db,
        'phase_deg': phase_deg,
        'coherence': coherence,
        'input_power': Pxx,
        'valid': valid,
    })


def find_bandwidth(response):
    low_band = response[
        response['valid']
        & (response['frequency_hz'] >= FREQ_MIN)
        & (response['frequency_hz'] <= LOW_GAIN_MAX_HZ)
    ]
    if low_band.empty:
        low_band = response[
            response['valid']
            & (response['frequency_hz'] >= FREQ_MIN)
            & (response['frequency_hz'] <= BANDWIDTH_SEARCH_START_HZ)
        ]

    if low_band.empty:
        return np.nan, 0.0

    reference_gain_db = float(low_band['magnitude_db'].median())
    relative_mag_db = response['magnitude_db'] - reference_gain_db

    valid_search_indices = response.index[
        response['valid']
        & (response['frequency_hz'] >= BANDWIDTH_SEARCH_START_HZ)
    ].to_numpy()

    for idx in valid_search_indices:
        if relative_mag_db.iloc[idx] > -3.0:
            continue

        previous_indices = valid_search_indices[valid_search_indices < idx]
        if len(previous_indices) == 0:
            return float(response['frequency_hz'].iloc[idx]), reference_gain_db

        prev_idx = previous_indices[-1]
        f0 = float(response['frequency_hz'].iloc[prev_idx])
        f1 = float(response['frequency_hz'].iloc[idx])
        m0 = float(relative_mag_db.iloc[prev_idx])
        m1 = float(relative_mag_db.iloc[idx])

        if m1 == m0:
            return f1, reference_gain_db

        fraction = (-3.0 - m0) / (m1 - m0)
        bandwidth_hz = f0 + fraction * (f1 - f0)
        return float(bandwidth_hz), reference_gain_db

    return np.nan, reference_gain_db


def plot_results(df, response, bandwidth_hz, metadata):
    saved_paths = {}

    fig_time, ax_time = plt.subplots(figsize=figure_size('single', 2.35))
    ax_time.plot(
        df['time'],
        df['cmd_pos_deg'],
        color='0.0',
        linestyle='--',
        label='Command'
    )
    ax_time.plot(
        df['time'],
        df['joint_pos_deg'],
        color='0.35',
        linestyle='-',
        label='Measured'
    )
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Position (deg)')
    style_axis(ax_time)
    ax_time.legend(loc='best')
    fig_time.tight_layout(pad=0.3)
    for fmt, path in save_ieee_figure(
        fig_time,
        TIME_FIGURE_BASENAME,
        metadata=metadata
    ).items():
        saved_paths[f'time_{fmt}'] = path

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig_time)

    fig_freq, axes = plt.subplots(
        2,
        1,
        figsize=figure_size('single', 3.15),
        sharex=True
    )

    axes[0].semilogx(
        response['frequency_hz'],
        response['magnitude_relative_db'],
        color='0.0',
        linestyle='-',
        linewidth=1.1
    )
    axes[0].axhline(-3.0, color='0.25', linestyle='--', linewidth=0.9, label='-3 dB')
    if not np.isnan(bandwidth_hz):
        axes[0].axvline(bandwidth_hz, color='0.25', linestyle=':', linewidth=0.9)
        axes[0].text(
            bandwidth_hz,
            -9.0,
            f"{bandwidth_hz:.2f} Hz",
            ha='left',
            va='center',
            fontsize=7
        )
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_xlim(FREQ_MIN, FREQ_MAX)
    axes[0].set_ylim(-30, 10)
    style_axis(axes[0])
    axes[0].legend(loc='best')
    add_panel_label(axes[0], '(a)')

    axes[1].semilogx(
        response['frequency_hz'],
        response['coherence'],
        color='0.0',
        linestyle='-'
    )
    axes[1].axhline(
        COHERENCE_THRESHOLD,
        color='0.25',
        linestyle='--',
        linewidth=0.9,
        label='Threshold'
    )
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Coherence')
    axes[1].set_xlim(FREQ_MIN, FREQ_MAX)
    axes[1].set_ylim(0, 1.05)
    style_axis(axes[1])
    axes[1].legend(loc='best')
    add_panel_label(axes[1], '(b)')

    fig_freq.tight_layout(pad=0.3)
    for fmt, path in save_ieee_figure(
        fig_freq,
        FREQUENCY_FIGURE_BASENAME,
        metadata=metadata
    ).items():
        saved_paths[f'frequency_{fmt}'] = path

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig_freq)

    return saved_paths


if __name__ == '__main__':
    main()
