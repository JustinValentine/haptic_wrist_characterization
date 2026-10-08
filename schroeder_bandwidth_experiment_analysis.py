"""Closed-loop position frequency response, bandwidth, and model fit (multisine).

For each amplitude level the first period is discarded as a transient. In
every remaining period the measured position and the command are fitted, by
least squares at the excited frequencies (exact for non-uniform timestamps),
to complex amplitudes Y and U; H = Y/U per period, then the mean and its
standard error across periods. From H:

* -3 dB bandwidth relative to the low-frequency gain (log-frequency
  interpolation), the -90 deg phase crossing, and the resonance peak;
* a parametric fit of the joint inertia J, viscous damping b, and an
  effective loop delay tau with the known PD gains:
      H = e^{-s tau} (Kp + FF Kd s) / (J s^2 + b s + e^{-s tau} (Kp + Kd s)),
  FF = 1 when the velocity feedforward was sent. The fitted model also gives
  an extrapolated bandwidth when -3 dB lies beyond the measured range.

Levels whose controller torque sat at the torque cap for more than 1% of the
samples are reported but excluded (the response is no longer linear).
Data from the old single-period script are analyzed with a one-period DFT at
the design bins and flagged as legacy.

Run:  python schroeder_bandwidth_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import device_config
from analysis_common import (export_figure, fit_sinusoids, load_run, make_parser, print_header,
                             print_truth_comparison, save_outputs, saturation_fraction, sim_truth)
from ieee_plot import (INK, SERIES, add_panel_label, configure_ieee_style, direct_label,
                       figure_size, log_ticks_hz, ordinal_colors, reference_line, style_axis,
                       zero_line)

EXPERIMENT = 'bandwidth'
LEGACY = [('bandwidth_data.csv', 'bandwidth_collection_metadata.json')]
BODE_BASENAME = 'fig_bandwidth_bode'
TIME_BASENAME = 'fig_bandwidth_time'
DIAGNOSTIC_BASENAME = 'diag_bandwidth_lines'
LOW_FREQ_LINES = 3            # Valid lines averaged for the low-frequency gain
MIN_SNR = 10.0                # |H| / SE below this marks a line invalid
SATURATION_LIMIT = 0.01       # Max fraction of samples at the torque cap
TIME_WINDOW_S = 1.0
MARKERS = ('o', 's', '^', 'D', 'v')


# --- estimation ---------------------------------------------------------------------
def estimate_level(frame, freqs, period, discard=1):
    """H per period at the excited lines; mean, standard error and SNR."""
    t = frame['t_local'].to_numpy()
    y = frame['joint_pos_deg'].to_numpy()
    u = frame['cmd_joint_deg'].to_numpy()
    dt = float(np.median(np.diff(t)))
    n_periods = int(math.floor((t[-1] + 2 * dt) / period + 1e-9))
    H, U = [], []
    for p in range(discard, n_periods):
        mask = (t >= p * period) & (t < (p + 1) * period)
        if mask.sum() < 4 * len(freqs):
            continue
        Yp, _ = fit_sinusoids(t[mask], y[mask], freqs)
        Up, _ = fit_sinusoids(t[mask], u[mask], freqs)
        H.append(Yp / Up)
        U.append(np.abs(Up))
    if not H:
        return None
    H = np.array(H)
    mean = H.mean(axis=0)
    if len(H) > 1:
        se = np.sqrt(H.real.var(axis=0, ddof=1) + H.imag.var(axis=0, ddof=1)) / math.sqrt(len(H))
    else:
        se = np.full(len(freqs), np.nan)
    return {'H': mean, 'se': se, 'u_amp_deg': np.mean(U, axis=0), 'num_periods': len(H)}


def legacy_estimate(df, meta):
    """Old script: one 40 s Schroeder period at 100 Hz; DFT at the design bins."""
    fs = float(meta.get('sample_frequency_commanded_hz', 100.0))
    period = float(meta.get('duration_s', 40.0))
    f_min = float(meta.get('freq_min_hz', 0.1))
    f_max = float(meta.get('freq_max_hz', 20.0))
    u = df['cmd_pos_deg' if 'cmd_pos_deg' in df else 'cmd_joint_deg'].to_numpy()
    y = df['joint_pos_deg'].to_numpy()
    n = int(round(period * fs))
    if len(u) < n:
        print(f'Legacy record has {len(u)} samples, fewer than one period ({n}); '
              'the DFT will leak between lines.')
        n = len(u)
    bins = np.arange(int(math.ceil(f_min * period - 1e-9)), int(math.floor(f_max * period + 1e-9)) + 1)
    bins = bins[(bins > 0) & (bins < n // 2)]
    U = np.fft.rfft(u[:n])[bins]
    Y = np.fft.rfft(y[:n])[bins]
    freqs = bins / period
    return freqs, {'H': Y / U, 'se': np.full(len(bins), np.nan),
                   'u_amp_deg': 2.0 * np.abs(U) / n, 'num_periods': 1}


def phase_deg(H):
    return np.degrees(np.unwrap(np.angle(H)))


def crossing(freqs, values, level):
    """First frequency where values falls below level (log-f interpolation)."""
    below = np.where(values < level)[0]
    if len(below) == 0:
        return math.nan
    i = below[0]
    if i == 0:
        return float(freqs[0])
    f0, f1 = math.log(freqs[i - 1]), math.log(freqs[i])
    v0, v1 = values[i - 1], values[i]
    frac = (level - v0) / (v1 - v0) if v1 != v0 else 0.0
    return float(math.exp(f0 + frac * (f1 - f0)))


def response_metrics(freqs, H, valid):
    f = freqs[valid]
    if len(f) < LOW_FREQ_LINES + 1:
        return None
    mag = 20.0 * np.log10(np.abs(H[valid]))
    phase = phase_deg(H[valid])
    ref = float(np.mean(mag[:LOW_FREQ_LINES]))
    rel = mag - ref
    bw = crossing(f, rel, -3.0)
    peak = int(np.argmax(rel))
    in_range = 0 < peak < len(rel) - 1
    return {
        'low_freq_gain_db': ref,
        'bandwidth_hz': bw,
        'bandwidth_above_range': not math.isfinite(bw),
        'bandwidth_lower_bound_hz': float(f[-1]) if not math.isfinite(bw) else bw,
        'phase_90_hz': crossing(f, phase, -90.0),
        'resonance_peak_db': float(rel[peak]),
        'resonance_freq_hz': float(f[peak]),
        'resonance_in_range': bool(in_range),   # False: still rising at the band edge
    }


# --- parametric model -------------------------------------------------------------
def model_response(freqs, J, b, tau, kp, kd, ff):
    s = 2j * np.pi * np.asarray(freqs, dtype=float)
    delay = np.exp(-s * tau)
    return delay * (kp + ff * kd * s) / (J * s ** 2 + b * s + delay * (kp + kd * s))


def fit_model(freqs, H, se, kp, kd, ff):
    """Weighted complex least squares for J, b and the loop delay."""
    sigma = np.where(np.isfinite(se), se, 0.0)
    sigma = np.sqrt(sigma ** 2 + (0.01 * np.abs(H)) ** 2)

    def residual(x):
        err = (model_response(freqs, math.exp(x[0]), x[1], x[2], kp, kd, ff) - H) / sigma
        return np.concatenate([err.real, err.imag])

    x0 = [math.log(device_config.NOMINAL_JOINT_INERTIA_KG_M2),
          device_config.NOMINAL_JOINT_DAMPING_NMS_RAD, 5e-4]
    result = optimize.least_squares(residual, x0, bounds=([math.log(1e-5), -2.0, 0.0],
                                                          [math.log(10.0), 20.0, 0.02]))
    J, b, tau = math.exp(result.x[0]), float(result.x[1]), float(result.x[2])
    fitted = model_response(freqs, J, b, tau, kp, kd, ff)
    err_db = 20.0 * np.log10(np.abs(H) / np.abs(fitted))
    dense = np.logspace(-2, 3, 4000)
    model = model_response(dense, J, b, tau, kp, kd, ff)
    rel = 20.0 * np.log10(np.abs(model) / np.abs(model[0]))
    return {
        'fit_J_kg_m2': J,
        'fit_b_nms_rad': b,
        'fit_delay_ms': 1000.0 * tau,
        'fit_bandwidth_hz': crossing(dense, rel, -3.0),
        'fit_phase_90_hz': crossing(dense, phase_deg(model), -90.0),
        'fit_rms_error_db': float(np.sqrt(np.mean(err_db ** 2))),
        'fit_max_error_db': float(np.max(np.abs(err_db))),
        'fit_success': bool(result.success),
    }


# --- figures -------------------------------------------------------------------------
def plot_bode(levels, fit, kp, kd, ff, headline, f_range, f_measured_max):
    fig, (ax_m, ax_p) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 3.6),
                                     gridspec_kw={'hspace': 0.12})
    shown = [lv for lv in levels if lv['valid']]
    colors = ordinal_colors(len(shown)) if shown else []
    mag_all, ph_all = [], []
    for i, lv in enumerate(shown):
        ok = lv['line_valid']
        f, H, se = lv['freqs'][ok], lv['H'][ok], lv['se'][ok]
        mag = 20.0 * np.log10(np.abs(H))
        ph = phase_deg(H)
        rel_se = np.where(np.isfinite(se), se / np.abs(H), 0.0)
        if len(f) > 120:   # Dense (legacy) responses read better as a line
            style = dict(linestyle='-', linewidth=0.8, color=colors[i], ecolor=colors[i],
                         elinewidth=0.4)
        else:
            style = dict(linestyle='None', marker=MARKERS[i % len(MARKERS)], markersize=3.0,
                         markerfacecolor='white' if i % 2 == 0 else colors[i],
                         markeredgecolor=colors[i], ecolor=colors[i], elinewidth=0.6)
        label = (f"{lv['amplitude_deg']:g}° amplitude" if lv['amplitude_deg']
                 else 'Measured')
        ax_m.errorbar(f, mag, yerr=2 * 8.686 * rel_se, label=label, **style)
        ax_p.errorbar(f, ph, yerr=2 * np.degrees(rel_se), **style)
        mag_all.extend(mag)
        ph_all.extend(ph)
    first = min(lv['freqs'][0] for lv in levels)
    dense = np.logspace(math.log10(first), math.log10(f_measured_max), 400)
    if fit:
        model = model_response(dense, fit['fit_J_kg_m2'], fit['fit_b_nms_rad'],
                               fit['fit_delay_ms'] / 1000.0, kp, kd, ff)
        ax_m.plot(dense, 20 * np.log10(np.abs(model)), color=INK, linestyle='--',
                  linewidth=0.9, label='Fitted model', zorder=3)
        ax_p.plot(dense, phase_deg(model), color=INK, linestyle='--', linewidth=0.9, zorder=3)
    if headline:
        ref = headline['low_freq_gain_db']
        reference_line(ax_m, ref - 3.0)
        # Left end, between the low-frequency plateau and the line (always empty).
        text = '−3 dB'
        if not headline['bandwidth_above_range']:
            text += f": {headline['bandwidth_hz']:.1f} Hz"
        direct_label(ax_m, f_range[0], ref - 3.0, text, dx=2, dy=1, ha='left', va='bottom')
        if headline['bandwidth_above_range']:
            text = f"−3 dB not reached below {f_measured_max:g} Hz"
            if fit and math.isfinite(fit['fit_bandwidth_hz']):
                text += f"\n(model: {fit['fit_bandwidth_hz']:.0f} Hz)"
            ax_m.text(0.03, 0.06, text, transform=ax_m.transAxes, ha='left', va='bottom',
                      fontsize=7, color=INK)
        else:
            reference_line(ax_m, headline['bandwidth_hz'], orientation='v', linestyle=':')
    ax_m.set_ylabel('Magnitude (dB)')
    floor = -9.0 if (headline and headline['bandwidth_above_range']) else -6.0
    lo = min(floor, (min(mag_all) if mag_all else 0) - 1.0)
    hi = max(3.0, (max(mag_all) if mag_all else 0) + 2.0)
    ax_m.set_ylim(lo, hi)
    style_axis(ax_m)
    handles, labels = ax_m.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == 'Fitted model')
    ax_m.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left',
                fontsize=6.5, ncol=2 if len(labels) > 3 else 1, columnspacing=1.0)
    add_panel_label(ax_m, '(a)', x=-0.12)
    ax_p.set_xscale('log')
    ax_p.set_xlim(*f_range)
    log_ticks_hz(ax_p)
    if ph_all and min(ph_all) < -60:
        reference_line(ax_p, -90.0)
    ax_p.set_xlabel('Frequency (Hz)')
    ax_p.set_ylabel('Phase (deg)')
    if ph_all:
        ax_p.set_ylim(min(-20.0, min(ph_all) - 10.0), max(10.0, max(ph_all) + 10.0))
    style_axis(ax_p)
    add_panel_label(ax_p, '(b)', x=-0.12)
    fig.align_ylabels([ax_m, ax_p])
    return fig


def plot_time(frame, period, amplitude):
    t = frame['t_local'].to_numpy()
    cmd = frame['cmd_joint_deg'].to_numpy()
    pos = frame['joint_pos_deg'].to_numpy()
    # A 1 s window after the transient period, centered on the largest command.
    start_period = period if t[-1] > 2 * period else 0.0
    mask = (t >= start_period) & (t < start_period + period)
    i_peak = np.flatnonzero(mask)[np.argmax(np.abs(cmd[mask]))]
    t0 = max(start_period, t[i_peak] - TIME_WINDOW_S / 2)
    window = (t >= t0) & (t <= t0 + TIME_WINDOW_S)
    fig, (ax_y, ax_e) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 2.9),
                                     gridspec_kw={'height_ratios': [1.6, 1.0], 'hspace': 0.12})
    tt = t[window] - t0
    ax_y.plot(tt, pos[window], color=INK, linewidth=0.9, label='Measured')
    ax_y.plot(tt, cmd[window], color=SERIES[0], linestyle='--', linewidth=0.9, label='Command')
    ax_y.set_ylabel('Position (deg)')
    style_axis(ax_y)
    ax_y.legend(loc='best', ncol=2)
    add_panel_label(ax_y, '(a)', x=-0.12)
    ax_e.plot(tt, pos[window] - cmd[window], color=INK, linewidth=0.8)
    zero_line(ax_e)
    ax_e.set_xlim(0, TIME_WINDOW_S)
    ax_e.set_xlabel('Time (s)')
    ax_e.set_ylabel('Error (deg)')
    style_axis(ax_e)
    add_panel_label(ax_e, '(b)', x=-0.12)
    fig.align_ylabels([ax_y, ax_e])
    return fig


def plot_diagnostics(levels, f_range):
    fig, (ax_u, ax_s) = plt.subplots(2, 1, sharex=True, figsize=figure_size('single', 3.2))
    colors = ordinal_colors(min(len(levels), 5)) if levels else []
    for i, lv in enumerate(levels[:5]):
        label = f"{lv['amplitude_deg']:g}°" + ('' if lv['valid'] else ' (excluded)')
        ax_u.loglog(lv['freqs'], lv['u_amp_deg'], marker=MARKERS[i % 5], linestyle='None',
                    markersize=2.5, color=colors[i], label=label)
        snr = np.abs(lv['H']) / lv['se']
        ax_s.loglog(lv['freqs'], snr, marker=MARKERS[i % 5], linestyle='None', markersize=2.5,
                    color=colors[i])
    reference_line(ax_s, MIN_SNR, label='Validity threshold')
    ax_u.set_ylabel('Command line (deg)')
    ax_s.set_ylabel('|H| / SE')
    ax_s.set_xlabel('Frequency (Hz)')
    ax_s.set_xlim(*f_range)
    log_ticks_hz(ax_s)
    for ax in (ax_u, ax_s):
        style_axis(ax, minor=False)
    ax_u.legend(loc='lower left', fontsize=6)
    ax_s.legend(loc='center left', fontsize=6)
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data, LEGACY)
    df, meta = run.df, run.meta
    kp = float(run.meta_value('kp_joint_nm_rad', 450.0, column='kp_joint_nm_rad'))
    kd = float(run.meta_value('kd_joint_nms_rad', 0.75, column='kd_joint_nms_rad'))
    ff_value = meta.get('send_velocity_feedforward', meta.get('send_command_velocity'))
    if ff_value is None and 'sent_velocity_feedforward' in df:
        ff_value = bool(df['sent_velocity_feedforward'].iloc[0])
    ff = 1.0 if (ff_value is None or bool(ff_value)) else 0.0
    legacy = 'multisine' not in meta

    levels = []
    if legacy:
        freqs, est = legacy_estimate(df, meta)
        sat = saturation_fraction(df)
        levels.append({'level': 0, 'amplitude_deg': float(meta.get('amplitude_deg', math.nan)),
                       'freqs': freqs, 'saturation_fraction': sat, 'frame': df, **est})
        period = float(meta.get('duration_s', 40.0))
    else:
        design = meta['multisine']
        freqs = np.asarray(design['freqs_hz'])
        period = float(design['period_s'])
        blocks = df[df['phase'] == 'multisine']
        for level, frame in blocks.groupby('level', sort=True):
            est = estimate_level(frame, freqs, period)
            if est is None:
                print(f'Level {level}: no complete period after the transient; skipped.')
                continue
            levels.append({'level': int(level), 'amplitude_deg': float(frame['amplitude_deg'].iloc[0]),
                           'freqs': freqs, 'saturation_fraction': saturation_fraction(frame),
                           'frame': frame, **est})
    if not levels:
        raise SystemExit('No multisine data to analyze.')

    rows, level_rows = [], []
    for lv in levels:
        sat = lv['saturation_fraction']
        lv['valid'] = not (math.isfinite(sat) and sat > SATURATION_LIMIT)
        snr = np.abs(lv['H']) / lv['se']
        lv['line_valid'] = np.where(np.isfinite(snr), snr >= MIN_SNR, True)
        metrics = response_metrics(lv['freqs'], lv['H'], lv['line_valid'])
        lv['metrics'] = metrics
        level_rows.append({'level': lv['level'], 'amplitude_deg': lv['amplitude_deg'],
                           'saturation_fraction': sat, 'valid': lv['valid'],
                           'num_periods': lv['num_periods'],
                           'num_valid_lines': int(lv['line_valid'].sum()),
                           **(metrics or {})})
        phases = phase_deg(lv['H'])
        for i, f in enumerate(lv['freqs']):
            rows.append({'level': lv['level'], 'amplitude_deg': lv['amplitude_deg'],
                         'frequency_hz': f, 'H_real': lv['H'][i].real, 'H_imag': lv['H'][i].imag,
                         'magnitude_db': 20 * math.log10(abs(lv['H'][i])),
                         'phase_deg': phases[i], 'standard_error': lv['se'][i],
                         'snr': snr[i], 'command_amplitude_deg': lv['u_amp_deg'][i],
                         'line_valid': bool(lv['line_valid'][i]), 'level_valid': lv['valid']})
    response = pd.DataFrame(rows)
    level_table = pd.DataFrame(level_rows)

    valid_levels = [lv for lv in levels if lv['valid'] and lv['metrics']]
    if not valid_levels:
        raise SystemExit('Every level saturated the torque cap; lower amplitudes_deg.')
    headline_level = max(valid_levels, key=lambda lv: (lv['amplitude_deg']
                                                       if math.isfinite(lv['amplitude_deg']) else 0))
    headline = headline_level['metrics']

    f_fit = np.concatenate([lv['freqs'][lv['line_valid']] for lv in valid_levels])
    H_fit = np.concatenate([lv['H'][lv['line_valid']] for lv in valid_levels])
    se_fit = np.concatenate([lv['se'][lv['line_valid']] for lv in valid_levels])
    fit = fit_model(f_fit, H_fit, se_fit, kp, kd, ff)
    times = df['time'].to_numpy()
    sample_rate = float(1.0 / np.median(np.diff(times))) if len(times) > 1 else math.nan

    summary = {
        **headline,
        'bandwidth_amplitude_deg': headline_level['amplitude_deg'],
        **fit,
        'kp_joint_nm_rad': kp,
        'kd_joint_nms_rad': kd,
        'velocity_feedforward': bool(ff),
        'num_levels': len(levels),
        'num_levels_valid': len(valid_levels),
        'max_saturation_fraction': float(np.nanmax([lv['saturation_fraction'] for lv in levels]))
        if any(math.isfinite(lv['saturation_fraction']) for lv in levels) else math.nan,
        'freq_min_hz': float(freqs[0]),
        'freq_max_hz': float(freqs[-1]),
        'sample_rate_hz': sample_rate,
        'sample_rate_warning': bool(sample_rate < 20.0 * float(freqs[-1])),
        'num_lines': int(len(freqs)),
        'legacy': legacy,
    }
    metadata = {
        'analysis': 'schroeder_bandwidth_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('legacy one-period DFT at design bins' if legacy else
                   'per-period least-squares sinusoid fits at excited lines, mean and SE '
                   'across periods after discarding the first'),
        'min_snr': MIN_SNR,
        'saturation_limit': SATURATION_LIMIT,
        'low_freq_lines': LOW_FREQ_LINES,
        'model': 'H = e^{-s tau}(Kp + FF Kd s)/(J s^2 + b s + e^{-s tau}(Kp + Kd s))',
        'note_fit_b': ('effective damping including friction at the excitation amplitudes; '
                       'weakly constrained when Kd >> b'),
    }
    save_outputs(run, {'frequency_response': response, 'levels': level_table}, summary, metadata)

    print_header('CLOSED-LOOP POSITION BANDWIDTH' + (' (LEGACY DATA)' if legacy else ''))
    if summary['sample_rate_warning']:
        print(f"WARNING: sampled at {summary['sample_rate_hz']:.0f} Hz, under 20x the highest line. "
              'The held/interpolated command distorts the upper lines (about -1 to -2 dB near '
              'the top of the band), which biases the fitted J and b and the extrapolated '
              'bandwidth. Re-collect at 1 kHz for publication.')
    print(f"Kp = {kp:g} N m/rad, Kd = {kd:g} N m s/rad, velocity feedforward {bool(ff)}; "
          f"{len(freqs)} lines {freqs[0]:.2f}-{freqs[-1]:.1f} Hz")
    for row in level_rows:
        status = 'ok' if row['valid'] else f"EXCLUDED (saturated {row['saturation_fraction']:.1%})"
        bw = row.get('bandwidth_hz', math.nan)
        bw_text = (f'{bw:.2f} Hz' if bw is not None and math.isfinite(bw)
                   else f"> {row.get('bandwidth_lower_bound_hz', math.nan):.1f} Hz")
        edge = '' if row.get('resonance_in_range', True) else ' (still rising at band edge)'
        print(f"  {row['amplitude_deg']:5.1f} deg: -3 dB {bw_text:>10}, max "
              f"{row.get('resonance_peak_db', math.nan):+.2f} dB at "
              f"{row.get('resonance_freq_hz', math.nan):.1f} Hz{edge}, "
              f"{row['num_valid_lines']} valid lines [{status}]")
    if headline['bandwidth_above_range']:
        print(f"  Bandwidth: above the measured range (> {headline['bandwidth_lower_bound_hz']:.1f} Hz)"
              f"; fitted model predicts {fit['fit_bandwidth_hz']:.1f} Hz")
    else:
        print(f"  Bandwidth ({headline_level['amplitude_deg']:g} deg): "
              f"{headline['bandwidth_hz']:.2f} Hz")
    print(f"  Fit: J = {fit['fit_J_kg_m2']:.5g} kg m^2, b = {fit['fit_b_nms_rad']:.4g} N m s/rad, "
          f"delay = {fit['fit_delay_ms']:.3f} ms, rms error {fit['fit_rms_error_db']:.2f} dB")
    print(f"  (b is an effective damping: it includes friction acting at these amplitudes and "
          f"is weakly constrained when Kd = {kd:g} dominates the loop damping; use the step "
          f"or friction-sweep experiments for b.)")
    truth = sim_truth(run)
    if truth:
        J_true = truth['motor_inertia'] + truth['output_inertia']
        b_true = truth['viscous'] + truth['output_viscous']
        print_truth_comparison([('Inertia J (FRF fit)', fit['fit_J_kg_m2'], J_true, 'kg m^2')])
        print(f"  Effective b {fit['fit_b_nms_rad']:.3g} N m s/rad is not comparable to the "
              f"viscous truth {b_true:.3g}: Kd and friction dominate the loop damping.")
        true_H = model_response(f_fit, J_true, b_true, 0.0, kp, kd, ff)
        dev = 20 * np.log10(np.abs(H_fit) / np.abs(true_H))
        print(f'  Measured vs true rigid model: max |error| {np.max(np.abs(dev)):.2f} dB')
        summary['sim_max_error_vs_true_model_db'] = float(np.max(np.abs(dev)))

    f_range = (freqs[0] * 0.8, freqs[-1] * 1.25)
    fig = plot_bode(levels, fit, kp, kd, ff, headline, f_range, float(freqs[-1]))
    paths = export_figure(fig, run, BODE_BASENAME, {**metadata, **summary}, args.paper_dir)
    if not legacy:
        fig_t = plot_time(headline_level['frame'], period, headline_level['amplitude_deg'])
        export_figure(fig_t, run, TIME_BASENAME, metadata, args.paper_dir)
        diag = plot_diagnostics(levels, f_range)
        export_figure(diag, run, DIAGNOSTIC_BASENAME, metadata, paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
