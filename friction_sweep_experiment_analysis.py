"""Friction versus velocity and position from the constant-velocity sweeps.

During each cruise the reported torque is friction(v) + conservative(theta).
Within each speed, forward and backward passes cover the same positions, so
on 1 deg position bins

    friction(v)         = (tau_forward - tau_backward) / 2
    conservative(theta) = (tau_forward + tau_backward) / 2.

The conservative profile averaged over all speeds is then subtracted from
every pass to get one signed friction value per pass, and the Stribeck +
viscous model

    tau_f(v) = sign(v) [fc + (fs - fc) exp(-(|v|/vs)^2)] + b v

is fitted to those values. The conservative profile is summarized by its
offset and slope (gravity, cable preload) and by the ripple left after a
low-order polynomial trend is removed (transmission ripple).

Cogging repeats far faster than these bins (every ~0.22 deg of joint travel for
the GL60 II at ~9.8:1), so it is identified separately from the slowest passes,
which sample it densely and are slow enough for the loop to follow it: the
peak of the averaged position spectrum gives its cycles per motor revolution,
and a least-squares harmonic series in absolute motor position gives its
amplitude and phase. That cogging model is saved for the other analyses, and
every position bin above is widened to whole cogging periods so cogging
averages out of the friction and ripple estimates.

Run:  python friction_sweep_experiment_analysis.py [run_dir_or_csv] [--no-show]
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import device_config
from analysis_common import (cogging_torque, export_figure, load_run, loop_delays, make_parser,
                             motor_torques, run_j2mp, run_joint_index, run_nodes, print_header,
                             print_truth_comparison, save_outputs, sim_truth)
from experiment_common import latest_summary
from ieee_plot import (INK, MUTED, TORQUE_UNIT, add_panel_label, configure_ieee_style,
                       figure_size, log_ticks_hz, nice_limits, reference_line, series_style,
                       style_axis, zero_line)

EXPERIMENT = 'friction_sweep'
LEGACY = []
VELOCITY_FIGURE = 'fig_friction_velocity'
POSITION_FIGURE = 'fig_friction_position'
DIAGNOSTIC_FIGURE = 'diag_friction_sweep'
BIN_DEG = 1.0                 # Bins for pairing forward/backward passes ...
FINE_BIN_DEG = 0.25           # ... and for the ripple profile, both rounded to whole
                              # cogging periods when cogging is detected
DROP_FRACTION = 0.2           # Drop at least this much of each cruise ...
SETTLE_PERIODS = 1.5          # ... and at least this many loop settling times
POSITION_FIGURE_SPEED = 10.0  # Speed shown in the position figure (closest available)
POSITION_FIGURE_BIN_DEG = 2.0 # Display bins
RIPPLE_DIAGNOSTIC = 'diag_friction_ripple'
COGGING_MAX_SPEED_DEG_S = 2.0      # Passes slow enough to sample and track cogging
COGGING_BAND_CPR = (6.0, 2000.0)   # Search band, cycles per motor revolution
COGGING_GRID_REV = 1e-4            # Motor-position grid for the spectrum (rev)
COGGING_HARMONICS = 3              # Harmonics in the cogging model
COGGING_MIN_SNR = 10.0             # Spectral peak over the band median for a detection
COGGING_PHASE_BINS = 24            # Phase bins for the folded cogging waveform
COGGING_MIN_SPAN_REV = 0.05        # Shortest slow pass used for the cogging fit
VELOCITY_FILTER_HZ = 400.0         # moteus velocity-estimate bandwidth (runs without loop data)
MAX_DRAG_FRACTION = 0.2            # Passes whose predicted cogging drag exceeds this
                                   # fraction of their friction are left out of the fit


# --- data preparation ----------------------------------------------------------------
def torque_column(df):
    for column in ('joint_torque_nm', 'control_joint_torque_nm'):
        if column in df.columns and df[column].notna().any():
            return column
    raise SystemExit('No torque column in the data.')


def settle_time(run):
    """Approximate settling time of the tracking loop, 4 / (zeta wn) = 8 J / (Kd + b)."""
    kd = float(run.meta_value('kd_joint_nms_rad', column='kd_joint_nms_rad', default=1.0))
    dev = run.meta.get('device_config', {})
    j = dev.get('NOMINAL_JOINT_INERTIA_KG_M2', device_config.NOMINAL_JOINT_INERTIA_KG_M2)
    b = dev.get('NOMINAL_JOINT_DAMPING_NMS_RAD', device_config.NOMINAL_JOINT_DAMPING_NMS_RAD)
    return 8.0 * j / max(kd + b, 1e-6)


def cruise_passes(df, column, settle_s):
    """Per pass: steady part of the cruise (transient after the blend dropped)."""
    cruise = df[df['phase'] == 'cruise']
    passes = {}
    for pass_id, frame in cruise.groupby('pass_id', sort=True):
        frame = frame.sort_values('time')
        t = frame['time'].to_numpy()
        duration = t[-1] - t[0]
        drop = min(max(DROP_FRACTION * duration, SETTLE_PERIODS * settle_s), 0.5 * duration)
        steady = frame[t >= t[0] + drop]
        if len(steady) >= 10:
            passes[int(pass_id)] = steady
    return passes


def bin_index(theta_deg, width):
    return np.floor(np.asarray(theta_deg) / width).astype(int)


def binned_means(theta_deg, values, width):
    frame = pd.DataFrame({'bin': bin_index(theta_deg, width), 'value': values})
    grouped = frame.groupby('bin')['value']
    return grouped.mean(), grouped.count(), grouped.std(ddof=1)


# --- friction model ----------------------------------------------------------------------
def stribeck(v, fc, fs, vs, b):
    v = np.asarray(v, dtype=float)
    return np.sign(v) * (fc + (fs - fc) * np.exp(-(np.abs(v) / vs) ** 2)) + b * v


def fit_stribeck(v, f):
    """Least-squares Stribeck + viscous fit to signed (velocity, friction) pairs."""
    v, f = np.asarray(v, float), np.asarray(f, float)
    speed, mag = np.abs(v), np.abs(f)
    order = np.argsort(speed)
    fast = order[len(order) // 2:]
    slope, intercept = np.polyfit(speed[fast], mag[fast], 1) if len(fast) >= 2 else (0.0, mag.mean())
    fc0 = max(float(intercept), 1e-4)
    b0 = max(float(slope), 0.0)
    fs0 = max(float(mag[order[0]]), fc0 * 1.05)
    vs0 = max(float(np.median(speed)) * 0.3, 1e-3)
    result = optimize.least_squares(
        lambda x: stribeck(v, *x) - f, x0=[fc0, fs0, vs0, b0],
        bounds=([0.0, 0.0, 1e-4, 0.0], [np.inf, np.inf, 10.0, np.inf]))
    fc, fs, vs, b = result.x
    residual = result.fun
    dof = max(len(f) - 4, 1)
    s2 = float(residual @ residual) / dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * s2
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
    model_mag = np.abs(stribeck(v, fc, fs, vs, b))
    ss_res = float(np.sum((mag - model_mag) ** 2))
    ss_tot = float(np.sum((mag - mag.mean()) ** 2))
    return {
        'fc': fc, 'fs': fs, 'vs': vs, 'b': b,
        'fc_se': se[0], 'fs_se': se[1], 'vs_se': se[2], 'b_se': se[3],
        'rms': float(np.sqrt(np.mean(residual ** 2))),
        'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
    }


# --- conservative torque ------------------------------------------------------------------
def trend_fit(theta_deg, values, weights):
    """Low-order polynomial trend (gravity, cable preload) over the covered span.

    A sin/cos gravity model is not identifiable over a typical wrist range
    (it is nearly collinear with a constant), so the trend is a polynomial
    whose order grows with the span: linear below 40 deg, up to cubic.
    """
    theta_deg = np.asarray(theta_deg, float)
    span = np.ptp(theta_deg)
    order = 1 if span < 40.0 else (2 if span < 80.0 else 3)
    coef = np.polyfit(theta_deg, values, order, w=np.sqrt(weights))
    slope = np.polyfit(theta_deg, values, 1, w=np.sqrt(weights))[0]
    return np.polyval(coef, theta_deg), float(slope), order


def whole_periods(width_deg, period_deg):
    """Bin width rounded to a whole number of cogging periods, so cogging averages out."""
    if not period_deg or not math.isfinite(period_deg) or period_deg <= 0:
        return width_deg
    return period_deg * max(1, round(width_deg / period_deg))


# --- cogging -----------------------------------------------------------------------------
def gear_ratio(run):
    dev = run.meta.get('device_config', {})
    return float(dev.get('GEAR_RATIO', device_config.GEAR_RATIO))


def slow_cruises(passes_df, passes, column):
    """(motor rev, torque with a cubic trend removed) for each pass slow enough
    to sample cogging densely and for the loop to follow it."""
    out = []
    slow = passes_df[passes_df['speed_deg_s'] <= COGGING_MAX_SPEED_DEG_S]
    for pass_id in slow['pass_id']:
        frame = passes[int(pass_id)]
        m = frame['motor_pos_rev'].to_numpy(float)
        y = frame[column].to_numpy(float)
        ok = np.isfinite(m) & np.isfinite(y)
        m, y = m[ok], y[ok]
        if len(m) < 50 or np.ptp(m) < COGGING_MIN_SPAN_REV:
            continue
        x = m - m.mean()
        out.append((m, y - np.polyval(np.polyfit(x, y, 3), x)))
    return out


def pass_spectrum(m, y, nfft):
    """Hann-windowed power spectrum of one pass, resampled on the motor-position grid."""
    order = np.argsort(m)
    m, y = m[order], y[order]
    n = int((m[-1] - m[0]) / COGGING_GRID_REV) + 1
    idx = np.minimum(((m - m[0]) / COGGING_GRID_REV).astype(int), n - 1)
    counts = np.bincount(idx, minlength=n)
    sums = np.bincount(idx, weights=y, minlength=n)
    have = np.flatnonzero(counts)
    values = np.interp(np.arange(n), have, sums[have] / counts[have])
    window = np.hanning(n)
    return np.abs(np.fft.rfft(values * window, nfft)) ** 2 / np.sum(window ** 2)


def harmonic_design(m, cpr, harmonics):
    columns = []
    for k in range(1, harmonics + 1):
        arg = 2.0 * math.pi * k * cpr * np.asarray(m, float)
        columns += [np.cos(arg), np.sin(arg)]
    return np.column_stack(columns)


def fit_cogging(slow, gear):
    """Cogging frequency, amplitude and phase from the slow cruises.

    The averaged spectrum's peak inside COGGING_BAND_CPR locates the frequency;
    maximizing the variance a sinusoid explains in the raw samples refines it;
    a least-squares harmonic series in absolute motor position then gives the
    model that cogging_torque() evaluates.
    """
    if not slow:
        return {'cogging_detected': False, 'cogging_note': 'no passes slow enough'}
    longest = max(int(np.ptp(m) / COGGING_GRID_REV) + 1 for m, _ in slow)
    nfft = 1 << int(math.ceil(math.log2(8 * longest)))
    freqs = np.fft.rfftfreq(nfft, d=COGGING_GRID_REV)
    power = np.mean([pass_spectrum(m, y, nfft) for m, y in slow], axis=0)
    # At least four cycles in the shortest pass.
    lo = max(COGGING_BAND_CPR[0], 4.0 / min(np.ptp(m) for m, _ in slow))
    hi = min(COGGING_BAND_CPR[1], 0.25 / COGGING_GRID_REV)
    band = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    peak = band[np.argmax(power[band])]
    snr = float(power[peak] / np.median(power[band]))

    m_all = np.concatenate([m for m, _ in slow])
    y_all = np.concatenate([y for _, y in slow])
    step = freqs[1] - freqs[0]

    def explained(cpr):
        X = harmonic_design(m_all, cpr, 1)
        coef, *_ = np.linalg.lstsq(X, y_all, rcond=None)
        return float(np.sum((X @ coef) ** 2))

    candidates = np.linspace(freqs[peak] - 2.0 * step, freqs[peak] + 2.0 * step, 161)
    cpr = float(candidates[int(np.argmax([explained(c) for c in candidates]))])
    X = harmonic_design(m_all, cpr, COGGING_HARMONICS)
    coef, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    residual = y_all - X @ coef
    s2 = float(residual @ residual) / max(len(y_all) - X.shape[1], 1)
    cov = np.linalg.inv(X.T @ X) * s2
    a1, b1 = float(coef[0]), float(coef[1])
    amplitude = math.hypot(a1, b1)
    grad = np.array([a1, b1]) / amplitude if amplitude > 0 else np.zeros(2)
    phase = np.linspace(0.0, 1.0, 721) / cpr
    wave = harmonic_design(phase, cpr, COGGING_HARMONICS) @ coef
    folded = ((m_all * cpr) % 1.0 * COGGING_PHASE_BINS).astype(int)
    bins = pd.DataFrame({'bin': folded, 'value': y_all}).groupby('bin')['value']
    detected = snr >= COGGING_MIN_SNR
    return {
        'cogging_detected': bool(detected),
        'cogging_cycles_per_motor_rev': cpr,
        'cogging_period_deg': 360.0 / (cpr * gear),
        'cogging_amplitude_nm': amplitude,
        'cogging_amplitude_se_nm': float(math.sqrt(max(grad @ cov[:2, :2] @ grad, 0.0))),
        'cogging_p2p_nm': float(np.ptp(wave)),
        'cogging_harmonic_amplitudes_nm': [float(math.hypot(coef[2 * i], coef[2 * i + 1]))
                                           for i in range(COGGING_HARMONICS)],
        'cogging_snr': snr,
        'cogging_num_passes': len(slow),
        'cogging_model': {'periods_per_motor_rev': cpr,
                          'harmonics': [[k + 1, float(coef[2 * k]), float(coef[2 * k + 1])]
                                        for k in range(COGGING_HARMONICS)]},
        'cogging_note': ('' if detected else
                         f'spectral peak only {snr:.1f}x the band median; not a detection'),
        '_folded': pd.DataFrame({'phase': (bins.mean().index.to_numpy() + 0.5)
                                 / COGGING_PHASE_BINS,
                                 'mean_nm': bins.mean().to_numpy(),
                                 'se_nm': (bins.std(ddof=1) / np.sqrt(bins.count())).to_numpy()}),
        '_wave': pd.DataFrame({'phase': phase * cpr, 'model_nm': wave}),
    }


def fit_motor_cogging(run, passes_df, passes, cpr):
    """Each moving motor's cogging from its own measured torque against its own angle.

    At cruise speeds this slow the motor is quasi-static: its torque is its own
    cogging plus its share of friction (a smooth trend per pass, removed), whatever
    the position loops do. So the per-motor fit holds however stiffly the loops
    track and however the other joint of the differential moves, where a fit
    against the joint angle mixes the two motors' phases. Returns None without
    per-motor currents and angles (moteus runs, unmapped current)."""
    T, nodes, joint = run_j2mp(run), run_nodes(run), run_joint_index(run)
    if T is None:
        return None
    slow = passes_df[passes_df['speed_deg_s'] <= COGGING_MAX_SPEED_DEG_S]
    motors, amplitudes = {}, {}
    for i, node in enumerate(nodes):
        if T[i][joint] == 0.0:
            continue
        ms, ys = [], []
        for pass_id in slow['pass_id']:
            frame = passes[int(pass_id)]
            torques = motor_torques(frame, run.meta)
            if torques is None or f'motor_angle_{node}' not in frame:
                return None
            m = frame[f'motor_angle_{node}'].to_numpy(float) / (2.0 * math.pi)   # motor turns
            y = torques[node]
            ok = np.isfinite(m) & np.isfinite(y)
            m, y = m[ok], y[ok]
            if len(m) < 50 or np.ptp(m) < COGGING_MIN_SPAN_REV:
                continue
            x = m - m.mean()
            ms.append(m)
            ys.append(y - np.polyval(np.polyfit(x, y, 3), x))
        if not ms:
            continue
        m_all, y_all = np.concatenate(ms), np.concatenate(ys)
        X = harmonic_design(m_all, cpr, COGGING_HARMONICS)
        coef, *_ = np.linalg.lstsq(X, y_all, rcond=None)
        motors[str(node)] = [[k + 1, float(coef[2 * k]), float(coef[2 * k + 1])]
                             for k in range(COGGING_HARMONICS)]
        amplitudes[str(node)] = float(math.hypot(coef[0], coef[1]))
    if not motors:
        return None
    return {'motors': motors, 'amplitudes': amplitudes, 'nodes': nodes, 'j2mp': T}


def loop_inertia(run):
    """Joint inertia for the drag model: the newest step-response estimate, else nominal."""
    found = latest_summary('step_response', run.simulated)
    if found is not None:
        value = found[1].get('J_kg_m2')
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            return float(value), f'step_response {found[0].name}'
    dev = run.meta.get('device_config', {})
    return float(dev.get('NOMINAL_JOINT_INERTIA_KG_M2',
                         device_config.NOMINAL_JOINT_INERTIA_KG_M2)), 'nominal'


def cogging_drag(speed_rad_s, cog, kp, kd, inertia, filter_hz=VELOCITY_FILTER_HZ, delays=None):
    """Mean torque the position loop draws from the cogging ripple at a cruise speed.

    Linearized about the cruise, cogging harmonic h, A_h sin(h k x), forces a ripple at
    w = h k v against the loop's dynamic stiffness Z(jw) = Kp P(jw) - J w^2 + jw Kd H(jw),
    with P the spring's delay and H the velocity path (filter, and for the p4wrist loop
    its delay). The mean extra torque is A_h^2 (h k)^2 v Re{jw Z_loop}/(w |Z|^2) with the
    loop's dissipative part: the energy it dissipates per unit travel. It grows like
    viscous friction at low speed and peaks where h k v nears the loop's resonance.
    """
    delays = delays or {'position': 0.0, 'velocity': 0.0}
    k = 2.0 * math.pi / math.radians(cog['cogging_period_deg'])
    v = abs(speed_rad_s)
    total = 0.0
    for h, a, b in cog['cogging_model']['harmonics']:
        w = h * k * v
        lag = np.exp(-1j * w * delays['velocity']) / (1.0 + 1j * w / (2.0 * math.pi * filter_hz))
        spring = kp * np.exp(-1j * w * delays['position'])
        z = spring - inertia * w ** 2 + 1j * w * kd * lag
        # Dissipative part of the loop's torque per unit velocity (the delayed
        # spring adds -Kp sin(w Tp)/w, which can be negative).
        damping = (kd * lag).real + (spring.imag / w if w > 0 else 0.0)
        total += (a * a + b * b) * (h * k) ** 2 * v * damping / (2.0 * abs(z) ** 2)
    return total


# --- figures ---------------------------------------------------------------------------
def plot_velocity(passes_df, fit):
    fig, ax = plt.subplots(figsize=figure_size('single', 2.35))
    for i, (direction, label) in enumerate([(1, 'Positive'), (-1, 'Negative')]):
        group = passes_df[passes_df['direction'] == direction]
        ax.plot(group['speed_meas_deg_s'].abs(), group['friction_nm'].abs(), label=label,
                **series_style(i, line=False))
    speeds = passes_df['speed_meas_deg_s'].abs()
    v = np.logspace(np.log10(speeds.min() / 1.25), np.log10(speeds.max() * 1.25), 200)
    model = stribeck(np.radians(v), fit['fc'], fit['fs'], fit['vs'], fit['b'])
    ax.plot(v, model, color=INK, linestyle='--', linewidth=1.0, label='Stribeck + viscous fit')
    reference_line(ax, fit['fc'], label=f"Coulomb level ({fit['fc']:.3f} {TORQUE_UNIT})")
    ax.set_xscale('log')
    log_ticks_hz(ax)
    ax.set_xlim(v[0], v[-1])
    top = max(passes_df['friction_nm'].abs().max(), model.max())
    ax.set_ylim(0, top * 1.45)
    ax.set_xlabel('Speed (deg/s)')
    ax.set_ylabel(f'Friction torque ({TORQUE_UNIT})')
    style_axis(ax)
    ax.legend(loc='upper left', ncol=2, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    return fig


def plot_position(passes, passes_df, column, speed, width):
    """Binned torque vs position for both directions at one speed, and their midline,
    over the positions both directions cover."""
    fig, ax = plt.subplots(figsize=figure_size('single', 2.35))
    chosen = passes_df[passes_df['speed_deg_s'] == speed]
    curves = {}
    for direction in (1, -1):
        ids = chosen[chosen['direction'] == direction]['pass_id']
        if ids.empty:
            continue
        frame = pd.concat([passes[int(k)] for k in ids])
        means, counts, _ = binned_means(frame['joint_pos_deg'], frame[column], width)
        curves[direction] = means[counts >= 3]
    common = None
    for curve in curves.values():
        common = curve.index if common is None else common.intersection(curve.index)
    values = []
    for i, (direction, label) in enumerate([(1, 'Positive pass'), (-1, 'Negative pass')]):
        if direction not in curves:
            continue
        curve = curves[direction].loc[common]
        ax.plot((common.to_numpy() + 0.5) * width, curve.to_numpy(), label=label,
                linewidth=1.0, **series_style(i, marker=False))
        values.append(curve.to_numpy())
    if len(curves) == 2:
        midline = 0.5 * (curves[1].loc[common] + curves[-1].loc[common])
        ax.plot((common.to_numpy() + 0.5) * width, midline.to_numpy(), color=INK,
                linewidth=0.9, label='Conservative (midline)')
        values.append(midline.to_numpy())
    zero_line(ax)
    lo, hi = nice_limits(np.concatenate(values), pad=0.15, include_zero=True)
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.set_xlabel('Joint position (deg)')
    ax.set_ylabel(f'Torque ({TORQUE_UNIT})')
    style_axis(ax)
    ax.legend(loc='upper center', ncol=2, columnspacing=1.0,
              title=f'{speed:g} deg/s, {width:.3g} deg bins', title_fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def plot_diagnostics(df, column, passes, passes_df):
    fig, (ax_p, ax_t) = plt.subplots(2, 1, figsize=figure_size('double', 3.6), sharex=True)
    ax_p.plot(df['time'], df['joint_pos_deg'], color=INK, linewidth=0.5)
    ax_p.set_ylabel('Position (deg)')
    ax_t.plot(df['time'], df[column], color=MUTED, linewidth=0.3)
    for _, row in passes_df.iterrows():
        frame = passes[int(row['pass_id'])]
        i = 0 if row['direction'] > 0 else 1
        ax_t.plot(frame['time'], frame[column], color=series_style(i)['color'], linewidth=0.4)
        ax_t.plot(frame['time'].mean(), row['torque_mean_nm'], linestyle='None',
                  marker='o', markersize=3, markerfacecolor='white', markeredgecolor=INK)
    ax_t.set_ylabel(f'Torque ({TORQUE_UNIT})')
    ax_t.set_xlabel('Time (s)')
    for ax in (ax_p, ax_t):
        style_axis(ax, minor=False)
    add_panel_label(ax_p, '(a)')
    add_panel_label(ax_t, '(b)')
    fig.tight_layout(pad=0.3)
    return fig


def plot_ripple(profile, cog, fine_bin):
    """(a) Cogging folded onto one period, with the fitted model; (b) the conservative
    torque on whole-period bins and its low-order trend."""
    fig, (ax_c, ax_r) = plt.subplots(2, 1, figsize=figure_size('single', 3.9))
    if '_folded' in cog:
        period = cog['cogging_period_deg']
        folded, wave = cog['_folded'], cog['_wave']
        ax_c.errorbar(folded['phase'] * period, folded['mean_nm'], yerr=1.96 * folded['se_nm'],
                      markersize=3, elinewidth=0.6, label='Slow passes, folded',
                      **series_style(0, line=False))
        ax_c.plot(wave['phase'] * period, wave['model_nm'], color=INK, linewidth=0.9,
                  label=f"Model, {cog['cogging_cycles_per_motor_rev']:.1f} cycles/motor rev")
        zero_line(ax_c)
        ax_c.set_xlabel('Position within one cogging period (deg)')
        ax_c.legend(loc='upper right')
    else:
        ax_c.text(0.5, 0.5, 'No passes slow enough to resolve cogging', ha='center',
                  va='center', transform=ax_c.transAxes, color=MUTED)
    ax_c.set_ylabel(f'Cogging ({TORQUE_UNIT})')
    ax_r.plot(profile['center_deg'], profile['conservative_nm'], color=INK, linewidth=0.6,
              label=f'Conservative torque ({fine_bin:.3g} deg bins)')
    ax_r.plot(profile['center_deg'], profile['trend_nm'], color='#2a78d6', linestyle='--',
              linewidth=1.0, label='Low-order trend')
    lo, hi = nice_limits(profile['conservative_nm'], pad=0.05)
    ax_r.set_ylim(lo, hi + 0.5 * (hi - lo))
    ax_r.set_xlabel('Joint position (deg)')
    ax_r.set_ylabel(f'Torque ({TORQUE_UNIT})')
    ax_r.legend(loc='upper center', ncol=2)
    for ax in (ax_c, ax_r):
        style_axis(ax)
    add_panel_label(ax_c, '(a)')
    add_panel_label(ax_r, '(b)')
    fig.tight_layout(pad=0.3)
    return fig


# --- main ------------------------------------------------------------------------------
def main(argv=None):
    args = make_parser(__doc__.splitlines()[0]).parse_args(argv)
    configure_ieee_style()
    run = load_run(EXPERIMENT, args.data, LEGACY)
    df = run.df.sort_values('time').reset_index(drop=True)
    column = torque_column(df)
    settle_s = settle_time(run)
    passes = cruise_passes(df, column, settle_s)
    if not passes:
        raise SystemExit('No cruise segments found.')

    rows = []
    for pass_id, frame in passes.items():
        rows.append({
            'pass_id': pass_id,
            'speed_deg_s': float(frame['speed_deg_s'].iloc[0]),
            'direction': int(frame['direction'].iloc[0]),
            'round_trip': int(frame['round_trip'].iloc[0]),
            'speed_meas_deg_s': float(frame['joint_vel_deg_s'].mean()),
            'torque_mean_nm': float(frame[column].mean()),
            'torque_std_nm': float(frame[column].std(ddof=1)),
            'pos_min_deg': float(frame['joint_pos_deg'].min()),
            'pos_max_deg': float(frame['joint_pos_deg'].max()),
            'num_samples': int(len(frame)),
        })
    passes_df = pd.DataFrame(rows)

    # Cogging first: its period sets the bin widths below.
    gear = gear_ratio(run)
    if 'motor_pos_rev' in df.columns:
        cog = fit_cogging(slow_cruises(passes_df, passes, column), gear)
    else:
        cog = {'cogging_detected': False, 'cogging_note': 'no motor position column'}
    if cog['cogging_detected']:
        per_motor = fit_motor_cogging(run, passes_df, passes, cog['cogging_cycles_per_motor_rev'])
        if per_motor:
            cog['cogging_model'].update({'motors': per_motor['motors'], 'nodes': per_motor['nodes'],
                                         'j2mp': per_motor['j2mp']})
            cog['cogging_motor_amplitudes_nm'] = per_motor['amplitudes']
            # The joint's cogging the motor models predict at the sweep's mean pose,
            # moving only this joint (the other joint of a differential sets the phase
            # between the two motors' ripples, so this amplitude belongs to that pose).
            joint, nodes, T = run_joint_index(run), per_motor['nodes'], per_motor['j2mp']
            n_eff = max(abs(T[i][joint]) for i in range(3))
            span = 2.0 * math.pi / (cog['cogging_cycles_per_motor_rev'] * n_eff)
            dq = np.linspace(0.0, span, 400, endpoint=False)
            pose = {f'motor_angle_{n}': float(df[f'motor_angle_{n}'].mean()) + T[i][joint] * dq
                    for i, n in enumerate(nodes)}
            predicted = cogging_torque(cog['cogging_model'], pose, joint)
            X = harmonic_design(dq * n_eff / (2.0 * math.pi), cog['cogging_cycles_per_motor_rev'], 1)
            coef, *_ = np.linalg.lstsq(X, predicted, rcond=None)
            cog['cogging_amplitude_from_motors_nm'] = float(math.hypot(*coef))
    period = cog['cogging_period_deg'] if cog['cogging_detected'] else None
    pair_bin = whole_periods(BIN_DEG, period)
    fine_bin = whole_periods(FINE_BIN_DEG, period)
    display_bin = whole_periods(POSITION_FIGURE_BIN_DEG, period)

    # Pair directions within each speed on position bins.
    per_speed, cons_parts = [], []
    for speed, group in passes_df.groupby('speed_deg_s'):
        sides = {}
        for direction in (1, -1):
            ids = group[group['direction'] == direction]['pass_id']
            if ids.empty:
                continue
            frame = pd.concat([passes[int(k)] for k in ids])
            sides[direction] = binned_means(frame['joint_pos_deg'], frame[column], pair_bin)[0]
        if len(sides) < 2:
            continue
        common = sides[1].index.intersection(sides[-1].index)
        if len(common) == 0:
            continue
        fwd, bwd = sides[1].loc[common], sides[-1].loc[common]
        friction_bins = 0.5 * (fwd - bwd)
        cons_parts.append(pd.DataFrame({'bin': common, 'conservative': 0.5 * (fwd + bwd).values}))
        per_speed.append({
            'speed_deg_s': speed,
            'friction_nm': float(friction_bins.mean()),
            'friction_se_nm': float(friction_bins.std(ddof=1) / math.sqrt(len(common)))
            if len(common) > 1 else math.nan,
            'num_bins': int(len(common)),
        })
    if not per_speed:
        raise SystemExit('Need forward and backward passes over the same positions.')
    per_speed = pd.DataFrame(per_speed)
    cons_all = pd.concat(cons_parts)
    cons_global = cons_all.groupby('bin')['conservative'].mean()
    centers = (cons_global.index.to_numpy() + 0.5) * pair_bin

    # Signed friction per pass after removing the shared conservative torque.
    friction = []
    for _, row in passes_df.iterrows():
        frame = passes[int(row['pass_id'])]
        cons = np.interp(frame['joint_pos_deg'], centers, cons_global.to_numpy())
        friction.append(float((frame[column] - cons).mean()))
    passes_df['friction_raw_nm'] = friction

    # Remove the drag the loop draws from the cogging ripple (see cogging_drag).
    kp = float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad'))
    kd = float(run.meta_value('kd_joint_nms_rad', column='kd_joint_nms_rad'))
    inertia, inertia_source = loop_inertia(run)
    filter_hz = float(run.meta.get('velocity_filter_hz') or VELOCITY_FILTER_HZ)
    delays = loop_delays(run)
    if cog['cogging_detected']:
        drag = passes_df['speed_meas_deg_s'].apply(
            lambda s: cogging_drag(math.radians(s), cog, kp, kd, inertia, filter_hz, delays))
    else:
        drag = pd.Series(0.0, index=passes_df.index)
    raw = passes_df['friction_raw_nm']
    passes_df['cogging_drag_nm'] = drag
    passes_df['friction_nm'] = raw - np.sign(raw) * drag
    passes_df['used_in_fit'] = drag <= MAX_DRAG_FRACTION * raw.abs()
    for speed in per_speed['speed_deg_s']:
        mask = per_speed['speed_deg_s'] == speed
        sel_speed = passes_df['speed_deg_s'] == speed
        per_speed.loc[mask, 'cogging_drag_nm'] = passes_df.loc[sel_speed, 'cogging_drag_nm'].mean()
        for direction, key in ((1, 'friction_pos_nm'), (-1, 'friction_neg_nm')):
            sel = sel_speed & (passes_df['direction'] == direction)
            per_speed.loc[mask, key] = passes_df.loc[sel, 'friction_nm'].abs().mean()
    per_speed['friction_corrected_nm'] = per_speed['friction_nm'] - per_speed['cogging_drag_nm']

    used = passes_df[passes_df['used_in_fit']]
    if len(used) < 4:
        raise SystemExit('Too few passes left after excluding cogging-drag-dominated speeds.')
    fit = fit_stribeck(np.radians(used['speed_meas_deg_s']), used['friction_nm'])

    # Fine conservative profile: torque minus the fitted friction of each pass.
    fine = []
    for _, row in passes_df.iterrows():
        frame = passes[int(row['pass_id'])]
        f_model = stribeck(math.radians(row['speed_meas_deg_s']), fit['fc'], fit['fs'],
                           fit['vs'], fit['b'])
        fine.append(pd.DataFrame({'theta': frame['joint_pos_deg'].to_numpy(),
                                  'value': frame[column].to_numpy() - float(f_model),
                                  'pass_id': int(row['pass_id'])}))
    fine = pd.concat(fine)
    fine['bin'] = bin_index(fine['theta'], fine_bin)
    grouped = fine.groupby('bin')['value']
    means, counts = grouped.mean(), grouped.count()
    # Standard error from the scatter between passes: cogging inside a whole-period
    # bin inflates the within-bin spread but not the bin mean.
    pass_means = fine.groupby(['bin', 'pass_id'])['value'].mean().groupby(level='bin')
    n_pass = pass_means.count().reindex(means.index).fillna(0)
    se = (pass_means.std(ddof=1) / np.sqrt(pass_means.count())).reindex(means.index)
    keep = ((counts >= 5) & (n_pass >= 2)).to_numpy()
    profile = pd.DataFrame({
        'center_deg': (means.index.to_numpy()[keep] + 0.5) * fine_bin,
        'conservative_nm': means.to_numpy()[keep],
        'count': counts.to_numpy()[keep],
        'se_nm': se.to_numpy()[keep],
    })
    trend, slope, trend_order = trend_fit(profile['center_deg'], profile['conservative_nm'],
                                          profile['count'])
    ripple = profile['conservative_nm'].to_numpy() - trend
    profile['trend_nm'] = trend
    profile['ripple_nm'] = ripple
    offset = float(np.average(profile['conservative_nm'], weights=profile['count']))
    noise_floor = float(np.sqrt(np.nanmean(profile['se_nm'] ** 2)))
    ripple_std = float(np.std(ripple, ddof=1))
    tau_max = device_config.MAX_JOINT_TORQUE_NM

    summary = {
        'coulomb_friction_nm': fit['fc'],
        'coulomb_friction_se_nm': fit['fc_se'],
        'static_friction_stribeck_nm': fit['fs'],
        'static_friction_stribeck_se_nm': fit['fs_se'],
        'stribeck_velocity_rad_s': fit['vs'],
        'stribeck_velocity_deg_s': math.degrees(fit['vs']),
        'viscous_damping_nms_rad': fit['b'],
        'viscous_damping_se_nms_rad': fit['b_se'],
        'fit_r2': fit['r2'],
        'fit_rms_nm': fit['rms'],
        'coulomb_percent_of_max_torque': 100.0 * fit['fc'] / tau_max,
        'static_percent_of_max_torque': 100.0 * fit['fs'] / tau_max,
        'friction_at_max_speed_nm': float(per_speed['friction_corrected_nm'].iloc[-1]),
        'max_speed_deg_s': float(per_speed['speed_deg_s'].iloc[-1]),
        'conservative_offset_nm': offset,
        'conservative_slope_nm_per_deg': slope,
        'conservative_trend_p2p_nm': float(np.ptp(trend)),
        'ripple_std_nm': ripple_std,
        'ripple_p2p_nm': float(np.ptp(ripple)),
        'ripple_noise_floor_nm': noise_floor,
        'ripple_std_corrected_nm': math.sqrt(max(ripple_std ** 2 - noise_floor ** 2, 0.0)),
        **{k: v for k, v in cog.items() if not k.startswith('_')},
        'pair_bin_deg': pair_bin,
        'fine_bin_deg': fine_bin,
        'max_cogging_drag_nm': float(passes_df['cogging_drag_nm'].max()),
        'speed_of_max_cogging_drag_deg_s': float(
            passes_df.loc[passes_df['cogging_drag_nm'].idxmax(), 'speed_deg_s']),
        'num_passes_in_fit': int(passes_df['used_in_fit'].sum()),
        'num_passes': int(len(passes_df)),
        'num_speeds': int(len(per_speed)),
        'torque_column': column,
        'kp_joint_nm_rad': float(run.meta_value('kp_joint_nm_rad', column='kp_joint_nm_rad')),
        'kd_joint_nms_rad': float(run.meta_value('kd_joint_nms_rad', column='kd_joint_nms_rad')),
    }
    metadata = {
        'analysis': 'friction_sweep_experiment_analysis.py',
        'source_data': str(run.source),
        'method': ('forward/backward pairing on position bins of whole cogging periods; '
                   'Stribeck + viscous least squares on per-pass friction; cogging from the '
                   'spectrum peak and a harmonic least-squares fit on the slowest passes'),
        'bin_deg': BIN_DEG,
        'fine_bin_deg': FINE_BIN_DEG,
        'pair_bin_deg_used': pair_bin,
        'fine_bin_deg_used': fine_bin,
        'cogging_max_speed_deg_s': COGGING_MAX_SPEED_DEG_S,
        'cogging_band_cycles_per_motor_rev': list(COGGING_BAND_CPR),
        'cogging_grid_rev': COGGING_GRID_REV,
        'cogging_harmonics': COGGING_HARMONICS,
        'cogging_min_snr': COGGING_MIN_SNR,
        'gear_ratio': gear,
        'cogging_drag': ('per-pass friction minus the mean torque the loop draws from the '
                         'cogging ripple, sum_h A_h^2 (h k)^2 v Re{Kd H} / (2 |Z(j h k v)|^2), '
                         'Z = Kp - J w^2 + j w Kd H; passes whose drag exceeds '
                         f'{MAX_DRAG_FRACTION:g} of their friction are left out of the fit'),
        'cogging_drag_inertia_kg_m2': inertia,
        'cogging_drag_inertia_source': inertia_source,
        'velocity_filter_hz': filter_hz,
        'ripple_trend_order': trend_order,
        'drop_fraction': DROP_FRACTION,
        'settle_periods': SETTLE_PERIODS,
        'settle_time_s': settle_s,
        'note': ('Torques are in the torque scale the drives report; multiply by the torque '
                 'calibration ratio if one was measured. Bins span whole cogging periods, so '
                 'the ripple excludes cogging, which is reported separately; '
                 'ripple_std_corrected_nm removes the noise floor estimated from the scatter '
                 'between passes. The cogging model is in absolute motor position and holds '
                 'until the encoder is re-zeroed.'),
    }
    save_outputs(run, {'passes': passes_df, 'per_speed': per_speed,
                       'conservative_profile': profile}, summary, metadata)

    print_header('FRICTION VS VELOCITY (CONSTANT-VELOCITY SWEEPS)')
    print(f"Passes: {summary['num_passes']}  speeds: {summary['num_speeds']}  "
          f'torque from {column}')
    print(f"  Coulomb friction fc      {fit['fc']:.4g} N m  (SE {fit['fc_se']:.2g})")
    print(f"  Static (Stribeck) fs     {fit['fs']:.4g} N m  (SE {fit['fs_se']:.2g})")
    print(f"  Stribeck velocity vs     {fit['vs']:.4g} rad/s ({math.degrees(fit['vs']):.3g} deg/s)")
    print(f"  Viscous damping b        {fit['b']:.4g} N m s/rad  (SE {fit['b_se']:.2g})")
    print(f"  Fit R^2 {fit['r2']:.4f}, RMS {fit['rms']:.3g} N m, "
          f"{summary['num_passes_in_fit']}/{summary['num_passes']} passes")
    if cog['cogging_detected']:
        drags = ', '.join(f"{r.speed_deg_s:g}: {1e3 * r.cogging_drag_nm:.1f}"
                          for r in per_speed.itertuples())
        print(f'  Cogging drag removed (deg/s: mN m): {drags}')
    print(f"  Friction = {summary['coulomb_percent_of_max_torque']:.2f}% (Coulomb) / "
          f"{summary['static_percent_of_max_torque']:.2f}% (static) of {tau_max} N m max torque")
    print(f"  Conservative torque: offset {offset:.3g} N m, slope {slope:.3g} N m/deg, "
          f"trend p2p {summary['conservative_trend_p2p_nm']:.3g} N m")
    print(f"  Position ripple excluding cogging: SD {summary['ripple_std_corrected_nm']:.3g} N m "
          f"(noise-corrected), p2p {summary['ripple_p2p_nm']:.3g} N m (incl. noise), "
          f'{fine_bin:.3g} deg bins')
    if 'cogging_amplitude_nm' in cog:
        print(f"  Cogging{'' if cog['cogging_detected'] else ' (NOT DETECTED)'}: amplitude "
              f"{cog['cogging_amplitude_nm']:.4g} N m (SE {cog['cogging_amplitude_se_nm']:.2g}), "
              f"p2p {cog['cogging_p2p_nm']:.4g} N m, {cog['cogging_cycles_per_motor_rev']:.2f} "
              f"cycles/motor rev = every {cog['cogging_period_deg']:.4g} deg at the joint; "
              f"spectral SNR {cog['cogging_snr']:.0f}, {cog['cogging_num_passes']} passes "
              f"<= {COGGING_MAX_SPEED_DEG_S:g} deg/s")
    else:
        print(f"  Cogging not identified: {cog['cogging_note']}")
    truth = sim_truth(run)
    if truth:
        rows = [
            ('Coulomb fc', fit['fc'], truth['coulomb'], 'N m'),
            ('Static fs', fit['fs'], truth['static'], 'N m'),
            ('Stribeck vs', fit['vs'], truth['stribeck_velocity'], 'rad/s'),
            ('Viscous b', fit['b'], truth['viscous'] + truth['output_viscous'], 'N m s/rad'),
        ]
        if 'cogging_amplitude_nm' in cog:
            rows += [('Cogging amplitude', cog['cogging_amplitude_nm'],
                      truth.get('cogging_amplitude'), 'N m'),
                     ('Cogging period', cog['cogging_period_deg'],
                      truth.get('cogging_period_deg'), 'deg')]
        if 'cogging_amplitude_from_motors_nm' in cog:
            rows.append(('Cogging (from motor models)', cog['cogging_amplitude_from_motors_nm'],
                         truth.get('cogging_amplitude'), 'N m'))
        motor_truth = (run.meta.get('sim_params') or {}).get('cogging_amplitude_motor')
        for node, amplitude in (cog.get('cogging_motor_amplitudes_nm') or {}).items():
            i = run_nodes(run).index(int(node))
            value = motor_truth[i] if isinstance(motor_truth, (list, tuple)) else motor_truth
            rows.append((f'Cogging, motor {node}', amplitude, value, 'N m'))
        print_truth_comparison(rows)

    fig = plot_velocity(used, fit)
    paths = export_figure(fig, run, VELOCITY_FIGURE, {**metadata, **summary}, args.paper_dir)
    full_window = passes_df.groupby('speed_deg_s').apply(
        lambda g: g['pos_max_deg'].max() - g['pos_min_deg'].min(), include_groups=False)
    candidates = full_window[full_window >= 0.8 * full_window.max()].index.to_numpy()
    speed = float(candidates[np.argmin(np.abs(np.log(candidates / POSITION_FIGURE_SPEED)))])
    export_figure(plot_position(passes, passes_df, column, speed, display_bin), run,
                  POSITION_FIGURE, {**metadata, 'speed_deg_s': speed}, args.paper_dir)
    export_figure(plot_diagnostics(df, column, passes, passes_df), run, DIAGNOSTIC_FIGURE,
                  metadata, paper_dir='')
    export_figure(plot_ripple(profile, cog, fine_bin), run, RIPPLE_DIAGNOSTIC, metadata,
                  paper_dir='')
    print(f"Saved results to {run.out_dir}; paper figure {paths['pdf']}")
    if args.no_show:
        plt.close('all')
    else:
        plt.show()
    return summary


if __name__ == '__main__':
    main()
