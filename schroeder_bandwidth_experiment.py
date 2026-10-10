"""Closed-loop position bandwidth from a periodic, Schroeder-phased multisine.

The joint tracks a multisine reference under a stiff PD controller. The
reference repeats with period T and only excites frequencies on the 1/T grid,
so after the first (transient) period every line can be estimated exactly by
least squares, period by period, giving the frequency response with error
bars. Lines are log-spaced from FREQ_MIN to FREQ_MAX; the amplitude spectrum
is flat up to a corner and rolled off above it so the inertial torque stays
inside a budget. Several amplitude levels show how friction (small motions)
and saturation (large motions) change the result.

Analyze with schroeder_bandwidth_experiment_analysis.py.

Run:  python schroeder_bandwidth_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

import device_config
from experiment_common import (Command, WristSession, Segment, confirm, describe_plan, hold,
                               make_arg_parser, run_cli)

EXPERIMENT = 'bandwidth'
SIM_PARAMS = None   # Tests may inject a p4_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class BandwidthConfig:
    kp_joint_nm_rad: float = 80.0           # Stiff PD (nominal operating gains)
    kd_joint_nms_rad: float = 1.0 
    send_velocity_feedforward: bool = True  # Changes the closed-loop transfer function
    amplitudes_deg: tuple = (2.0, 5.0, 10.0)   # Peak command per level
    period_s: float = 10.0                  # Frequency resolution is 1 / period
    periods_per_level: int = 5              # The first period is discarded (transient)
    freq_min_hz: float = 0.2
    freq_max_hz: float = 30.0
    num_lines: int = 50                     # Log-spaced excited lines (unique DFT bins)
    corner_hz: float = 3.0                  # Flat amplitude below, rolled off above
    rolloff_order: int = 1                  # 1: constant velocity, 2: constant acceleration
    torque_budget_fraction: float = 0.5     # Predicted peak torque / torque cap
    crest_iterations: int = 40              # Clipping iterations to lower the crest factor
    rest_s: float = 1.0                     # Hold at zero between levels
    sample_rate_hz: float = 500.0


QUICK = dict(amplitudes_deg=(3.0, 10.0), period_s=5.0, periods_per_level=3,
             freq_min_hz=0.4, num_lines=30)


# --- multisine design -----------------------------------------------------------------
def excited_bins(cfg):
    """Unique integer DFT bins, log-spaced between freq_min and freq_max."""
    k_min = max(1, int(round(cfg.freq_min_hz * cfg.period_s)))
    k_max = int(math.floor(cfg.freq_max_hz * cfg.period_s))
    if k_max <= k_min:
        raise SystemExit('freq_max_hz must exceed freq_min_hz by at least one bin.')
    n = cfg.num_lines
    bins = np.unique(np.round(np.geomspace(k_min, k_max, n)).astype(int))
    while len(bins) < min(cfg.num_lines, k_max - k_min + 1):
        n += 5
        bins = np.unique(np.round(np.geomspace(k_min, k_max, n)).astype(int))
    return bins


def amplitude_shape(freqs, corner_hz, order):
    return np.where(freqs <= corner_hz, 1.0, (corner_hz / freqs) ** order)


def schroeder_phases(amplitudes):
    """Schroeder phases for an arbitrary power spectrum (phi_1 = 0)."""
    power = amplitudes ** 2 / np.sum(amplitudes ** 2)
    k = np.arange(1, len(amplitudes) + 1)
    phases = np.zeros(len(amplitudes))
    for i in range(1, len(amplitudes)):
        phases[i] = -2.0 * np.pi * np.sum((k[i] - k[:i]) * power[:i])
    return np.mod(phases, 2.0 * np.pi)


def synthesize(bins, amplitudes, phases, n):
    """One period on an n-point grid: sum A cos(2 pi k i / n + phi)."""
    spectrum = np.zeros(n // 2 + 1, dtype=complex)
    spectrum[bins] = amplitudes * np.exp(1j * phases)
    return np.fft.irfft(spectrum, n) * (n / 2.0)


def reduce_crest_factor(bins, amplitudes, phases, n, iterations, clip=0.9):
    """Clip-and-restore iterations (keep amplitudes, adapt phases)."""
    def crest(x):
        return np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))

    best_phases = phases
    best = crest(synthesize(bins, amplitudes, phases, n))
    for _ in range(iterations):
        x = synthesize(bins, amplitudes, phases, n)
        peak = np.max(np.abs(x))
        clipped = np.clip(x, -clip * peak, clip * peak)
        phases = np.angle(np.fft.rfft(clipped)[bins])
        value = crest(synthesize(bins, amplitudes, phases, n))
        if value < best:
            best, best_phases = value, phases
    return np.mod(best_phases, 2.0 * np.pi), best


class Multisine:
    """Unit-peak multisine; position and velocity at any time (deg, deg/s per deg)."""

    def __init__(self, freqs_hz, amplitudes, phases, t_offset_s):
        self.freqs = np.asarray(freqs_hz, dtype=float)
        self.w = 2.0 * np.pi * self.freqs
        self.amplitudes = np.asarray(amplitudes, dtype=float)
        self.phases = np.asarray(phases, dtype=float)
        self.t_offset = t_offset_s

    def __call__(self, t):
        arg = self.w * (t + self.t_offset) + self.phases
        return (float(np.dot(self.amplitudes, np.cos(arg))),
                float(-np.dot(self.amplitudes * self.w, np.sin(arg))))


def design_multisine(cfg, torque_cap_nm):
    """Lines, amplitudes, phases and start offset meeting the torque budget."""
    bins = excited_bins(cfg)
    freqs = bins / cfg.period_s
    n = 1 << int(math.ceil(math.log2(max(1024, 20 * cfg.freq_max_hz * cfg.period_s))))
    dt = cfg.period_s / n
    j = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    b = device_config.NOMINAL_JOINT_DAMPING_NMS_RAD
    fc = device_config.NOMINAL_COULOMB_FRICTION_NM
    budget = cfg.torque_budget_fraction * torque_cap_nm
    a_max = max(cfg.amplitudes_deg)
    corner = min(cfg.corner_hz, cfg.freq_max_hz)
    while True:
        shape = amplitude_shape(freqs, corner, cfg.rolloff_order)
        phases = schroeder_phases(shape)
        phases, crest = reduce_crest_factor(bins, shape, phases, n, cfg.crest_iterations)
        x = synthesize(bins, shape, phases, n)
        scale = 1.0 / np.max(np.abs(x))         # Unit peak position
        amplitudes = shape * scale
        w = 2.0 * np.pi * freqs
        vel = synthesize(bins, amplitudes * w, phases + np.pi / 2, n)
        acc = synthesize(bins, amplitudes * w ** 2, phases + np.pi, n)
        peak_vel = float(np.max(np.abs(vel)))   # deg/s per deg of amplitude
        peak_acc = float(np.max(np.abs(acc)))
        torque = (j * math.radians(peak_acc * a_max) + b * math.radians(peak_vel * a_max) + fc)
        if torque <= budget or corner <= freqs[0] * 1.0001:
            break
        corner = max(corner * 0.85, freqs[0])
    if torque > budget:
        raise SystemExit(
            f'Even with the corner at {corner:.2f} Hz the predicted peak torque '
            f'{torque:.2f} N m exceeds the budget {budget:.2f} N m. Lower amplitudes_deg or '
            'freq_max_hz, or raise rolloff_order.')
    # Start where position and velocity are both near zero (smallest jump).
    pos = x * scale
    score = np.abs(pos) + np.abs(vel) / peak_vel
    t_offset = float(np.argmin(score) * dt)
    return {
        'bins': bins.tolist(),
        'freqs_hz': freqs.tolist(),
        'amplitudes_per_deg': amplitudes.tolist(),
        'phases_rad': phases.tolist(),
        't_offset_s': t_offset,
        'period_s': cfg.period_s,
        'corner_hz': float(corner),
        'rolloff_order': cfg.rolloff_order,
        'crest_factor': float(crest),
        'peak_velocity_deg_s_per_deg': peak_vel,
        'peak_accel_deg_s2_per_deg': peak_acc,
        'levels': [{
            'amplitude_deg': a,
            'predicted_peak_velocity_deg_s': peak_vel * a,
            'predicted_peak_accel_deg_s2': peak_acc * a,
            'predicted_peak_torque_nm': (j * math.radians(peak_acc * a)
                                         + b * math.radians(peak_vel * a) + fc),
        } for a in cfg.amplitudes_deg],
    }


def build_segments(cfg, design):
    signal = Multisine(design['freqs_hz'], design['amplitudes_per_deg'],
                       design['phases_rad'], design['t_offset_s'])
    duration = cfg.period_s * cfg.periods_per_level
    ff = cfg.send_velocity_feedforward
    segments = [hold(cfg.rest_s, 0.0, phase='rest')]
    for level, amplitude in enumerate(cfg.amplitudes_deg):
        def command(t, _, a=amplitude):
            pos, vel = signal(t)
            return Command(position_deg=a * pos, velocity_deg_s=a * vel if ff else 0.0)

        segments.append(Segment('multisine', duration, command,
                                {'level': level, 'amplitude_deg': amplitude}))
        segments.append(hold(cfg.rest_s, 0.0, phase='rest'))
    return segments


def plan_lines(cfg, design):
    lo, hi = device_config.JOINT_LIMITS_DEG
    reach = 1.5 * max(cfg.amplitudes_deg)
    if reach > min(abs(lo), abs(hi)):
        raise SystemExit(f'Amplitudes may reach {reach:.0f} deg, beyond JOINT_LIMITS_DEG.')
    if cfg.freq_max_hz > cfg.sample_rate_hz / 10.0:
        raise SystemExit('Keep freq_max_hz below sample_rate_hz / 10.')
    if (cfg.freq_min_hz * cfg.period_s) < 1.0:
        raise SystemExit('freq_min_hz must be at least 1 / period_s.')
    lines = [f"  {len(design['freqs_hz'])} lines {design['freqs_hz'][0]:.2f}-"
             f"{design['freqs_hz'][-1]:.1f} Hz, corner {design['corner_hz']:.2f} Hz "
             f"(1/f^{cfg.rolloff_order} above), crest factor {design['crest_factor']:.2f}"]
    for level in design['levels']:
        vel = level['predicted_peak_velocity_deg_s']
        if vel > 0.8 * device_config.MAX_JOINT_VELOCITY_DEG_S:
            raise SystemExit(f"Level {level['amplitude_deg']} deg needs {vel:.0f} deg/s, "
                             'too close to MAX_JOINT_VELOCITY_DEG_S.')
        lines.append(f"  {level['amplitude_deg']:5.1f} deg: peak {vel:6.0f} deg/s, "
                     f"{level['predicted_peak_accel_deg_s2']:8.0f} deg/s^2, "
                     f"~{level['predicted_peak_torque_nm']:.2f} N m")
    return lines


async def main(args):
    cfg = BandwidthConfig(**(QUICK if args.quick else {}))
    design = design_multisine(cfg, device_config.MAX_JOINT_TORQUE_NM)
    segments = build_segments(cfg, design)
    lines = plan_lines(cfg, design)
    async with WristSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, cfg.kd_joint_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        confirm(args, ['Closed-loop position bandwidth (periodic multisine)',
                       f'  Kp = {cfg.kp_joint_nm_rad} N m/rad, Kd = {cfg.kd_joint_nms_rad} '
                       f'N m s/rad, velocity feedforward {cfg.send_velocity_feedforward}',
                       *describe_plan(segments, session.max_torque_nm), *lines])
        session.metadata['multisine'] = design
        session.metadata['send_velocity_feedforward'] = cfg.send_velocity_feedforward
        session.metadata['periods_per_level'] = cfg.periods_per_level
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
