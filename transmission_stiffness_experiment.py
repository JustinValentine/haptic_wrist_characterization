"""Locked-output transmission test: cable stiffness, lost motion, and resonance.

With the output clamped to the frame, every motor-side rotation is cable
stretch (plus any backlash), so torque versus motor-side angle measures the
transmission directly:

  Phase 1 (quasi-static): slow triangular torque cycles 0 -> +T -> -T -> 0.
      The tau-theta hysteresis loop gives the stiffness (slope), lost motion
      (gap between the positive and negative branches), and friction
      hysteresis.
  Phase 2 (dynamic): a torque multisine about a preload (which keeps the cable
      out of any backlash). The compliance FRF theta/tau of the motor side on
      the cable gives the dynamic stiffness, the reflected motor-side inertia,
      and the transmission resonance that ultimately bounds control bandwidth.

All quantities are joint-referred; the motor encoder angle is divided by
GEAR_RATIO. transmission_stiffness_experiment_analysis.py does the analysis.

Run:  python transmission_stiffness_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

import device_config
from experiment_common import (Command, WristSession, SafetyLimits, Segment, confirm,
                               describe_plan, hold, make_arg_parser, min_jerk, run_cli)

EXPERIMENT = 'transmission'
SIM_PARAMS = None   # Tests inject a p4_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class TransmissionConfig:
    # Peak cycle torque as a fraction of MAX_JOINT_TORQUE_NM. High, because one P4
    # motor count is ~0.3 N m of cable torque at FE (0.0117 deg at ~1500 N m/rad).
    test_torque_fraction: float = 0.8
    cycle_period_s: float = 8.0             # One 0 -> +T -> -T -> 0 cycle (quasi-static)
    cycles: int = 3
    # Multisine phase. Off by default on the P4 wrist: its motion is below one count
    # of the 4096-count motor encoders (see plan_checks); the static loops still work.
    dynamic: bool = False
    preload_fraction: float = 0.3           # Multisine offset, fraction of the cycle torque
    multisine_amplitude_fraction: float = 0.15   # Multisine peak, fraction of the cycle torque
    multisine_period_s: float = 2.0         # Frequency resolution = 1 / period
    multisine_periods: int = 6              # The first period is discarded as transient
    freq_min_hz: float = 1.0
    freq_max_hz: float = 200.0              # Past the locked-output resonance; Nyquist is 250 Hz
    num_lines: int = 40                     # Log-spaced excited frequencies
    max_deflection_deg: float = 3.0         # Motor-side joint angle; more means the lock slipped
    max_speed_deg_s: float = 600.0          # Resonance can be lively; a slipped lock is faster
    sample_rate_hz: float = 500.0


QUICK = dict(cycles=2, cycle_period_s=4.0, multisine_periods=4)


def triangle(s):
    """0 -> 1 -> 0 -> -1 -> 0 over s in [0, 1)."""
    s = s % 1.0
    if s < 0.25:
        return 4.0 * s
    if s < 0.75:
        return 2.0 - 4.0 * s
    return 4.0 * s - 4.0


def multisine_design(cfg, test_torque, seed=0, candidates=300):
    """Log-spaced lines on the DFT grid of one period, low-crest-factor phases."""
    period = cfg.multisine_period_s
    nyquist_guard = cfg.sample_rate_hz / 2.5
    f_max = min(cfg.freq_max_hz, nyquist_guard)
    harmonics = np.unique(np.round(np.logspace(
        math.log10(max(cfg.freq_min_hz * period, 1.0)), math.log10(f_max * period),
        cfg.num_lines)).astype(int))
    freqs = harmonics / period
    k = np.arange(1, len(freqs) + 1)
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, period, 4000, endpoint=False)
    arg = 2.0 * np.pi * np.outer(grid, freqs)
    phase_sets = [-np.pi * k * (k - 1) / len(freqs)]
    phase_sets += [rng.uniform(0, 2 * np.pi, len(freqs)) for _ in range(candidates)]
    best = None
    for phases in phase_sets:
        x = np.cos(arg + phases).sum(axis=1)
        crest = np.max(np.abs(x)) / np.sqrt(np.mean(x ** 2))
        if best is None or crest < best[0]:
            best = (crest, phases, x)
    crest, phases, x = best
    # Start where both the signal and its slope are small (gentle onset).
    dx = (-np.sin(arg + phases) * freqs).sum(axis=1)
    score = np.abs(x) / np.max(np.abs(x)) + np.abs(dx) / np.max(np.abs(dx))
    shift = grid[int(np.argmin(score))]
    phases = phases + 2.0 * np.pi * freqs * shift
    amplitude = cfg.multisine_amplitude_fraction * test_torque / np.max(np.abs(x))
    return {
        'period_s': period,
        'periods': cfg.multisine_periods,
        'freqs_hz': freqs.tolist(),
        'amplitudes_nm': [float(amplitude)] * len(freqs),
        'phases_rad': np.mod(phases, 2 * np.pi).tolist(),
        'preload_nm': cfg.preload_fraction * test_torque,
        'peak_nm': cfg.multisine_amplitude_fraction * test_torque,
        'crest_factor': float(crest),
    }


def multisine_torque(design):
    freqs = 2.0 * np.pi * np.asarray(design['freqs_hz'])
    amps = np.asarray(design['amplitudes_nm'])
    phases = np.asarray(design['phases_rad'])
    preload = design['preload_nm']

    def torque(t):
        return preload + float(np.dot(amps, np.cos(freqs * t + phases)))

    return torque


def torque_ramp(start, end, duration, phase):
    def command(t, _):
        frac, _ = min_jerk(t / duration)
        return Command(kp_scale=0.0, kd_scale=0.0, torque_nm=start + (end - start) * frac)

    return Segment(phase, duration, command)


def build_segments(cfg, test_torque, design):
    torque_only = dict(kp_scale=0.0, kd_scale=0.0)
    period = cfg.cycle_period_s
    segments = [
        hold(1.0, phase='dwell', torque_nm=0.0, **torque_only),
        Segment('cycle', cfg.cycles * period,
                lambda t, _: Command(torque_nm=test_torque * triangle(t / period), **torque_only),
                info={'cycle_period_s': period, 'test_torque_nm': test_torque}),
        hold(0.5, phase='dwell', torque_nm=0.0, **torque_only),
    ]
    if design is not None:
        torque = multisine_torque(design)
        start = torque(0.0)
        segments += [
            torque_ramp(0.0, start, 1.0, 'preload'),
            Segment('multisine', design['periods'] * design['period_s'],
                    lambda t, _: Command(torque_nm=torque(t), **torque_only),
                    info={'multisine_period_s': design['period_s']}),
            torque_ramp(torque(design['periods'] * design['period_s']), 0.0, 1.0, 'unload'),
            hold(0.5, phase='dwell', torque_nm=0.0, **torque_only),
        ]
    return segments


def plan_checks(cfg):
    if not 0.0 < cfg.test_torque_fraction <= 0.9:
        raise SystemExit('test_torque_fraction must be in (0, 0.9].')
    test_torque = cfg.test_torque_fraction * device_config.MAX_JOINT_TORQUE_NM
    if cfg.dynamic:
        if cfg.multisine_amplitude_fraction >= cfg.preload_fraction:
            raise SystemExit('The multisine peak must stay below the preload so the cable '
                             'never unloads into backlash.')
        if cfg.preload_fraction + cfg.multisine_amplitude_fraction > 1.0:
            raise SystemExit('Preload + multisine exceed the cycle torque.')
        if cfg.freq_max_hz > cfg.sample_rate_hz / 2.5:
            print(f'NOTE: freq_max_hz limited to {cfg.sample_rate_hz / 2.5:.0f} Hz by the '
                  'sample rate.')
    k_guess = 1000.0   # N m/rad, only for the printed estimate
    count_deg = 360.0 / (4096 * device_config.GEAR_RATIO)   # one motor count at the joint
    if cfg.dynamic:
        amplitude = cfg.multisine_amplitude_fraction * test_torque
        motion_deg = math.degrees(amplitude / k_guess)
        if motion_deg < 5 * count_deg:
            print(f'WARNING: the multisine moves the joint ~{motion_deg:.4f} deg against a stiff '
                  f'cable, {motion_deg / count_deg:.1f} motor counts ({count_deg:.4f} deg each): '
                  'too little for the dynamic fit. Raise multisine_amplitude_fraction or '
                  'test with an output encoder.')
    lines = [f'  Cycle torque +/-{test_torque:.2f} N m; expected deflection about '
             f'{math.degrees(test_torque / k_guess):.2f} deg at {k_guess:.0f} N m/rad.',
             f'  The run stops if the joint moves more than {cfg.max_deflection_deg} deg '
             '(lock slipped).']
    return test_torque, lines


async def main(args):
    cfg = TransmissionConfig(**(QUICK if args.quick else {}))
    test_torque, lines = plan_checks(cfg)
    design = multisine_design(cfg, test_torque) if cfg.dynamic else None
    segments = build_segments(cfg, test_torque, design)
    limits = SafetyLimits(joint_min_deg=-cfg.max_deflection_deg,
                          joint_max_deg=cfg.max_deflection_deg,
                          max_joint_velocity_deg_s=cfg.max_speed_deg_s)
    async with WristSession(EXPERIMENT, args, 0.0, 0.0, config=asdict(cfg),
                             rate_hz=cfg.sample_rate_hz, sim_params=SIM_PARAMS,
                             limits=limits) as session:
        session.require_locked_output(
            'Clamp the output so it cannot rotate; the motor will push against the clamp.')
        confirm(args, ['Transmission stiffness test (output locked, torque control)',
                       *describe_plan(segments, session.max_torque_nm), *lines])
        session.metadata['test_torque_nm'] = test_torque
        session.metadata['cycle_period_s'] = cfg.cycle_period_s
        session.metadata['cycles'] = cfg.cycles
        session.metadata['multisine'] = design
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
