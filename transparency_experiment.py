"""Transparency (backdrivability): what the operator feels when moving the handle.

The operator moves the handle back and forth in prompted phases (slow,
medium, fast) while the motor renders either zero torque or a feedforward
friction compensation

    tau = gain * (fc_hat * tanh(omega / omega_eps) + b_hat * omega)   (capped)

computed from the last measured velocity. An external torque sensor between
the hand and the handle measures the interaction torque; without one, the
analysis falls back to a clearly labeled model-based estimate. The analysis
fits the apparent inertia, damping, and friction felt in each condition.

The safest place for this test is with the wrist mounted as in use, the
operator seated, and a low torque cap: the motor only ever adds compensation
torque in the direction the handle is already moving.

Run:  python transparency_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (Command, MoteusSession, SafetyLimits, Segment, confirm, hold,
                               latest_summary, make_arg_parser, run_cli)

EXPERIMENT = 'transparency'
SIM_PARAMS = None   # Tests inject a moteus_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class TransparencyConfig:
    conditions: tuple = ('zero_torque', 'compensated')
    speeds: tuple = (('slow', 0.3), ('medium', 1.0), ('fast', 2.0))   # name, target Hz
    trial_s: float = 12.0                 # Per condition x speed
    amplitude_deg: float = 30.0           # Guidance for the operator (+/-)
    compensation_gain: float = 0.8        # Fraction of identified friction to cancel
    compensation_cap_nm: float = 0.5      # Never add more than this
    coulomb_smoothing_deg_s: float = 5.0  # tanh width of the Coulomb term
    max_torque_nm: float = 1.0            # Session torque cap
    max_velocity_deg_s: float = 1000.0    # Abort above this speed
    countdown_s: float = 3.0              # Pause between phases with --yes
    rest_s: float = 1.0                   # Zero torque before each trial
    sample_rate_hz: float = 1000.0
    sim_human_stiffness: float = 40.0     # Simulated operator (N m/rad)
    sim_human_damping: float = 0.8        # Simulated operator (N m s/rad)


QUICK = dict(trial_s=6.0, rest_s=0.5)

FRICTION_SOURCES = [
    # (experiment, coulomb keys, viscous keys)
    ('friction_sweep', ('fc_nm', 'coulomb_nm', 'coulomb_friction_nm'),
     ('b_nms_rad', 'viscous_nms_rad', 'viscous_damping_nms_rad')),
    ('step_response', ('fc_nm',), ('b_nms_rad',)),
]


def first_key(summary, keys):
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def friction_estimates(simulated):
    """Coulomb/viscous/inertia estimates for compensation and model-based torque.

    Uses the newest analyzed run of the same kind (real or simulated) only.
    """
    estimates = {'fc_nm': device_config.NOMINAL_COULOMB_FRICTION_NM,
                 'b_nms_rad': device_config.NOMINAL_JOINT_DAMPING_NMS_RAD,
                 'J_kg_m2': device_config.NOMINAL_JOINT_INERTIA_KG_M2,
                 'source': 'device_config nominal values'}
    step = latest_summary('step_response', simulated)
    if step:
        J = first_key(step[1], ('J_kg_m2',))
        if J:
            estimates['J_kg_m2'] = J
            estimates['inertia_source'] = str(step[0])
    for experiment, fc_keys, b_keys in FRICTION_SOURCES:
        found = latest_summary(experiment, simulated)
        if not found:
            continue
        fc, b = first_key(found[1], fc_keys), first_key(found[1], b_keys)
        if fc is not None and b is not None:
            estimates.update(fc_nm=fc, b_nms_rad=b, source=str(found[0]))
            break
    return estimates


def compensation_torque(cfg, estimates, velocity_deg_s):
    omega = math.radians(velocity_deg_s)
    tau = cfg.compensation_gain * (
        estimates['fc_nm'] * math.tanh(velocity_deg_s / cfg.coulomb_smoothing_deg_s)
        + estimates['b_nms_rad'] * omega)
    return max(-cfg.compensation_cap_nm, min(cfg.compensation_cap_nm, tau))


ZERO = Command(position_deg=math.nan, kp_scale=0.0, kd_scale=0.0, torque_nm=0.0)


def trial_segment(cfg, estimates, condition, info):
    if condition == 'zero_torque':
        return Segment('trial', cfg.trial_s, lambda t, last: ZERO, info)
    if condition != 'compensated':
        raise ValueError(f'Unknown condition {condition!r}')

    def command(t, last):
        velocity = last['joint_vel_deg_s'] if last else 0.0
        if not math.isfinite(velocity):
            velocity = 0.0
        return Command(position_deg=math.nan, kp_scale=0.0, kd_scale=0.0,
                       torque_nm=compensation_torque(cfg, estimates, velocity))

    return Segment('trial', cfg.trial_s, command, info)


def sim_motion(t0, duration, amplitude_deg, freq_hz, ramp_s=1.0):
    """Sinusoid about 0 with raised-cosine ramp in/out (zero velocity at both ends)."""
    amp = math.radians(amplitude_deg)
    w = 2.0 * math.pi * freq_hz
    ramp = min(ramp_s, duration / 3.0)

    def envelope(tau):
        if tau <= 0.0 or tau >= duration:
            return 0.0, 0.0
        if tau < ramp:
            x = math.pi * tau / ramp
            return 0.5 * (1 - math.cos(x)), 0.5 * math.pi / ramp * math.sin(x)
        if tau > duration - ramp:
            x = math.pi * (duration - tau) / ramp
            return 0.5 * (1 - math.cos(x)), -0.5 * math.pi / ramp * math.sin(x)
        return 1.0, 0.0

    def motion(t):
        tau = t - t0
        e, de = envelope(tau)
        s, c = math.sin(w * tau), math.cos(w * tau)
        return amp * e * s, amp * (de * s + e * w * c)

    return motion


def plan_checks(cfg):
    lo, hi = device_config.JOINT_LIMITS_DEG
    if cfg.amplitude_deg >= min(abs(lo), abs(hi)):
        raise SystemExit(f'amplitude_deg {cfg.amplitude_deg} reaches JOINT_LIMITS_DEG {lo, hi}.')
    if cfg.compensation_cap_nm > cfg.max_torque_nm:
        raise SystemExit('compensation_cap_nm must not exceed max_torque_nm.')
    for condition in cfg.conditions:
        if condition not in ('zero_torque', 'compensated'):
            raise SystemExit(f'Unknown condition {condition!r}.')


async def announce(session, args, cfg, condition, speed, freq_hz):
    message = (f"[{condition}] {speed}: move the handle about +/-{cfg.amplitude_deg:.0f} deg at "
               f"~{freq_hz:g} Hz for {cfg.trial_s:.0f} s. ")
    info = {'condition': condition, 'speed': speed}
    if session.is_sim:
        plant = session.plant
        if plant.human is None:
            import moteus_sim
            plant.human = moteus_sim.HumanModel(motion=lambda t: (0.0, 0.0),
                                                stiffness=cfg.sim_human_stiffness,
                                                damping=cfg.sim_human_damping)
        plant.human.motion = lambda t: (0.0, 0.0)
        await session.run_segment(hold(cfg.rest_s, phase='rest', info=info,
                                       kp_scale=0.0, kd_scale=0.0))
        start = session.clock.now()
        plant.human.motion = sim_motion(start, cfg.trial_s, cfg.amplitude_deg, freq_hz)
        return
    if args.yes:
        print(message + f'Starting in {cfg.countdown_s:.0f} s.')
        await session.run_segment(hold(cfg.countdown_s, phase='wait', info=info,
                                       kp_scale=0.0, kd_scale=0.0))
        return
    # Keep commanding zero torque while waiting so the moteus watchdog stays fed.
    await session.prompt_while_holding(message + 'Press Enter to start... ', ZERO,
                                       phase='wait', info=info)


async def main(args):
    cfg = TransparencyConfig(**(QUICK if args.quick else {}))
    plan_checks(cfg)
    estimates = friction_estimates(args.sim)
    limits = SafetyLimits(max_joint_velocity_deg_s=cfg.max_velocity_deg_s)
    trials = [(c, s, f) for c in cfg.conditions for s, f in cfg.speeds]
    async with MoteusSession(EXPERIMENT, args, 0.0, 0.0, config=asdict(cfg),
                             rate_hz=cfg.sample_rate_hz, needs_sensor=True,
                             sim_params=SIM_PARAMS, limits=limits,
                             max_torque_nm=cfg.max_torque_nm) as session:
        continuous = session.sensor.continuous
        lines = [
            'Transparency: the operator moves the handle; the motor renders zero torque or '
            'friction compensation.',
            f"  {len(trials)} trials of {cfg.trial_s:.0f} s ({', '.join(cfg.conditions)} x "
            f"{', '.join(s for s, _ in cfg.speeds)})",
            f"  Compensation: gain {cfg.compensation_gain} x (fc {estimates['fc_nm']:.3g} N m, "
            f"b {estimates['b_nms_rad']:.3g} N m s/rad) from {estimates['source']}, "
            f'capped at {cfg.compensation_cap_nm} N m',
            f'  Torque cap {session.max_torque_nm:.2f} N m. Hold the handle the way a user would.',
        ]
        if not continuous:
            lines.append('  NOTE: no continuous torque sensor; the analysis will report a '
                         'MODEL-BASED estimate of the interaction torque.')
        confirm(args, lines)
        session.metadata.update({
            'protocol': 'operator-driven back-and-forth motion in timed phases',
            'compensation': {**estimates, 'gain': cfg.compensation_gain,
                             'cap_nm': cfg.compensation_cap_nm,
                             'coulomb_smoothing_deg_s': cfg.coulomb_smoothing_deg_s},
            'sensor_continuous': continuous,
            'trials': [],
        })
        for trial_id, (condition, speed, freq_hz) in enumerate(trials):
            await announce(session, args, cfg, condition, speed, freq_hz)
            info = {'trial_id': trial_id, 'condition': condition, 'speed': speed,
                    'target_hz': freq_hz}
            await session.run_segment(trial_segment(cfg, estimates, condition, info))
            session.metadata['trials'].append({**info, 'amplitude_deg': cfg.amplitude_deg,
                                               'duration_s': cfg.trial_s})
        if session.is_sim:
            session.plant.human.motion = lambda t: (0.0, 0.0)
        await session.run_segment(hold(0.5, phase='rest', kp_scale=0.0, kd_scale=0.0))


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
