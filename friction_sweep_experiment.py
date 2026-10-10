"""Constant-velocity sweeps: friction versus velocity and position.

A stiff PD controller with velocity feedforward drives the joint through
constant-velocity passes across a position window, in both directions, at
log-spaced speeds. During each cruise the inertial torque is zero, so the
torque the drives report is friction(v) + conservative(theta) (gravity,
cogging, cable preload). Pairing the forward and backward passes over the
same positions separates the two:

    friction(v)        = (tau_forward - tau_backward) / 2
    conservative(theta) = (tau_forward + tau_backward) / 2

friction_sweep_experiment_analysis.py fits a Stribeck + viscous model
(Coulomb, static, Stribeck velocity, viscous), identifies the motor's cogging
from the slowest passes, and reports the remaining torque ripple.

Run:  python friction_sweep_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (Command, WristSession, Segment, confirm, describe_plan, hold,
                               make_arg_parser, min_jerk_move, run_cli)

EXPERIMENT = 'friction_sweep'
SIM_PARAMS = None   # Tests can inject a p4_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class SweepConfig:
    # Tracking must be stiffer than the cogging's peak stiffness (amplitude x
    # 2 pi / period; ~14 N m/rad at FE for the simulated ILM50 motors); a softer
    # loop snaps from detent to detent at slow speeds and inflates low-speed
    # friction. The 500 Hz loop's delay caps stable stiffness near 80 N m/rad.
    kp_joint_nm_rad: float = 80.0
    kd_joint_nms_rad: float = 1.0
    speeds_deg_s: tuple = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
    window_deg: tuple = (-40.0, 40.0)    # Position range covered by each pass
    max_cruise_s: float = 15.0           # Caps the cruise length at slow speeds
    min_cruise_s: float = 0.2            # Shortest acceptable cruise (fast speeds)
    round_trips: int = 2                 # Forward + backward passes per speed
    accel_deg_s2: float = 2000.0         # Peak acceleration into and out of each cruise
    dwell_s: float = 0.3                 # Hold between passes
    reposition_speed_deg_s: float = 30.0
    sample_rate_hz: float = 500.0


QUICK = dict(speeds_deg_s=(1.0, 3.0, 10.0, 30.0, 100.0), window_deg=(-15.0, 15.0),
             max_cruise_s=6.0, round_trips=1)


def blend_time(speed, accel):
    """Duration of the sinusoidal speed blend whose peak acceleration is accel."""
    return math.pi * speed / (2.0 * accel)


def pass_geometry(cfg, speed):
    """Start/end of a pass at this speed, centered in the window."""
    lo, hi = cfg.window_deg
    blend_dist = 0.5 * speed * blend_time(speed, cfg.accel_deg_s2)
    cruise = min(hi - lo - 2.0 * blend_dist, speed * cfg.max_cruise_s)
    center = 0.5 * (lo + hi)
    half = 0.5 * cruise + blend_dist
    return center - half, center + half, cruise, blend_dist


def constant_velocity_pass(start, end, speed, accel, info):
    """Blend up to speed, cruise, blend down, as velocity-only commands.

    Position is sent as nan so the p4wrist loop integrates the setpoint itself
    at its control rate. Re-sending an absolute position every cycle would make the
    setpoint jump by speed x (latency jitter) on each command, which shows up
    as torque ripple correlated with the sample instants and biases the mean
    torque low at high speed. The dwell holds between passes re-anchor the
    absolute position. The speed blend is v (1 - cos)/2, so acceleration
    returns smoothly to zero at the start of the cruise.
    """
    direction = 1.0 if end > start else -1.0
    t_b = blend_time(speed, accel)
    x_b = 0.5 * speed * t_b
    cruise_dist = abs(end - start) - 2.0 * x_b
    t_cruise = cruise_dist / speed
    w = math.pi / t_b

    def accel_cmd(t, _):
        t = min(t, t_b)
        return Command(velocity_deg_s=direction * 0.5 * speed * (1.0 - math.cos(w * t)))

    def cruise_cmd(t, _):
        return Command(velocity_deg_s=direction * speed)

    def decel_cmd(t, _):
        t = min(t, t_b)
        return Command(velocity_deg_s=direction * 0.5 * speed * (1.0 + math.cos(w * t)))

    return [Segment('accel', t_b, accel_cmd, info),
            Segment('cruise', t_cruise, cruise_cmd, info),
            Segment('decel', t_b, decel_cmd, info)]


def build_segments(cfg):
    segments = [hold(1.0, 0.0, phase='settle')]
    position = 0.0
    pass_id = 0
    for speed_index, speed in enumerate(cfg.speeds_deg_s):
        start, end, _, _ = pass_geometry(cfg, speed)
        segments.append(min_jerk_move(position, start, cfg.reposition_speed_deg_s))
        segments.append(hold(cfg.dwell_s, start, phase='dwell'))
        for trip in range(cfg.round_trips):
            for a, b in ((start, end), (end, start)):
                info = {'pass_id': pass_id, 'speed_index': speed_index, 'speed_deg_s': speed,
                        'direction': 1 if b > a else -1, 'round_trip': trip}
                segments += constant_velocity_pass(a, b, speed, cfg.accel_deg_s2, info)
                segments.append(hold(cfg.dwell_s, b, phase='dwell'))
                pass_id += 1
        position = start
    segments.append(min_jerk_move(position, 0.0, cfg.reposition_speed_deg_s))
    segments.append(hold(0.5, 0.0, phase='settle'))
    return segments


def plan_warnings(cfg):
    lines = []
    lo, hi = cfg.window_deg
    limit_lo, limit_hi = device_config.JOINT_LIMITS_DEG
    if lo < limit_lo + 2.0 or hi > limit_hi - 2.0:
        raise SystemExit(f'window_deg {cfg.window_deg} is too close to JOINT_LIMITS_DEG '
                         f'{device_config.JOINT_LIMITS_DEG}.')
    top = max(cfg.speeds_deg_s)
    if top > 0.8 * device_config.MAX_JOINT_VELOCITY_DEG_S:
        raise SystemExit(f'Speed {top} deg/s is near MAX_JOINT_VELOCITY_DEG_S.')
    for speed in cfg.speeds_deg_s:
        _, _, cruise, _ = pass_geometry(cfg, speed)
        if cruise / speed < cfg.min_cruise_s:
            raise SystemExit(f'At {speed} deg/s the cruise lasts only {cruise / speed:.2f} s; '
                             'widen window_deg or raise accel_deg_s2.')
    j = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    accel_torque = j * math.radians(cfg.accel_deg_s2)
    lines.append(f'  Speeds {cfg.speeds_deg_s} deg/s over {cfg.window_deg} deg, '
                 f'{cfg.round_trips} round trip(s) each.')
    lines.append(f'  Acceleration torque ~{accel_torque:.2f} N m (nominal J = {j} kg m^2).')
    slow = min(cfg.speeds_deg_s)
    _, _, cruise, _ = pass_geometry(cfg, slow)
    lines.append(f'  Slowest pass covers {cruise:.1f} deg (cruise capped at {cfg.max_cruise_s} s).')
    return lines


async def main(args):
    cfg = SweepConfig(**(QUICK if args.quick else {}))
    segments = build_segments(cfg)
    warnings = plan_warnings(cfg)
    async with WristSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, cfg.kd_joint_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        confirm(args, ['Constant-velocity friction sweep (stiff PD + velocity feedforward)',
                       f'  Kp = {cfg.kp_joint_nm_rad} N m/rad, Kd = {cfg.kd_joint_nms_rad} '
                       'N m s/rad',
                       *describe_plan(segments, session.max_torque_nm), *warnings])
        session.metadata['protocol'] = {
            'passes': [
                dict(zip(('start_deg', 'end_deg', 'cruise_deg', 'accel_deg'),
                         pass_geometry(cfg, speed)), speed_deg_s=speed)
                for speed in cfg.speeds_deg_s],
        }
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
