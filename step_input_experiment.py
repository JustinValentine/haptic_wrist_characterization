"""Step-input identification of joint inertia, viscous damping, and Coulomb friction.

With a pure P controller (Kd = 0) the joint behaves as a spring-mass system
with friction:

    J th'' + b th' + fc sign(th') = Kp (th_cmd - th)

Each step makes the joint ring down. step_input_experiment_analysis.py
identifies J, b, and fc from the sequence of turning points. Several step
sizes are used because Coulomb friction makes the decay amplitude-dependent,
which is what separates it from viscous damping.

Run:  python step_input_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (MoteusSession, confirm, describe_plan, hold, make_arg_parser,
                               min_jerk_move, run_cli)

EXPERIMENT = 'step_response'


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class StepConfig:
    kp_joint_nm_rad: float = 15.0           # Soft P-only spring (Kd = 0)
    step_sizes_deg: tuple = (5.0, 10.0, 20.0)
    steps_per_size: int = 6                 # Alternating up/down steps per size
    center_deg: float = 0.0                 # Steps are centered here
    hold_s: float = 3.0                     # Per step; must cover the full ring-down
    settle_s: float = 2.0                   # Before each block of steps
    reposition_speed_deg_s: float = 20.0
    sample_rate_hz: float = 1000.0


QUICK = dict(step_sizes_deg=(10.0, 20.0), steps_per_size=4, hold_s=2.0, settle_s=1.0)


def build_segments(cfg):
    p_only = dict(kd_scale=0.0)
    segments = [hold(cfg.settle_s, 0.0, phase='settle', **p_only)]
    position = 0.0
    step_id = 0
    for size in cfg.step_sizes_deg:
        low = cfg.center_deg - size / 2.0
        high = cfg.center_deg + size / 2.0
        segments.append(min_jerk_move(position, low, cfg.reposition_speed_deg_s, **p_only))
        segments.append(hold(cfg.settle_s, low, phase='settle', **p_only))
        position = low
        for _ in range(cfg.steps_per_size):
            target = high if position == low else low
            segments.append(hold(cfg.hold_s, target, phase='step', info={
                'step_id': step_id,
                'step_size_deg': size,
                'direction': 1 if target > position else -1,
                'prev_target_deg': position,
            }, **p_only))
            position = target
            step_id += 1
    segments.append(min_jerk_move(position, 0.0, cfg.reposition_speed_deg_s, **p_only))
    segments.append(hold(1.0, 0.0, phase='settle', **p_only))
    return segments


def plan_warnings(cfg):
    lines = []
    largest = max(cfg.step_sizes_deg)
    peak_torque = cfg.kp_joint_nm_rad * math.radians(largest)
    if peak_torque > 0.9 * device_config.MAX_JOINT_TORQUE_NM:
        raise SystemExit(f'Kp x largest step = {peak_torque:.2f} N m saturates the torque cap; '
                         'reduce kp_joint_nm_rad or the step size.')
    # A lightly damped spring can overshoot by nearly a full step.
    reach = abs(cfg.center_deg) + 1.5 * largest
    lo, hi = device_config.JOINT_LIMITS_DEG
    if reach > min(abs(lo), abs(hi)):
        raise SystemExit(f'Steps may reach {reach:.0f} deg, beyond JOINT_LIMITS_DEG {lo, hi}.')
    j = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    fn = math.sqrt(cfg.kp_joint_nm_rad / j) / (2 * math.pi)
    lines.append(f'  Peak spring torque {peak_torque:.2f} N m; expected ringing ~{fn:.1f} Hz '
                 f'(nominal J = {j} kg m^2).')
    lines.append('  The joint is intentionally underdamped and will oscillate.')
    return lines


async def main(args):
    cfg = StepConfig(**(QUICK if args.quick else {}))
    segments = build_segments(cfg)
    warnings = plan_warnings(cfg)
    async with MoteusSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, 0.0,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz) as session:
        confirm(args, ['Step-response identification (P-only, Kd = 0)',
                       f'  Kp = {cfg.kp_joint_nm_rad} N m/rad, steps {cfg.step_sizes_deg} deg '
                       f'x {cfg.steps_per_size}',
                       *describe_plan(segments, session.max_torque_nm), *warnings])
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
