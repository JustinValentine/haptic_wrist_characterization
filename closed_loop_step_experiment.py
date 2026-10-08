"""Closed-loop step performance and dynamic capability under stiff PD control.

Steps of increasing size are commanded with the nominal (stiff) position
gains. Small steps characterize the linear closed loop (rise time, overshoot,
settling time, steady-state error). Large steps drive the torque into the cap,
so the joint accelerates at its limit; they measure the achievable
acceleration, speed and mechanical power, which is what "highly dynamic"
claims rest on.

Analyze with closed_loop_step_experiment_analysis.py.

Run:  python closed_loop_step_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (MoteusSession, confirm, describe_plan, hold, make_arg_parser,
                               min_jerk_move, run_cli)

EXPERIMENT = 'closed_loop_step'
SIM_PARAMS = None   # Tests may inject a moteus_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class ClosedLoopStepConfig:
    kp_joint_nm_rad: float = 450.0          # Same stiff gains as the bandwidth test
    kd_joint_nms_rad: float = 0.75
    step_sizes_deg: tuple = (1.0, 2.0, 5.0, 10.0, 20.0)   # 1 deg stays below the cap
    steps_per_size: int = 6                 # Alternating up/down steps per size
    center_deg: float = 0.0
    hold_s: float = 1.0                     # Per step; long enough to settle
    settle_s: float = 0.5                   # Before each block of steps
    reposition_speed_deg_s: float = 60.0
    sample_rate_hz: float = 1000.0


QUICK = dict(step_sizes_deg=(1.0, 5.0, 20.0), steps_per_size=2, hold_s=0.6, settle_s=0.3)


def build_segments(cfg):
    segments = [hold(cfg.settle_s, 0.0, phase='settle')]
    position = 0.0
    step_id = 0
    for size in cfg.step_sizes_deg:
        low = cfg.center_deg - size / 2.0
        high = cfg.center_deg + size / 2.0
        segments.append(min_jerk_move(position, low, cfg.reposition_speed_deg_s))
        segments.append(hold(cfg.settle_s, low, phase='settle'))
        position = low
        for _ in range(cfg.steps_per_size):
            target = high if position == low else low
            segments.append(hold(cfg.hold_s, target, phase='step', info={
                'step_id': step_id,
                'step_size_deg': size,
                'direction': 1 if target > position else -1,
                'prev_target_deg': position,
            }))
            position = target
            step_id += 1
    segments.append(min_jerk_move(position, 0.0, cfg.reposition_speed_deg_s))
    segments.append(hold(cfg.settle_s, 0.0, phase='settle'))
    return segments


def plan_lines(cfg, torque_cap_nm):
    j = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    accel = torque_cap_nm / j                       # rad/s^2 at the torque cap
    largest = max(cfg.step_sizes_deg)
    # A saturated PD loop brakes late, so it accelerates over nearly the whole
    # step: v ~ sqrt(2 a d) (back-EMF derating lowers this on real hardware).
    peak_speed = math.degrees(math.sqrt(2.0 * accel * math.radians(largest)))
    limit = device_config.MAX_JOINT_VELOCITY_DEG_S
    if peak_speed > 0.9 * limit:
        raise SystemExit(f'A {largest:g} deg step can reach ~{peak_speed:.0f} deg/s, too close to '
                         f'MAX_JOINT_VELOCITY_DEG_S ({limit:g}). Reduce the largest step.')
    reach = abs(cfg.center_deg) + 0.5 * largest + 0.6 * largest   # Allow 60% overshoot
    lo, hi = device_config.JOINT_LIMITS_DEG
    if reach > min(abs(lo), abs(hi)):
        raise SystemExit(f'Steps may reach {reach:.0f} deg, beyond JOINT_LIMITS_DEG {lo, hi}.')
    wn = math.sqrt(cfg.kp_joint_nm_rad / j)
    zeta = cfg.kd_joint_nms_rad / (2.0 * math.sqrt(cfg.kp_joint_nm_rad * j))
    linear = torque_cap_nm / cfg.kp_joint_nm_rad   # Initial torque is Kp * step
    return [f'  Nominal loop: {wn / (2 * math.pi):.1f} Hz, damping ratio {zeta:.2f} '
            f'(J = {j} kg m^2)',
            f'  Steps above {math.degrees(linear):.2f} deg saturate the '
            f'{torque_cap_nm:g} N m cap (intended: measures peak acceleration '
            f'~{accel:.0f} rad/s^2, speed up to ~{peak_speed:.0f} deg/s).']


async def main(args):
    cfg = ClosedLoopStepConfig(**(QUICK if args.quick else {}))
    segments = build_segments(cfg)
    lines = plan_lines(cfg, device_config.MAX_JOINT_TORQUE_NM)
    async with MoteusSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, cfg.kd_joint_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        confirm(args, ['Closed-loop step performance (stiff PD, torque cap engaged)',
                       f'  Kp = {cfg.kp_joint_nm_rad} N m/rad, Kd = {cfg.kd_joint_nms_rad} '
                       f'N m s/rad, steps {cfg.step_sizes_deg} deg x {cfg.steps_per_size}',
                       *describe_plan(segments, session.max_torque_nm), *lines,
                       '  Large steps move fast: keep hands and cables clear.'])
        session.metadata['torque_cap_nm'] = session.max_torque_nm
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
