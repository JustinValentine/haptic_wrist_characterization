"""Ramp-and-hold measurement of static (breakaway) friction.

A soft P-only spring (Kd = 0) drags the joint: while the joint is stuck, the
spring torque Kp (th_cmd - th) rises at Kp * ramp rate, and the joint breaks
away once that torque exceeds static friction. Each ramp is followed by a
pause so the joint comes to rest before the next ramp. The staircase covers
0 -> +limit -> 0 -> -limit -> 0, so breakaway is measured in both directions
across the workspace. ramp_input_experiment_analysis.py detects breakaway from
displacement (robust to encoder noise) and reports the spring torque at onset.

Run:  python ramp_input_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (Command, WristSession, Segment, confirm, describe_plan, hold,
                               make_arg_parser, run_cli)

EXPERIMENT = 'friction_ramp'
SIM_PARAMS = None   # Tests can inject a p4_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class RampConfig:
    kp_joint_nm_rad: float = 3.0         # Very soft spring: torque rises slowly
    ramp_step_deg: float = 5.0           # Command increment per ramp
    ramp_rate_deg_s: float = 2.5         # Command speed during a ramp
    pause_s: float = 2.0                 # Rest between ramps
    workspace_limit_deg: float = 45.0    # Staircase spans +/- this
    cycles: int = 1                      # Repeats of the full staircase
    settle_s: float = 2.0
    sample_rate_hz: float = 500.0


QUICK = dict(workspace_limit_deg=15.0, cycles=1, settle_s=1.0)


def build_ramp_targets(step_deg, workspace_limit_deg):
    """Command levels for 0 -> +limit -> 0 -> -limit -> 0 in step increments."""
    n = int(round(workspace_limit_deg / step_deg))
    if n < 1 or abs(n * step_deg - workspace_limit_deg) > 1e-9:
        raise SystemExit('workspace_limit_deg must be a positive multiple of ramp_step_deg.')
    up = [k * step_deg for k in range(1, n + 1)]
    down = [k * step_deg for k in range(n - 1, -1, -1)]
    return up + down + [-v for v in up] + [-v for v in down]


def ramp_segment(start_deg, end_deg, rate_deg_s, info):
    duration = abs(end_deg - start_deg) / rate_deg_s
    velocity = math.copysign(rate_deg_s, end_deg - start_deg)

    def command(t, _):
        frac = min(t / duration, 1.0)
        return Command(position_deg=start_deg + (end_deg - start_deg) * frac,
                       velocity_deg_s=velocity if frac < 1.0 else 0.0, kd_scale=0.0)

    return Segment('ramp', duration, command, info)


def build_segments(cfg):
    targets = build_ramp_targets(cfg.ramp_step_deg, cfg.workspace_limit_deg)
    segments = [hold(cfg.settle_s, 0.0, phase='settle', kd_scale=0.0)]
    position = 0.0
    ramp_id = 0
    for cycle in range(cfg.cycles):
        for target in targets:
            info = {
                'ramp_id': ramp_id,
                'cycle': cycle,
                'direction': 1 if target > position else -1,
                'ramp_start_deg': position,
                'ramp_end_deg': target,
            }
            segments.append(ramp_segment(position, target, cfg.ramp_rate_deg_s, info))
            segments.append(hold(cfg.pause_s, target, phase='pause', info=info, kd_scale=0.0))
            position = target
            ramp_id += 1
    return segments


def plan_warnings(cfg):
    lines = []
    lo, hi = device_config.JOINT_LIMITS_DEG
    # The joint lags the command by up to static friction / Kp and can slip past it.
    lag_deg = math.degrees(3.0 * device_config.NOMINAL_COULOMB_FRICTION_NM / cfg.kp_joint_nm_rad)
    reach = cfg.workspace_limit_deg + lag_deg + 2.0
    if reach > min(abs(lo), abs(hi)):
        raise SystemExit(f'The staircase may reach {reach:.0f} deg, beyond JOINT_LIMITS_DEG {lo, hi}.')
    expected_static = 1.4 * device_config.NOMINAL_COULOMB_FRICTION_NM
    torque_per_ramp = cfg.kp_joint_nm_rad * math.radians(cfg.ramp_step_deg)
    if torque_per_ramp < 2.0 * expected_static:
        lines.append(f'  WARNING: one ramp raises the spring torque by only {torque_per_ramp:.3f} N m '
                     f'(< 2 x expected static friction {expected_static:.3f} N m); breakaway may '
                     'need several ramps. Increase kp_joint_nm_rad or ramp_step_deg.')
    rate = cfg.kp_joint_nm_rad * math.radians(cfg.ramp_rate_deg_s)
    lines.append(f'  Spring torque rises at {rate:.3f} N m/s during ramps '
                 f'(Kp = {cfg.kp_joint_nm_rad} N m/rad, {cfg.ramp_rate_deg_s} deg/s).')
    return lines


async def main(args):
    cfg = RampConfig(**(QUICK if args.quick else {}))
    segments = build_segments(cfg)
    warnings = plan_warnings(cfg)
    async with WristSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, 0.0, config=asdict(cfg),
                             rate_hz=cfg.sample_rate_hz, sim_params=SIM_PARAMS) as session:
        confirm(args, ['Static friction ramp-and-hold (P-only, Kd = 0)',
                       f'  +/-{cfg.workspace_limit_deg} deg in {cfg.ramp_step_deg} deg ramps, '
                       f'{cfg.cycles} cycle(s)',
                       *describe_plan(segments, session.max_torque_nm), *warnings])
        session.metadata['protocol'] = {
            'targets_deg': build_ramp_targets(cfg.ramp_step_deg, cfg.workspace_limit_deg),
            'ramp_duration_s': cfg.ramp_step_deg / cfg.ramp_rate_deg_s,
            'spring_torque_rate_nm_s': cfg.kp_joint_nm_rad * math.radians(cfg.ramp_rate_deg_s),
        }
        await session.run(segments)


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
