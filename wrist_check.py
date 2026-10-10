"""Bench bring-up checks for the P4 wrist (run these before any experiment).

  status   Pucks up, loop running, drives DISABLED: prints joint angles, motor
           counts, statuswords and bus voltage. Move each joint by hand in its
           positive direction and check that its angle increases.
  nudge    Drives enabled with NO position gains: a small positive torque pulse
           on one joint at a time. Each joint must move positive; if one moves
           negative, the torque and position signs disagree and any PD loop
           would push it away (fix TORQUE_SIGN/POSITION_SIGN or J2MP first).
  hold     Holds the start pose with the HOLD gains until Ctrl-C.

Every command zeroes the wrist the way the experiments do (kept zero, the
single-turn encoders, or placement at home).

Run:  python wrist_check.py status|nudge|hold [--seconds S] [--sim]
"""
import argparse
import asyncio
import math
import sys

import device_config
from experiment_common import Command, WristSession

NUDGE_TORQUE_NM = 0.3
NUDGE_S = 0.05            # ~4 deg of travel on a ~0.005 kg m^2 joint
NUDGE_MIN_DEG = 0.5


def row_line(row):
    return (f"t {row['time']:7.2f}s  PS {row['q_ps_deg']:+8.2f}  FE {row['q_fe_deg']:+8.2f}  "
            f"UR {row['q_ur_deg']:+8.2f} deg  counts "
            + ' '.join(f"{row[f'counts_{n}']:+7d}" for n in device_config.NODES)
            + f"  bus {row['voltage_v']:.1f} V")


async def status(session, seconds):
    """Loop runs, drives disabled: watch the joints while moving them by hand."""
    session.wrist.disable(1.0)
    end = session.clock.now() + seconds
    next_print = 0.0
    while session.clock.now() < end:
        await session.clock.sleep(0.02)   # (disabled: no watchdog to feed)
        sample = session.wrist.sample()
        if session.clock.now() >= next_print:
            row = {'time': sample.time, 'voltage_v': sample.bus_voltage[0],
                   **{f'q_{n}_deg': math.degrees(sample.q[k]) for k, n in enumerate(('ps', 'fe', 'ur'))},
                   **{f'counts_{n}': sample.counts[i] for i, n in enumerate(device_config.NODES)}}
            print(row_line(row))
            next_print = session.clock.now() + 0.25


async def nudge(session):
    """Positive torque on each joint in turn: it must move positive."""
    problems = []
    for j, name in enumerate(device_config.JOINT_NAMES):
        session.j = j          # the session commands this joint; it holds the others
        session.start_q = list(session.wrist.sample().q)
        before = session.wrist.sample().q[j]
        await session.run_segment(_torque_segment(NUDGE_TORQUE_NM, NUDGE_S, name))
        moved = math.degrees(session.wrist.sample().q[j] - before)
        await session.brake(0.3, phase='brake', info={'nudged': name})
        verdict = ('OK' if moved > NUDGE_MIN_DEG else
                   'DID NOT MOVE (friction? raise NUDGE_TORQUE_NM)' if abs(moved) <= NUDGE_MIN_DEG
                   else 'MOVED NEGATIVE: torque and position signs disagree')
        print(f'{name}: +{NUDGE_TORQUE_NM} N m for {NUDGE_S} s moved it {moved:+.2f} deg  {verdict}')
        if moved < -NUDGE_MIN_DEG:
            problems.append(name)
    if problems:
        print('\nDo not run experiments yet. A joint that moves negative under positive torque '
              'makes every PD loop positive feedback. Check TORQUE_SIGN against POSITION_SIGN '
              f'for the motors of {", ".join(problems)} (device_config).')
        return 1
    print('\nTorque and position directions agree on every joint.')
    return 0


def _torque_segment(torque, duration, name):
    from experiment_common import Segment
    command = Command(position_deg=math.nan, torque_nm=torque, kp_scale=0.0, kd_scale=0.0)
    return Segment('nudge', duration, lambda t, s: command, {'nudged': name})


async def hold(session, seconds):
    end = session.clock.now() + seconds
    while session.clock.now() < end:
        result = await session.run_segment(_hold_segment(min(1.0, end - session.clock.now())))
        if result.rows:
            print(row_line(result.rows[-1]) + f"  tau {result.rows[-1]['control_joint_torque_nm']:+.3f} N m")


def _hold_segment(duration):
    from experiment_common import hold as hold_segment
    return hold_segment(duration, 0.0, phase='hold')


async def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('check', choices=('status', 'nudge', 'hold'))
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--out', help='where to save the log (default data/wrist_check/...)')
    parser.add_argument('--interface', default=device_config.CAN_INTERFACE)
    args = parser.parse_args(argv)
    name = device_config.JOINT
    kp = device_config.HOLD_KP_NM_RAD[name]
    kd = device_config.HOLD_KD_NMS_RAD[name]
    async with WristSession('wrist_check', args, kp, kd, max_torque_nm=1.0) as session:
        if args.check == 'status':
            await status(session, args.seconds)
            return 0
        if args.check == 'nudge':
            return await nudge(session)
        await hold(session, args.seconds)
        return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
