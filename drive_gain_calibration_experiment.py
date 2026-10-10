"""Drive gain calibration: what the P4's own position/velocity loop gains render.

The P4's onboard loops (CSP: position P -> velocity PI -> torque) take raw gains
whose units are not documented. This experiment runs the drive loop with known
raw gains, velocity feedforward and integral off, so each drive's torque demand is

    demand = k_unit * kv * g_p * (target - position) - b_unit * kv * velocity
             (+ torque offset)                        g_p = kp * kout / 100

with demand in per mille of rated torque, positions in counts and velocity in
counts/s. The joint follows slow target sinusoids (handle free), and
drive_gain_calibration_experiment_analysis.py fits k_unit and b_unit from each
drive's own torque demand, position and target: the drive's control law, not the
plant, so friction and inertia do not enter. A torque-offset step checks whether
0x60B2 is added in CSP (feedforward in the drive loop depends on it).

Two gain sets (the drive's own defaults, then twice the position gain) check
that the law is linear in the gains. Because the units are firmware constants,
one calibration holds for every puck; WristSession.use_drive_loop converts joint
stiffness and damping into raw gains with it.

Run:  python drive_gain_calibration_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (Command, Segment, WristSession, confirm, hold, make_arg_parser,
                               run_cli)

EXPERIMENT = 'drive_gain_calibration'
SIM_PARAMS = None   # Tests may inject a p4_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class DriveCalibrationConfig:
    # Raw gain sets as multiples of each drive's own gains at connect (kp, kv).
    gain_scales: tuple = ((1.0, 1.0), (2.0, 1.0))
    frequencies_hz: tuple = (0.5, 1.0, 2.0)   # Target sinusoids, slow so velocities agree
    amplitude_deg: float = 6.0                # Joint amplitude of the target motion
    cycles: int = 3                           # Per frequency
    offset_nm: float = 0.1                    # Joint torque-offset step (0x60B2 check)
    offset_s: float = 0.6
    settle_s: float = 0.5
    sample_rate_hz: float = 500.0


QUICK = dict(gain_scales=((1.0, 1.0), (2.0, 1.0)), frequencies_hz=(1.0, 2.0), cycles=2,
             offset_s=0.4, settle_s=0.3)


def sinusoid_segment(f, amplitude, cycles, info):
    w = 2.0 * math.pi * f

    def command(t, _):
        return Command(position_deg=amplitude * math.sin(w * t),
                       velocity_deg_s=amplitude * w * math.cos(w * t))

    return Segment('sinusoid', cycles / f, command, {**info, 'frequency_hz': f})


async def main(args):
    cfg = DriveCalibrationConfig(**(QUICK if args.quick else {}))
    lo, hi = device_config.JOINT_LIMITS_DEG
    if cfg.amplitude_deg >= min(abs(lo), abs(hi)):
        raise SystemExit('amplitude_deg reaches JOINT_LIMITS_DEG.')
    hold_kp = device_config.HOLD_KP_NM_RAD[device_config.JOINT]
    hold_kd = device_config.HOLD_KD_NMS_RAD[device_config.JOINT]
    async with WristSession(EXPERIMENT, args, hold_kp, hold_kd, config=asdict(cfg),
                            rate_hz=cfg.sample_rate_hz, sim_params=SIM_PARAMS) as session:
        defaults = [d['gains'] for d in session.metadata['drive_info']]
        confirm(args, [
            "Drive gain calibration: the pucks' own position loops move the joint (handle free)",
            f"  {len(cfg.gain_scales)} gain sets x {len(cfg.frequencies_hz)} frequencies, "
            f"+/-{cfg.amplitude_deg} deg, then a {cfg.offset_nm} N m torque-offset step",
            '  Drive gains at connect: ' + ', '.join(
                f"kp {g['kp']} kv {g['kv']:.3g} ki {g['ki']:.3g}" for g in defaults),
            '  Velocity feedforward and integral are off during the calibration.',
        ])
        if any(g['kp'] <= 0 or g['kv'] <= 0 for g in defaults):
            raise SystemExit('A drive reported zero position or velocity gain; set its gains '
                             'with PuckTuner first.')
        session.metadata['gain_sets'] = []
        for set_id, (kp_scale, kv_scale) in enumerate(cfg.gain_scales):
            gains = [{'kp': int(round(g['kp'] * kp_scale)), 'kv': g['kv'] * kv_scale, 'ki': 0.0,
                      'kvff': 0, 'kaff': 0, 'kout': g['kout'] or 100} for g in defaults]
            session.metadata['gain_sets'].append({'gain_set': set_id, 'gains': gains})
            info = {'gain_set': set_id}
            await session.use_drive_loop_raw(gains)
            await session.run_segment(hold(cfg.settle_s, 0.0, phase='settle', info=info))
            for f in cfg.frequencies_hz:
                await session.run_segment(sinusoid_segment(f, cfg.amplitude_deg, cfg.cycles, info))
            await session.run_segment(hold(cfg.settle_s, 0.0, phase='settle', info=info))
            await session.run_segment(hold(cfg.offset_s, 0.0, phase='offset', info=info,
                                           torque_nm=cfg.offset_nm))
            await session.run_segment(hold(cfg.settle_s, 0.0, phase='settle', info=info))
        await session.use_pc_loop()
        await session.run_segment(hold(cfg.settle_s, 0.0, phase='settle'))


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
