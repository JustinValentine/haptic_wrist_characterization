"""Torque calibration: commanded joint torque vs an external torque sensor.

The output is locked against a load cell (at a lever arm) or through an F/T
sensor, and the motor is torque-controlled through a staircase that sweeps
up to +tau_test, down to -tau_test, and back to zero. Each level is held, and
the sensor reading is averaged (continuous sensor) or typed in by the
operator (manual force gauge). torque_calibration_experiment_analysis.py fits

    measured = k * commanded + offset

where k (actual / commanded torque) corrects every torque-derived quantity
(J, b, fc, stiffness) identified from drive-reported torques, and reports
nonlinearity and hysteresis. Approaching each level from both sides exposes
transmission friction as hysteresis.

Run:  python torque_calibration_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import device_config
from experiment_common import (Command, WristSession, SafetyAbort, SafetyLimits, Segment,
                               confirm, describe_plan, make_arg_parser, min_jerk, run_cli)

EXPERIMENT = 'torque_calibration'
SIM_PARAMS = None   # Tests inject a p4_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class TorqueCalibrationConfig:
    test_torque_nm: float = 0.7 * device_config.MAX_JOINT_TORQUE_NM   # Largest |command|
    levels_per_side: int = 5             # Nonzero levels per sign, continuous sensor
    manual_levels_per_side: int = 2      # With a hand-read gauge: 9 readings per loop
    cycles: int = 1                      # Full up/down loops
    transition_s: float = 0.5            # Smooth torque change between levels
    settle_s: float = 1.0                # Let the transmission settle before averaging
    average_s: float = 2.0               # Averaging window (continuous sensor)
    lock_check_torque_nm: float = 0.5    # Small torque used to verify the lock
    lock_check_max_deg: float = 1.0      # Allowed deflection during the lock check
    max_deflection_deg: float = 3.0      # Abort if the motor side moves further (lock slipped)
    max_slip_velocity_deg_s: float = 60.0
    manual_readings_are_magnitudes: bool = True   # Apply the commanded sign to typed values
    sample_rate_hz: float = 500.0


QUICK = dict(levels_per_side=3, settle_s=0.5, average_s=1.0, transition_s=0.3)


def torque_levels(cfg, manual):
    """Visit order: 0 -> +max -> -max -> 0, so each level is approached from both sides.

    The first zero ('start') is read before any torque is applied.
    """
    n = cfg.manual_levels_per_side if manual else cfg.levels_per_side
    step = cfg.test_torque_nm / n
    up_pos = [k * step for k in range(1, n + 1)]
    down = [k * step for k in range(n - 1, -n - 1, -1)]
    up_neg = [k * step for k in range(-n + 1, 1)]
    visits = [0.0]
    for _ in range(cfg.cycles):
        visits += up_pos + down + up_neg
    levels = []
    for i, level in enumerate(visits):
        previous = visits[i - 1] if i else None
        if previous is None:
            branch = 'start'
        else:
            branch = 'up' if level > previous else 'down'
        levels.append({'level_id': i, 'level_nm': round(level, 9), 'branch': branch,
                       'previous_nm': previous if previous is not None else 0.0})
    return levels


def torque_command(torque_nm):
    return Command(position_deg=math.nan, kp_scale=0.0, kd_scale=0.0, torque_nm=torque_nm)


def transition(start_nm, end_nm, duration, info):
    def command(t, _):
        frac, _ = min_jerk(t / duration)
        return torque_command(start_nm + (end_nm - start_nm) * frac)
    return Segment('transition', duration, command, info)


def constant(phase, duration, torque_nm, info):
    cmd = torque_command(torque_nm)
    return Segment(phase, duration, lambda t, _: cmd, info)


def level_segments(cfg, level, include_measure):
    info = {'level_id': level['level_id'], 'level_nm': level['level_nm'],
            'branch': level['branch']}
    segments = [transition(level['previous_nm'], level['level_nm'], cfg.transition_s, info),
                constant('settle', cfg.settle_s, level['level_nm'], info)]
    if include_measure:
        segments.append(constant('measure', cfg.average_s, level['level_nm'], info))
    return segments


def lock_check_segment(cfg):
    check_nm = min(cfg.lock_check_torque_nm, cfg.test_torque_nm)

    def command(t, _):
        frac, _ = min_jerk(t / 0.5)
        return torque_command(check_nm * frac)

    def too_far(t, row):
        if abs(row['joint_pos_deg']) > cfg.lock_check_max_deg:
            return f"deflected {row['joint_pos_deg']:.2f} deg under {check_nm:.2f} N m"
        return None

    return Segment('lock_check', 1.0, command, {'level_id': -1}, until=too_far)


def plan_checks(cfg):
    if cfg.test_torque_nm > device_config.MAX_JOINT_TORQUE_NM:
        raise SystemExit(f'test_torque_nm {cfg.test_torque_nm} exceeds MAX_JOINT_TORQUE_NM '
                         f'{device_config.MAX_JOINT_TORQUE_NM}.')
    if cfg.levels_per_side < 1 or cfg.manual_levels_per_side < 1:
        raise SystemExit('Need at least one level per side.')


def sensor_note(sensor_config):
    kind = (sensor_config or {}).get('type')
    if kind == 'manual':
        arm = sensor_config.get('lever_arm_m')
        where = f'at the lever arm of {arm} m set in device_config' if arm else 'on the axis'
        return (f'Block the handle with the load cell / force gauge {where}. The motor pushes '
                'both ways: the sensor must measure both directions (for a one-way gauge, '
                're-mount it between the positive and negative halves).')
    if kind == 'ati_netft':
        return ('Clamp the output to the frame through the F/T sensor with the configured '
                'axis aligned to the wrist axis; bias the sensor with no load first.')
    return 'Clamp the output to the frame through the external torque sensor.'


async def manual_reading(session, cfg, level, total):
    sensor = session.sensor
    message = (f"Level {level['level_id'] + 1}/{total}: commanded {level['level_nm']:+.3f} N m. "
               f'Enter the reading ({sensor.prompt_units()}): ')
    info = {'level_id': level['level_id'], 'level_nm': level['level_nm'],
            'branch': level['branch']}
    while True:
        text = await session.prompt_while_holding(message, torque_command(level['level_nm']),
                                                  phase='measure', info=info)
        try:
            value = sensor.to_torque(text)
            break
        except ValueError:
            print('Not a number; try again.')
    if cfg.manual_readings_are_magnitudes and level['level_nm'] != 0.0:
        value = math.copysign(abs(value), level['level_nm'])
    return {'level_id': level['level_id'], 'level_nm': level['level_nm'],
            'branch': level['branch'], 'reading_text': text.strip(), 'torque_nm': value}


async def main(args):
    cfg = TorqueCalibrationConfig(**(QUICK if args.quick else {}))
    plan_checks(cfg)
    sensor_config = device_config.EXTERNAL_TORQUE_SENSOR
    if not args.sim and not sensor_config:
        raise SystemExit('Torque calibration needs an external sensor: set '
                         'EXTERNAL_TORQUE_SENSOR in device_config.py.')
    manual_planned = (not args.sim) and sensor_config.get('type') == 'manual'
    levels = torque_levels(cfg, manual_planned)
    planned = []
    for i, level in enumerate(levels):
        planned += level_segments(cfg, level, include_measure=True)
        if i == 0:
            planned.append(lock_check_segment(cfg))
    limits = SafetyLimits(joint_min_deg=-cfg.max_deflection_deg,
                          joint_max_deg=cfg.max_deflection_deg,
                          max_joint_velocity_deg_s=cfg.max_slip_velocity_deg_s)

    async with WristSession(EXPERIMENT, args, 0.0, 0.0, config=asdict(cfg),
                             rate_hz=cfg.sample_rate_hz, needs_sensor=True,
                             sim_params=SIM_PARAMS, limits=limits,
                             max_torque_nm=cfg.test_torque_nm * 1.05) as session:
        manual = not session.sensor.continuous
        if manual and not hasattr(session.sensor, 'to_torque'):
            raise SystemExit('The configured sensor provides no readings.')
        lines = ['Torque calibration (output locked against the external sensor)',
                 f'  Sensor: {session.sensor.describe()}',
                 f"  {len(levels)} torque levels up to +/-{cfg.test_torque_nm:.2f} N m"
                 + (' (operator types each reading)' if manual else ''),
                 *describe_plan(planned, session.max_torque_nm)]
        if manual:
            lines.append('  Duration depends on how quickly readings are entered.')
        confirm(args, lines)
        session.require_locked_output(sensor_note(sensor_config))
        session.metadata.update({
            'protocol': 'torque staircase 0 -> +max -> -max -> 0 with locked output',
            'levels': levels,
            'sensor_mode': 'manual' if manual else 'continuous',
            'manual_readings': [],
        })

        for i, level in enumerate(levels):
            await session.run(level_segments(cfg, level, include_measure=not manual))
            if manual:
                reading = await manual_reading(session, cfg, level, len(levels))
                session.metadata['manual_readings'].append(reading)
            if i == 0:
                # Zero-torque reading taken; now verify the lock before loading it.
                result = await session.run_segment(lock_check_segment(cfg))
                if result.ended_by:
                    raise SafetyAbort(f'Output does not appear locked: {result.ended_by}.')
                await session.run_segment(transition(
                    min(cfg.lock_check_torque_nm, cfg.test_torque_nm), 0.0, cfg.transition_s,
                    {'level_id': -1}))
        await session.run([transition(levels[-1]['level_nm'], 0.0, cfg.transition_s,
                                      {'level_id': -1}),
                           constant('settle', 0.5, 0.0, {'level_id': -1})])
    print('Done. Remove the clamp before running free-motion experiments.')


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
