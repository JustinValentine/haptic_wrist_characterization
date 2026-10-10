"""Continuous torque rating from winding temperature rise (output locked).

The output is clamped and the motor holds constant torque levels while the
winding temperature is logged. Copper loss scales with torque squared, so the
steady-state temperature rise does too; thermal_experiment_analysis.py fits a
first-order thermal response to each level, then dT_ss = c * torque^2, and
reports the torque that would settle exactly at the temperature limit (the
continuous torque rating) alongside the peak torque cap.

Between levels the motor cools at zero torque. Each level stops early when
the temperature rise or the absolute temperature approaches its limit.

Uses the motor thermistor (P4 0x3010:03, converted by device_config.MOTOR_THERMISTOR).
Without it the P4 amplifier temperature is used instead, which rates the
driver, not the motor.

Run:  python thermal_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

import device_config
from experiment_common import (Command, WristSession, SafetyLimits, Segment, confirm,
                               describe_plan, hold, make_arg_parser, run_cli)

EXPERIMENT = 'thermal'
SIM_PARAMS = None   # Tests inject a p4_sim.SimParams here.


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class ThermalConfig:
    torque_fractions: tuple = (0.3, 0.5, 0.7)   # Of MAX_JOINT_TORQUE_NM
    hold_s: float = 600.0              # Longest time per level (aim for >= 3 time constants)
    max_rise_c: float = 40.0           # End a level after this much winding rise
    temperature_margin_c: float = 10.0 # End a level this far below the temperature limit
    baseline_s: float = 10.0           # Zero-torque baseline to measure the start temperature
    cool_s: float = 600.0              # Longest zero-torque cool-down between levels
    cool_within_c: float = 3.0         # End the cool-down this close to the start temperature
    alternate_sign: bool = True        # Alternate torque direction between levels
    max_deflection_deg: float = 3.0    # Motor-side joint angle; more means the lock slipped
    sample_rate_hz: float = 25.0       # Keep well above 1/WATCHDOG_TIMEOUT_S


QUICK = dict(hold_s=60.0, cool_s=60.0, baseline_s=3.0)


def torque_levels(cfg):
    cap = device_config.MAX_JOINT_TORQUE_NM
    levels = []
    for i, fraction in enumerate(cfg.torque_fractions):
        sign = -1.0 if (cfg.alternate_sign and i % 2) else 1.0
        levels.append(sign * fraction * cap)
    return levels


def choose_channel(rows):
    """Prefer the winding thermistor; fall back to the board sensor."""
    motor = np.array([r.get('motor_temp_c', math.nan) for r in rows], dtype=float)
    board = np.array([r.get('board_temp_c', math.nan) for r in rows], dtype=float)
    ok = np.isfinite(motor) & (motor > -20.0) & (motor < 150.0) & (motor != 0.0)
    if ok.mean() > 0.9:
        return 'motor_temp_c', float(np.nanmedian(motor)), device_config.MAX_MOTOR_TEMPERATURE_C
    print('\n' + '!' * 72)
    print('WARNING: no usable motor temperature (set device_config.MOTOR_THERMISTOR). Falling')
    print('back to the P4 AMPLIFIER temperature. This rates the driver, NOT the motor winding.')
    print('!' * 72 + '\n')
    return 'board_temp_c', float(np.nanmedian(board)), device_config.MAX_BOARD_TEMPERATURE_C


def heat_segment(level, torque, cfg, channel, limit_c):
    state = {'start': None}

    def until(_, row):
        temp = row.get(channel, math.nan)
        if not math.isfinite(temp):
            return None
        if state['start'] is None:
            state['start'] = temp
        if temp - state['start'] >= cfg.max_rise_c:
            return 'rise_limit'
        if temp >= limit_c - cfg.temperature_margin_c:
            return 'temperature_limit'
        return None

    command = Command(kp_scale=0.0, kd_scale=0.0, torque_nm=torque)
    return Segment('heat', cfg.hold_s, lambda t, _: command,
                   info={'level': level, 'level_torque_nm': torque}, until=until)


def cool_segment(level, cfg, channel, start_c):
    def until(_, row):
        temp = row.get(channel, math.nan)
        return 'cooled' if math.isfinite(temp) and temp <= start_c + cfg.cool_within_c else None

    command = Command(kp_scale=0.0, kd_scale=0.0, torque_nm=0.0)
    return Segment('cool', cfg.cool_s, lambda t, _: command, info={'level': level}, until=until)


def plan_checks(cfg):
    if not cfg.torque_fractions or max(cfg.torque_fractions) > 1.0 or min(cfg.torque_fractions) <= 0:
        raise SystemExit('torque_fractions must be in (0, 1].')
    if cfg.sample_rate_hz < 2.5 / device_config.WATCHDOG_TIMEOUT_S:
        raise SystemExit('sample_rate_hz is too low for the p4wrist watchdog '
                         f'({device_config.WATCHDOG_TIMEOUT_S} s).')
    levels = torque_levels(cfg)
    worst = cfg.baseline_s + len(levels) * cfg.hold_s + (len(levels) - 1) * cfg.cool_s
    return levels, [
        '  Levels: ' + ', '.join(f'{t:+.2f} N m' for t in levels),
        f'  Up to {worst / 60:.0f} min; each level stops after {cfg.max_rise_c:.0f} C of rise '
        f'or {cfg.temperature_margin_c:.0f} C below the limit.',
    ]


async def main(args):
    cfg = ThermalConfig(**(QUICK if args.quick else {}))
    levels, lines = plan_checks(cfg)
    torque_only = dict(kp_scale=0.0, kd_scale=0.0)
    limits = SafetyLimits(joint_min_deg=-cfg.max_deflection_deg,
                          joint_max_deg=cfg.max_deflection_deg)
    plan = [hold(cfg.baseline_s, phase='baseline', **torque_only)]
    plan += [Segment('heat', cfg.hold_s, lambda t, _, tq=tq: Command(torque_nm=tq, **torque_only))
             for tq in levels]
    async with WristSession(EXPERIMENT, args, 0.0, 0.0, config=asdict(cfg),
                             rate_hz=cfg.sample_rate_hz, sim_params=SIM_PARAMS,
                             limits=limits) as session:
        session.require_locked_output(
            'Clamp the output so it cannot rotate; the motor will hold torque against it.')
        confirm(args, ['Thermal / continuous torque test (output locked, torque control)',
                       *describe_plan(plan, session.max_torque_nm), *lines])
        baseline = await session.run_segment(
            hold(cfg.baseline_s, phase='baseline', torque_nm=0.0, **torque_only))
        channel, start_c, limit_c = choose_channel(baseline.rows)
        session.metadata.update({
            'temperature_channel': channel,
            'start_temperature_c': start_c,
            'temperature_limit_c': limit_c,
            'levels_nm': levels,
            'torque_fractions': list(cfg.torque_fractions),
            'level_end_reasons': [],
        })
        print(f'Start temperature {start_c:.1f} C on {channel}; limit {limit_c:.0f} C.')
        for i, torque in enumerate(levels):
            result = await session.run_segment(heat_segment(i, torque, cfg, channel, limit_c))
            reason = result.ended_by or 'duration'
            session.metadata['level_end_reasons'].append(reason)
            temp = result.rows[-1].get(channel, math.nan) if result.rows else math.nan
            print(f'Level {i}: {torque:+.2f} N m ended by {reason} at {temp:.1f} C')
            if i < len(levels) - 1:
                await session.run_segment(cool_segment(i, cfg, channel, start_c))


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
