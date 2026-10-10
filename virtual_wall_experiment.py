"""Virtual wall impacts: how hard a rendered wall feels and whether it is passive.

The wrist launches itself, under stiff tracking, into a unilateral virtual
wall at WALL_DEG and is released just before contact, so the impact is
governed by the wall alone:

    'rt' mode:      the p4wrist loop's stay-within mode with the upper bound
                    at the wall; the loop renders K and B at 500 Hz.
    'host_<rate>':  the host computes tau = -K (th - th_w) - B th' while the
                    joint is inside the wall, from the previous sample, and
                    sends it as feedforward torque (a PC-side haptic loop).

Each impact gives the coefficient of restitution e = |v_out| / |v_in| (e > 1:
the wall injected energy; physical walls give e < 1), the work the wall did
on the joint (integral of torque over penetration; positive = energy
generated), the rate-hardness (initial rate of change of the wall torque per
unit impact velocity; Lawrence et al., IEEE TRA 16(4), 2000), the effective
stiffness, peak penetration, contact time, and the number of contacts. For
each mode, damping, and impact speed the stiffness is raised until the wall
chatters or launches the handle back too fast.

Run with the handle free (no hand on it): the wrist moves itself.

Run:  python virtual_wall_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass

import numpy as np

import device_config
from experiment_common import (Command, WristSession, Segment, confirm, hold, latest_summary,
                               make_arg_parser, min_jerk_move, run_cli)

EXPERIMENT = 'virtual_wall'
SIM_PARAMS = None   # Tests may inject a p4_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class VirtualWallConfig:
    modes: tuple = ('rt', 'host_250')   # 'rt' and/or 'host_<rate_hz>' (max 3)
    stiffness_nm_rad: tuple = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
    damping_nms_rad: tuple = (0.0, 0.1)
    impact_speeds_deg_s: tuple = (60.0, 120.0)
    repeats: int = 2
    wall_deg: float = 10.0                   # Wall position relative to the start pose
    release_before_deg: float = 1.5          # Tracking stops this far before the wall
    approach_accel_deg_s2: float = 1500.0    # Peak acceleration of the run-up
    cruise_s: float = 0.12                   # Constant-speed run-up before release
    tracking_k_nm_rad: float = 60.0          # Approach/return tracking gains (stable in the
    tracking_b_nms_rad: float = 0.8          # 500 Hz loop)
    catch_s: float = 0.08                    # Keep rendering after leaving the wall
    max_wall_s: float = 1.0                  # Longest wall phase (e.g. joint stuck inside)
    catch_speed_deg_s: float = 5.0           # Damped catch ends below this speed
    settle_s: float = 0.3
    brake_s: float = 0.1
    return_speed_deg_s: float = 60.0
    max_penetration_deg: float = 6.0         # Trial stops (unstable) beyond this
    max_velocity_deg_s: float = 600.0
    max_contacts: int = 4                    # More contacts in one trial = chatter
    restitution_stop: float = 2.0            # Stop raising K once e exceeds this
    torque_cap_fraction: float = 0.9         # Of the device torque cap, during the wall
    sample_rate_hz: float = 500.0            # Approach, return, and rt-mode logging


QUICK = dict(stiffness_nm_rad=(20.0, 100.0, 500.0), damping_nms_rad=(0.0,),
             impact_speeds_deg_s=(120.0,), repeats=1, settle_s=0.2)

VEL_WINDOW_S = 0.010        # Position-slope window for entry/exit speeds (5 samples at 500 Hz)
RH_WINDOW_FRACTION = 0.25   # Rate-hardness window: first quarter of the contact ...
RH_WINDOW_MIN_S = 0.002     # ... clipped to [min, max]
RH_WINDOW_MAX_S = 0.020
DELAY_BAND = (0.1, 0.5)     # Torque delay: line through the rise between these x peak
RELEASE_FRACTION = 0.05     # Wall torque counts as released below this x peak
UNSTABLE_REASONS = ('penetration limit', 'velocity limit', 'chatter')


def parse_mode(name, cfg):
    if name == 'rt':
        return {'name': name, 'renderer': 'rt', 'rate_hz': cfg.sample_rate_hz}
    if name.startswith('host_'):
        try:
            rate = float(name.split('_', 1)[1])
        except ValueError:
            raise SystemExit(f'Bad mode {name!r}; use host_<rate_hz>, e.g. host_250.')
        return {'name': name, 'renderer': 'host', 'rate_hz': rate}
    raise SystemExit(f'Unknown mode {name!r}; use "rt" or "host_<rate_hz>".')


# --- impact metrics (also used by the analysis) --------------------------------------
def _slope(t, y):
    if len(t) < 3 or np.ptp(t) <= 0:
        return math.nan
    return float(np.polyfit(t - t[0], y, 1)[0])


def _crossing(t, p, i):
    """Time where p crosses zero between samples i-1 and i."""
    if i <= 0 or p[i] == p[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (0.0 - p[i - 1]) / (p[i] - p[i - 1]) * (t[i] - t[i - 1]))


def impact_metrics(t, pos_deg, torque_nm, wall_deg, inertia, coulomb_nm=0.0, viscous=0.0):
    """Metrics of the first impact in a wall phase.

    t: s; pos_deg: joint position (deg); torque_nm: measured joint torque
    (negative pushes out of the wall); inertia: kg m^2 for the energy terms.
    Velocities are position slopes over VEL_WINDOW_S just before entry and
    just after the wall torque has released (a delayed wall keeps pushing past
    exit), so they do not depend on the loop's velocity filter.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(pos_deg, dtype=float) - wall_deg
    tau = np.asarray(torque_nm, dtype=float)
    out = {key: math.nan for key in (
        't_in_s', 't_out_s', 'v_in_deg_s', 'v_out_deg_s', 'restitution', 'peak_penetration_deg',
        'contact_ms', 'peak_torque_nm', 'k_eff_nm_rad', 'k_eff_peak_nm_rad',
        'rate_hardness_nm_rad', 'wall_work_mj', 'kinetic_change_mj', 'friction_loss_mj',
        'wall_work_balance_mj', 'impact_energy_mj', 'wall_energy_gain_percent',
        'rate_hardness_window_ms', 'torque_delay_ms', 'torque_release_ms')}
    out.update({'num_contacts': 0, 'exited': False, 'contact': False})
    if len(t) < 5:
        return out
    inside = p > 0.0
    entries = list(np.flatnonzero(inside[1:] & ~inside[:-1]) + 1)
    out['num_contacts'] = len(entries)
    if not entries:
        return out
    out['contact'] = True
    i_in = int(entries[0])
    t_in = _crossing(t, p, i_in)
    pre = (t >= t_in - VEL_WINDOW_S - 1e-9) & (t < t_in)
    v_in = _slope(t[pre], p[pre])
    out.update({'t_in_s': t_in, 'v_in_deg_s': v_in})
    after = np.flatnonzero(~inside[i_in:])
    i_out = i_in + int(after[0]) if after.size else len(t)
    contact = slice(i_in, i_out)
    pen = p[contact]
    push = -tau[contact]
    i_max = int(np.argmax(pen))
    peak_pen = float(pen[i_max])
    out['peak_penetration_deg'] = peak_pen
    out['peak_torque_nm'] = float(np.max(push))
    if peak_pen > 0:
        out['k_eff_nm_rad'] = float(push[i_max] / math.radians(peak_pen))
        out['k_eff_peak_nm_rad'] = float(np.max(push) / math.radians(peak_pen))
    peak = out['peak_torque_nm']
    i_off = len(t)
    if after.size:
        t_out = _crossing(t, p, i_out)
        # A delayed wall keeps pushing after the handle has left: find where
        # the wall torque has actually released before measuring v_out.
        quiet = np.flatnonzero(np.abs(tau[i_out:]) < RELEASE_FRACTION * max(peak, 1e-9))
        i_off = i_out + int(quiet[0]) if quiet.size else len(t)
        t_off = float(t[min(i_off, len(t) - 1)])
        # (+1 ns: on a regular sample clock the window edge lands on a sample,
        # whose time must not depend on rounding, e.g. a CSV round trip.)
        post = (t > t_off) & (t <= t_off + VEL_WINDOW_S + 1e-9)
        v_out = _slope(t[post], p[post])
        out.update({'exited': True, 't_out_s': t_out, 'v_out_deg_s': v_out,
                    'contact_ms': 1e3 * (t_out - t_in),
                    'torque_release_ms': 1e3 * max(t_off - t_out, 0.0)})
        if math.isfinite(v_in) and v_in > 0 and math.isfinite(v_out):
            out['restitution'] = abs(v_out) / v_in
        duration = t_out - t_in
    else:
        duration = t[-1] - t_in
    # Rate-hardness: the rate of rise of the wall torque per unit impact speed,
    # as a least-squares line through (t_in, 0) over the first quarter of the
    # contact. For a delay-free spring this is close to K (slightly low from
    # the curvature of the torque rise and the current-loop lag, about the
    # same at every K); a torque jump from damping raises it, and a delayed
    # torque onset (host loop) lowers it.
    window = min(max(RH_WINDOW_FRACTION * duration, RH_WINDOW_MIN_S), RH_WINDOW_MAX_S)
    early = (t > t_in) & (t <= t_in + window) & inside
    if early.sum() >= 1 and math.isfinite(v_in) and v_in > 0:
        dt = t[early] - t_in
        slope = float(np.sum(dt * -tau[early]) / np.sum(dt * dt))
        out['rate_hardness_nm_rad'] = slope / math.radians(v_in)
        out['rate_hardness_window_ms'] = 1e3 * window
    # Effective torque delay: a line through the rising torque between
    # DELAY_BAND x peak, extrapolated back to zero torque, relative to entry.
    # Near zero for a delay-free wall; about the loop delay for the host loop.
    if peak > 0:
        tc = t[contact] - t_in
        rise = np.flatnonzero((push >= DELAY_BAND[0] * peak) & (push <= DELAY_BAND[1] * peak))
        rise = rise[rise <= int(np.argmax(push))]
        if rise.size >= 2 and np.ptp(tc[rise]) > 0:
            slope, intercept = np.polyfit(tc[rise], push[rise], 1)
            if slope > 0:
                out['torque_delay_ms'] = 1e3 * float(-intercept / slope)
        else:
            above = np.flatnonzero(push >= DELAY_BAND[0] * peak)
            if above.size:
                out['torque_delay_ms'] = 1e3 * float(tc[above[0]])   # Upper bound
    # Work the wall did on the joint (positive = energy injected), from the
    # measured torque and the encoder angle (trapezoid), from just before entry
    # until the wall torque has released after exit.
    lo, hi = max(i_in - 1, 0), min(i_off + 1, len(t))
    theta = np.radians(p[lo:hi])
    tq = tau[lo:hi]
    out['wall_work_mj'] = 1e3 * float(np.sum(0.5 * (tq[1:] + tq[:-1]) * np.diff(theta)))
    if out['exited'] and math.isfinite(v_in) and math.isfinite(out['v_out_deg_s']):
        e_in = 0.5 * inertia * math.radians(v_in) ** 2
        e_out = 0.5 * inertia * math.radians(out['v_out_deg_s']) ** 2
        omega = np.gradient(theta, t[lo:hi]) if hi - lo > 2 else np.zeros_like(theta)
        friction = (coulomb_nm * 2.0 * math.radians(peak_pen)
                    + viscous * float(np.sum(0.5 * (omega[1:] ** 2 + omega[:-1] ** 2)
                                             * np.diff(t[lo:hi]))))
        out.update({
            'impact_energy_mj': 1e3 * e_in,
            'kinetic_change_mj': 1e3 * (e_out - e_in),
            'friction_loss_mj': 1e3 * friction,
            'wall_work_balance_mj': 1e3 * (e_out - e_in + friction),
            'wall_energy_gain_percent': 100.0 * out['wall_work_mj'] / (1e3 * e_in) if e_in > 0 else math.nan,
        })
    return out


# --- trial construction --------------------------------------------------------------
def approach_profile(start_deg, speed, accel):
    """Smooth run-up: v(t) = speed (1 - cos(pi t / t_b)) / 2, then constant speed."""
    t_b = math.pi * speed / (2.0 * accel)
    x_b = 0.5 * speed * t_b

    def state(t):
        if t < t_b:
            v = 0.5 * speed * (1.0 - math.cos(math.pi * t / t_b))
            x = 0.5 * speed * (t - t_b / math.pi * math.sin(math.pi * t / t_b))
        else:
            v = speed
            x = x_b + speed * (t - t_b)
        return start_deg + x, v

    return state, t_b, x_b


def run_up_start(cfg, speed):
    t_b = math.pi * speed / (2.0 * cfg.approach_accel_deg_s2)
    x_b = 0.5 * speed * t_b
    return cfg.wall_deg - cfg.release_before_deg - speed * cfg.cruise_s - x_b


def wall_command(mode, K, B, wall_deg, base, cap):
    if mode['renderer'] == 'rt':
        cmd = Command(stay_within=(math.nan, wall_deg), torque_nm=0.0,
                      kp_scale=K / base[0], kd_scale=(B / base[1]) if base[1] > 0 else 0.0,
                      max_torque_nm=cap)
        return lambda t, last: cmd

    def command(t, last):
        torque = 0.0
        if last is not None:
            pen = math.radians(last['joint_pos_deg'] - wall_deg)
            if pen > 0.0:
                torque = -K * pen - B * math.radians(last['joint_vel_deg_s'])
                torque = max(-cap, min(cap, torque))
        return Command(position_deg=math.nan, torque_nm=torque, kp_scale=0.0, kd_scale=0.0,
                       max_torque_nm=cap)

    return command


def wall_stop(cfg):
    """Ends the wall phase after the rebound, or early when the wall misbehaves."""
    state = {'inside': False, 'contacts': 0, 't_exit': None}

    def until(t, row):
        pos = row['joint_pos_deg']
        if pos - cfg.wall_deg > cfg.max_penetration_deg:
            return 'penetration limit'
        if abs(row['joint_vel_deg_s']) > cfg.max_velocity_deg_s:
            return 'velocity limit'
        inside = pos > cfg.wall_deg
        if inside and not state['inside']:
            state['contacts'] += 1
            state['t_exit'] = None
            if state['contacts'] > cfg.max_contacts:
                return 'chatter'
        if not inside and state['inside']:
            state['t_exit'] = t
        state['inside'] = inside
        if state['t_exit'] is not None and t - state['t_exit'] >= cfg.catch_s:
            return 'released'
        if t >= cfg.max_wall_s:
            return 'stuck in wall' if inside else 'no exit'
        return None

    return until


def catch_stop(cfg):
    def until(t, row):
        return 'stopped' if abs(row['joint_vel_deg_s']) < cfg.catch_speed_deg_s else None
    return until


async def run_trial(session, cfg, mode, K, B, speed, repeat, trial_id, base, cap, params):
    info = {'trial_id': trial_id, 'render_mode': mode['name'], 'virtual_k_nm_rad': K,
            'virtual_b_nms_rad': B, 'impact_speed_deg_s': speed, 'repeat': repeat,
            'wall_deg': cfg.wall_deg}
    track = dict(kp_scale=cfg.tracking_k_nm_rad / base[0],
                 kd_scale=cfg.tracking_b_nms_rad / base[1])
    start = run_up_start(cfg, speed)
    here = await session.command(Command(position_deg=math.nan, **track))
    await session.run([
        min_jerk_move(here['joint_pos_deg'], start, cfg.return_speed_deg_s, phase='return',
                      info=info, **track),
        hold(cfg.settle_s, start, phase='settle', info=info, **track),
    ])
    profile, t_b, _ = approach_profile(start, speed, cfg.approach_accel_deg_s2)
    release_at = cfg.wall_deg - cfg.release_before_deg

    def approach(t, _):
        position, velocity = profile(t)
        # Never drive the setpoint toward the wall: if release fails to trigger
        # (e.g. the handle is blocked), the setpoint stops just past it.
        if position > release_at + 0.5:
            position, velocity = release_at + 0.5, 0.0
        return Command(position_deg=position, velocity_deg_s=velocity, **track)

    def released(t, row):
        return 'release' if row['joint_pos_deg'] >= release_at else None

    run_up = (release_at - start) / speed + t_b
    approach_result = await session.run_segment(Segment(
        'approach', run_up + 0.5, approach, info, until=released))
    if approach_result.ended_by != 'release':
        ended_by = 'no release'
        wall_rows = []
    else:
        wall = await session.run_segment(Segment(
            'wall', cfg.max_wall_s + 0.05, wall_command(mode, K, B, cfg.wall_deg, base, cap),
            info, until=wall_stop(cfg), rate_hz=mode['rate_hz']))
        ended_by = wall.ended_by or 'duration'
        wall_rows = wall.rows

    if ended_by in UNSTABLE_REASONS:
        session.log_event('unstable_wall', trial_id=trial_id, ended_by=ended_by)
    else:
        # Damped catch: pure damping toward zero velocity, then a short brake.
        await session.run_segment(Segment(
            'catch', 0.4, lambda t, _: Command(position_deg=math.nan, kp_scale=0.0,
                                               kd_scale=track['kd_scale']),
            info, until=catch_stop(cfg)))
    await session.brake(cfg.brake_s, info=info)

    metrics = impact_metrics([r['t_local'] for r in wall_rows],
                             [r['joint_pos_deg'] for r in wall_rows],
                             [r['joint_torque_nm'] for r in wall_rows],
                             cfg.wall_deg, params['inertia'], params['coulomb'], params['viscous'])
    unstable = ended_by in UNSTABLE_REASONS or metrics['num_contacts'] > cfg.max_contacts
    record = {'trial_id': trial_id, 'mode': mode['name'], 'renderer': mode['renderer'],
              'render_rate_hz': mode['rate_hz'], 'virtual_k_nm_rad': K,
              'virtual_b_nms_rad': B, 'impact_speed_deg_s': speed, 'repeat': repeat,
              'ended_by': ended_by, 'unstable': bool(unstable),
              **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in metrics.items()}}
    session.metadata.setdefault('impacts', []).append(record)
    e = metrics['restitution']
    text = 'UNSTABLE' if unstable else (f'e = {e:.2f}' if math.isfinite(e) else ended_by)
    print(f"  {mode['name']:<10} B={B:<5.3g} v={speed:<5.0f} K={K:<7.4g} #{repeat}: {text}"
          + (f", RH {metrics['rate_hardness_nm_rad']:.0f} N m/rad"
             if math.isfinite(metrics['rate_hardness_nm_rad']) else ''))
    return record


def base_gains(cfg, mode, K, B):
    if mode['renderer'] == 'rt':
        return max(cfg.tracking_k_nm_rad, K), max(cfg.tracking_b_nms_rad, B)
    return cfg.tracking_k_nm_rad, cfg.tracking_b_nms_rad


# --- planning ---------------------------------------------------------------------------
def reference_params(simulated):
    """Inertia and friction for the energy terms (same kind of run only)."""
    params = {'inertia': device_config.NOMINAL_JOINT_INERTIA_KG_M2,
              'coulomb': device_config.NOMINAL_COULOMB_FRICTION_NM,
              'viscous': device_config.NOMINAL_JOINT_DAMPING_NMS_RAD,
              'source': 'device_config nominal'}
    step = latest_summary('step_response', simulated)
    if step and isinstance(step[1].get('J_kg_m2'), (int, float)):
        params.update(inertia=float(step[1]['J_kg_m2']),
                      coulomb=float(step[1].get('fc_nm', params['coulomb'])),
                      viscous=float(step[1].get('b_nms_rad', params['viscous'])),
                      source=f'step_response {step[0].name}')
    sweep = latest_summary('friction_sweep', simulated)
    if sweep and isinstance(sweep[1].get('coulomb_friction_nm'), (int, float)):
        params.update(coulomb=float(sweep[1]['coulomb_friction_nm']),
                      viscous=float(sweep[1].get('viscous_damping_nms_rad', params['viscous'])))
        params['source'] += f"; friction from friction_sweep {sweep[0].name}"
    return params


def check_plan(cfg, modes, cap):
    if not modes or len(modes) > 3:
        raise SystemExit('Configure one to three modes.')
    if any(k <= 0 for k in cfg.stiffness_nm_rad) or any(b < 0 for b in cfg.damping_nms_rad):
        raise SystemExit('Stiffness must be positive and damping non-negative.')
    lo, hi = device_config.JOINT_LIMITS_DEG
    v_max = max(cfg.impact_speeds_deg_s)
    if v_max > 0.5 * device_config.MAX_JOINT_VELOCITY_DEG_S:
        raise SystemExit('Impact speeds must stay below half of MAX_JOINT_VELOCITY_DEG_S.')
    if cfg.max_velocity_deg_s >= device_config.MAX_JOINT_VELOCITY_DEG_S:
        raise SystemExit('max_velocity_deg_s must be below MAX_JOINT_VELOCITY_DEG_S.')
    if cfg.wall_deg + cfg.max_penetration_deg > hi - 2.0:
        raise SystemExit('wall_deg + max_penetration_deg must stay inside JOINT_LIMITS_DEG.')
    start = min(run_up_start(cfg, v) for v in cfg.impact_speeds_deg_s)
    # Worst rebound: restitution_stop x the fastest impact, coasting for the catch window
    # and stopped by the tracking damping.
    j = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    v_back = math.radians(cfg.restitution_stop * v_max)
    rebound = math.degrees(v_back * (cfg.catch_s + j / cfg.tracking_b_nms_rad))
    if min(start, cfg.wall_deg - rebound) < lo + 2.0:
        raise SystemExit(f'The run-up (start {start:.1f} deg) or a rebound (~{rebound:.0f} deg) '
                         f'leaves JOINT_LIMITS_DEG {lo, hi}; lower the speeds or move wall_deg.')
    k_min, k_max = min(cfg.stiffness_nm_rad), max(cfg.stiffness_nm_rad)
    pen = math.degrees(math.radians(v_max) * math.sqrt(j / k_min))
    torque = math.radians(v_max) * math.sqrt(j * k_max)
    if pen > cfg.max_penetration_deg:
        raise SystemExit(f'K = {k_min:g} at {v_max:g} deg/s penetrates ~{pen:.1f} deg, beyond '
                         'max_penetration_deg; raise the stiffness or lower the speed.')
    lines = [f'  Run-up from {start:.1f} deg to the wall at {cfg.wall_deg:+.1f} deg; release '
             f'{cfg.release_before_deg} deg before it.',
             f'  Predicted penetration {pen:.2f} deg (softest wall, fastest impact); peak wall '
             f'torque ~{torque:.2f} N m (stiffest; cap {cap:.2f} N m).']
    if torque > cap:
        lines.append('  NOTE: the stiffest walls will hit the torque cap and feel softer than '
                     'commanded.')
    return lines


async def main(args):
    cfg = VirtualWallConfig(**(QUICK if args.quick else {}))
    modes = [parse_mode(m, cfg) for m in cfg.modes]
    cap_nm = cfg.torque_cap_fraction * device_config.MAX_JOINT_TORQUE_NM
    lines = check_plan(cfg, modes, cap_nm)
    params = reference_params(args.sim)
    n_max = (len(modes) * len(cfg.damping_nms_rad) * len(cfg.impact_speeds_deg_s)
             * len(cfg.stiffness_nm_rad) * cfg.repeats)
    async with WristSession(EXPERIMENT, args, cfg.tracking_k_nm_rad, cfg.tracking_b_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        cap = cfg.torque_cap_fraction * session.max_torque_nm
        confirm(args, [
            'Virtual wall impacts (restitution, wall energy, rate-hardness)',
            f"  Modes: {', '.join(m['name'] for m in modes)}; K = {cfg.stiffness_nm_rad} N m/rad; "
            f'B = {cfg.damping_nms_rad} N m s/rad; speeds {cfg.impact_speeds_deg_s} deg/s',
            f'  Up to {n_max} impacts (~{n_max * 2.5 / 60:.0f} min).',
            *lines,
            '  The HANDLE MUST BE FREE: the wrist launches itself into the wall.',
            f"  Energy terms use J = {params['inertia']:.4g} kg m^2 ({params['source']}).",
        ])
        session.metadata['virtual_wall_protocol'] = {
            'modes': modes, 'wall_torque_cap_nm': cap, 'reference': params,
            'vel_window_s': VEL_WINDOW_S, 'rh_window_fraction': RH_WINDOW_FRACTION,
            'rh_window_min_s': RH_WINDOW_MIN_S, 'rh_window_max_s': RH_WINDOW_MAX_S,
            'torque_delay_band': DELAY_BAND, 'release_fraction': RELEASE_FRACTION,
            'unstable_reasons': list(UNSTABLE_REASONS),
        }
        await session.run([hold(0.3, 0.0, phase='start')])
        await session.brake(cfg.brake_s)
        base = (cfg.tracking_k_nm_rad, cfg.tracking_b_nms_rad)
        trial_id = 0
        for mode in modes:
            print(f"Mode {mode['name']} ({mode['renderer']} rendering at {mode['rate_hz']:g} Hz)")
            for B in cfg.damping_nms_rad:
                for speed in cfg.impact_speeds_deg_s:
                    for K in cfg.stiffness_nm_rad:
                        wanted = base_gains(cfg, mode, K, B)
                        if wanted != base:
                            await session.set_gains(*wanted)   # Controller is braking here
                            base = wanted
                        records = []
                        for repeat in range(cfg.repeats):
                            records.append(await run_trial(session, cfg, mode, K, B, speed,
                                                           repeat, trial_id, base, cap, params))
                            trial_id += 1
                        es = [r['restitution'] for r in records if math.isfinite(r['restitution'])]
                        if any(r['unstable'] for r in records) or (
                                es and max(es) > cfg.restitution_stop):
                            print(f'    stopping the K sweep for {mode["name"]}, B={B:g}, '
                                  f'v={speed:g}')
                            break
        if base != (cfg.tracking_k_nm_rad, cfg.tracking_b_nms_rad):
            await session.set_gains(cfg.tracking_k_nm_rad, cfg.tracking_b_nms_rad)
        here = await session.command(Command(position_deg=math.nan))
        await session.run([min_jerk_move(here['joint_pos_deg'], 0.0, cfg.return_speed_deg_s,
                                         phase='end'),
                           hold(0.5, 0.0, phase='end')])


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
