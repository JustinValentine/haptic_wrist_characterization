"""Z-width: the range of virtual stiffness and damping the wrist renders stably.

A virtual spring-damper

    tau = -K (th - th0) - B th'

is rendered about a fixed pose, either by the moteus position loop at its
internal rate ('moteus' mode, the hardware/firmware limit) or by this host,
which computes the torque from the latest measurement and sends it as a
feedforward torque ('host_<rate>' modes, the situation of a PC-side haptic
simulation with one sample of delay plus communication latency). For each B
the stiffness K is raised geometrically until a short torque pulse produces a
sustained or growing oscillation; the largest stable K per B traces the
Z-width boundary (Colgate & Brown, 1994).

Instability is expected. Each trial is contained by a deviation, velocity,
and torque-chatter stop, a brief brake, and a return to a safe hold.
zwidth_experiment_analysis.py reclassifies every trial from the raw data and
plots the boundary.

Run:  python zwidth_experiment.py [--sim] [--quick]
"""
import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

import device_config
from experiment_common import (Command, MoteusSession, Segment, confirm, hold, make_arg_parser,
                               min_jerk_move, run_cli)

EXPERIMENT = 'zwidth'
SIM_PARAMS = None   # Tests may inject a moteus_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class ZWidthConfig:
    modes: tuple = ('host_1000', 'moteus')   # 'moteus' and/or 'host_<rate_hz>' (max 3)
    damping_values_nms_rad: tuple = (0.0, 0.05, 0.1, 0.2, 0.5)
    k_start_host_nm_rad: float = 5.0
    k_max_host_nm_rad: float = 1000.0
    k_start_moteus_nm_rad: float = 50.0
    k_max_moteus_nm_rad: float = 10000.0    # Raise to find the firmware limit
    k_factor: float = 1.4                    # Geometric stiffness increment
    refine_steps: int = 1                    # Bisection trials between stable and unstable
    center_deg: float = 0.0                  # Pose the impedance is rendered about
    safe_k_nm_rad: float = 20.0              # Hold between trials (moteus PD)
    safe_b_nms_rad: float = 0.3
    settle_s: float = 0.6                    # Safe hold before each trial (noise floor)
    pulse_start_s: float = 0.2               # Test impedance active before the pulse
    pulse_width_s: float = 0.005             # Short, so it excites up to ~100 Hz
    pulse_torque_nm: float = 1.0             # Disturbance torque pulse
    observe_s: float = 1.5                   # Observation after the pulse
    brake_s: float = 0.3                     # Brake after every trial
    torque_cap_fraction: float = 0.5         # Of the device torque cap, during trials
    max_deviation_deg: float = 8.0           # Trial stops (unstable) beyond this
    max_velocity_deg_s: float = 600.0
    return_speed_deg_s: float = 30.0
    sample_rate_hz: float = 1000.0           # Holds and moteus-mode trials


QUICK = dict(damping_values_nms_rad=(0.0, 0.3), k_factor=2.0, refine_steps=1,
             observe_s=1.0, settle_s=0.4)

# Classification thresholds (shared with the analysis).
EARLY_WINDOW_S = 0.25     # After the pulse
DECAY_CHECK_S = (0.5, 0.75)   # Must have halved by this window after the pulse
LATE_WINDOW_S = 0.5       # End of the observation
DECAY_RATIO_LIMIT = 0.5   # Late/early oscillation RMS above this is not decaying
NOISE_FACTOR = 5.0
MIN_POSITION_RMS_DEG = 1e-3
MIN_TORQUE_RMS_FRACTION = 0.03   # Of the trial torque cap
CHATTER_WINDOW = 40              # Samples
CHATTER_SPAN_FRACTION = 1.5      # Peak-to-peak torque above this x cap ...
CHATTER_SIGN_CHANGES = 8         # ... with this many sign changes -> chatter


def parse_mode(name, cfg):
    if name == 'moteus':
        return {'name': name, 'renderer': 'moteus', 'rate_hz': cfg.sample_rate_hz,
                'k_start': cfg.k_start_moteus_nm_rad, 'k_max': cfg.k_max_moteus_nm_rad}
    if name.startswith('host_'):
        try:
            rate = float(name.split('_', 1)[1])
        except ValueError:
            raise SystemExit(f'Bad mode {name!r}; use host_<rate_hz>, e.g. host_1000.')
        return {'name': name, 'renderer': 'host', 'rate_hz': rate,
                'k_start': cfg.k_start_host_nm_rad, 'k_max': cfg.k_max_host_nm_rad}
    raise SystemExit(f'Unknown mode {name!r}; use "moteus" or "host_<rate_hz>".')


# --- stability classification (also used by the analysis) --------------------------
def oscillation_frequency(t, x, band):
    """Mean oscillation frequency (Hz) from zero crossings with hysteresis.

    x is detrended with a line; crossings count only after the signal leaves
    the +/- band, so noise and a friction offset do not register. Returns nan
    if fewer than two full crossings occur.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) < 8:
        return math.nan
    x = x - np.polyval(np.polyfit(t - t[0], x, 1), t - t[0])
    state = 0
    crossings = []
    for ti, xi in zip(t, x):
        if xi > band and state <= 0:
            if state < 0:
                crossings.append(ti)
            state = 1
        elif xi < -band and state >= 0:
            if state > 0:
                crossings.append(ti)
            state = -1
    if len(crossings) < 3:
        return math.nan
    return float((len(crossings) - 1) / (2.0 * (crossings[-1] - crossings[0])))


def noise_floor(t, deviation_deg, torque_nm, window_s=0.25):
    """Position and torque noise (std) at the end of a quiet hold."""
    t = np.asarray(t, dtype=float)
    if len(t) < 5:
        return math.nan, math.nan
    sel = t >= t[-1] - window_s
    return (float(np.std(np.asarray(deviation_deg, dtype=float)[sel])),
            float(np.std(np.asarray(torque_nm, dtype=float)[sel])))


def classify_trial(t, deviation_deg, torque_nm, *, pulse_end_s, planned_end_s, noise_deg,
                   noise_nm, cap_nm, ended_by=None):
    """Stable if the pulse response decays to the noise floor.

    Unstable if the trial was stopped early; if the position oscillation has
    not fallen below DECAY_RATIO_LIMIT of its post-pulse value within
    DECAY_CHECK_S (ringing that friction only stops much later still feels
    like a buzzing wall); or if, in the last LATE_WINDOW_S, the position or
    torque oscillation is still above DECAY_RATIO_LIMIT of its post-pulse
    value and above the noise floor (sustained oscillation / limit cycle).
    Marginal stability therefore counts as unstable.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(deviation_deg, dtype=float)
    tq = np.asarray(torque_nm, dtype=float)
    out = {'early_rms_deg': math.nan, 'late_rms_deg': math.nan, 'rms_ratio': math.nan,
           'decay_check_rms_deg': math.nan,
           'early_torque_rms_nm': math.nan, 'late_torque_rms_nm': math.nan,
           'osc_freq_hz': math.nan, 'peak_deviation_deg': math.nan}
    if len(t) == 0:
        return {'stable': False, 'reason': 'no data', **out}
    post = t >= pulse_end_s
    band = 3.0 * (noise_deg if math.isfinite(noise_deg) else 0.0) + MIN_POSITION_RMS_DEG
    if post.sum() >= 2:
        out['peak_deviation_deg'] = float(np.max(np.abs(x[post])))
        out['osc_freq_hz'] = oscillation_frequency(t[post], x[post], band)
    early = post & (t < pulse_end_s + EARLY_WINDOW_S)
    if early.sum() >= 5:
        out['early_rms_deg'] = float(np.std(x[early]))
        out['early_torque_rms_nm'] = float(np.std(tq[early]))
    if ended_by:
        return {'stable': False, 'reason': ended_by, **out}
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if t[-1] < planned_end_s - max(5 * dt, 0.05):
        return {'stable': False, 'reason': 'truncated', **out}
    late = t >= t[-1] - LATE_WINDOW_S
    out['late_rms_deg'] = float(np.std(x[late]))
    out['late_torque_rms_nm'] = float(np.std(tq[late]))
    early_pos = out['early_rms_deg'] if math.isfinite(out['early_rms_deg']) else 0.0
    early_tq = out['early_torque_rms_nm'] if math.isfinite(out['early_torque_rms_nm']) else 0.0
    if early_pos > 0:
        out['rms_ratio'] = out['late_rms_deg'] / early_pos
    pos_floor = max(NOISE_FACTOR * (noise_deg if math.isfinite(noise_deg) else 0.0),
                    MIN_POSITION_RMS_DEG)
    tq_floor = max(NOISE_FACTOR * (noise_nm if math.isfinite(noise_nm) else 0.0),
                   MIN_TORQUE_RMS_FRACTION * cap_nm)
    check = (t >= pulse_end_s + DECAY_CHECK_S[0]) & (t < pulse_end_s + DECAY_CHECK_S[1])
    if check.sum() >= 5:
        out['decay_check_rms_deg'] = float(np.std(x[check]))
        if out['decay_check_rms_deg'] > max(DECAY_RATIO_LIMIT * early_pos, pos_floor):
            return {'stable': False, 'reason': 'slow decay', **out}
    if out['late_rms_deg'] > max(DECAY_RATIO_LIMIT * early_pos, pos_floor):
        return {'stable': False, 'reason': 'sustained oscillation', **out}
    if out['late_torque_rms_nm'] > max(DECAY_RATIO_LIMIT * early_tq, tq_floor):
        return {'stable': False, 'reason': 'torque chatter', **out}
    return {'stable': True, 'reason': 'decayed', **out}


# --- trial construction ---------------------------------------------------------------
def host_command(K, B, theta0, cap, cfg):
    """Host-rendered spring-damper from the previous measurement, plus the pulse."""
    t_on, t_off = cfg.pulse_start_s, cfg.pulse_start_s + cfg.pulse_width_s

    def command(t, last):
        torque = 0.0
        if last is not None:
            torque = (-K * math.radians(last['joint_pos_deg'] - theta0)
                      - B * math.radians(last['joint_vel_deg_s']))
        if t_on <= t < t_off:
            torque += cfg.pulse_torque_nm
        torque = max(-cap, min(cap, torque))
        return Command(position_deg=math.nan, torque_nm=torque, kp_scale=0.0, kd_scale=0.0,
                       max_torque_nm=cap)

    return command


def moteus_command(theta0, cap, cfg):
    """moteus renders K and B (set via set_gains); the host only adds the pulse."""
    t_on, t_off = cfg.pulse_start_s, cfg.pulse_start_s + cfg.pulse_width_s

    def command(t, _):
        torque = cfg.pulse_torque_nm if t_on <= t < t_off else 0.0
        return Command(position_deg=theta0, torque_nm=torque, max_torque_nm=cap)

    return command


def make_stop_check(theta0, cap, cfg):
    """Early stop for clearly unstable trials (deviation, speed, saturating chatter)."""
    recent = deque(maxlen=CHATTER_WINDOW)
    watch_after = cfg.pulse_start_s + cfg.pulse_width_s + 0.05

    def until(t, row):
        if abs(row['joint_pos_deg'] - theta0) > cfg.max_deviation_deg:
            return 'deviation limit'
        if abs(row['joint_vel_deg_s']) > cfg.max_velocity_deg_s:
            return 'velocity limit'
        recent.append(row['joint_torque_nm'])
        if t > watch_after and len(recent) == CHATTER_WINDOW:
            values = np.asarray(recent)
            if values.max() - values.min() > CHATTER_SPAN_FRACTION * cap:
                centered = values - values.mean()
                if np.count_nonzero(np.diff(np.sign(centered))) >= CHATTER_SIGN_CHANGES:
                    return 'torque chatter'
        return None

    return until


def trial_duration(cfg):
    return cfg.pulse_start_s + cfg.pulse_width_s + cfg.observe_s


async def run_trial(session, cfg, mode, K, B, trial_id, theta0, cap):
    info = {'trial_id': trial_id, 'render_mode': mode['name'], 'virtual_k_nm_rad': K,
            'virtual_b_nms_rad': B, 'render_rate_hz': mode['rate_hz']}
    settle = await session.run_segment(hold(cfg.settle_s, theta0, phase='settle', info=info))
    noise_deg, noise_nm = noise_floor([r['t_local'] for r in settle.rows],
                                      [r['joint_pos_deg'] - theta0 for r in settle.rows],
                                      [r['joint_torque_nm'] for r in settle.rows])

    if mode['renderer'] == 'moteus':
        await session.set_gains(K, B)
        command = moteus_command(theta0, cap, cfg)
    else:
        command = host_command(K, B, theta0, cap, cfg)
    trial = await session.run_segment(Segment(
        'trial', trial_duration(cfg), command, info,
        until=make_stop_check(theta0, cap, cfg), rate_hz=mode['rate_hz']))

    # Contain: brake (passive damping), restore the safe hold, return to center.
    session.log_event('brake', trial_id=trial_id, ended_by=trial.ended_by)
    await session.brake(cfg.brake_s, info={'trial_id': trial_id})
    if mode['renderer'] == 'moteus':
        await session.set_gains(cfg.safe_k_nm_rad, cfg.safe_b_nms_rad)
    here = await session.command(Command(position_deg=math.nan))   # Hold where it stopped
    await session.run_segment(min_jerk_move(here['joint_pos_deg'], theta0,
                                            cfg.return_speed_deg_s, phase='return',
                                            info={'trial_id': trial_id}))

    rows = trial.rows
    result = classify_trial(
        [r['t_local'] for r in rows], [r['joint_pos_deg'] - theta0 for r in rows],
        [r['joint_torque_nm'] for r in rows],
        pulse_end_s=cfg.pulse_start_s + cfg.pulse_width_s,
        planned_end_s=trial_duration(cfg), noise_deg=noise_deg, noise_nm=noise_nm,
        cap_nm=cap, ended_by=trial.ended_by)
    record = {'trial_id': trial_id, 'mode': mode['name'], 'renderer': mode['renderer'],
              'render_rate_hz': mode['rate_hz'], 'virtual_k_nm_rad': K,
              'virtual_b_nms_rad': B, 'ended_by': trial.ended_by,
              'noise_deg': noise_deg, 'noise_nm': noise_nm, **result}
    session.metadata.setdefault('trials', []).append(record)
    verdict = 'stable' if result['stable'] else f"UNSTABLE ({result['reason']})"
    print(f"  {mode['name']:<10} B={B:<6.3g} K={K:<8.4g} {verdict}")
    return record


async def sweep(session, cfg, mode, theta0, cap, next_id):
    boundary = []
    for B in cfg.damping_values_nms_rad:
        last_stable = first_unstable = None
        K = mode['k_start']
        while K <= mode['k_max'] * (1 + 1e-9):
            record = await run_trial(session, cfg, mode, K, B, next_id(), theta0, cap)
            if record['stable']:
                last_stable = K
                K *= cfg.k_factor
            else:
                first_unstable = K
                break
        for _ in range(cfg.refine_steps):
            if last_stable is None or first_unstable is None:
                break
            K = math.sqrt(last_stable * first_unstable)
            record = await run_trial(session, cfg, mode, K, B, next_id(), theta0, cap)
            if record['stable']:
                last_stable = K
            else:
                first_unstable = K
        boundary.append({'mode': mode['name'], 'virtual_b_nms_rad': B,
                         'k_last_stable_nm_rad': last_stable,
                         'k_first_unstable_nm_rad': first_unstable})
    return boundary


# --- planning ---------------------------------------------------------------------------
def check_plan(cfg, modes):
    if not modes:
        raise SystemExit('No rendering modes configured.')
    if len(modes) > 3:
        raise SystemExit('At most three modes (the figure shows three categorical series).')
    if any(b < 0 for b in cfg.damping_values_nms_rad):
        raise SystemExit('Negative virtual damping is not supported by this protocol.')
    if cfg.k_factor <= 1.05:
        raise SystemExit('k_factor must be > 1.05.')
    if not 0.0 < cfg.torque_cap_fraction <= 1.0:
        raise SystemExit('torque_cap_fraction must be in (0, 1].')
    cap = cfg.torque_cap_fraction * device_config.MAX_JOINT_TORQUE_NM
    if cfg.pulse_torque_nm >= cap:
        raise SystemExit('pulse_torque_nm must be below the trial torque cap.')
    lo, hi = device_config.JOINT_LIMITS_DEG
    if not (lo < cfg.center_deg - cfg.max_deviation_deg and cfg.center_deg + cfg.max_deviation_deg < hi):
        raise SystemExit('center_deg +/- max_deviation_deg must lie inside JOINT_LIMITS_DEG.')
    if cfg.max_velocity_deg_s >= device_config.MAX_JOINT_VELOCITY_DEG_S:
        raise SystemExit('max_velocity_deg_s must be below MAX_JOINT_VELOCITY_DEG_S so trials '
                         'stop before the session safety limit.')
    for mode in modes:
        if mode['renderer'] == 'host' and not 50.0 <= mode['rate_hz'] <= 4000.0:
            raise SystemExit(f"Host rate {mode['rate_hz']} Hz is outside 50-4000 Hz.")
        if mode['k_start'] >= mode['k_max']:
            raise SystemExit(f"{mode['name']}: k_start must be below k_max.")


def worst_case_trials(cfg, modes):
    total = 0
    for mode in modes:
        steps = math.floor(math.log(mode['k_max'] / mode['k_start']) / math.log(cfg.k_factor)) + 1
        total += len(cfg.damping_values_nms_rad) * (steps + cfg.refine_steps)
    return total


async def main(args):
    cfg = ZWidthConfig(**(QUICK if args.quick else {}))
    modes = [parse_mode(m, cfg) for m in cfg.modes]
    check_plan(cfg, modes)
    per_trial_s = cfg.settle_s + trial_duration(cfg) + cfg.brake_s + 0.5
    async with MoteusSession(EXPERIMENT, args, cfg.safe_k_nm_rad, cfg.safe_b_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        cap = cfg.torque_cap_fraction * session.max_torque_nm
        n_max = worst_case_trials(cfg, modes)
        confirm(args, [
            'Z-width (renderable impedance range)',
            f"  Modes: {', '.join(m['name'] for m in modes)}; "
            f'B = {cfg.damping_values_nms_rad} N m s/rad',
            f'  Adaptive sweep, at most {n_max} trials (~{n_max * per_trial_s / 60:.0f} min worst case)',
            f'  Trial torque cap {cap:.2f} N m; trials stop at {cfg.max_deviation_deg} deg, '
            f'{cfg.max_velocity_deg_s} deg/s, or torque chatter',
            '  The HANDLE MUST BE FREE: no hand on it, nothing in the workspace.',
            '  Unstable (oscillating) trials are expected; each is stopped, braked,',
            '  and returned to a safe hold automatically.',
        ])
        session.metadata['zwidth_protocol'] = {
            'modes': modes,
            'trial_torque_cap_nm': cap,
            'pulse_end_s': cfg.pulse_start_s + cfg.pulse_width_s,
            'planned_trial_s': trial_duration(cfg),
            'classification': {
                'early_window_s': EARLY_WINDOW_S, 'late_window_s': LATE_WINDOW_S,
                'decay_check_window_s': DECAY_CHECK_S,
                'decay_ratio_limit': DECAY_RATIO_LIMIT, 'noise_factor': NOISE_FACTOR,
                'min_position_rms_deg': MIN_POSITION_RMS_DEG,
                'min_torque_rms_fraction_of_cap': MIN_TORQUE_RMS_FRACTION,
            },
        }
        theta0 = cfg.center_deg
        await session.run([hold(0.5, 0.0, phase='start'),
                           min_jerk_move(0.0, theta0, cfg.return_speed_deg_s, phase='start')]
                          if theta0 != 0.0 else [hold(0.5, 0.0, phase='start')])
        counter = iter(range(10 ** 6))
        boundary = []
        for mode in modes:
            print(f"Sweeping {mode['name']} ({mode['renderer']} rendering at "
                  f"{mode['rate_hz']:g} Hz)")
            boundary += await sweep(session, cfg, mode, theta0, cap, lambda: next(counter))
        session.metadata['boundary'] = boundary
        await session.run([min_jerk_move(theta0, 0.0, cfg.return_speed_deg_s, phase='end'),
                           hold(0.5, 0.0, phase='end')] if theta0 != 0.0
                          else [hold(0.5, 0.0, phase='end')])
        print('Boundary (largest stable K):')
        for b in boundary:
            k = b['k_last_stable_nm_rad']
            note = '' if b['k_first_unstable_nm_rad'] is not None else ' (no instability up to k_max)'
            print(f"  {b['mode']:<10} B={b['virtual_b_nms_rad']:<6.3g} K_max="
                  f"{'none stable' if k is None else f'{k:.4g}'} N m/rad{note}")


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
