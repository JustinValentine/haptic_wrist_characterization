"""Large-signal dynamic envelope: how far the wrist can move at each frequency.

Small-signal bandwidth (schroeder_bandwidth_experiment.py) describes the
closed loop for small motions, far from the actuator's limits. A highly
dynamic, high-force device is better described by its large-signal
capability: for each frequency, the largest sinusoidal amplitude it tracks
before the response departs from its small-signal (linear) behavior. That
amplitude is bounded by

    torque:  A_tau(f) = tau_cap / (J (2 pi f)^2)   inertia driven at the torque cap
    speed:   A_v(f)   = v_max / (2 pi f)           back-EMF / speed limit
    range:   A_rom                                 range of motion

At each frequency the commanded amplitude starts from a small reference (which
defines the linear response G0), jumps to half the predicted limit, and grows
geometrically. Each trial is a raised-cosine-windowed sinusoid tracked by the
stiff PD loop with velocity feedforward. A trial is linear while |G/G0| stays
within 10%, the phase within 10 deg, the harmonic distortion below 5%, the
controller torque sits at the cap for under 2% of the steady window, and the
measured torque keeps up with the controller (no voltage limit). The first
non-linear amplitude ends the sweep at that frequency, after bisection steps
that tighten the bracket to within 5% (the first failure can lie far above the
last linear trial when back-EMF, which the torque-line prediction ignores,
limits first). When the safety caps (range, speed, predicted torque) are
reached first, the result is a lower bound.

Analyze with dynamic_envelope_experiment_analysis.py.

Run:  python dynamic_envelope_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

import device_config
from analysis_common import fit_sinusoids
from experiment_common import (Command, MoteusSession, Segment, confirm, hold, latest_summary,
                               make_arg_parser, min_jerk_move, run_cli)

EXPERIMENT = 'dynamic_envelope'
SIM_PARAMS = None   # Tests may inject a moteus_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class EnvelopeConfig:
    kp_joint_nm_rad: float = 450.0          # Stiff PD (bandwidth-test gains)
    kd_joint_nms_rad: float = 0.75
    send_velocity_feedforward: bool = True
    frequencies_hz: tuple = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0)
    reference_fraction: float = 0.1         # Small-signal reference / predicted limit
    start_fraction: float = 0.5             # First large amplitude / predicted limit
    growth: float = 1.25                    # Amplitude factor between trials
    refine_ratio: float = 1.05              # Bisect until last-linear/first-failing is this tight
    max_refine_steps: int = 6               # ... in at most this many extra trials
    max_amplitude_deg: float = 40.0         # Range-of-motion cap on the joint amplitude
    speed_fraction: float = 0.8             # Joint speed cap / MAX_JOINT_VELOCITY_DEG_S
    torque_overdrive: float = 1.5           # Command up to this x the torque-limited amplitude
    speed_limit_deg_s: Optional[float] = None   # Speed line in the figure (default: speed cap)
    ramp_periods: float = 2.0               # Raised-cosine ramp in and out ...
    min_ramp_s: float = 0.3                 # ... but at least this long
    steady_periods: int = 4                 # Steady window ...
    min_steady_s: float = 1.0               # ... but at least this long
    rest_s: float = 0.3
    center_deg: float = 0.0
    gain_tolerance: float = 0.10            # Linear: |G/G0| within 1 +/- this
    phase_tolerance_deg: float = 10.0       # ... phase change within this
    max_thd: float = 0.05                   # ... harmonic distortion below this
    max_saturation: float = 0.02            # ... controller torque at the cap below this fraction
    max_current_shortfall: float = 0.02     # ... measured torque short of the controller's
    reference_regions: tuple = ()           # Optional shaded regions for the figure:
    # ({'label': ..., 'f_min': ..., 'f_max': ..., 'amp_min': ..., 'amp_max': ...}, ...)
    # with amplitudes in deg; cite your source in the caption.
    sample_rate_hz: float = 1000.0


QUICK = dict(frequencies_hz=(1.0, 8.0, 16.0), ramp_periods=1.0, min_ramp_s=0.2,
             steady_periods=3, min_steady_s=0.6, growth=1.35)

SATURATION_THRESHOLD = 0.98   # |controller torque| >= this x cap counts as saturated
SHORTFALL_THRESHOLD = 0.10    # Controller - measured torque > this x cap: current can't follow


# --- planning --------------------------------------------------------------------------
def model_gain(f, kp, kd, J, b, ff):
    """|G| of the rigid closed loop: (Kp + FF Kd s) / (J s^2 + (b + Kd) s + Kp)."""
    s = 2j * math.pi * f
    return abs((kp + ff * kd * s) / (J * s ** 2 + (b + kd) * s + kp))


def reference_inertia(simulated):
    """J (and b) from the newest step_response analysis of the same kind."""
    found = latest_summary('step_response', simulated)
    if found is not None:
        run_dir, summary = found
        J = summary.get('J_kg_m2')
        if isinstance(J, (int, float)) and math.isfinite(J) and J > 0:
            b = summary.get('b_nms_rad')
            b = b if isinstance(b, (int, float)) and math.isfinite(b) else \
                device_config.NOMINAL_JOINT_DAMPING_NMS_RAD
            return float(J), float(b), f'step_response analysis ({run_dir.name})'
    return (device_config.NOMINAL_JOINT_INERTIA_KG_M2, device_config.NOMINAL_JOINT_DAMPING_NMS_RAD,
            'device_config nominal')


def limit_amplitudes(f, torque_cap_nm, J, speed_deg_s, rom_deg):
    """Joint-amplitude limits (deg) at frequency f."""
    w = 2.0 * math.pi * f
    return {'torque': math.degrees(torque_cap_nm / (J * w ** 2)),
            'speed': speed_deg_s / w,
            'range': rom_deg}


def trial_timing(cfg, f):
    ramp = max(cfg.ramp_periods / f, cfg.min_ramp_s)
    periods = max(cfg.steady_periods, int(math.ceil(cfg.min_steady_s * f - 1e-9)))
    return ramp, periods / f


def plan_frequency(cfg, f, J, b, torque_cap_nm):
    """Predicted limit and safety cap on the commanded amplitude at f."""
    ff = 1.0 if cfg.send_velocity_feedforward else 0.0
    gain = model_gain(f, cfg.kp_joint_nm_rad, cfg.kd_joint_nms_rad, J, b, ff)
    speed_cap = cfg.speed_fraction * device_config.MAX_JOINT_VELOCITY_DEG_S
    limits = limit_amplitudes(f, torque_cap_nm, J, speed_cap, cfg.max_amplitude_deg)
    binding = min(limits, key=limits.get)
    caps = dict(limits, torque=cfg.torque_overdrive * limits['torque'])
    cap_binding = min(caps, key=caps.get)
    ramp_s, steady_s = trial_timing(cfg, f)
    return {
        'frequency_hz': f,
        'model_gain': gain,
        'limit_torque_deg': limits['torque'],
        'limit_speed_deg': limits['speed'],
        'limit_range_deg': limits['range'],
        'predicted_binding': binding,
        'predicted_command_deg': limits[binding] / gain,
        'cap_command_deg': caps[cap_binding] / gain,
        'cap_binding': cap_binding,
        'ramp_s': ramp_s,
        'steady_s': steady_s,
    }


def worst_case_trials(cfg, plan):
    span = plan['cap_command_deg'] / (cfg.start_fraction * plan['predicted_command_deg'])
    steps = max(1, math.ceil(math.log(max(span, 1.0)) / math.log(cfg.growth)) + 1)
    return 1 + steps + cfg.max_refine_steps


def trial_duration(cfg, plan):
    return 2 * plan['ramp_s'] + plan['steady_s'] + cfg.rest_s


# --- one trial ----------------------------------------------------------------------------
def sine_command(center, amplitude, f, ramp_s, steady_s, feedforward):
    """Raised-cosine-windowed sinusoid with matching velocity feedforward."""
    w = 2.0 * math.pi * f
    total = 2.0 * ramp_s + steady_s

    def envelope(t):
        if t < ramp_s:
            x = math.pi * t / ramp_s
            return 0.5 * (1.0 - math.cos(x)), 0.5 * math.pi / ramp_s * math.sin(x)
        if t > ramp_s + steady_s:
            x = math.pi * (total - t) / ramp_s
            return 0.5 * (1.0 - math.cos(x)), -0.5 * math.pi / ramp_s * math.sin(x)
        return 1.0, 0.0

    def command(t, _):
        t = min(max(t, 0.0), total)
        e, de = envelope(t)
        s, c = math.sin(w * t), math.cos(w * t)
        velocity = amplitude * (de * s + e * w * c)
        return Command(position_deg=center + amplitude * e * s,
                       velocity_deg_s=velocity if feedforward else 0.0)

    return command, total


def make_guard(center, cfg):
    """Stop a trial before the session's hard limits trip."""
    speed_guard = 0.95 * device_config.MAX_JOINT_VELOCITY_DEG_S
    range_guard = cfg.max_amplitude_deg + 5.0

    def until(_, row):
        if abs(row['joint_vel_deg_s']) > speed_guard:
            return 'speed guard'
        if abs(row['joint_pos_deg'] - center) > range_guard:
            return 'range guard'
        return None

    return until


def trial_metrics(t, joint_deg, command_deg, joint_vel_deg_s, measured_torque_nm,
                  control_torque_nm, cap_nm, f, ramp_s, steady_s):
    """Steady-window response at f: gain, phase, distortion, torque use."""
    t = np.asarray(t, dtype=float)
    sel = (t >= ramp_s) & (t < ramp_s + steady_s)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if sel.sum() < 20 or t[-1] < ramp_s + steady_s - 2.0 * dt:
        return None   # Trial stopped before the steady window was complete.
    freqs = [f, 2.0 * f, 3.0 * f]
    Y, residual = fit_sinusoids(t[sel], np.asarray(joint_deg, float)[sel], freqs)
    U, _ = fit_sinusoids(t[sel], np.asarray(command_deg, float)[sel], freqs)
    G = Y[0] / U[0]
    error = (np.asarray(joint_deg, float) - np.asarray(command_deg, float))[sel]
    error = error - error.mean()
    control = np.abs(np.asarray(control_torque_nm, float)[sel])
    measured = np.abs(np.asarray(measured_torque_nm, float)[sel])
    cap = np.abs(np.asarray(cap_nm, float)[sel])
    velocity = np.asarray(joint_vel_deg_s, float)[sel]
    power = np.abs(np.asarray(measured_torque_nm, float)[sel] * np.radians(velocity))
    valid = np.isfinite(control) & np.isfinite(cap) & (cap > 0)
    saturated = control[valid] >= SATURATION_THRESHOLD * cap[valid]
    shortfall = (control[valid] - measured[valid]) > SHORTFALL_THRESHOLD * cap[valid]
    return {
        'command_amplitude_deg': float(abs(U[0])),
        'joint_amplitude_deg': float(abs(Y[0])),
        'gain': float(abs(G)),
        'phase_deg': float(math.degrees(np.angle(G))),
        'G_real': float(G.real),
        'G_imag': float(G.imag),
        'thd': float(np.sqrt(abs(Y[1]) ** 2 + abs(Y[2]) ** 2) / abs(Y[0])),
        'error_rms_percent': float(100.0 * np.sqrt(np.mean(error ** 2)) / abs(U[0])),
        'fit_residual_deg': residual,
        'saturation_fraction': float(saturated.mean()) if valid.any() else math.nan,
        'current_shortfall_fraction': float(shortfall.mean()) if valid.any() else math.nan,
        'peak_velocity_deg_s': float(np.max(np.abs(velocity))),
        'peak_torque_nm': float(np.nanmax(measured)),
        'peak_control_torque_nm': float(np.nanmax(control)),
        'peak_power_w': float(np.nanmax(power)),
        'num_samples': int(sel.sum()),
    }


def classify(metrics, reference, thresholds, ended_by=None):
    """Linear or not, relative to the small-signal reference at the same frequency."""
    if metrics is None:
        return {'linear': False, 'reason': ended_by or 'no data', 'all_reasons': ended_by or '',
                'gain_ratio': math.nan, 'phase_change_deg': math.nan}
    G = complex(metrics['G_real'], metrics['G_imag'])
    G0 = complex(reference['G_real'], reference['G_imag']) if reference else G
    ratio = abs(G) / abs(G0)
    dphase = math.degrees(np.angle(G / G0))
    fails = []
    if ended_by:
        fails.append(ended_by)
    if metrics['saturation_fraction'] >= thresholds['max_saturation']:
        fails.append('torque saturation')
    if metrics['current_shortfall_fraction'] >= thresholds['max_current_shortfall']:
        fails.append('voltage limit')
    if abs(ratio - 1.0) > thresholds['gain_tolerance']:
        fails.append('gain change')
    if abs(dphase) > thresholds['phase_tolerance_deg']:
        fails.append('phase change')
    if metrics['thd'] > thresholds['max_thd']:
        fails.append('distortion')
    return {'linear': not fails, 'reason': fails[0] if fails else 'linear',
            'all_reasons': ';'.join(fails), 'gain_ratio': ratio, 'phase_change_deg': dphase}


def thresholds_from(config):
    keys = ('gain_tolerance', 'phase_tolerance_deg', 'max_thd', 'max_saturation',
            'max_current_shortfall')
    defaults = asdict(EnvelopeConfig())
    return {k: float((config or {}).get(k, defaults[k])) for k in keys}


def metrics_from_rows(rows, f, ramp_s, steady_s):
    return trial_metrics(
        [r['t_local'] for r in rows], [r['joint_pos_deg'] for r in rows],
        [r['cmd_joint_deg'] for r in rows], [r['joint_vel_deg_s'] for r in rows],
        [r['joint_torque_nm'] for r in rows], [r['control_joint_torque_nm'] for r in rows],
        [r['max_torque_nm'] for r in rows], f, ramp_s, steady_s)


async def run_trial(session, cfg, plan, amplitude, role, trial_id, reference, thresholds):
    f = plan['frequency_hz']
    command, total = sine_command(cfg.center_deg, amplitude, f, plan['ramp_s'],
                                  plan['steady_s'], cfg.send_velocity_feedforward)
    info = {'trial_id': trial_id, 'frequency_hz': f, 'command_amplitude_deg': amplitude,
            'role': role, 'ramp_s': plan['ramp_s'], 'steady_s': plan['steady_s']}
    result = await session.run_segment(Segment('trial', total, command, info,
                                               until=make_guard(cfg.center_deg, cfg)))
    if result.ended_by:
        # Stopped mid-sinusoid: damp the motion, then return smoothly.
        session.log_event('trial_stopped', trial_id=trial_id, ended_by=result.ended_by)
        await session.brake(0.2, info={'trial_id': trial_id})
        here = await session.command(Command(position_deg=math.nan))
        await session.run_segment(min_jerk_move(here['joint_pos_deg'], cfg.center_deg, 60.0,
                                                phase='return', info={'trial_id': trial_id}))
    await session.run_segment(hold(cfg.rest_s, cfg.center_deg, phase='rest',
                                   info={'trial_id': trial_id}))
    metrics = metrics_from_rows(result.rows, f, plan['ramp_s'], plan['steady_s'])
    verdict = classify(metrics, reference, thresholds, result.ended_by)
    record = {**info, 'ended_by': result.ended_by, **(metrics or {}), **verdict}
    session.metadata.setdefault('trials', []).append(record)
    label = 'linear' if verdict['linear'] else f"NON-LINEAR ({verdict['all_reasons']})"
    joint = metrics['joint_amplitude_deg'] if metrics else math.nan
    print(f"  {f:5.1f} Hz  command {amplitude:7.3f} deg  joint {joint:7.3f} deg  {label}")
    return record, metrics


async def sweep_frequency(session, cfg, plan, next_id, thresholds):
    """Reference, geometric growth until non-linear or the cap, then bisection."""
    pred, cap = plan['predicted_command_deg'], plan['cap_command_deg']
    reference_amp = min(cfg.reference_fraction * pred, cap)
    record, reference = await run_trial(session, cfg, plan, reference_amp, 'reference',
                                        next_id(), None, thresholds)
    if not record['linear']:
        return {'frequency_hz': plan['frequency_hz'], 'last_linear_command_deg': None,
                'first_nonlinear_command_deg': reference_amp, 'stopped_by': 'reference failed'}
    last_linear, first_fail, stopped_by = reference_amp, None, None
    amplitude = max(cfg.start_fraction * pred, reference_amp * cfg.growth)
    while True:
        amplitude = min(amplitude, cap)
        record, _ = await run_trial(session, cfg, plan, amplitude, 'sweep', next_id(),
                                    reference, thresholds)
        if not record['linear']:
            first_fail = amplitude
            break
        last_linear = amplitude
        if amplitude >= cap * (1.0 - 1e-9):
            stopped_by = f"safety cap ({plan['cap_binding']})"
            break
        amplitude *= cfg.growth
    for _ in range(cfg.max_refine_steps if first_fail is not None else 0):
        if first_fail / last_linear <= cfg.refine_ratio:
            break
        amplitude = math.sqrt(last_linear * first_fail)
        record, _ = await run_trial(session, cfg, plan, amplitude, 'refine', next_id(),
                                    reference, thresholds)
        if record['linear']:
            last_linear = amplitude
        else:
            first_fail = amplitude
    return {'frequency_hz': plan['frequency_hz'], 'last_linear_command_deg': last_linear,
            'first_nonlinear_command_deg': first_fail, 'stopped_by': stopped_by}


# --- planning checks and main --------------------------------------------------------------
def check_plan(cfg):
    if not cfg.frequencies_hz or min(cfg.frequencies_hz) <= 0:
        raise SystemExit('frequencies_hz must be positive.')
    if max(cfg.frequencies_hz) > cfg.sample_rate_hz / 20.0:
        raise SystemExit('Keep every frequency below sample_rate_hz / 20 '
                         '(>= 20 samples per period).')
    if not 0.0 < cfg.reference_fraction < cfg.start_fraction <= 1.0:
        raise SystemExit('Need 0 < reference_fraction < start_fraction <= 1.')
    if cfg.growth <= 1.05:
        raise SystemExit('growth must be > 1.05.')
    if not 0.0 < cfg.speed_fraction < 1.0:
        raise SystemExit('speed_fraction must be in (0, 1).')
    lo, hi = device_config.JOINT_LIMITS_DEG
    reach = abs(cfg.center_deg) + cfg.max_amplitude_deg + 5.0
    if reach > min(abs(lo), abs(hi)):
        raise SystemExit(f'center_deg + max_amplitude_deg + 5 deg = {reach:.0f} deg exceeds '
                         f'JOINT_LIMITS_DEG {lo, hi}.')


async def main(args):
    cfg = EnvelopeConfig(**(QUICK if args.quick else {}))
    check_plan(cfg)
    J, b, source = reference_inertia(args.sim)
    torque_cap = device_config.MAX_JOINT_TORQUE_NM
    plans = [plan_frequency(cfg, f, J, b, torque_cap) for f in sorted(cfg.frequencies_hz)]
    worst = sum(worst_case_trials(cfg, p) * trial_duration(cfg, p) for p in plans)
    thresholds = thresholds_from(asdict(cfg))
    lines = ['Large-signal dynamic envelope (stiff PD + velocity feedforward)',
             f'  Kp = {cfg.kp_joint_nm_rad} N m/rad, Kd = {cfg.kd_joint_nms_rad} N m s/rad, '
             f'torque cap {torque_cap:g} N m, J = {J:.4g} kg m^2 ({source})',
             f'  Worst case ~{worst / 60:.1f} min; speed capped at '
             f'{cfg.speed_fraction * device_config.MAX_JOINT_VELOCITY_DEG_S:.0f} deg/s, '
             f'amplitude at {cfg.max_amplitude_deg:g} deg.']
    for p in plans:
        lines.append(f"  {p['frequency_hz']:5.1f} Hz: predicted limit "
                     f"{p['predicted_command_deg']:6.2f} deg command ({p['predicted_binding']}), "
                     f"cap {p['cap_command_deg']:6.2f} deg")
    lines.append('  Large, fast oscillations at full torque: keep hands, cables and people clear.')
    async with MoteusSession(EXPERIMENT, args, cfg.kp_joint_nm_rad, cfg.kd_joint_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        confirm(args, lines)
        session.metadata['envelope_protocol'] = {
            'J_planning_kg_m2': J, 'b_planning_nms_rad': b, 'J_planning_source': source,
            'torque_cap_nm': session.max_torque_nm,
            'speed_cap_deg_s': cfg.speed_fraction * device_config.MAX_JOINT_VELOCITY_DEG_S,
            'plans': plans, 'thresholds': thresholds,
            'saturation_threshold': SATURATION_THRESHOLD,
            'shortfall_threshold': SHORTFALL_THRESHOLD,
        }
        start = [hold(0.5, 0.0, phase='start')]
        if cfg.center_deg != 0.0:
            start.append(min_jerk_move(0.0, cfg.center_deg, 30.0, phase='start'))
        await session.run(start)
        counter = iter(range(10 ** 6))
        envelope = []
        for plan in plans:
            print(f"Frequency {plan['frequency_hz']:g} Hz")
            envelope.append(await sweep_frequency(session, cfg, plan, lambda: next(counter),
                                                  thresholds))
        session.metadata['envelope_online'] = envelope
        end = [min_jerk_move(cfg.center_deg, 0.0, 30.0, phase='end')] if cfg.center_deg else []
        await session.run(end + [hold(0.5, 0.0, phase='end')])


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
