"""Rendered impedance spectrum: the stiffness and damping the wrist actually displays.

A target spring-damper Z_d = K + B s is rendered about a fixed pose, either by
the p4wrist 500 Hz loop ('rt' mode: torque 2 SYNC periods after its sample)
or by this host from the previous measurement ('host_<rate>' modes, a PC-side
haptic loop, which adds its own period and sample age to that delay). A periodic torque multisine
is added on top. The measured compliance C = Theta / T_d gives the displayed
impedance

    Z(jw) = 1 / C(jw) = -J w^2 + j w b + Z_r(jw),
    Z_r(jw) = (K + j w B) e^{-j w T_eff}   (delayed rendering loop),

so the rendered stiffness and damping can be read off per frequency. Delay
turns stiffness into negative damping, B_r ~= B - K T_eff, which predicts the
stability limit K_crit ~= (b + B) / T_eff that the Z-width experiment
measures. A near-free target (weak centering spring) gives the intrinsic
impedance of the device: what an operator feels in free space.

Each target is checked for stability first (torque pulse, same rule as the
Z-width experiment) and skipped if it rings; unstable targets never get the
multisine. impedance_fidelity_experiment_analysis.py does the analysis.

Run:  python impedance_fidelity_experiment.py [--sim] [--quick]
"""
import math
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

import device_config
from experiment_common import (Command, WristSession, Segment, confirm, hold, latest_summary,
                               make_arg_parser, min_jerk_move, run_cli)
from schroeder_bandwidth_experiment import (Multisine, excited_bins, reduce_crest_factor,
                                            schroeder_phases, synthesize)
from zwidth_experiment import classify_trial, make_stop_check, noise_floor

EXPERIMENT = 'impedance'
SIM_PARAMS = None   # Tests may inject a p4_sim.SimParams


# --- USER CONFIGURATION -----------------------------------------------------------
@dataclass
class ImpedanceConfig:
    # (mode, K N m/rad, B N m s/rad). 'rt' or 'host_<rate_hz>'. The first
    # near-free target (small K, B = 0) measures the intrinsic device impedance;
    # a matched (K, B) in both loops compares firmware and host rendering.
    targets: tuple = (('rt', 5.0, 0.0),
                      ('rt', 20.0, 0.2),
                      ('rt', 60.0, 0.5),
                      ('host_250', 5.0, 0.0),
                      ('host_250', 20.0, 0.2),
                      ('host_250', 30.0, 0.3),    # Stiffest: a damping error dB biases the
                                                    # delay by dB / K
                      ('drive', 20.0, 0.2),       # The pucks' own loops (needs a drive-gain
                      ('drive', 100.0, 0.5))      # calibration; B > 0)
    center_deg: float = 0.0
    period_s: float = 4.0                  # Frequency resolution = 1 / period
    periods: int = 5                       # The first is discarded (transient)
    freq_min_hz: float = 0.5
    freq_max_hz: float = 20.0              # Below rate/10 of the slowest host loop (250 Hz)
    num_lines: int = 30
    line_displacement_deg: float = 0.25    # Target motion per line (displacement-flat design)
    max_motion_deg: float = 4.0            # Predicted peak motion of the whole multisine
    torque_budget_fraction: float = 0.35   # Peak disturbance torque / trial torque cap
    crest_iterations: int = 40
    torque_cap_fraction: float = 0.6       # Of the device torque cap, during targets
    min_center_k_nm_rad: float = 2.0       # Below this the joint drifts; not allowed
    zwidth_margin: float = 0.7             # Skip targets above this x measured Z-width K_max
    safe_k_nm_rad: float = 20.0            # Hold between targets (loop PD)
    safe_b_nms_rad: float = 0.3
    settle_s: float = 0.4
    precheck_pulse_nm: float = 1.0         # Stability pre-check: torque pulse ...
    precheck_pulse_s: float = 0.01         # ... this long (an impulse comparable to the multisine)...
    precheck_observe_s: float = 1.0        # ... then watch the response
    brake_s: float = 0.2
    max_deviation_deg: float = 10.0        # Stop a target beyond this
    max_velocity_deg_s: float = 600.0
    return_speed_deg_s: float = 30.0
    sample_rate_hz: float = 500.0


QUICK = dict(targets=(('rt', 5.0, 0.0), ('rt', 20.0, 0.2),
                      ('host_250', 5.0, 0.0), ('host_250', 20.0, 0.2), ('drive', 20.0, 0.2)),
             period_s=2.0, periods=3, freq_min_hz=1.0, num_lines=20)

PRECHECK_START_S = 0.2   # Target active this long before the pulse


def parse_target(index, target, cfg):
    mode, K, B = target
    K, B = float(K), float(B)
    if mode in ('rt', 'drive'):
        renderer, rate = mode, cfg.sample_rate_hz
        if mode == 'drive' and not B > 0:
            raise SystemExit('Drive-loop targets need B > 0 (its velocity loop makes the torque).')
    elif isinstance(mode, str) and mode.startswith('host_'):
        try:
            rate = float(mode.split('_', 1)[1])
        except ValueError:
            raise SystemExit(f'Bad mode {mode!r}; use host_<rate_hz>, e.g. host_250.')
        renderer = 'host'
    else:
        raise SystemExit(f'Unknown mode {mode!r}; use "rt", "drive" or "host_<rate_hz>".')
    return {'target_id': index, 'mode': mode, 'renderer': renderer, 'rate_hz': rate,
            'k_nm_rad': K, 'b_nms_rad': B}


# --- excitation -------------------------------------------------------------------------
def design_excitation(cfg, K, B, cap_nm):
    """Torque multisine whose displacement per line is roughly constant.

    Torque per line is theta_line |Z_model| (Z_model from the target and the
    nominal inertia/damping), capped per line, then scaled so the peak torque
    stays within the budget and the predicted peak motion within max_motion.
    """
    spec = SimpleNamespace(freq_min_hz=cfg.freq_min_hz, freq_max_hz=cfg.freq_max_hz,
                           period_s=cfg.period_s, num_lines=cfg.num_lines)
    bins = excited_bins(spec)
    freqs = bins / cfg.period_s
    w = 2.0 * np.pi * freqs
    J = device_config.NOMINAL_JOINT_INERTIA_KG_M2
    b = device_config.NOMINAL_JOINT_DAMPING_NMS_RAD
    z_model = K + 1j * w * (B + b) - J * w ** 2
    n = 1 << int(math.ceil(math.log2(max(1024, 20 * cfg.freq_max_hz * cfg.period_s))))
    dt = cfg.period_s / n
    budget = cfg.torque_budget_fraction * cap_nm
    line_cap = budget / math.sqrt(len(freqs))
    amps = np.minimum(math.radians(cfg.line_displacement_deg) * np.abs(z_model), line_cap)
    phases = schroeder_phases(amps)
    phases, crest = reduce_crest_factor(bins, amps, phases, n, cfg.crest_iterations)
    torque = synthesize(bins, amps, phases, n)
    motion = synthesize(bins, amps / np.abs(z_model), phases - np.angle(z_model), n)
    scale = min(1.0, budget / np.max(np.abs(torque)),
                math.radians(cfg.max_motion_deg) / np.max(np.abs(motion)))
    amps = amps * scale
    torque = torque * scale
    motion = motion * scale
    slope = synthesize(bins, amps * w, phases + np.pi / 2, n)
    score = np.abs(torque) / np.max(np.abs(torque)) + np.abs(slope) / np.max(np.abs(slope))
    t_offset = float(np.argmin(score) * dt)
    return {
        'bins': bins.tolist(),
        'freqs_hz': freqs.tolist(),
        'amplitudes_nm': amps.tolist(),
        'phases_rad': np.mod(phases, 2 * np.pi).tolist(),
        't_offset_s': t_offset,
        'period_s': cfg.period_s,
        'periods': cfg.periods,
        'crest_factor': float(crest),
        'peak_torque_nm': float(np.max(np.abs(torque))),
        'predicted_peak_motion_deg': float(np.degrees(np.max(np.abs(motion)))),
        'predicted_line_motion_deg': np.degrees(amps / np.abs(z_model)).tolist(),
    }


def disturbance(design):
    """tau_d(t) in N m; the analysis rebuilds it from the stored design."""
    signal = Multisine(design['freqs_hz'], design['amplitudes_nm'], design['phases_rad'],
                       design['t_offset_s'])
    return lambda t: signal(t)[0]


# --- rendering ----------------------------------------------------------------------------
def render_command(target, theta0, cap, tau_d=None, pulse=None):
    """Command factory: target impedance plus disturbance torque (or a pulse)."""
    K, B = target['k_nm_rad'], target['b_nms_rad']

    def extra(t):
        if tau_d is not None:
            return tau_d(t)
        if pulse is not None and pulse[0] <= t < pulse[1]:
            return pulse[2]
        return 0.0

    if target['renderer'] in ('rt', 'drive'):
        def command(t, _):
            return Command(position_deg=theta0, torque_nm=extra(t), max_torque_nm=cap)
    else:
        def command(t, last):
            torque = extra(t)
            if last is not None:
                torque += (-K * math.radians(last['joint_pos_deg'] - theta0)
                           - B * math.radians(last['joint_vel_deg_s']))
            torque = max(-cap, min(cap, torque))
            return Command(position_deg=math.nan, torque_nm=torque, kp_scale=0.0,
                           kd_scale=0.0, max_torque_nm=cap)
    return command


def stop_check(theta0, cap, cfg, watch_after):
    limits = SimpleNamespace(pulse_start_s=watch_after, pulse_width_s=0.0,
                             max_deviation_deg=cfg.max_deviation_deg,
                             max_velocity_deg_s=cfg.max_velocity_deg_s)
    return make_stop_check(theta0, cap, limits)


async def contain(session, cfg, target, theta0):
    """Brake, restore the safe hold, and return to the center pose."""
    info = {'target_id': target['target_id']}
    await session.brake(cfg.brake_s, info=info)
    if target['renderer'] == 'drive':
        await session.use_pc_loop()
    if target['renderer'] in ('rt', 'drive'):
        await session.set_gains(cfg.safe_k_nm_rad, cfg.safe_b_nms_rad)
    here = await session.command(Command(position_deg=math.nan))
    await session.run_segment(min_jerk_move(here['joint_pos_deg'], theta0,
                                            cfg.return_speed_deg_s, phase='return', info=info))


async def run_target(session, cfg, target, theta0, cap):
    # Segment info is merged into logged rows: never use logged column names
    # such as 'mode' (the logged mode column) here.
    info = {'target_id': target['target_id'], 'render_mode': target['mode'],
            'virtual_k_nm_rad': target['k_nm_rad'], 'virtual_b_nms_rad': target['b_nms_rad'],
            'render_rate_hz': target['rate_hz']}
    settle = await session.run_segment(hold(cfg.settle_s, theta0, phase='settle', info=info))
    noise_deg, noise_nm = noise_floor([r['t_local'] for r in settle.rows],
                                      [r['joint_pos_deg'] - theta0 for r in settle.rows],
                                      [r['joint_torque_nm'] for r in settle.rows])
    if target['renderer'] == 'rt':
        await session.set_gains(target['k_nm_rad'], target['b_nms_rad'])
    elif target['renderer'] == 'drive':
        await session.use_drive_loop(target['k_nm_rad'], target['b_nms_rad'])

    pulse_end = PRECHECK_START_S + cfg.precheck_pulse_s
    planned = pulse_end + cfg.precheck_observe_s
    check = await session.run_segment(Segment(
        'precheck', planned,
        render_command(target, theta0, cap,
                       pulse=(PRECHECK_START_S, pulse_end, cfg.precheck_pulse_nm)),
        info, until=stop_check(theta0, cap, cfg, pulse_end + 0.05),
        rate_hz=target['rate_hz']))
    verdict = classify_trial(
        [r['t_local'] for r in check.rows], [r['joint_pos_deg'] - theta0 for r in check.rows],
        [r['joint_torque_nm'] for r in check.rows], pulse_end_s=pulse_end,
        planned_end_s=planned, noise_deg=noise_deg, noise_nm=noise_nm, cap_nm=cap,
        ended_by=check.ended_by)
    record = {**target, 'precheck_stable': bool(verdict['stable']),
              'precheck_reason': verdict['reason'], 'skipped': None, 'ended_by': None}
    if not verdict['stable']:
        record['skipped'] = f"unstable at pre-check ({verdict['reason']})"
        print(f"  target {target['target_id']}: {target['mode']} K={target['k_nm_rad']:g} "
              f"B={target['b_nms_rad']:g} SKIPPED: {record['skipped']}")
        await contain(session, cfg, target, theta0)
        return record

    design = target['design']
    excite = await session.run_segment(Segment(
        'excite', design['period_s'] * design['periods'],
        render_command(target, theta0, cap, tau_d=disturbance(design)), info,
        until=stop_check(theta0, cap, cfg, 0.05), rate_hz=target['rate_hz']))
    record['ended_by'] = excite.ended_by
    status = f'stopped ({excite.ended_by})' if excite.ended_by else 'measured'
    print(f"  target {target['target_id']}: {target['mode']} K={target['k_nm_rad']:g} "
          f"B={target['b_nms_rad']:g} {status}")
    await contain(session, cfg, target, theta0)
    return record


# --- planning -------------------------------------------------------------------------------
def zwidth_limits(simulated):
    """Largest stable K per mode from the newest same-kind Z-width analysis."""
    found = latest_summary('zwidth', simulated)
    if found is None:
        return None, {}
    run_dir, summary = found
    limits = {}
    for key, value in summary.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        for tag in ('b0', 'bmin'):
            prefix, suffix = 'zwidth_k_max_', f'_{tag}_nm_rad'
            if key.startswith(prefix) and key.endswith(suffix):
                limits.setdefault(key[len(prefix):-len(suffix)], {})['b0'] = float(value)
        if key.startswith('zwidth_k_max_') and key.endswith('_nm_rad') and \
                not key.endswith(('_b0_nm_rad', '_bmin_nm_rad')):
            limits.setdefault(key[len('zwidth_k_max_'):-len('_nm_rad')], {})['any_b'] = float(value)
    return run_dir, limits


def check_plan(cfg, targets, cap):
    if not targets:
        raise SystemExit('No targets configured.')
    lo, hi = device_config.JOINT_LIMITS_DEG
    if not (lo < cfg.center_deg - cfg.max_deviation_deg and
            cfg.center_deg + cfg.max_deviation_deg < hi):
        raise SystemExit('center_deg +/- max_deviation_deg must lie inside JOINT_LIMITS_DEG.')
    if cfg.max_velocity_deg_s >= device_config.MAX_JOINT_VELOCITY_DEG_S:
        raise SystemExit('max_velocity_deg_s must be below MAX_JOINT_VELOCITY_DEG_S.')
    if cfg.freq_min_hz * cfg.period_s < 1.0:
        raise SystemExit('freq_min_hz must be at least 1 / period_s.')
    if cfg.periods < 2:
        raise SystemExit('Need at least 2 periods (the first is discarded).')
    if cfg.max_motion_deg >= cfg.max_deviation_deg:
        raise SystemExit('max_motion_deg must stay below max_deviation_deg.')
    if cfg.precheck_pulse_nm >= cap:
        raise SystemExit('precheck_pulse_nm must be below the trial torque cap.')
    for t in targets:
        if t['k_nm_rad'] < cfg.min_center_k_nm_rad:
            raise SystemExit(f"Target {t['target_id']}: K = {t['k_nm_rad']} is below "
                             f'min_center_k_nm_rad ({cfg.min_center_k_nm_rad}); the joint would '
                             'drift. Use a weak centering spring for free-space targets.')
        if t['b_nms_rad'] < 0:
            raise SystemExit('Negative virtual damping is not supported.')
        if cfg.freq_max_hz > t['rate_hz'] / 10.0:
            raise SystemExit(f"freq_max_hz must stay below rate/10 for {t['mode']}.")
        if t['renderer'] == 'host' and not 50.0 <= t['rate_hz'] <= 4000.0:
            raise SystemExit(f"Host rate {t['rate_hz']} Hz is outside 50-4000 Hz.")


async def main(args):
    cfg = ImpedanceConfig(**(QUICK if args.quick else {}))
    targets = [parse_target(i, t, cfg) for i, t in enumerate(cfg.targets)]
    cap = cfg.torque_cap_fraction * device_config.MAX_JOINT_TORQUE_NM
    check_plan(cfg, targets, cap)
    zw_run, zw = zwidth_limits(args.sim)
    for t in targets:
        t['design'] = design_excitation(cfg, t['k_nm_rad'], t['b_nms_rad'], cap)
        mode_limits = zw.get(t['mode'], {})
        limit = mode_limits.get('b0') if t['b_nms_rad'] == 0 else mode_limits.get('any_b')
        t['zwidth_limit_nm_rad'] = limit
        t['planned_skip'] = (f'K above {cfg.zwidth_margin:g} x Z-width K_max ({limit:.4g})'
                             if limit is not None and t['k_nm_rad'] > cfg.zwidth_margin * limit
                             else None)
    per_target = (cfg.settle_s + PRECHECK_START_S + cfg.precheck_pulse_s + cfg.precheck_observe_s
                  + cfg.period_s * cfg.periods + cfg.brake_s + 0.5)
    theta0 = cfg.center_deg
    async with WristSession(EXPERIMENT, args, cfg.safe_k_nm_rad, cfg.safe_b_nms_rad,
                             config=asdict(cfg), rate_hz=cfg.sample_rate_hz,
                             sim_params=SIM_PARAMS) as session:
        if any(t['renderer'] == 'drive' for t in targets):
            ok, why = session.drive_loop_available()
            if not ok:
                print(f'WARNING: skipping drive-loop targets: {why}')
                for t in targets:
                    if t['renderer'] == 'drive':
                        t['planned_skip'] = f'drive loop unavailable: {why}'
        lines = ['Rendered impedance spectrum (target impedance + torque multisine)',
                 f'  {len(targets)} targets, {cfg.periods} x {cfg.period_s:g} s each, '
                 f'{len(targets[0]["design"]["freqs_hz"])} lines {cfg.freq_min_hz:g}-'
                 f'{cfg.freq_max_hz:g} Hz (~{len(targets) * per_target / 60:.1f} min)',
                 f'  Trial torque cap {cap:.2f} N m; stops at {cfg.max_deviation_deg} deg, '
                 f'{cfg.max_velocity_deg_s} deg/s, or torque chatter']
        for t in targets:
            d = t['design']
            note = f"  SKIP: {t['planned_skip']}" if t['planned_skip'] else ''
            lines.append(f"  [{t['target_id']}] {t['mode']:<10} K={t['k_nm_rad']:<6g} "
                         f"B={t['b_nms_rad']:<5g} peak torque {d['peak_torque_nm']:.2f} N m, "
                         f"motion ~{d['predicted_peak_motion_deg']:.1f} deg{note}")
        if zw_run is None:
            lines.append('  No Z-width results found: every target relies on its stability '
                         'pre-check (run zwidth_experiment.py first for an extra safeguard).')
        lines.append('  The HANDLE MUST BE FREE (no hand on it); targets that ring at the '
                     'pre-check are skipped.')
        confirm(args, lines)
        session.metadata['impedance_protocol'] = {
            'theta0_deg': theta0,
            'trial_torque_cap_nm': cap,
            'zwidth_source': str(zw_run) if zw_run else None,
            'precheck': {'start_s': PRECHECK_START_S, 'pulse_s': cfg.precheck_pulse_s,
                         'pulse_nm': cfg.precheck_pulse_nm, 'observe_s': cfg.precheck_observe_s},
        }
        await session.run_segment(hold(0.5, 0.0, phase='start'))
        if theta0 != 0.0:
            await session.run_segment(min_jerk_move(0.0, theta0, cfg.return_speed_deg_s,
                                                    phase='start'))
        records = []
        for target in targets:
            if target['planned_skip']:
                print(f"  target {target['target_id']}: SKIPPED ({target['planned_skip']})")
                records.append({**target, 'precheck_stable': None, 'precheck_reason': None,
                                'skipped': target['planned_skip'], 'ended_by': None})
                continue
            records.append(await run_target(session, cfg, target, theta0, cap))
            session.metadata['targets'] = records
        session.metadata['targets'] = records
        if theta0 != 0.0:
            await session.run_segment(min_jerk_move(theta0, 0.0, cfg.return_speed_deg_s,
                                                    phase='end'))
        await session.run_segment(hold(0.5, 0.0, phase='end'))


if __name__ == '__main__':
    parser = make_arg_parser(__doc__.splitlines()[0])
    run_cli(lambda: main(parser.parse_args()))
