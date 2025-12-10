#!/usr/bin/env python3
"""
Step input characterization for a single-DOF joint driven by a moteus controller.

Implements the "Step Input Experiment: Inertia, Viscous Damping, and Kinetic
Friction" procedure from the OpenWrist paper. The script commands a square-wave
position input with a soft host-side PD (torque) controller, logs the measured
response, and estimates inertia J, viscous damping b, and kinetic friction fk via
logarithmic decrement.

Notes:
  - Position/velocity from moteus are in revolutions and rev/s. They are converted
    to radians internally.
  - Kp/Kd are host-side gains in Nm/rad and Nm*s/rad, applied as torque commands.
  - To match the paper, use a soft Kp with Kd=0 for the active joint, and stiffen
    any other joints externally (mechanical lock) so the dynamics are dominated by
    the single DOF.
"""

import argparse
import asyncio
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import moteus
import numpy as np


def rev_to_rad(value: float) -> float:
    return value * 2.0 * math.pi


def rad_to_rev(value: float) -> float:
    return value / (2.0 * math.pi)


def deg_to_rad(value: float) -> float:
    return math.radians(value)


def sign_or_prev(values: np.ndarray) -> np.ndarray:
    """Replace zeros in sign vector with the previous non-zero entry."""
    result = values.copy()
    for i in range(1, len(result)):
        if result[i] == 0.0:
            result[i] = result[i - 1]
    if len(result) and result[0] == 0.0:
        # If the first element is zero, propagate the first non-zero sign forward.
        for val in result:
            if val != 0.0:
                result[0] = val
                break
    return result


def find_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of peaks and valleys for an oscillatory signal."""
    if len(signal) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)
    dx = np.diff(signal)
    sign = sign_or_prev(np.sign(dx))
    changes = np.diff(sign)
    peaks = np.where(changes < 0)[0] + 1
    valleys = np.where(changes > 0)[0] + 1
    return peaks, valleys


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return None


def plot_step_response(time_s: np.ndarray, target: np.ndarray, position: np.ndarray, torque_cmd: np.ndarray, torque_meas: np.ndarray, save_path: Path = None) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(time_s, np.rad2deg(target), label="target (deg)", linewidth=1.5)
    ax1.plot(time_s, np.rad2deg(position), label="position (deg)", linewidth=1.0)
    ax1.set_ylabel("Position (deg)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_s, torque_cmd, label="command torque (Nm)", linewidth=1.0)
    ax2.plot(time_s, torque_meas, label="measured torque (Nm)", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (Nm)")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def extrema_period(times: np.ndarray, indices: np.ndarray) -> float:
    """Average period between successive extrema indices."""
    if len(indices) < 2:
        return float("nan")
    return float(np.mean(np.diff(times[indices])))


def estimate_parameters(
    time_s: np.ndarray,
    target_rad: np.ndarray,
    position_rad: np.ndarray,
    kp: float,
    step_rad: float,
) -> Dict[str, float]:
    """Estimate J, b, fk from logged step responses."""
    if kp <= 0.0:
        raise ValueError("Kp must be > 0 for parameter estimation.")

    # Identify command transitions.
    step_threshold = 0.25 * abs(step_rad)
    change_indices = np.where(np.abs(np.diff(target_rad)) > step_threshold)[0]
    if len(change_indices) == 0:
        raise ValueError("No command transitions found in log; cannot estimate.")

    j_estimates: List[float] = []
    b_estimates: List[float] = []
    fk_estimates: List[float] = []

    for idx, start_idx in enumerate(change_indices):
        end_idx = change_indices[idx + 1] if idx + 1 < len(change_indices) else len(target_rad) - 1
        seg_slice = slice(start_idx + 1, end_idx)
        if seg_slice.start >= seg_slice.stop:
            continue
        segment_time = time_s[seg_slice] - time_s[seg_slice.start]
        seg_target = target_rad[start_idx + 1]  # after the transition
        error = position_rad[seg_slice] - seg_target

        peaks, valleys = find_extrema(error)
        ordered_indices = np.sort(np.concatenate([peaks, valleys]))
        if len(ordered_indices) < 4:
            continue

        extrema_vals = error[ordered_indices]
        extrema_times = segment_time[ordered_indices]

        # Logarithmic decrement (phi) over sliding windows of 4 extrema.
        phis: List[float] = []
        for i in range(2, len(extrema_vals) - 1):
            num = extrema_vals[i + 1] - extrema_vals[i - 1]
            den = extrema_vals[i] - extrema_vals[i - 2]
            if den == 0.0:
                continue
            ratio = abs(num / den)
            if ratio <= 0.0:
                continue
            phi = -(1.0 / math.pi) * math.log(ratio)
            if math.isfinite(phi) and phi > 0:
                phis.append(phi)

        if not phis:
            continue

        phi_avg = float(np.mean(phis))
        damping_ratio = math.sqrt(phi_avg * phi_avg / (phi_avg * phi_avg + 1.0))

        peak_period = extrema_period(extrema_times, peaks)
        valley_period = extrema_period(extrema_times, valleys)
        period_candidates = [p for p in [peak_period, valley_period] if math.isfinite(p) and p > 0.0]
        if not period_candidates:
            continue
        period = float(np.mean(period_candidates))
        if period <= 0.0:
            continue

        wd = 2.0 * math.pi / period
        if damping_ratio >= 1.0:
            continue
        wn = wd / math.sqrt(1.0 - damping_ratio * damping_ratio)

        j_val = kp / (wn * wn)
        b_val = 2.0 * j_val * damping_ratio * wn

        exp_term = math.exp(-phi_avg * math.pi)
        xk_terms: List[float] = []
        for i in range(1, len(extrema_vals) - 1):
            num = extrema_vals[i + 1] - extrema_vals[i] + exp_term * (extrema_vals[i] - extrema_vals[i - 1])
            denom = 2.0 * ((-1) ** i) * (exp_term + 1.0)
            if denom == 0.0:
                continue
            xk_terms.append(abs(num / denom))

        if not xk_terms:
            continue

        fk_val = kp * float(np.mean(xk_terms))

        j_estimates.append(j_val)
        b_estimates.append(b_val)
        fk_estimates.append(fk_val)

    def avg_and_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.mean(values)), float(np.std(values))

    j_mean, j_std = avg_and_std(j_estimates)
    b_mean, b_std = avg_and_std(b_estimates)
    fk_mean, fk_std = avg_and_std(fk_estimates)

    return {
        "J_mean": j_mean,
        "J_std": j_std,
        "b_mean": b_mean,
        "b_std": b_std,
        "fk_mean": fk_mean,
        "fk_std": fk_std,
        "samples_used": len(j_estimates),
    }


def _get_value(state: moteus.Result, register: moteus.Register) -> float:
    return state.values.get(register, float("nan"))


async def collect_step_response(args) -> List[Dict[str, float]]:
    controller = moteus.Controller(id=args.id)
    await controller.set_stop()

    # Pre-read for initial state.
    last_state = await controller.query()
    data: List[Dict[str, float]] = []

    dt = 1.0 / args.rate_hz
    dwell = args.dwell
    base = deg_to_rad(args.offset_deg)
    step = deg_to_rad(args.step_deg)
    first = base
    second = base + step
    targets = [first]
    for _ in range(args.cycles):
        targets.extend([second, first])

    start_time = time.monotonic()
    next_tick = start_time

    for target in targets:
        segment_end = time.monotonic() + dwell
        while time.monotonic() < segment_end:
            now = time.monotonic()
            # PD torque using previous measurement (one-cycle lag, fine at high rate).
            pos_rev = _get_value(last_state, moteus.Register.POSITION)
            vel_rev_s = _get_value(last_state, moteus.Register.VELOCITY)
            pos_rad = rev_to_rad(pos_rev) if math.isfinite(pos_rev) else float("nan")
            vel_rad_s = rev_to_rad(vel_rev_s) if math.isfinite(vel_rev_s) else 0.0

            pos_err = target - pos_rad if math.isfinite(pos_rad) else 0.0
            vel_err = -vel_rad_s
            torque_cmd = args.kp * pos_err + args.kd * vel_err
            torque_cmd = float(np.clip(torque_cmd, -args.max_torque, args.max_torque))

            state = await controller.set_position(
                position=math.nan,
                velocity=None,
                feedforward_torque=torque_cmd,
                kp_scale=0.0,
                kd_scale=0.0,
                maximum_torque=args.max_torque,
                query=True,
            )

            timestamp = time.monotonic() - start_time
            meas_pos_rev = _get_value(state, moteus.Register.POSITION)
            meas_vel_rev_s = _get_value(state, moteus.Register.VELOCITY)
            meas_torque = _get_value(state, moteus.Register.TORQUE)

            data.append(
                {
                    "time_s": timestamp,
                    "target_rad": target,
                    "position_rad": rev_to_rad(meas_pos_rev) if math.isfinite(meas_pos_rev) else float("nan"),
                    "velocity_rad_s": rev_to_rad(meas_vel_rev_s) if math.isfinite(meas_vel_rev_s) else float("nan"),
                    "measured_torque_Nm": meas_torque,
                    "command_torque_Nm": torque_cmd,
                    "q_current_A": _get_value(state, moteus.Register.Q_CURRENT),
                    "d_current_A": _get_value(state, moteus.Register.D_CURRENT),
                    "voltage_V": _get_value(state, moteus.Register.VOLTAGE),
                    "temperature_C": _get_value(state, moteus.Register.TEMPERATURE),
                }
            )

            last_state = state
            next_tick += dt
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    await controller.set_stop()
    return data


def write_log(path: Path, metadata: Dict[str, float], rows: Iterable[Dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_s",
                "target_rad",
                "position_rad",
                "velocity_rad_s",
                "measured_torque_Nm",
                "command_torque_Nm",
                "q_current_A",
                "d_current_A",
                "voltage_V",
                "temperature_C",
            ],
        )
        f.write("# step_input_experiment\n")
        f.write("# " + ",".join(f"{k}={v}" for k, v in metadata.items()) + "\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(filter(lambda l: not l.startswith("#"), f))
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def summarize(results: Dict[str, float]) -> str:
    parts = [
        f"Inertia J: {results['J_mean']:.5f} kg*m^2 (std {results['J_std']:.5f})",
        f"Viscous damping b: {results['b_mean']:.5f} Nm*s/rad (std {results['b_std']:.5f})",
        f"Kinetic friction fk: {results['fk_mean']:.5f} Nm (std {results['fk_std']:.5f})",
        f"Segments used: {results['samples_used']}",
    ]
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step input characterization with moteus.")
    parser.add_argument("--id", type=int, default=1, help="moteus controller id")
    parser.add_argument("--rate-hz", type=float, default=500.0, help="command/query rate (Hz)")
    parser.add_argument("--dwell", type=float, default=1.5, help="seconds to hold each step level")
    parser.add_argument("--cycles", type=int, default=3, help="number of square-wave cycles (step up + down counts as one cycle)")
    parser.add_argument("--step-deg", type=float, default=20.0, help="step-to-step amplitude in degrees")
    parser.add_argument("--offset-deg", type=float, default=0.0, help="lower level of square wave in degrees (bias for workspace coverage)")
    parser.add_argument("--kp", type=float, default=5.0, help="host-side proportional gain (Nm/rad)")
    parser.add_argument("--kd", type=float, default=0.0, help="host-side derivative gain (Nm*s/rad)")
    parser.add_argument("--max-torque", type=float, default=2.0, help="torque limit in Nm")
    parser.add_argument("--log", type=Path, default=Path("step_input_log.csv"), help="path to write/read log csv")
    parser.add_argument("--analyze-only", action="store_true", help="skip hardware run, analyze existing log at --log")
    parser.add_argument("--plot", action="store_true", help="show time-series plots after run/analysis")
    parser.add_argument("--plot-file", type=Path, help="save plots to this path (png, pdf, etc)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        rows = read_log(args.log)
    else:
        print(f"Running step input experiment on moteus id {args.id} ...")
        rows = asyncio.run(collect_step_response(args))
        metadata = {
            "kp_Nm_per_rad": args.kp,
            "kd_Nm_s_per_rad": args.kd,
            "rate_hz": args.rate_hz,
            "dwell_s": args.dwell,
            "step_deg": args.step_deg,
            "offset_deg": args.offset_deg,
            "cycles": args.cycles,
            "max_torque_Nm": args.max_torque,
        }
        write_log(args.log, metadata, rows)
        print(f"Wrote log to {args.log}")

    time_s = np.array([r["time_s"] for r in rows])
    target_rad = np.array([r["target_rad"] for r in rows])
    position_rad = np.array([r["position_rad"] for r in rows])

    results = estimate_parameters(
        time_s=time_s,
        target_rad=target_rad,
        position_rad=position_rad,
        kp=args.kp,
        step_rad=deg_to_rad(args.step_deg),
    )

    print(summarize(results))
    if args.plot or args.plot_file:
        torque_cmd = np.array([r["command_torque_Nm"] for r in rows])
        torque_meas = np.array([r["measured_torque_Nm"] for r in rows])
        plot_step_response(time_s, target_rad, position_rad, torque_cmd, torque_meas, args.plot_file)


if __name__ == "__main__":
    main()
