#!/usr/bin/env python3
"""
Ramp input characterization for static friction on a single moteus-driven DOF.

Implements the ramp experiment from the OpenWrist paper (Section 3.3):
  - Command a slow position ramp (default 5 deg over 2 s) with a soft host-side
    PD torque controller (Kd defaults to 0).
  - Hold for a pause duration, then ramp in the opposite direction, repeating for
    a set number of ramps.
  - Static friction is inferred as the commanded torque one control cycle before
    the measured velocity first exceeds a small threshold at each ramp start.
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


def _get_value(state: moteus.Result, register: moteus.Register) -> float:
    return state.values.get(register, float("nan"))


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return None


def plot_ramp(time_s: np.ndarray, target: np.ndarray, position: np.ndarray, torque_cmd: np.ndarray, torque_meas: np.ndarray, save_path: Path = None) -> None:
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


async def collect_ramp_response(args) -> List[Dict[str, float]]:
    controller = moteus.Controller(id=args.id)
    await controller.set_stop()
    last_state = await controller.query()

    dt = 1.0 / args.rate_hz
    ramp_step = deg_to_rad(args.ramp_deg)
    ramp_time = args.ramp_time
    hold_time = args.hold_time
    ramps = args.ramps
    start_target = deg_to_rad(args.start_deg)
    direction = 1.0 if args.first_up else -1.0

    ramp_index = 0
    phase_start = time.monotonic()
    phase = "ramp"
    ramp_start_target = start_target
    data: List[Dict[str, float]] = []
    t0 = phase_start
    next_tick = phase_start

    while ramp_index < ramps:
        now = time.monotonic()
        elapsed_global = now - t0
        phase_elapsed = now - phase_start

        if phase == "ramp":
            alpha = min(1.0, phase_elapsed / ramp_time)
            target = ramp_start_target + direction * ramp_step * alpha
            if alpha >= 1.0:
                phase = "hold"
                phase_start = now
        else:  # hold
            target = ramp_start_target + direction * ramp_step
            if phase_elapsed >= hold_time:
                ramp_index += 1
                direction *= -1.0
                ramp_start_target = target
                phase = "ramp"
                phase_start = now
                continue  # recompute target next loop

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

        data.append(
            {
                "time_s": elapsed_global,
                "target_rad": target,
                "position_rad": rev_to_rad(_get_value(state, moteus.Register.POSITION)),
                "velocity_rad_s": rev_to_rad(_get_value(state, moteus.Register.VELOCITY)),
                "measured_torque_Nm": _get_value(state, moteus.Register.TORQUE),
                "command_torque_Nm": torque_cmd,
                "q_current_A": _get_value(state, moteus.Register.Q_CURRENT),
                "d_current_A": _get_value(state, moteus.Register.D_CURRENT),
                "voltage_V": _get_value(state, moteus.Register.VOLTAGE),
                "temperature_C": _get_value(state, moteus.Register.TEMPERATURE),
                "ramp_index": ramp_index,
                "direction": direction,
                "phase": phase,
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
                "ramp_index",
                "direction",
                "phase",
            ],
        )
        f.write("# ramp_input_experiment\n")
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


def estimate_static_friction(
    rows: List[Dict[str, float]],
    vel_threshold_rad_s: float,
) -> Dict[str, float]:
    friction_samples: List[Tuple[int, float]] = []
    by_ramp: Dict[int, List[Dict[str, float]]] = {}
    for r in rows:
        idx = int(r["ramp_index"])
        by_ramp.setdefault(idx, []).append(r)

    for idx, samples in by_ramp.items():
        if len(samples) < 2:
            continue
        # Detect first movement.
        for i in range(1, len(samples)):
            vel = samples[i]["velocity_rad_s"]
            if not math.isfinite(vel):
                continue
            if abs(vel) >= vel_threshold_rad_s:
                torque_nm = samples[i - 1]["command_torque_Nm"]
                if math.isfinite(torque_nm):
                    friction_samples.append((idx, torque_nm))
                break

    values = [v for _, v in friction_samples]
    if not values:
        return {
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "count": 0,
        }

    abs_vals = [abs(v) for v in values]
    return {
        "mean_abs": float(np.mean(abs_vals)),
        "max_abs": float(np.max(abs_vals)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "count": len(values),
    }


def summarize(results: Dict[str, float]) -> str:
    return (
        f"Static friction (abs) mean: {results['mean_abs']:.5f} Nm, "
        f"max: {results['max_abs']:.5f} Nm\n"
        f"Signed mean: {results['mean']:.5f} Nm, min: {results['min']:.5f} Nm\n"
        f"Ramps used: {results['count']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ramp input static friction experiment with moteus.")
    parser.add_argument("--id", type=int, default=1, help="moteus controller id")
    parser.add_argument("--rate-hz", type=float, default=500.0, help="command/query rate (Hz)")
    parser.add_argument("--ramp-deg", type=float, default=5.0, help="ramp amplitude in degrees")
    parser.add_argument("--ramp-time", type=float, default=2.0, help="time to complete one ramp (s)")
    parser.add_argument("--hold-time", type=float, default=2.0, help="pause after each ramp (s)")
    parser.add_argument("--ramps", type=int, default=12, help="number of ramps to execute")
    parser.add_argument("--start-deg", type=float, default=0.0, help="starting position for the first ramp (deg)")
    parser.add_argument("--first-up", action="store_true", help="start ramping in positive direction (default is negative)")
    parser.add_argument("--kp", type=float, default=2.0, help="host-side proportional gain (Nm/rad)")
    parser.add_argument("--kd", type=float, default=0.0, help="host-side derivative gain (Nm*s/rad)")
    parser.add_argument("--max-torque", type=float, default=2.0, help="torque limit in Nm")
    parser.add_argument("--vel-threshold", type=float, default=0.02, help="velocity threshold (rad/s) for motion detection")
    parser.add_argument("--log", type=Path, default=Path("ramp_input_log.csv"), help="path to write/read log csv")
    parser.add_argument("--analyze-only", action="store_true", help="skip hardware run and analyze an existing log at --log")
    parser.add_argument("--plot", action="store_true", help="show time-series plots after run/analysis")
    parser.add_argument("--plot-file", type=Path, help="save plots to this path (png, pdf, etc)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        rows = read_log(args.log)
    else:
        print(f"Running ramp input experiment on moteus id {args.id} ...")
        rows = asyncio.run(collect_ramp_response(args))
        metadata = {
            "kp_Nm_per_rad": args.kp,
            "kd_Nm_s_per_rad": args.kd,
            "rate_hz": args.rate_hz,
            "ramp_deg": args.ramp_deg,
            "ramp_time_s": args.ramp_time,
            "hold_time_s": args.hold_time,
            "ramps": args.ramps,
            "start_deg": args.start_deg,
            "first_up": args.first_up,
            "max_torque_Nm": args.max_torque,
            "vel_threshold_rad_s": args.vel_threshold,
        }
        write_log(args.log, metadata, rows)
        print(f"Wrote log to {args.log}")

    results = estimate_static_friction(
        rows=rows,
        vel_threshold_rad_s=args.vel_threshold,
    )
    print(summarize(results))
    if args.plot or args.plot_file:
        time_s = np.array([r["time_s"] for r in rows])
        target = np.array([r["target_rad"] for r in rows])
        position = np.array([r["position_rad"] for r in rows])
        torque_cmd = np.array([r["command_torque_Nm"] for r in rows])
        torque_meas = np.array([r["measured_torque_Nm"] for r in rows])
        plot_ramp(time_s, target, position, torque_cmd, torque_meas, args.plot_file)


if __name__ == "__main__":
    main()
