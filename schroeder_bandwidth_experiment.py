#!/usr/bin/env python3
"""
Schroeder multisine input experiment for closed-loop position bandwidth (single DOF).

Implements the OpenWrist Section 3.4 test using a Schroeder-phased multisine
position reference and a host-side PD torque controller. The script:
  - Generates a multisine between fmin and fmax with minimized peak-to-peak amplitude.
  - Drives the joint with PD torque (kp, kd) and logs target/actual position,
    velocity, and torques.
  - Estimates the closed-loop position frequency response and -3 dB bandwidth
    by taking the ratio of output to input at the excited frequencies.
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


def deg_to_rad(value: float) -> float:
    return math.radians(value)


def rev_to_rad(value: float) -> float:
    return value * 2.0 * math.pi


def _get_value(state: moteus.Result, register: moteus.Register) -> float:
    return state.values.get(register, float("nan"))


def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return None


def plot_schroeder(
    time_s: np.ndarray,
    target: np.ndarray,
    position: np.ndarray,
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    low_gain: float,
    bandwidth_hz: float,
    save_path: Path = None,
) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(time_s, np.rad2deg(target), label="target (deg)", linewidth=1.5)
    ax1.plot(time_s, np.rad2deg(position), label="position (deg)", linewidth=1.0)
    ax1.set_ylabel("Position (deg)")
    ax1.legend()
    ax1.grid(True)

    ax2.semilogx(freqs, 20 * np.log10(magnitudes), marker="o", linewidth=1.0)
    ax2.axhline(20 * math.log10(low_gain) - 3.0, color="r", linestyle="--", label="-3 dB")
    if math.isfinite(bandwidth_hz):
        ax2.axvline(bandwidth_hz, color="g", linestyle=":", label=f"bw {bandwidth_hz:.2f} Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Gain (dB)")
    ax2.grid(True, which="both")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()


def generate_schroeder(
    duration_s: float,
    rate_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    amplitude_rad: float,
    n_components: int,
    offset_rad: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, position reference, velocity reference, and frequency list."""
    t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    freqs = np.linspace(fmin_hz, fmax_hz, n_components)
    k = np.arange(1, n_components + 1)
    phases = -np.pi * k * (k - 1) / n_components  # Schroeder phase

    # Build multisine and its derivative.
    pos = np.zeros_like(t)
    vel = np.zeros_like(t)
    for f, phi in zip(freqs, phases):
        omega = 2.0 * math.pi * f
        pos += np.cos(omega * t + phi)
        vel += -omega * np.sin(omega * t + phi)

    # Normalize peak amplitude to 1 then scale.
    peak = np.max(np.abs(pos))
    scale = amplitude_rad / peak if peak > 0 else 0.0
    pos = offset_rad + scale * pos
    vel = scale * vel
    return t, pos, vel, freqs


async def run_signal(args) -> List[Dict[str, float]]:
    controller = moteus.Controller(id=args.id)
    await controller.set_stop()
    last_state = await controller.query()

    t, ref_pos, ref_vel, freqs = generate_schroeder(
        duration_s=args.duration,
        rate_hz=args.rate_hz,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        amplitude_rad=deg_to_rad(args.amplitude_deg),
        n_components=args.components,
        offset_rad=deg_to_rad(args.offset_deg),
    )

    dt = 1.0 / args.rate_hz
    start = time.monotonic()
    next_tick = start
    data: List[Dict[str, float]] = []

    for i, target in enumerate(ref_pos):
        target_vel = ref_vel[i]

        pos_rev = _get_value(last_state, moteus.Register.POSITION)
        vel_rev_s = _get_value(last_state, moteus.Register.VELOCITY)
        pos_rad = rev_to_rad(pos_rev) if math.isfinite(pos_rev) else float("nan")
        vel_rad_s = rev_to_rad(vel_rev_s) if math.isfinite(vel_rev_s) else 0.0

        pos_err = target - pos_rad if math.isfinite(pos_rad) else 0.0
        vel_err = target_vel - vel_rad_s
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

        now = time.monotonic()
        data.append(
            {
                "time_s": now - start,
                "target_rad": target,
                "target_vel_rad_s": target_vel,
                "position_rad": rev_to_rad(_get_value(state, moteus.Register.POSITION)),
                "velocity_rad_s": rev_to_rad(_get_value(state, moteus.Register.VELOCITY)),
                "measured_torque_Nm": _get_value(state, moteus.Register.TORQUE),
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
                "target_vel_rad_s",
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
        f.write("# schroeder_bandwidth_experiment\n")
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


def estimate_bandwidth(
    rows: List[Dict[str, float]],
    rate_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    n_components: int,
) -> Dict[str, float]:
    t = np.array([r["time_s"] for r in rows])
    ref = np.array([r["target_rad"] for r in rows])
    pos = np.array([r["position_rad"] for r in rows])

    # Use only the portion that matches the duration implied by rate.
    n = min(len(ref), int(rate_hz * (t[-1] - t[0] + 1.0 / rate_hz)))
    ref = ref[:n]
    pos = pos[:n]

    # Remove DC offset to avoid leakage.
    ref = ref - np.mean(ref)
    pos = pos - np.mean(pos)

    freqs = np.linspace(fmin_hz, fmax_hz, n_components)
    fft_ref = np.fft.rfft(ref)
    fft_pos = np.fft.rfft(pos)
    fft_freqs = np.fft.rfftfreq(len(ref), d=1.0 / rate_hz)

    mags: List[float] = []
    for f in freqs:
        idx = int(np.argmin(np.abs(fft_freqs - f)))
        if idx <= 0 or idx >= len(fft_ref):
            mags.append(float("nan"))
            continue
        if fft_ref[idx] == 0:
            mags.append(float("nan"))
            continue
        tf = fft_pos[idx] / fft_ref[idx]
        mags.append(abs(tf))

    mags = np.array(mags)
    low_mag = mags[0] if mags.size else float("nan")
    cutoff = low_mag / math.sqrt(2) if math.isfinite(low_mag) else float("nan")
    bandwidth = float("nan")
    for f, m in zip(freqs, mags):
        if not math.isfinite(m):
            continue
        if m <= cutoff:
            bandwidth = f
            break

    return {
        "bandwidth_hz": bandwidth,
        "low_freq_gain": low_mag,
        "freqs_hz": freqs.tolist(),
        "magnitudes": mags.tolist(),
    }


def summarize(results: Dict[str, float]) -> str:
    bw = results["bandwidth_hz"]
    if math.isfinite(bw):
        return f"Estimated -3 dB bandwidth: {bw:.2f} Hz (low-frequency gain {results['low_freq_gain']:.3f})"
    return "Bandwidth could not be determined (no -3 dB crossing found)."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schroeder multisine bandwidth experiment with moteus.")
    parser.add_argument("--id", type=int, default=1, help="moteus controller id")
    parser.add_argument("--rate-hz", type=float, default=500.0, help="command/query rate (Hz)")
    parser.add_argument("--duration", type=float, default=15.0, help="signal duration (s)")
    parser.add_argument("--fmin", type=float, default=0.5, help="minimum frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=15.0, help="maximum frequency (Hz)")
    parser.add_argument("--components", type=int, default=40, help="number of sinusoidal components")
    parser.add_argument("--amplitude-deg", type=float, default=10.0, help="peak amplitude of multisine in degrees")
    parser.add_argument("--offset-deg", type=float, default=0.0, help="position offset in degrees")
    parser.add_argument("--kp", type=float, default=10.0, help="host-side proportional gain (Nm/rad)")
    parser.add_argument("--kd", type=float, default=0.5, help="host-side derivative gain (Nm*s/rad)")
    parser.add_argument("--max-torque", type=float, default=3.0, help="torque limit (Nm)")
    parser.add_argument("--log", type=Path, default=Path("schroeder_log.csv"), help="path to write/read log csv")
    parser.add_argument("--analyze-only", action="store_true", help="analyze an existing log at --log without running hardware")
    parser.add_argument("--plot", action="store_true", help="show plots after run/analysis")
    parser.add_argument("--plot-file", type=Path, help="save plots to this path (png, pdf, etc)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        rows = read_log(args.log)
    else:
        print(f"Running Schroeder multisine experiment on moteus id {args.id} ...")
        rows = asyncio.run(run_signal(args))
        metadata = {
            "kp_Nm_per_rad": args.kp,
            "kd_Nm_s_per_rad": args.kd,
            "rate_hz": args.rate_hz,
            "duration_s": args.duration,
            "fmin_hz": args.fmin,
            "fmax_hz": args.fmax,
            "components": args.components,
            "amplitude_deg": args.amplitude_deg,
            "offset_deg": args.offset_deg,
            "max_torque_Nm": args.max_torque,
        }
        write_log(args.log, metadata, rows)
        print(f"Wrote log to {args.log}")

    results = estimate_bandwidth(
        rows=rows,
        rate_hz=args.rate_hz,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        n_components=args.components,
    )
    print(summarize(results))
    if args.plot or args.plot_file:
        time_s = np.array([r["time_s"] for r in rows])
        target = np.array([r["target_rad"] for r in rows])
        position = np.array([r["position_rad"] for r in rows])
        freqs = np.array(results["freqs_hz"])
        mags = np.array(results["magnitudes"])
        plot_schroeder(time_s, target, position, freqs, mags, results["low_freq_gain"], results["bandwidth_hz"], args.plot_file)


if __name__ == "__main__":
    main()
