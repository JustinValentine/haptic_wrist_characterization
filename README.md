# Characterization Scripts (single DOF, moteus)

Three quick scripts that mirror the OpenWrist experiments for a single joint driven by a moteus controller. All of them disable the moteus position loop (`kp_scale=0`) and use host-side PD torque. Methods follow the OpenWrist thesis (E. Pezent, “Design, Characterization, and Validation of the OpenWrist Exoskeleton,” 2017) Sections 3.2 (step), 3.3 (ramp), and 3.4 (Schroeder multisine).

## Common setup
- Axis under test parallel to gravity; other joints locked; passive sliders fixed mid-range.
- Requirements: `python3`, `numpy`, `moteus`. Add `--plot` to see figures (`--plot-file path.png` to save).

## 1) Step Input (inertia, viscous damping, kinetic friction) — Section 3.2
Script: `step_input_experiment.py`

Run (20° steps, 3 cycles):
```bash
python step_input_experiment.py \
  --id 1 --step-deg 20 --offset-deg 0 \
  --kp 5.0 --kd 0.0 --max-torque 2.0 \
  --rate-hz 500 --dwell 1.5 --cycles 3 \
  --log step_ps_0deg.csv --plot
```
- Logs target/position/velocity/torques; estimates J, b, fk via log decrement.
- Sweep workspace by changing `--offset-deg` (e.g., -50, 0, 50).
- Refit a log: `... --log step_ps_0deg.csv --analyze-only`.

## 2) Ramp Input (static friction) — Section 3.3
Script: `ramp_input_experiment.py`

Run (12 ramps, 5° over 2 s):
```bash
python ramp_input_experiment.py \
  --id 1 --ramp-deg 5 --ramp-time 2 --hold-time 2 \
  --ramps 12 --start-deg 0 \
  --kp 2.0 --kd 0.0 --max-torque 2.0 \
  --rate-hz 500 --log ramp_ps_0deg.csv --plot
```
- Static friction = command torque one cycle before velocity crosses `--vel-threshold`.
- Re-analyze: `... --log ramp_ps_0deg.csv --analyze-only`.
- If motion never starts, lower `--vel-threshold` or bump `--kp` slightly.

## 3) Schroeder Multisine (closed-loop position bandwidth) — Section 3.4
Script: `schroeder_bandwidth_experiment.py`

Run (15 s, 0.5–15 Hz, 40 components, ±10°):
```bash
python schroeder_bandwidth_experiment.py \
  --id 1 --duration 15 --fmin 0.5 --fmax 15 --components 40 \
  --amplitude-deg 10 --offset-deg 0 \
  --kp 10.0 --kd 0.5 --max-torque 3.0 \
  --rate-hz 500 --log schroeder_ps.csv --plot
```
- Estimates Bode magnitude at excited frequencies and reports -3 dB cutoff.
- Re-analyze: `... --log schroeder_ps.csv --analyze-only`.
- If no -3 dB crossing, increase `--fmax`/`--components` or tune PD for critical damping.

## Safety/notes
- Start with conservative torque limits; soften gains if you see overshoot.
- Keep gravity effects out (axis level), or results will skew.
