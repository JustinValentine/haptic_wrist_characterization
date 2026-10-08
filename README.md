# Haptic Wrist Characterization with moteus

Characterization experiments for a single-DOF, cable-driven haptic wrist actuated by a
[moteus](https://mjbots.github.io/moteus/) controller. Every experiment writes raw data plus
metadata, has a paired analysis that produces IEEE-ready figures and a summary, and can be
dry-run against a simulated wrist (`--sim`) before touching hardware.

| What it measures | Experiment | Setup | Paper figure |
|---|---|---|---|
| Inertia, viscous damping, Coulomb friction | `step_input_experiment.py` | free | `fig_step_identification` |
| Static (breakaway) friction vs position | `ramp_input_experiment.py` | free | `fig_static_friction` |
| Friction vs velocity (Stribeck), cogging/gravity vs position | `friction_sweep_experiment.py` | free | `fig_friction_velocity`, `fig_friction_position` |
| Closed-loop position bandwidth (Bode) | `schroeder_bandwidth_experiment.py` | free | `fig_bandwidth_bode` |
| Rise/settling time, peak velocity and acceleration | `closed_loop_step_experiment.py` | free | `fig_closed_loop_steps` |
| Renderable impedance range (Z-width) | `zwidth_experiment.py` | free, no hand contact | `fig_zwidth` |
| Cable stiffness, lost motion, transmission resonance | `transmission_stiffness_experiment.py` | output clamped | `fig_transmission_stiffness`, `fig_transmission_frf` |
| Torque accuracy (scale factor, hysteresis, linearity) | `torque_calibration_experiment.py` | output clamped against a load cell / F/T | `fig_torque_calibration` |
| Continuous vs peak torque | `thermal_experiment.py` | output clamped, motor thermistor | `fig_thermal` |
| Transparency: friction/damping/inertia felt at the handle | `transparency_experiment.py` | operator moves handle, F/T sensor | `fig_transparency` |
| **Rendered stiffness/damping vs frequency, loop delay, predicted stability limit** | `impedance_fidelity_experiment.py` | free, no hand contact | `fig_impedance_spectrum`, `fig_impedance_free` |
| **Virtual-wall impacts: restitution, wall energy, rate-hardness** | `virtual_wall_experiment.py` | free, no hand contact | `fig_virtual_wall`, `fig_virtual_wall_hardness` |
| **Large-signal dynamic envelope: largest linear amplitude vs frequency** | `dynamic_envelope_experiment.py` | free | `fig_dynamic_envelope` |
| One summary table (CSV, Markdown, LaTeX) | `characterization_report.py` | after the analyses | `report/characterization_table.tex` |

Bold rows are the experiments designed for this wrist (see
[Positioning against prior work](#positioning-against-prior-work)).

## Contents
1. [Setup](#setup)
2. [Safety](#safety)
3. [Workflow](#workflow)
4. [Simulator and tests](#simulator-and-tests)
5. [Positioning against prior work](#positioning-against-prior-work)
6. [Experiments](#experiments)
7. [Figures](#figures)
8. [Data layout](#data-layout)
9. [Hardware notes](#hardware-notes)
10. [Reporting in the paper](#reporting-in-the-paper)

---

## Setup

```bash
pip install -r requirements.txt
```

Edit **`device_config.py`** once for your wrist, and set `JOINT` to the haptic joint under
test ('PS' or 'FE') before each run: gear ratio per joint, controller ID, joint travel and
speed limits, the torque cap sent with every command, optional output encoder and external
torque sensor, and nominal inertia/friction used only to size excitation signals.
Experiment-specific settings (gains, amplitudes, durations) are the `Config` dataclass at the
top of each experiment script.

### Units and moteus assumptions
* moteus is configured with `motor_position.rotor_to_output_ratio = 1`, so moteus positions
  are motor revolutions and torques are motor-side N·m. The scripts check this at startup.
* `GEAR_RATIO` is the external cable reduction: joint rev = motor rev / N, joint torque ≈
  N · motor torque (ideal, before cable losses).
* Joint-side gains convert to moteus gains as `Kp_moteus = Kp_joint · 2π / N²` (same for Kd).
* Joint positions are relative to the motor position when the experiment starts, so start each
  run with the handle at its neutral pose.
* Positions come from the **motor encoder**. Cable stretch is therefore invisible to the
  free-motion experiments; the transmission experiment measures it directly (and an output
  encoder, if configured in `device_config.OUTPUT_ENCODER`, is logged alongside).
* Torques are the torque moteus believes it applies (from current and its torque constant).
  The torque calibration experiment measures the true scale factor; the report applies it.

### What every experiment does to the controller
At startup the session stops the motor, stops diagnostic telemetry, snapshots the moteus
configuration, and then for the duration of the run:
* sets `servo.pid_position.kp/kd` from the joint-side gains and `ki = 0`;
* sets `servo.default_velocity_limit`, `servo.default_accel_limit`, and
  `servo.max_position_slip` to `nan` (any of these silently turns steps into trajectories or
  drags the setpoint, which corrupts step and frequency responses);
* sends `maximum_torque` and a `watchdog_timeout` with every command.

On exit (normal, Ctrl-C, or a safety abort) it brakes if aborting, stops the motor, restores
every configuration value it changed, and saves the data with a `completed` flag and the abort
reason. Nothing is written to flash.

## Safety

* Each sample is checked against `device_config` limits: joint travel, joint speed, motor and
  board temperature, and moteus fault/timeout modes. A violation brakes (moteus brake mode:
  shorted windings, passive damping), stops the motor, and saves partial data, including the
  sample that tripped.
* Every command carries `maximum_torque` (the device cap, or a tighter per-experiment cap) and a
  watchdog (`WATCHDOG_TIMEOUT_S`): if the host stops commanding, moteus enters timeout mode.
  Scripts never block for operator input while the motor is in position mode; prompts either
  happen before commanding starts or keep commanding a hold while waiting.
* Each experiment prints a plan (duration, commanded range, torque cap, warnings) and waits for
  Enter. `--yes` skips the prompt; do not use it on hardware until you have watched a run.
* Z-width trials deliberately go unstable; they are stopped by deviation, speed, and torque
  chatter checks, braked, and returned to a safe hold. Run it with the handle free.
* Locked-output experiments use a ±3° travel limit, so a slipping clamp stops the run.

## Workflow

```bash
# 1. Dry run in simulation (fast; no hardware)
python step_input_experiment.py --sim --quick
python step_input_experiment_analysis.py

# 2. On hardware
python step_input_experiment.py            # prints the plan, waits for Enter
python step_input_experiment_analysis.py   # analyzes the latest run of that experiment

# 3. After running the experiments you need
python characterization_report.py          # real runs only; add --include-sim for dry runs
```

Recommended order on hardware (later experiments reuse earlier results where noted):
1. `step_input_experiment.py`: J, b, fc (used to sanity-check later fits)
2. `ramp_input_experiment.py`: static friction (used by the step analysis model replay)
3. `friction_sweep_experiment.py`: friction model (used by transparency compensation)
4. `schroeder_bandwidth_experiment.py`, `closed_loop_step_experiment.py`,
   `dynamic_envelope_experiment.py` (uses J from step 1 to plan its amplitudes)
5. `zwidth_experiment.py`, then `impedance_fidelity_experiment.py` (skips targets above 0.7x the
   measured Z-width limit) and `virtual_wall_experiment.py`
6. Clamp the output: `transmission_stiffness_experiment.py`, `torque_calibration_experiment.py`,
   `thermal_experiment.py`
7. Unclamp: `transparency_experiment.py`
8. `characterization_report.py`

Cross-experiment lookups only use runs of the same kind: a `--sim` dry run never feeds
parameters into a hardware experiment, and vice versa.

Common options for every experiment: `--sim`, `--quick` (shortened protocol), `--yes`,
`--out DIR`, `--seed N` (simulator), `--controller-id`, and the moteus transport options
(`--fdcanusb`, `--can-iface`, ...). Every analysis accepts a run directory or CSV (default:
the latest run), `--no-show`, and `--paper-dir` (default `figures/ieee`; `""` to skip).

## Simulator and tests

`moteus_sim.py` implements the parts of the moteus API these scripts use (`set_position`,
`set_stop`, `set_brake`, `query`, and `conf get/set` on the diagnostic stream) on top of a
two-mass cable-driven wrist: motor-side and output inertia joined by a compliant, damped cable
(optional backlash), Karnopp stick-slip friction with Stribeck effect, viscous friction,
cogging, optional gravity, the moteus position/velocity/feedforward control law with its
velocity filter, current-loop lag, torque saturation and back-EMF derating, encoder noise,
host round-trip latency and jitter, a winding thermal model, an output lock, an external
torque sensor, and an optional human operator. Ground truth is stored in each simulated run's
metadata, and every analysis prints estimate-vs-truth when it analyzes a simulated run.

Default ground truth (joint side): J = 0.008 kg·m² (0.004 motor side + 0.004 output),
b = 0.022 N·m·s/rad, fc = 0.05 N·m, fs = 0.07 N·m, Stribeck velocity 0.05 rad/s,
cable stiffness 2000 N·m/rad (locked-output resonance 112 Hz), current limit 12 N·m. The motor
is the CubeMars GL60 II KV28 at a 36 V supply, scaled by `device_config.GEAR_RATIO`: cogging of
0.015 N·m at the motor (the datasheet's 1.5 cN·m, taken as the amplitude) repeating 168 times per
motor revolution (24 slots, 28 poles), about 0.15 N·m every 0.22° at the joint; and a back-EMF
torque–speed line from the stall torque (0.0625 N·m per supply volt at the motor) to the no-load
speed (28 rpm/V), about 22 N·m and 10.8 rad/s at the joint. The rotor's 401 g·cm² × N² matches
the 0.004 kg·m² motor-side inertia.

Cogging this large reaches several analyses. The friction sweep identifies it as a model in
absolute motor position, and the ramp, transmission, torque-calibration and impedance analyses
subtract that model when a friction-sweep analysis of the same kind (real or simulated) exists,
so run the friction sweep and its analysis first. The model holds until the encoder is
re-zeroed.

The tests run each experiment's quick protocol in the simulator and check that its analysis
recovers the ground truth:

```bash
python -m unittest discover -s tests -t .      # or: pytest tests
```

---

## Positioning against prior work

The first three experiments follow the characterization protocol of OpenWrist and its MAHI-lab
predecessors (static friction, inertia and damping, closed-loop position bandwidth). The rest
are standard actuator/haptics measures, and three are designed for this wrist specifically.

| Experiment | Origin | What is different here |
|---|---|---|
| Step response | OpenWrist protocol [1] | The Liang–Feeny decrement relation that OpenWrist evaluates on groups of four extrema [7] is fitted once by least squares over every half-cycle; bootstrap CIs; model replay. (The earlier plain log-decrement script conflated Coulomb and viscous friction: +62% b and +134% fc in simulation.) |
| Static friction ramp | OpenWrist protocol [1] | Displacement-based breakaway with a cubic onset fit (a velocity threshold fails with any encoder noise) |
| Closed-loop bandwidth | OpenWrist protocol [1] | Periodic, torque-budgeted multisine; per-period estimates with error bars at several amplitudes; parametric fit (the single-period Welch estimate erred by up to 21 dB in simulation) |
| Friction sweep, closed-loop steps, transmission, torque calibration, thermal, transparency | Standard actuator and haptic-device measures [2] | Automated, simulation-validated protocols |
| Z-width | Haptics standard [3], [4] | Firmware vs host rendering loops; friction-aware stability criterion [5] |
| **Rendered-impedance spectrum** | Designed for this wrist; builds on the effective-impedance analysis of [8] | Displayed stiffness and damping vs frequency for firmware and host rendering, measured without a device model; the identified effective delay predicts the linear stability limit |
| **Virtual-wall impacts** | Designed for this wrist; metrics from [6] and the passivity view of [4] | Automated self-launched impacts: restitution and wall work as an energy-passivity measure, rate-hardness, firmware (stay-within) vs host walls |
| **Large-signal dynamic envelope** | Designed for this wrist | Largest linearly tracked amplitude vs frequency against the torque, speed, and range limits |

"Designed for this wrist" means these protocols are not part of the OpenWrist-lineage
characterizations; related ideas exist in the haptics literature, so check the literature for
the exact novelty claim you make. References (verify details before citing):

1. E. Pezent, C. G. Rose, A. D. Deshpande, M. K. O'Malley, "Design and characterization of the
   OpenWrist: A robotic wrist exoskeleton for coordinated hand-wrist rehabilitation," ICORR 2017.
2. V. Hayward and O. R. Astley, "Performance measures for haptic interfaces," Robotics Research
   (ISRR), 1996.
3. J. E. Colgate and J. M. Brown, "Factors affecting the Z-width of a haptic display," ICRA 1994.
4. J. E. Colgate and G. G. Schenkel, "Passivity of a class of sampled-data systems: Application
   to haptic interfaces," J. Robotic Systems, 14(1), 1997.
5. N. Diolaiti, G. Niemeyer, F. Barbagli, J. K. Salisbury, "Stability of haptic rendering:
   Discretization, quantization, time delay, and Coulomb effects," IEEE Trans. Robotics, 22(2),
   2006.
6. D. A. Lawrence, L. Y. Pao, A. M. Dougherty, M. A. Salada, Y. Pavlou, "Rate-hardness: A new
   performance metric for haptic interfaces," IEEE Trans. Robotics and Automation, 16(4),
   357–371, 2000.
7. J. W. Liang and B. F. Feeny, "Identifying Coulomb and viscous friction from free-vibration
   decrements," Nonlinear Dynamics, 16(4), 337–347, 1998.
8. N. Colonnese, A. F. Siu, C. M. Abbott, A. M. Okamura, "Rendered and characterized closed-loop
   accuracy of impedance-type haptic displays," IEEE Trans. Haptics, 2015.

## Experiments

### Step response: inertia, viscous damping, Coulomb friction
`step_input_experiment.py` → `step_input_experiment_analysis.py`

**Protocol.** A soft P-only spring (Kd = 0, default Kp = 15 N·m/rad) and alternating steps of
several sizes (default 5, 10, 20°, six steps each). The joint rings down after each step.

**Model.** With a = θ − θ_cmd: J ä + b ȧ + fc sgn(ȧ) = −Kp a + τ_bias. Between turning points
the motion direction d_n is constant, so each half-cycle starting from rest at a_n ends at

  a_{n+1} = −r a_n − (1 + r)(fc/Kp) d_n + (1 + r) τ_bias/Kp,  r = exp(−πζ/√(1 − ζ²)),

after exactly half a damped period π/ω_d.

**Analysis.** Turning points are detected with noise-scaled hysteresis and refined by parabolic
interpolation. A pooled linear regression over every half-cycle of every step gives r (viscous
damping) and fc (Coulomb friction) *separately*; the half-cycle durations give ω_d. Then
ω_n = ω_d/√(1 − ζ²), J = Kp/ω_n², b = 2Jζω_n. 95% confidence intervals come from bootstrapping
whole steps, per-size fits check amplitude dependence, and the identified model is replayed on
every step (piecewise-analytic solution, using the measured static friction when a ramp
analysis exists) to report a fit percentage. Half-cycles are left out of the regression when the
delivered torque falls short of the command by more than 20% of its peak (back-EMF on the first
swing of large steps) or when they start inside the dead band where static friction and cogging
trap the joint, estimated as twice the median rest offset; the replay skips the same parts.

Plain logarithmic decrement attributes Coulomb decay to viscous damping, and the
oscillation-center offset is dominated by viscous decay; in simulation those biased b by +62%
and fc by +134%. This method recovers J, b, and fc to within 0.3%, 5%, and 1%.

**Outputs.** `fig_step_identification`: (a) a normalized step response with the identified
model; (b) amplitude lost per half-cycle vs amplitude (slope = viscous, intercept = Coulomb).

### Static friction: ramp-and-hold
`ramp_input_experiment.py` → `ramp_input_experiment_analysis.py`

**Protocol.** A very soft P-only spring (Kp = 3 N·m/rad) drags the joint through a ramp-and-hold
staircase 0 → +45° → 0 → −45° → 0 in 5° ramps at 2.5 °/s with 2 s pauses. While the joint is
stuck the spring torque Kp(θ_cmd − θ) rises steadily until the joint breaks away.

**Analysis.** Breakaway is detected from *displacement*, not differentiated velocity: velocity
from raw encoder differences is dominated by noise at these speeds, and a velocity threshold
either never fires or fires on noise. The encoder noise σ is measured during the pauses and the
detection threshold is max(6σ, 0.005°). Any displacement threshold fires late, because right
after breakaway the net torque is nearly zero and grows with the ramp (displacement ∝
(t − t0)³), so the onset t0 is found by fitting θ0 + A·max(0, t − t0)^p around the crossing,
with p = 1, 2 or 3 (cogging lets the joint creep along a detent before it snaps, which a cubic
cannot follow). The static friction is the spring torque at t0 minus the cogging model's torque
at that motor position; the raw breakaway torque, which includes cogging, is reported too, as is
the plain threshold method. Results are given per direction with confidence intervals, plus a
test for position dependence.

**Outputs.** `fig_static_friction`: breakaway torque vs position per direction with means;
`fig_static_friction_example`: one ramp showing the torque rise and the onset fit.

### Friction vs velocity and position: constant-velocity sweeps
`friction_sweep_experiment.py` → `friction_sweep_experiment_analysis.py`

**Protocol.** A stiff PD loop (Kp = 600 N·m/rad, Kd = 1.5 N·m·s/rad, stiffer than the cogging's
peak stiffness of about 240 N·m/rad so the joint tracks through each detent) with velocity feedforward
drives constant-velocity passes over −40…+40° in both directions at 0.5–200 °/s (two round
trips each, smooth sinusoidal speed blends). During the cruise the setpoint is advanced by moteus
itself (velocity-only commands), because re-sending absolute positions under host timing jitter
injects torque ripple. In the cruise the inertial torque is zero, so the measured torque is
friction(v) + conservative(θ).

**Analysis.** Forward and backward passes are paired on position bins:
friction = (τ₊ − τ₋)/2 and conservative = (τ₊ + τ₋)/2 (gravity, cable preload). Cogging is
identified first from the passes at or below 2 °/s: the peak of their averaged position spectrum
gives its cycles per motor revolution, and a least-squares harmonic series in absolute motor
position gives its amplitude and phase, saved as the cogging model. Every position bin is then
widened to whole cogging periods so cogging averages out. The loop's damping also draws a drag
from the cogging ripple, largest where the cogging frequency nears the loop's resonance; it is
predicted from the cogging model and the loop's dynamic stiffness and subtracted, and passes where
it exceeds 20% of the friction are left out. The Stribeck + viscous model
τ_f = sgn(v)[f_c + (f_s − f_c)e^(−(|v|/v_s)²)] + bv is fitted by least squares with standard
errors. The conservative torque is summarized by its trend and by the remaining ripple after
noise correction. Friction is also reported as a percentage of the torque cap.

**Outputs.** `fig_friction_velocity`: friction vs speed (log axis) with the fitted Stribeck
curve; `fig_friction_position`: torque vs position for both directions and their midline.

### Closed-loop position bandwidth: periodic multisine
`schroeder_bandwidth_experiment.py` → `schroeder_bandwidth_experiment_analysis.py`

**Protocol.** Stiff PD (Kp = 450 N·m/rad, Kd = 0.75 N·m·s/rad) tracks a periodic multisine: about
50 log-spaced lines from 0.2 to 30 Hz on a 10 s period, Schroeder phases for the actual power
spectrum plus crest-factor reduction. The amplitude spectrum is flat up to a corner and rolled
off above it; the corner is lowered automatically until the predicted inertial torque fits in
half the torque cap (the previous flat 0.1–20 Hz signal demanded about 1150 °/s and saturated the
actuator). Three amplitude levels (2, 5, 10°) of five periods each; the first period of each
level is discarded as a transient. Commands are evaluated at the actual send time, at 1 kHz.
`send_velocity_feedforward` (default on) adds Kd·ṙ to the reference and changes the
closed-loop transfer function; state which one you used.

**Analysis.** In every period the command and the measured position are fitted at the excited
frequencies by least squares (exact with jittered timestamps), giving H per period, then the
mean and standard error across periods. Lines with SNR < 10 and levels that sat at the torque
cap more than 1% of the time are excluded. Reported: −3 dB bandwidth relative to the
low-frequency gain (or a lower bound when it lies above the tested range), −90° phase
frequency, resonance peak, and a fit of H = e^(−sτ)(Kp + FF·Kd·s)/(Js² + bs + e^(−sτ)(Kp + Kd s))
giving J (a cross-check of the step test), an effective loop delay, and a model-extrapolated
bandwidth. The fitted b is an *effective* damping dominated by Kd and friction; report damping
from the step and sweep tests instead. Data from the earlier single-period 100 Hz script are
re-analyzed with a one-period DFT and flagged as legacy.

**Outputs.** `fig_bandwidth_bode`: magnitude and phase per amplitude with ±2 SE error bars and
the fitted model; `fig_bandwidth_time`: a 1 s zoom of command, response, and error.

### Closed-loop steps: rise time, settling, peak acceleration
`closed_loop_step_experiment.py` → `closed_loop_step_experiment_analysis.py`

**Protocol.** The same stiff PD gains, alternating steps of 1, 2, 5, 10, and 20° (six each) with
the full torque cap available. Small steps characterize the linear loop; large steps saturate
the torque, so the joint accelerates at its limit (this is what "highly dynamic" claims rest
on).

**Analysis.** Per step: 10–90% rise time, overshoot, settling time into a 2% band around the
final value (friction and cogging leave a steady-state error, reported separately) and into an
absolute band, peak velocity, peak torque and mechanical power, and saturation. Over the first
interval at the cap, position is fitted to the double integral of the delivered torque minus the
friction-sweep friction (motor-side velocity rings with the cable for the first milliseconds,
which inflates derivative-based estimates). Because back-EMF lets the delivered torque fall below
the command as speed rises, this gives the acceleration per unit torque, and so a large-signal
inertia, where a plain quadratic fit would mistake derating for missing acceleration. The
acceleration at the cap is compared with cap/J using J from the step analysis, and the speed at
which the delivered torque first falls below 95% of the cap marks the back-EMF knee.

**Outputs.** `fig_closed_loop_steps`: (a) normalized responses per step size; (b) rise and
settling time vs step size.

### Z-width: renderable impedance range
`zwidth_experiment.py` → `zwidth_experiment_analysis.py`

**Protocol.** A virtual spring-damper τ = −K(θ − θ0) − Bθ' is rendered either by the moteus
position loop (`moteus` mode, the firmware limit) or by the host from the previous
measurement as feedforward torque (`host_<rate>` modes, the situation of a PC-side haptic
simulation with one sample of delay plus communication latency). For each B, K increases
geometrically (×1.4, then one bisection step) until a 1 N·m × 5 ms torque pulse produces a
response that does not decay. Each trial is contained by deviation, speed, and torque-chatter
stops, a brake, and a smooth return to a safe hold. Run with the handle free.

**Classification.** A trial is stable only if the pulse response halves within 0.5–0.75 s and has
decayed to the noise floor by the end of the observation. Ringing that friction stops only much
later still feels like a buzzing wall, so it counts as unstable; marginal stability counts as
unstable. Every trial is reclassified offline from the raw data.

**Outputs.** `fig_zwidth`: largest stable K vs B per mode, first unstable K marked, and lower
bounds (no instability up to the largest K tested) shown with arrows. With Coulomb friction the
boundary depends on the disturbance size, so state the pulse in the caption.

### Transmission stiffness, lost motion, and resonance (output clamped)
`transmission_stiffness_experiment.py` → `transmission_stiffness_experiment_analysis.py`

**Protocol.** With the output clamped, motor-side rotation is cable stretch plus any backlash.
Phase 1: slow triangular torque cycles 0 → +T → −T → 0. Phase 2: a low-crest-factor torque
multisine (1–200 Hz) about a preload that keeps the cable out of backlash.

**Analysis.** Stiffness is a common-slope fit with separate zero-torque intercepts per torque
side (so backlash does not bias the slope); lost motion is the intercept gap; the loading vs
unloading torque gap gives the friction hysteresis (which also widens the loop by 2f/k).
Phase 2: the commanded torque is integrated exactly as a piecewise-constant signal from each
command's arrival time and the motor angle is fitted at those times, giving the compliance
θ/τ per period; 1/(J_m s² + c s + k) is fitted for the dynamic stiffness, the reflected
motor-side inertia, and the transmission resonance √(k/J_m)/2π that ultimately bounds control
bandwidth.

**Outputs.** `fig_transmission_stiffness` (torque–deflection loop with fit),
`fig_transmission_frf` (compliance magnitude/phase with fit and resonance).

### Torque calibration (output clamped against a sensor)
`torque_calibration_experiment.py` → `torque_calibration_experiment_analysis.py`

**Protocol.** Torque staircase 0 → +max → −max → 0 so every level is approached from both
sides. A zero reading is taken first, then a 0.5 N·m lock check (aborts if the output moves).
With a continuous sensor each level is averaged; with `EXTERNAL_TORQUE_SENSOR = {'type':
'manual'}` the operator types each force-gauge reading while the motor keeps holding.

**Analysis.** measured = k · commanded + offset + h · branch (branch = −1 increasing, +1
decreasing): k is the torque scale factor (torque-constant error × static transmission
efficiency), h is half the hysteresis loop (≈ motor-side static friction), plus nonlinearity
(% full scale). A single line through both branches biases k because the extreme levels are
visited on one branch only. **Multiply torque-derived quantities from the other experiments by
k**; `characterization_report.py` does this in its "calibrated" column.

### Thermal: continuous torque (output clamped)
`thermal_experiment.py` → `thermal_experiment_analysis.py`

**Protocol.** Constant torque levels (default 30/50/70% of the cap) held against the clamp
until the winding rise or absolute temperature approaches its limit, with zero-torque
cool-downs between levels. Logs at 25 Hz. Needs a motor thermistor
(`servo.enable_motor_temperature`); otherwise it falls back loudly to the board temperature,
which rates the driver, not the motor.

**Analysis.** Each level is fitted with T(t) = T_ss + (T_0 − T_ss)e^(−t/τ) (incomplete cool-downs
allowed). Copper loss scales with torque², so ΔT_ss = c·τ² through the origin, and the
continuous torque is √((T_limit − T_start)/c). Levels held for less than one time constant are
flagged as extrapolated.

**Outputs.** `fig_thermal`: (a) temperature rise per level with fits; (b) steady-state rise vs
torque² with the continuous-torque construction.

### Transparency: what the operator feels
`transparency_experiment.py` → `transparency_experiment_analysis.py`

**Protocol.** The operator moves the handle back and forth (slow/medium/fast, ±30°) while the
motor renders zero torque or a capped feedforward friction compensation
gain · (fc tanh(ω/ω_ε) + b ω) using parameters from the latest friction-sweep or step analysis.
An external torque sensor between hand and handle measures the interaction torque.

**Analysis.** Per condition, τ_int = J_app θ̈ + b_app θ̇ + f_app sgn(θ̇) + offset gives the
apparent inertia, damping, and friction the operator feels, and RMS interaction torque per
speed. Without a continuous sensor the analysis reports a clearly labeled model-based estimate,
which only restates the identified model.

**Outputs.** `fig_transparency`: (a) resistive torque vs velocity, zero torque vs compensated;
(b) RMS interaction torque per speed.

### Rendered-impedance spectrum: what the rendering actually displays
`impedance_fidelity_experiment.py` → `impedance_fidelity_experiment_analysis.py`

**Protocol.** For each target (loop, K, B) a spring-damper is rendered about a fixed pose, either
by the moteus firmware loop or by the host from the previous sample, and a periodic torque
multisine (0.5–40 Hz, about 30 lines, designed for roughly equal motion per line, peak torque
within a budget) is added on top. Every target is first stability-checked with a torque pulse
(same rule as Z-width) and skipped if it rings; targets above 0.7× a measured Z-width limit are
skipped outright. Defaults: a near-free firmware target (K = 5, B = 0), firmware K = 40 and 200
(B = 0.2), host K = 10 (B = 0) and K = 40 (B = 0.2). Run with the handle free.

**Analysis.** The disturbance torque is rebuilt from the stored design and integrated exactly as
a zero-order hold from each command's arrival; position is fitted at the arrival times. That
gives the displayed impedance Z = T/Θ per period, with standard errors. Subtracting the near-free
target line by line, Z_r = Z_target − Z_free + K_c, removes inertia, friction and transmission
dynamics without a device model and leaves what the loop renders. Each target is fitted with
(K_eff + jωB)e^(−jωT_eff). Delay turns stiffness into negative damping, B_r ≈ B − K·T_eff, so the
measured loop delay predicts the linear stability limit K_crit ≈ b/T_eff (viscous b from the step
analysis). The analysis compares that prediction with the Z-width result. It also reports the
fidelity bandwidth (stiffness within 10%, phase within 10°) and the free-space apparent inertia
and damping across frequency.

**Outputs.** `fig_impedance_spectrum`: (a) rendered stiffness K_r/K and (b) damping shift
B_r − B vs frequency for firmware and host targets, with fits at each loop's measured delay.
`fig_impedance_free`: free-space apparent inertia and damping vs frequency. In simulation the
host 1 kHz delay is measured at 1.53 ms (expected 1.6 ms), its damping shift at K = 40 is −0.063
N·m·s/rad (−K·T predicts −0.064), and the predicted linear limit (13 N·m/rad) sits far below the
pulse-based Z-width (113 N·m/rad). Coulomb friction quenches small oscillations [5], so a pulse
test overstates how stiff a host-rendered wall can be under larger disturbances. In the damping
panel, the small offset of the firmware target is a friction-mismatch artifact of the
subtraction (friction depends on motion amplitude), not rendered damping; mention it in the
caption.

### Virtual-wall impacts: restitution, wall energy, rate-hardness
`virtual_wall_experiment.py` → `virtual_wall_experiment_analysis.py`

**Protocol.** The wrist launches itself under stiff tracking toward a unilateral virtual wall
(default at +10°), is released 1.5° before contact, and the wall alone governs the impact. The
wall is rendered either by the moteus firmware in stay-within mode (bound at the wall, gains
selected per phase by kp/kd scales) or by the host loop from the previous sample. Sweep: two
rendering loops × B ∈ {0, 0.1} × two impact speeds × K = 20–1000 N·m/rad, two repeats; the K
sweep stops once a wall chatters or launches the handle back at more than twice the impact
speed. Each impact ends with a damped catch and a brake. Run with the handle free.

**Metrics.** The coefficient of restitution e = |v_out|/|v_in| uses position slopes just
before entry and just after the wall torque releases (a delayed wall keeps pushing past exit);
e > 1 means the rendering injected energy. Wall work is ∫τ dθ over the contact, as a percentage of
the impact energy (positive = energy generated by the rendering). Rate-hardness [6] is the
initial rate of rise of the wall torque per unit impact speed, the perceptually relevant
hardness. The analysis also reports the effective torque delay, penetration, contact time and
effective stiffness.

**Outputs.** `fig_virtual_wall`: (a) torque–penetration loops of firmware vs host walls at the
same K (a counter-clockwise loop means energy injected); (b) restitution vs K.
`fig_virtual_wall_hardness`: (a) rate-hardness / K; (b) wall energy gain vs K. In simulation the
firmware wall stays passive (e 0.77–0.97) up to 1000 N·m/rad, while the 1 kHz host wall, with
its ~1.9 ms torque delay, injects energy from K ≈ 100 N·m/rad and loses most of its
rate-hardness.

### Large-signal dynamic envelope
`dynamic_envelope_experiment.py` → `dynamic_envelope_experiment_analysis.py`

**Protocol.** Small-signal bandwidth does not show what a high-force device can do. For each
frequency (0.5–20 Hz) the stiff PD loop tracks raised-cosine-windowed sinusoids of increasing
amplitude: a small-signal reference trial, then from half the predicted limit upward (×1.25)
until the response stops being linear, then bisection until the bracket is within 5% (back-EMF,
which the torque-line prediction ignores, can stop the sweep well below it). Commanded amplitudes are planned
from the predicted joint amplitude divided by the closed-loop gain, and speed and range guards
end a trial before the session limits trip.

**Analysis.** For each trial, least-squares fits at f, 2f, and 3f give the gain and phase relative
to the small-signal reference, the harmonic distortion, and the fraction of time the torque sits
at the cap or the measured torque falls short of the controller's (voltage limit). A trial is
linear while |G/G0| stays within ±10%, phase within 10°, THD below 5%, and saturation below 2%.
A_max(f) is the largest linear amplitude, with its limiting mechanism; points stopped by a
safety cap are lower bounds.

**Outputs.** `fig_dynamic_envelope`: (a) A_max vs frequency (log-log) against the torque limit
τ_cap/(Jω²), the speed cap, and the amplitude cap, with optional user-supplied reference regions
(`reference_regions`, empty by default; cite your source); (b) peak torque at the limit. In
simulation the envelope follows the torque line within 2% from 8 to 20 Hz.

---

## Figures

Analyses write every figure into the run's `analysis/figures/` and paper figures also into
`figures/ieee/` (`--paper-dir`), as vector PDF and EPS (TrueType fonts embedded) plus a 600 dpi
PNG and a JSON sidecar with the analysis settings and results. Design rules (`ieee_plot.py`):
* 3.5 in single-column width, Times, 8 pt labels and 7 pt ticks, no in-plot titles;
* never a second y-axis; panels stack instead;
* at most three categorical colors, from a colorblind-validated palette, always paired with
  distinct markers and line styles so figures survive grayscale print; ordered series (step
  sizes, torque levels) use a single-hue light-to-dark ramp;
* measured data in black, identified models dashed, thresholds muted, solid hairline grid.

Figures whose names start with `diag_` are diagnostics (turning points, trial traces, time
series) and are written only to the run directory. Check them before trusting a result.

## Data layout

```
data/<experiment>/<YYYYmmdd-HHMMSS>[_sim]/
    <experiment>_data.csv          one row per command/reply (joint-side units)
    <experiment>_metadata.json     config, moteus config snapshot, git commit, events, abort reason
    analysis/                      results tables, summary.csv/json, analysis metadata, figures/
figures/ieee/                      latest paper figures
report/                            characterization_summary.csv, characterization_table.tex/.md
```

Data columns include host timestamps (`time`, `t_send`), the joint-side command (position,
velocity feedforward, feedforward torque, gain scales, torque cap), measured joint position,
velocity and torque, q-axis current, moteus' internal setpoint and control torque, voltage,
temperatures, mode/fault, the phase and segment of the protocol, and optional output-encoder
and external-sensor columns. Simulated runs (`*_sim`) are git-ignored.

The analyses also read CSVs written by the earlier version of these scripts
(`step_response_data.csv`, `friction_ramp_data.csv`, `bandwidth_data.csv` in the working
directory), so existing hardware data can be re-analyzed with the corrected methods.

## Hardware notes

* The scripts have been validated end to end against the simulator only. Run each one first with
  `--sim`, then on hardware with the confirmation prompt, and watch the first run.
* Brake mode (used to stop aborted runs and between Z-width trials) needs moteus firmware with
  mode 15 (2023 or later).
* Set `MAX_JOINT_TORQUE_NM` at or below what `servo.max_current_A` allows (times the torque
  constant and gear ratio): saturation detection compares the controller torque with this cap,
  so a lower firmware current limit would go unnoticed.
* Update `NOMINAL_*` in `device_config.py` after the step and sweep tests; they size the
  bandwidth excitation and the safety predictions.
* Z-width changes gains over the diagnostic channel between trials; each `conf set` must finish
  inside `WATCHDOG_TIMEOUT_S`.
* The ATI Net F/T reader in `experiment_common.py` has not been tested on hardware; check its
  sign and scale against a known load.
* At 1 kHz a run writes roughly 1 MB of CSV per 6 s of protocol (the full friction sweep is
  about 50 MB).

## Reporting in the paper

* Report mean ± SD (or the bootstrap 95% CI) and the number of trials; the summaries include
  both.
* State that positions are motor-encoder based and joint-referred, and give the transmission
  stiffness and resonance alongside the free-motion results.
* Apply (or at least report) the torque scale factor from the calibration.
* For bandwidth, state the controller gains and whether velocity feedforward was sent, since it
  changes the closed-loop transfer function.
* For Z-width, state the rendering loop (moteus firmware vs host rate), the disturbance pulse,
  and the stability criterion.
