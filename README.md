# Haptic Wrist Characterization: P4 Wrist

Characterization experiments for the three-joint wrist built at Barrett (pronation–supination PS,
flexion–extension FE, ulnar–radial deviation UR) on **Barrett P4 pucks** 6, 7 and 8, with FE and
UR on a cable differential, run on the bench without the WAM or its passive J5. Every
experiment writes raw data plus metadata, has a paired analysis that produces IEEE-ready figures
and a summary, and can be dry-run against a simulated wrist (`--sim`) before touching hardware.

This branch (`p4-wrist`) adapts the suite written for the moteus-driven WHAM wrist
(`characterization-suite`). The wrist is driven by **`p4wrist/`**, a small real-time C++ library
written for it (no libbarrett dependency): CANopen setup, a 500 Hz SYNC loop with PD +
feedforward rendering through the differential, zeroing, and its own safety supervision, since
the bench has no safety board. See [`p4wrist/README.md`](p4wrist/README.md).

| What it measures | Experiment | Setup | Paper figure |
|---|---|---|---|
| Inertia, viscous damping, Coulomb friction | `step_input_experiment.py` | free | `fig_step_identification` |
| Static (breakaway) friction vs position | `ramp_input_experiment.py` | free | `fig_static_friction` |
| Friction vs velocity (Stribeck), cogging/gravity vs position | `friction_sweep_experiment.py` | free | `fig_friction_velocity`, `fig_friction_position` |
| Closed-loop position bandwidth (Bode) | `schroeder_bandwidth_experiment.py` | free | `fig_bandwidth_bode` |
| Rise/settling time, peak velocity and acceleration | `closed_loop_step_experiment.py` | free | `fig_closed_loop_steps` |
| Renderable impedance range (Z-width) | `zwidth_experiment.py` | free, no hand contact | `fig_zwidth` |
| Cable stiffness and lost motion (resonance optional) | `transmission_stiffness_experiment.py` | output clamped | `fig_transmission_stiffness` |
| Torque accuracy (scale factor, hysteresis, linearity) | `torque_calibration_experiment.py` | output clamped against a load cell / F/T | `fig_torque_calibration` |
| Continuous vs peak torque | `thermal_experiment.py` | output clamped, motor thermistor | `fig_thermal` |
| Transparency: friction/damping/inertia felt at the handle | `transparency_experiment.py` | operator moves handle, F/T sensor | `fig_transparency` |
| **Rendered stiffness/damping vs frequency, loop delay, predicted stability limit** | `impedance_fidelity_experiment.py` | free, no hand contact | `fig_impedance_spectrum`, `fig_impedance_free` |
| **Virtual-wall impacts: restitution, wall energy, rate-hardness** | `virtual_wall_experiment.py` | free, no hand contact | `fig_virtual_wall`, `fig_virtual_wall_hardness` |
| **Large-signal dynamic envelope: largest linear amplitude vs frequency** | `dynamic_envelope_experiment.py` | free | `fig_dynamic_envelope` |
| One summary table (CSV, Markdown, LaTeX) | `characterization_report.py` | after the analyses | `report/characterization_table.tex` |

Bold rows are the experiments designed for the WHAM wrist (see
[Positioning against prior work](#positioning-against-prior-work)).

## Contents
1. [Setup](#setup)
2. [Bench bring-up](#bench-bring-up)
3. [Safety](#safety)
4. [Workflow](#workflow)
5. [What the simulation predicts for this wrist](#what-the-simulation-predicts-for-this-wrist)
6. [Simulator and tests](#simulator-and-tests)
7. [Positioning against prior work](#positioning-against-prior-work)
8. [Experiments](#experiments)
9. [Figures](#figures)
10. [Data layout](#data-layout)
11. [Hardware notes](#hardware-notes)
12. [Reporting in the paper](#reporting-in-the-paper)

---

## Setup

```bash
pip install -r requirements.txt
cmake -S p4wrist -B p4wrist/build && cmake --build p4wrist/build     # the wrist library
```

On the Linux host, bring the CAN interface up at 1 Mbit/s
(`sudo ip link set can0 up type can bitrate 1000000`). For the real-time loop thread give
Python CAP_SYS_NICE (`sudo setcap cap_sys_nice+ep "$(readlink -f "$(which python3)")"`) or set
`REALTIME_PRIORITY = 0` in `device_config.py`.

Edit **`device_config.py`** once for your wrist and bench, and set `JOINT` to the joint under
test ('PS', 'FE' or 'UR') before each run: puck nodes, the transmission matrix (from
libbarrett-canopen's `wam8g.conf`), per-motor signs, home pose, the loop's absolute joint
limits, speed and torque caps, the gains that hold the joints not under test, the external
torque sensor, and nominal inertia/friction used only to size excitation signals.
Experiment-specific settings (gains, amplitudes, durations) are the `Config` dataclass at the
top of each experiment script.

### Units and conventions
* Motor angles are linear in joint angles, m = J2MP·q (motors 6, 7, 8; joints PS, FE, UR):
  m6 = −N2·PS, m7 = N1·(FE + UR), m8 = N1·(−FE + UR), N2 = 8.7456, N1 = 7.5077. FE is the
  difference of the differential motors and UR their sum. Joint torques map to motor torques
  through J2MP^−T; motors that share a joint saturate together so a clipped motor never turns an
  FE torque into a UR torque.
* Joint positions in the data are relative to the start pose of each run; the zeroed absolute
  angles of all three joints are logged too (`q_ps_deg`, `q_fe_deg`, `q_ur_deg`).
* `GEAR_RATIO` is the largest motor turn per joint turn of the joint under test (N2 for PS, N1
  for FE/UR); `motor_pos_rev` is the joint angle in those motor turns, so each motor's cogging
  stays periodic in it.
* Positions come from the **motor encoders** (4096 counts per motor turn: 0.010° at PS, 0.0117°
  at FE/UR). Cable stretch is therefore invisible to the free-motion experiments.
* Torques: `control_joint_torque_nm` is the loop's command after limits; `joint_torque_nm` is
  the drives' measured q-axis current times Kt (the P4 reports phase-peak current and Kt per
  A rms, so torque = Kt·I/√2, not yet checked on this wrist). The torque calibration experiment
  measures the true scale factor; the report applies it.

### What every experiment does to the wrist
At startup the session brings the pucks up (keeping the previous zero when no puck was reset,
else resetting them and zeroing from the single-turn encoders or the home pose), starts the
500 Hz loop, enables the drives and holds every joint at its start angle. During the run it
commands the joint under test (PD + velocity feedforward + feedforward torque, or a stay-within
wall) and keeps the other two at their start angles with the `HOLD_*` gains. Every loop cycle is
logged from the library's own log, tagged with the command it used. On exit (normal, Ctrl-C, or
a safety abort) it brakes if aborting, disables the drives, disarms the drives' SYNC-loss timer,
stops the loop, and saves the data with a `completed` flag and the abort reason.

## Bench bring-up

Before the first experiment on a newly assembled or re-cabled wrist:

```bash
python zero_calibration.py      # home pose + single-turn encoder survey -> wrist_calibration.json
python wrist_check.py status    # drives off: move each joint by hand; its angle must increase
python wrist_check.py nudge     # small positive torque per joint: each must move positive
python wrist_check.py hold      # holds the start pose with the HOLD gains
```

`nudge` is the important one: a joint that moves negative under positive torque turns every PD
loop into positive feedback. The positive directions of FE and UR have not been checked on the
hardware (libbarrett-canopen's 2026-09-22 notes); fix J2MP columns or the per-motor signs before
anything else.

## Safety

There is no safety board on the bench, so the p4wrist loop supervises everything itself, every
2 ms, independent of Python:
* absolute joint limits and joint speed (brake with grounded phases for `FAULT_BRAKE_S`, then
  disable); torque caps per joint and per motor;
* a host watchdog (`WATCHDOG_TIMEOUT_S`): if Python stops commanding, the loop brakes and
  disables;
* puck liveness: missing positions, missing heartbeats, any puck reset (which also invalidates
  the zero), drive faults (EMCY and statusword), drives dropping out of Operation Enabled;
* host death: the drives' SYNC-loss timeout (0x2406) is set to 20 ms once SYNC runs, so if the
  process dies the drives fault and ground their phases.

Faults latch until cleared; enabling is refused outside the joint limits or without a zero. On
top, the host checks every logged sample against `device_config` limits (travel relative to the
start pose, speed, temperatures) and aborts the run with a brake.

Each experiment prints a plan (duration, commanded range, torque cap, warnings) and waits for
Enter. `--yes` skips the prompt; do not use it on hardware until you have watched a run.
Z-width trials deliberately go unstable; they are stopped by deviation, speed, and torque
chatter checks, braked, and returned to a safe hold. Run them with the handle free.
Locked-output experiments use a ±3° travel limit, so a slipping clamp stops the run.

## Workflow

```bash
# 1. Dry run in simulation (fast; no hardware)
python step_input_experiment.py --sim --quick
python step_input_experiment_analysis.py

# 2. On hardware (Linux, SocketCAN)
python step_input_experiment.py            # prints the plan, waits for Enter
python step_input_experiment_analysis.py   # analyzes the latest run of that experiment

# 3. After running the experiments you need
python characterization_report.py          # real runs only; add --include-sim for dry runs
```

Recommended order on hardware (later experiments reuse earlier results where noted), for each
joint you characterize:
1. `step_input_experiment.py`: J, b, fc (used to sanity-check later fits)
2. `ramp_input_experiment.py`: static friction (used by the step analysis model replay)
3. `friction_sweep_experiment.py`: friction and cogging models (used by transparency compensation
   and subtracted by later analyses)
4. `schroeder_bandwidth_experiment.py`, `closed_loop_step_experiment.py`,
   `dynamic_envelope_experiment.py` (uses J from step 1 to plan its amplitudes)
5. `zwidth_experiment.py`, then `impedance_fidelity_experiment.py` (skips targets above 0.7x the
   measured Z-width limit) and `virtual_wall_experiment.py`
6. Clamp the output: `transmission_stiffness_experiment.py`, `torque_calibration_experiment.py`,
   `thermal_experiment.py`
7. Unclamp: `transparency_experiment.py`
8. `characterization_report.py`

Cross-experiment lookups only use runs of the same kind: a `--sim` dry run never feeds
parameters into a hardware experiment, and vice versa. They do not yet distinguish joints, so
analyze one joint's runs before moving to the next (or keep each joint's data in its own copy).

Common options for every experiment: `--sim`, `--quick` (shortened protocol), `--yes`,
`--out DIR`, `--seed N` (simulator), `--interface can0`. Every analysis accepts a run directory
or CSV (default: the latest run), `--no-show`, and `--paper-dir` (default `figures/ieee`; `""`
to skip).

## What the simulation predicts for this wrist

The simulator's motor parameters are placeholders of the right order (see below), but the loop
structure is the real one, and it shapes every haptic result:

* **Loop delay.** One SYNC every 2 ms; a puck latches the torque sent after a SYNC at the next
  SYNC and ramps to it over the following period (CST). A torque therefore reaches the motor
  **4 ms** after the position it was computed from, plus the drive's current loop. The velocity
  used for damping lags further (half a period of differencing plus the filter).
* **Stiffness ceiling.** A gain scan on FE and PS (J ≈ 0.0054 kg·m²) finds stable PD up to about
  40 N·m/rad with a 30 Hz velocity filter (libbarrett-canopen's wrist gains are 20/10/10) and
  about 80 N·m/rad at 100 Hz (`VELOCITY_FILTER_HZ`, default 100) or 120 at 250 Hz. The suite's
  "stiff" gains are therefore Kp = 80, Kd = 1.0 (the moteus WHAM wrist used 450–600).
* **Passivity.** With b ≈ 0.02 N·m·s/rad the loop renders an undamped spring passively only up
  to about b/(T/2) ≈ 10 N·m/rad; friction stretches that at small amplitudes. A host loop on top
  (Python, VR, teleoperation) adds its period and sample age to the 4 ms.
* **Resolution.** 4096 counts per motor turn give 0.0117° per count at FE/UR and 0.010° at PS.
  The cable's deflection at the test torque spans only ~8 counts, which limits the stiffness
  measurement to ~15% and rules out the dynamic transmission test without an output encoder.

## Simulator and tests

`p4_sim.py` runs the real library against emulated pucks on a virtual CAN bus:

* **Pucks**: `SimP4` extends barrett-puck's `MockP4` (vendored in `vendor/barrett_puck`,
  unmodified) with what the library depends on: physics (counters restarting at 0 on reset,
  absolute single-turn encoders, q-axis current), the CST torque ramp, IDLE with grounded phases,
  boot-up after an extra heartbeat, heartbeats on the 0x1017 period, SYNC-loss faults, the
  self-disable after ~55 SYNCs without a controlword, RPDOs applied on receipt (type 255),
  TPDO2 every 100 ms whatever its type (as measured on firmware 4.3.2), mode writes accepted
  only while Operation Enabled, and PDO mapping that refuses unmappable objects.
* **Plant**: three motors and three outputs coupled by the differential and per-joint cable
  compliance; Karnopp stick-slip friction, viscous friction and cogging per motor; output
  damping and optional gravity and backlash per joint; current-loop lag; a back-EMF line at the
  48 V supply; winding and amplifier thermal models; an output lock; an external torque sensor;
  and an optional human operator.

Default ground truth for FE (joint side): J = 0.0054 kg·m² (0.0034 reflected from the two
differential rotors + 0.002 output), b = 0.019 N·m·s/rad, fc = 0.086 N·m, fs = 0.12 N·m, cable
stiffness 1500 N·m/rad, drive limit ≈ 0.96 N·m per motor (the WAM wrist's 23.5 A peak setting).
Motor parameters are ILM50-class placeholders, not datasheet values: rotor 3e-5 kg·m², Kt 58
mNm/A (the pucks' configured value), cogging 2 mN·m at 60 cycles per turn (assuming 12 slots).
Ground truth for the joint under test is stored in each simulated run's metadata, and every
analysis prints estimate-vs-truth when it analyzes a simulated run.

```bash
python -m unittest discover -s tests -t .      # 42 tests: the library and every experiment
cmake --build p4wrist/build && p4wrist/build/p4wrist_tests   # C++ unit tests
```

`tests/test_p4wrist.py` exercises the library itself against the emulated pucks: bus layout,
warm and cold connects, rough-placement zeroing, tracking on all joints, differential torque
signs, saturation, stay-within, braking, and every fault path (limits, watchdog, lost puck, EMCY,
puck reset, host death, clean stop). The other tests run each experiment's quick protocol and
check its analysis against the ground truth or, for the loop's dynamics, against an independent
model of the delayed loop.

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
| Z-width | Haptics standard [3], [4] | The wrist's own 500 Hz loop vs a host loop; friction-aware stability criterion [5] |
| **Rendered-impedance spectrum** | Designed for this wrist; builds on the effective-impedance analysis of [8] | Displayed stiffness and damping vs frequency for the 500 Hz loop and host rendering, measured without a device model; the identified effective delay predicts the linear stability limit |
| **Virtual-wall impacts** | Designed for this wrist; metrics from [6] and the passivity view of [4] | Automated self-launched impacts: restitution and wall work as an energy-passivity measure, rate-hardness, 500 Hz loop (stay-within) vs host walls |
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

**Protocol.** A soft spring (default Kp = 5 N·m/rad) with a small known damping (Kd = 0.04
N·m·s/rad) and alternating steps of several sizes (default 5, 10, 20°, six steps each). The joint
rings down after each step. The loop applies its torque 2 SYNC periods (4 ms) after the position
it came from, which acts like −Kp × 4 ms of damping: a P-only ring-down would barely decay, so
the small Kd keeps it decaying and the analysis removes both.

**Model.** With a = θ − θ_cmd: J ä + b ȧ + fc sgn(ȧ) = −Kp a + τ_bias. Between turning points
the motion direction d_n is constant, so each half-cycle starting from rest at a_n ends at

  a_{n+1} = −r a_n − (1 + r)(fc/Kp) d_n + (1 + r) τ_bias/Kp,  r = exp(−πζ/√(1 − ζ²)),

after exactly half a damped period π/ω_d.

**Analysis.** Turning points are detected with noise-scaled hysteresis and refined by parabolic
interpolation. A pooled linear regression over every half-cycle of every step gives r (viscous
damping) and fc (Coulomb friction) *separately*; the half-cycle durations give ω_d. Then
ω_n = ω_d/√(1 − ζ²), J = Kp/ω_n², b = 2Jζω_n: the inertia and damping the delayed loop shows. To
second order in s, a PD loop with delays T_p (spring) and T_d (damping) adds −Kd·T_d + Kp·T_p²/2
to the apparent inertia and Kd − Kp·T_p to the damping, so J and b are corrected with the run's
logged loop delays (`analysis_common.correct_for_loop`). 95% confidence intervals come from bootstrapping
whole steps, per-size fits check amplitude dependence, and the identified model is replayed on
every step (piecewise-analytic solution, using the measured static friction when a ramp
analysis exists) to report a fit percentage. Half-cycles are left out of the regression when the
delivered torque falls short of the command by more than 20% of its peak (back-EMF on the first
swing of large steps) or when they start inside the dead band where static friction and cogging
trap the joint, estimated as twice the median rest offset; the replay skips the same parts.

Plain logarithmic decrement attributes Coulomb decay to viscous damping, and the
oscillation-center offset is dominated by viscous decay; in simulation those biased b by +62%
and fc by +134%. On the simulated P4 wrist this method recovers J, b, and fc to within 2%,
5%, and 1%.

**Outputs.** `fig_step_identification`: (a) a normalized step response with the identified
model; (b) amplitude lost per half-cycle vs amplitude (slope = viscous, intercept = Coulomb).

### Static friction: ramp-and-hold
`ramp_input_experiment.py` → `ramp_input_experiment_analysis.py`

**Protocol.** A very soft P-only spring (Kp = 3 N·m/rad; its −Kp × 4 ms of loop damping is far
below the friction) drags the joint through a ramp-and-hold
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

**Protocol.** A stiff PD loop (Kp = 80 N·m/rad, Kd = 1.0 N·m·s/rad: about the stiffest the 500 Hz
loop holds stably, and stiffer than the cogging's peak stiffness so the joint tracks through each
detent) with velocity feedforward drives constant-velocity passes over −40…+40° in both
directions at 0.5–200 °/s (two round trips each, smooth sinusoidal speed blends). During the
cruise the p4wrist loop advances the setpoint itself (velocity-only commands), because
re-sending absolute positions under host timing jitter injects torque ripple. In the cruise the inertial torque is zero, so the measured torque is
friction(v) + conservative(θ).

**Analysis.** Forward and backward passes are paired on position bins:
friction = (τ₊ − τ₋)/2 and conservative = (τ₊ + τ₋)/2 (gravity, cable preload). Cogging is
identified first from the passes at or below 2 °/s: the peak of their averaged position spectrum
gives its cycles per motor revolution, and a least-squares harmonic series in absolute motor
position gives its amplitude and phase, saved as the cogging model. Every position bin is then
widened to whole cogging periods so cogging averages out. The loop's damping also draws a drag
from the cogging ripple, largest where the cogging frequency nears the loop's resonance; it is
predicted from the cogging model and the loop's dynamic stiffness (with the logged velocity
filter and loop delays) and subtracted, and passes where
it exceeds 20% of the friction are left out. The Stribeck + viscous model
τ_f = sgn(v)[f_c + (f_s − f_c)e^(−(|v|/v_s)²)] + bv is fitted by least squares with standard
errors. The conservative torque is summarized by its trend and by the remaining ripple after
noise correction. Friction is also reported as a percentage of the torque cap.

Positions here are joint angles in motor turns (`motor_pos_rev` = GEAR_RATIO × joint turns):
both differential motors turn N1 times per FE or UR turn, so each motor's cogging stays periodic.
On FE and UR the two motors' ripples add with a phase set by the other joint's angle, so the
cogging model belongs to that pose. Because the loop is at most ~80 N·m/rad stiff against
cogging that is itself ~14 N·m/rad stiff, part of the ripple becomes position ripple: in
simulation the amplitude reads ~20–25% low (the period to 0.5%).

**Outputs.** `fig_friction_velocity`: friction vs speed (log axis) with the fitted Stribeck
curve; `fig_friction_position`: torque vs position for both directions and their midline.

### Closed-loop position bandwidth: periodic multisine
`schroeder_bandwidth_experiment.py` → `schroeder_bandwidth_experiment_analysis.py`

**Protocol.** Stiff PD (Kp = 80 N·m/rad, Kd = 1.0 N·m·s/rad) tracks a periodic multisine: about
50 log-spaced lines from 0.2 to 30 Hz on a 10 s period, Schroeder phases for the actual power
spectrum plus crest-factor reduction. The amplitude spectrum is flat up to a corner and rolled
off above it; the corner is lowered automatically until the predicted inertial torque fits in
half the torque cap (the previous flat 0.1–20 Hz signal demanded about 1150 °/s and saturated the
actuator). Three amplitude levels (2, 5, 10°) of five periods each; the first period of each
level is discarded as a transient. Commands are evaluated at the actual send time, at the
loop's 500 Hz.
`send_velocity_feedforward` (default on) adds Kd·ṙ to the reference and changes the
closed-loop transfer function; state which one you used.

**Analysis.** In every period the command and the measured position are fitted at the excited
frequencies by least squares (exact with jittered timestamps), giving H per period, then the
mean and standard error across periods. Lines with SNR < 10 and levels that sat at the torque
cap more than 1% of the time are excluded. Reported: −3 dB bandwidth relative to the
low-frequency gain (or a lower bound when it lies above the tested range), −90° phase
frequency, resonance peak, and a fit of H = e^(−sτ)(Kp + FF·Kd·s)/(Js² + bs + e^(−sτ)(Kp + Kd D(s)))
giving J (a cross-check of the step test) and a model-extrapolated bandwidth. For the p4wrist
loop D(s) = s·e^(−sT/2)/(1 + s/ω_f) (velocity from differenced, filtered positions) and τ is
fixed at the loop's 2-period delay; J and b trade off when Kd dominates the damping (J within
~12% in simulation), so a free-delay fit is reported only as a check. The fitted b is an *effective* damping dominated by Kd and friction; report damping
from the step and sweep tests instead. Data from the earlier single-period 100 Hz script are
re-analyzed with a one-period DFT and flagged as legacy.

**Outputs.** `fig_bandwidth_bode`: magnitude and phase per amplitude with ±2 SE error bars and
the fitted model; `fig_bandwidth_time`: a 1 s zoom of command, response, and error.

### Closed-loop steps: rise time, settling, peak acceleration
`closed_loop_step_experiment.py` → `closed_loop_step_experiment_analysis.py`

**Protocol.** The same stiff PD gains, alternating steps of 1, 2, 5 and 10° (six each) with
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

In simulation the saturated acceleration matches cap/J within 1.5%, and the 1° step overshoots
~27%: a delay-free loop with these gains would overshoot ~2%, so this is the 4 ms loop delay
(the test checks it against a model of the loop's sampling, filter and CST ramp).

**Outputs.** `fig_closed_loop_steps`: (a) normalized responses per step size; (b) rise and
settling time vs step size.

### Z-width: renderable impedance range
`zwidth_experiment.py` → `zwidth_experiment_analysis.py`

**Protocol.** A virtual spring-damper τ = −K(θ − θ0) − Bθ' is rendered either by the p4wrist
500 Hz loop (`rt` mode, the limit of the wrist's own loop) or by the host from the previous
measurement as feedforward torque (`host_<rate>` modes, the situation of a PC-side haptic
simulation, which adds its own period and sample age to the loop's delay). For each B, K increases
geometrically (×1.4, then one bisection step) until a 1 N·m × 5 ms torque pulse produces a
response that does not decay. Each trial is contained by deviation, speed, and torque-chatter
stops, a brake, and a smooth return to a safe hold. Run with the handle free.

**Classification.** A trial is stable only if the pulse response halves within 0.5–0.75 s and has
decayed to the noise floor by the end of the observation. Ringing that friction stops only much
later still feels like a buzzing wall, so it counts as unstable; marginal stability counts as
unstable. Every trial is reclassified offline from the raw data.

**Outputs.** `fig_zwidth`: largest stable K vs B per mode, first unstable K marked, and lower
bounds (no instability up to the largest K tested) shown with arrows. With Coulomb friction the
boundary depends on the disturbance size, so state the pulse in the caption. In simulation the
500 Hz loop holds about 2.5× the host loop's stiffness at B = 0, roughly the ratio of their delays.

### Transmission stiffness, lost motion, and resonance (output clamped)
`transmission_stiffness_experiment.py` → `transmission_stiffness_experiment_analysis.py`

**Protocol.** With the output clamped, motor-side rotation is cable stretch plus any backlash.
Phase 1: slow triangular torque cycles 0 → +T → −T → 0 (default T = 80% of the cap). Phase 2
(off by default on this wrist): a low-crest-factor torque multisine (1–200 Hz) about a preload
that keeps the cable out of backlash. One P4 motor count is 0.0117° at FE (0.010° at PS), about
0.3 N·m of cable torque at ~1500 N·m/rad, so the static loops span only ~8 counts (stiffness to
~15%) and the multisine's motion is below one count; enable it only with an output encoder or a
larger amplitude (the experiment warns when its motion is under 5 counts).

**Analysis.** Stiffness is a common-slope fit with separate zero-torque intercepts per torque
side (so backlash does not bias the slope); lost motion is the intercept gap; the loading vs
unloading torque gap gives the friction hysteresis (which also widens the loop by 2f/k).
Phase 2: the commanded torque is integrated exactly as the drive applies it (for the P4, a
linear ramp to each cycle's target ending 2 SYNC periods later; for moteus, piecewise constant
from each command's arrival) and the motor angle is fitted at the sample times, giving the compliance
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
cool-downs between levels. Commands at 25 Hz (the loop logs every 2 ms). Needs the motor
thermistor (P4 0x3010:03, mapped into a slow PDO) converted by `device_config.MOTOR_THERMISTOR`,
whose scale is not yet known; otherwise it falls back loudly to the P4 amplifier temperature,
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
by the p4wrist 500 Hz loop or by the host from the previous sample, and a periodic torque
multisine (0.5–20 Hz, about 30 lines, designed for roughly equal motion per line, peak torque
within a budget) is added on top. Every target is first stability-checked with a torque pulse
(same rule as Z-width) and skipped if it rings; targets above 0.7× a measured Z-width limit are
skipped outright. Defaults: a near-free target in the 500 Hz loop (K = 5, B = 0), loop K = 20
(B = 0.2) and 60 (B = 0.5), host (250 Hz) K = 5 (B = 0), 20 (B = 0.2) and 30 (B = 0.3). Run with
the handle free.

**Analysis.** The disturbance torque is rebuilt from the stored design at each command's send
time and integrated exactly as the drive applies it (a ramp ending 2 SYNC periods after the loop
cycle that used it); position is fitted at the SYNC sample times. That gives the displayed
impedance Z = T/Θ per period, with standard errors. Subtracting the near-free target line by
line, Z_r = Z_target − Z_free + K_c·e^(−jωT_loop), removes inertia, friction and transmission
dynamics without a device model and leaves what the loop renders (the reference's own spring
also reaches the joint through the loop delay; treating it as delay-free would bias every
target by −K_c·T of damping). Each target is fitted with
(K_eff + jωB)e^(−jωT_eff). Delay turns stiffness into negative damping, B_r ≈ B − K·T_eff, so the
measured loop delay predicts the linear stability limit K_crit ≈ b/T_eff (viscous b from the step
analysis). The analysis compares that prediction with the Z-width result. It also reports the
fidelity bandwidth (stiffness within 10%, phase within 10°) and the free-space apparent inertia
and damping across frequency.

**Outputs.** `fig_impedance_spectrum`: (a) rendered stiffness K_r/K and (b) damping shift
B_r − B vs frequency for the loop and host targets, with fits at each loop's measured delay.
`fig_impedance_free`: free-space apparent inertia and damping vs frequency. In simulation the
free-space inertia is within 0.1–3%, the 500 Hz loop's delay reads 4.7–5.6 ms (between its
spring path, 4.2 ms, and its filtered damping path, ~6.8 ms) and the 250 Hz host loop's ~9 ms
(nominal 8.2). Even the wrist's own loop is delayed enough that an undamped spring above ~4
N·m/rad is linearly unstable (K_crit ≈ b/T); friction keeps such springs quiet at small
amplitude [5], so a pulse test overstates how stiff a wall can be under larger disturbances.
With Coulomb friction at these ~2° motions the subtraction biases stiffness by up to ~10%
(~2% without friction); mention it in the caption.

### Virtual-wall impacts: restitution, wall energy, rate-hardness
`virtual_wall_experiment.py` → `virtual_wall_experiment_analysis.py`

**Protocol.** The wrist launches itself under stiff tracking toward a unilateral virtual wall
(default at +10°), is released 1.5° before contact, and the wall alone governs the impact. The
wall is rendered either by the p4wrist loop in stay-within mode (bound at the wall, gains
selected per phase by kp/kd scales) or by the host loop from the previous sample. Sweep: two
rendering loops × B ∈ {0, 0.1} × two impact speeds × K = 10–500 N·m/rad, two repeats; the K
sweep stops once a wall chatters or launches the handle back at more than twice the impact
speed. Each impact ends with a damped catch and a brake. Run with the handle free.

**Metrics.** The coefficient of restitution e = |v_out|/|v_in| uses position slopes over 10 ms
just before entry and just after the wall torque releases (a delayed wall keeps pushing past exit);
e > 1 means the rendering injected energy. Wall work is ∫τ dθ over the contact, as a percentage of
the impact energy (positive = energy generated by the rendering). Rate-hardness [6] is the
initial rate of rise of the wall torque per unit impact speed, the perceptually relevant
hardness. The analysis also reports the effective torque delay, penetration, contact time and
effective stiffness.

**Outputs.** `fig_virtual_wall`: (a) torque–penetration loops of the loop's vs the host's walls
at the same K (a counter-clockwise loop means energy injected); (b) restitution vs K.
`fig_virtual_wall_hardness`: (a) rate-hardness / K; (b) wall energy gain vs K. In simulation the
500 Hz loop's wall (measured torque delay 3.9 ms; its CST delay is 4 ms) is passive at 20 N·m/rad
(e 0.86) and injects energy from 100 N·m/rad (e 1.66); the 250 Hz host wall (5.1 ms) already
injects energy at 20 N·m/rad. Colgate's bound b > K·T/2 puts the wrist's passive limit near 10
N·m/rad without virtual damping; friction stretches it at these impact speeds.

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
at the cap or the measured torque falls short of the controller's (voltage limit; for the P4
the controller's torque from 2 SYNC periods earlier, when it is actually delivered). A trial is
linear while |G/G0| stays within ±10%, phase within 10°, THD below 5%, and saturation below 2%.
A_max(f) is the largest linear amplitude, with its limiting mechanism; points stopped by a
safety cap are lower bounds.

**Outputs.** `fig_dynamic_envelope`: (a) A_max vs frequency (log-log) against the torque limit
τ_cap/(Jω²), the speed cap, and the amplitude cap, with optional user-supplied reference regions
(`reference_regions`, empty by default; cite your source); (b) peak torque at the limit. In
simulation (without Coulomb friction, which makes the small-signal reference itself amplitude
dependent) the envelope follows the actuator limit within 15% at the top frequencies; with
friction, within 7% at 8 Hz.

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
    <experiment>_data.csv          one row per loop cycle (2 ms), joint-side units
    <experiment>_metadata.json     config, p4wrist loop config, drive info, zero, git commit,
                                   events, abort reason (and ground truth when simulated)
    analysis/                      results tables, summary.csv/json, analysis metadata, figures/
data/p4wrist_zero_record.json      the zero kept between runs (git-ignored)
wrist_calibration.json             home pose and single-turn encoder survey (zero_calibration.py)
figures/ieee/                      latest paper figures
report/                            characterization_summary.csv, characterization_table.tex/.md
```

Data columns include the loop time (`time`), the host time of the command used (`t_send`), the
joint-side command (position, velocity feedforward, feedforward torque, gain scales, torque cap),
measured joint position, velocity and torque, the primary motor's q-axis current, the loop's
setpoint and commanded torque, bus voltage, temperatures, mode/fault, the phase and segment of
the protocol, all three joints' angles and torques, every motor's counts, torque target and
current, and the loop's lateness. Simulated runs (`*_sim`) are git-ignored.

## Hardware notes

* Everything here has been validated against the simulator only. Run each experiment first with
  `--sim`, then on hardware with the confirmation prompt, and watch the first run.
* Run `zero_calibration.py` and `wrist_check.py nudge` before the first experiment, and again
  after any change to the cables, capstans or pucks.
* The joint limits in `device_config.JOINT_RANGE_DEG` are placeholders: set them inside the
  wrist's hard stops for your home pose.
* The P4 torque scale (rated torque 0x6076, Kt 0x3011:04 and the phase-peak/rms convention)
  comes from the drives' own configuration; the torque calibration measures what it really is.
* The motor thermistor's scale is unknown; until `MOTOR_THERMISTOR` is set, the thermal test
  rates the amplifier, not the winding.
* The ATI Net F/T reader in `experiment_common.py` has not been tested on hardware; check its
  sign and scale against a known load.
* A run writes roughly 1 MB of CSV per 4 s of protocol at 500 Hz.

## Reporting in the paper

* Report mean ± SD (or the bootstrap 95% CI) and the number of trials; the summaries include
  both.
* State that positions are motor-encoder based (4096 counts per motor turn) and joint-referred,
  and give the transmission stiffness alongside the free-motion results.
* State the loop: 500 Hz CANopen SYNC, cyclic synchronous torque (4 ms from sample to torque),
  the velocity filter, and for host-rendered results the host rate.
* Apply (or at least report) the torque scale factor from the calibration.
* For bandwidth, state the controller gains and whether velocity feedforward was sent, since it
  changes the closed-loop transfer function.
* For Z-width, state the rendering loop (the 500 Hz loop vs a host rate), the disturbance pulse,
  and the stability criterion.
