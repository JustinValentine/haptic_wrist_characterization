# Robotic Joint Characterization with Moteus

This repository implements a standard suite of characterization experiments for robotic manipulators using the [Moteus](https://mjbots.github.io/moteus/) motor controller. The protocols are derived from the *OpenWrist* exoskeleton validation and are designed to estimate the dynamic properties of a single degree-of-freedom (DOF) joint.

## Hardware and Units Assumptions
These scripts assume `motor_position.rotor_to_output_ratio = 1` on the moteus controller, so all moteus position readings are measured motor revolutions. The `GEAR_RATIO` constant in the scripts is the external cable/mechanical reduction from motor revolutions to the characterized joint/load:

* joint revs = motor revs / `GEAR_RATIO`
* ideal joint torque ~= motor torque * `GEAR_RATIO`
* moteus position PID gains are converted from load-side joint gains with:
  * `Kp_moteus = Kp_joint * 2*pi / GEAR_RATIO^2`
  * `Kd_moteus = Kd_joint * 2*pi / GEAR_RATIO^2`

The torque conversion is the ideal load-side value before losses, compliance, or cable efficiency effects. If you configure moteus to report output-side revolutions instead, update or remove these external `GEAR_RATIO` conversions.

The experiment scripts configure `servo.pid_position.kp`, `servo.pid_position.kd`, and `servo.pid_position.ki` at startup through the moteus diagnostic configuration channel, then use `kp_scale`/`kd_scale` in each command. By default, these gain changes are not persisted to flash.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Table of Contents
1. [Step Input Experiment: Inertia & Damping](#1-step-input-experiment-inertia--damping)
2. [Ramp Input Experiment: Static Friction](#2-ramp-input-experiment-static-friction)
3. [Schroeder Multisine Experiment: Bandwidth](#3-schroeder-multisine-experiment-bandwidth)
4. [Usage Guide](#usage-guide)
5. [IEEE-Ready Figures](#ieee-ready-figures)
6. [Data Files](#data-files)

---

## 1. Step Input Experiment: Inertia & Damping
**Objective:** Estimate the joint's total moment of inertia ($J$), viscous damping coefficient ($b$), and kinetic/Coulomb friction ($f_k$).

### Theory
The joint is modeled as a rotational mass-spring-damper system. By applying a step command with a pure Proportional (P) controller (setting $K_d=0$), the system response is dominated by its natural underdamped dynamics.

The equation of motion is:
$$J\ddot{\theta} + b\dot{\theta} + f_k \text{sign}(\dot{\theta}) = \tau$$

Where:
* $\tau = K_p(\theta_{cmd} - \theta)$ is the spring torque from the controller.
* $b$ represents the mechanical viscous damping. In this experiment $K_d=0$, so the controller should not add derivative damping.
* $f_k$ represents the kinetic (Coulomb) friction.

### Analysis Method
The analysis uses the **Logarithmic Decrement** method on the decaying oscillations. For each step segment, it subtracts the final equilibrium, detects same-signed extrema one oscillation period apart, and uses the median decay and period to estimate the damping ratio ($\zeta$) and natural frequency ($\omega_n$).

1.  **Logarithmic Decrement ($\delta$):**
    Calculated from the decay of same-signed oscillation peaks:
    $$\delta = \ln\left(\frac{|Y_i|}{|Y_{i+1}|}\right)$$

2.  **Damping Ratio ($\zeta$):**
    Derived from $\delta$:
    $$\zeta = \frac{\delta}{\sqrt{(2\pi)^2 + \delta^2}}$$

3.  **Parameter Extraction:**
    * **Natural Frequency ($\omega_n$):** Derived from the damped period $T_d$ and damping ratio.
        $$\omega_d = \frac{2\pi}{T_d}, \quad \omega_n = \frac{\omega_d}{\sqrt{1-\zeta^2}}$$
    * **Inertia ($J$):** Derived from the stiffness $K_p$ and natural frequency.
        $$J = \frac{K_p}{\omega_n^2}$$
    * **Viscous Damping ($b$):** Calculated using the inertia and damping ratio.
        $$b = 2 J \zeta \omega_n$$
    * **Kinetic Friction ($f_k$):** Estimated from the oscillation-center offset ($x_k$). Treat this as a rough Coulomb-friction estimate, not a direct measurement.
        $$f_k = x_k K_p$$

---

## 2. Ramp Input Experiment: Static Friction
**Objective:** Measure the Static Friction (Stiction) torque ($f_s$) required to initiate movement.

### Theory
Static friction is the torque threshold that must be overcome for the joint to transition from zero velocity to motion. This experiment uses a "quasi-static" approach where torque is increased linearly and slowly until the joint slips.

### Experimental Protocol
1.  **Controller:** Use a "soft" P-controller (low $K_p$, zero $K_d$).
2.  **Trajectory:** Command a slow ramp (e.g., 2.5°/s) followed by a pause. Repeat this "Ramp-and-Hold" pattern from zero to the positive limit, back to zero, to the negative limit, and back to zero.
3.  **Measurement:** Monitor the backward-differentiated joint velocity. When velocity along the ramp exceeds the configured breakaway threshold after a short settled period, record the **commanded torque** from the previous sample.
    * *Note:* Commanded torque is used because measured torque can be noisy at zero velocity.

$$f_s = \tau_{cmd}[i_{breakaway} - 1]$$

---

## 3. Schroeder Multisine Experiment: Bandwidth
**Objective:** Determine the closed-loop position bandwidth of the system: the frequency where tracking magnitude drops by -3 dB relative to the low-frequency gain.

### Theory
Bandwidth is a standard measure of a robot's dynamic performance and "transparency". To measure it efficiently, we use a **Schroeder Multisine** signal. This is a deterministic signal composed of a sum of sinusoids that excites a wide frequency range (e.g., 0.1 to 20 Hz) simultaneously.

The raw signal is constructed as a sum of unit cosines on frequency bins spaced by $1/T$, then normalized so its largest absolute position command is `AMPLITUDE_DEG`:
$$u_{raw}(t) = \sum_{k=1}^{N} \cos(2\pi f_k t + \phi_k)$$
$$f_k = f_{min} + \frac{k-1}{T}$$

The phases ($\phi_k$) are chosen to reduce the peak-to-peak amplitude of the summed signal, helping avoid actuator saturation while keeping energy spread across the requested frequency range:
$$\phi_k = -\frac{k(k-1)\pi}{N}$$

### Analysis Method
1.  **Excitation:** Drive the joint with the Schroeder signal using a stiff PD controller (representing nominal operation). The generated signal starts near zero to avoid a large initial jump.
2.  **Frequency Response:** Calculate the Transfer Function estimate ($H(f)$) between the Command Input ($U$) and Measured Position ($Y$) using the Cross-Spectral Density.
    $$H(f) = \frac{P_{xy}(f)}{P_{xx}(f)}$$
3.  **Bandwidth:** Normalize the magnitude by the low-frequency gain and find the cutoff frequency $f_c$ where the response drops below -3 dB ($ \approx 0.707$). The analysis also reports coherence and ignores bins with too little command excitation.

---

## Usage Guide
1. Set `GEAR_RATIO`, gains, amplitudes, and workspace limits in the relevant experiment script.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run one experiment script at a time while the device is in a safe test fixture.
4. Run the paired analysis script after the CSV is collected.

```bash
python step_input_experiment.py
python step_input_experiment_analysis.py

python ramp_input_experiment.py
python ramp_input_experiment_analysis.py

python schroeder_bandwidth_experiment.py
python schroeder_bandwidth_experiment_analysis.py
```

Review the generated plots before trusting the numerical results. The plots show the detected peaks, breakaway points, or frequency-response coherence used by the analysis.

---

## IEEE-Ready Figures
The analysis scripts export paper-ready line-art figures into `figures/ieee/` whenever they run. Each figure is saved as:

* `.pdf` vector artwork
* `.eps` vector artwork
* `.png` at 600 dpi
* `.json` metadata describing the source data and analysis settings

The figure styling is designed for IEEE manuscripts:

* single-column width by default (`3.5 in`)
* serif font with embedded PDF/EPS fonts where supported by Matplotlib
* 8 pt axis labels and 7 pt tick labels/legends
* no in-plot titles, so captions can be written in the paper
* grayscale-safe line styles and markers instead of color-only encodings
* SI-style axis units such as `Time (s)` and `Torque (N m)`

Generated figure basenames:

* `fig_step_response`
* `fig_static_friction`
* `fig_static_friction_breakaway_check`
* `fig_bandwidth_time_response`
* `fig_bandwidth_frequency_response`

Use the vector `.pdf` or `.eps` files for manuscript submission when possible. Use the `.png` files for quick previews or systems that require raster graphics.

---

## Data Files
The experiment scripts write:

* `step_response_data.csv`
* `friction_ramp_data.csv`
* `bandwidth_data.csv`

The experiment scripts also write collection metadata:

* `step_response_collection_metadata.json`
* `friction_ramp_collection_metadata.json`
* `bandwidth_collection_metadata.json`

The analysis scripts write:

* `step_response_results.csv`
* `step_response_summary.csv`
* `static_friction_results.csv`
* `static_friction_summary.csv`
* `bandwidth_response.csv`
* `bandwidth_summary.csv`
* analysis metadata JSON files for each processed experiment

Each collected CSV includes measured motor revolutions, relative joint position, command position, motor torque, estimated joint torque, configured gains, and the gear ratio used for conversion.
