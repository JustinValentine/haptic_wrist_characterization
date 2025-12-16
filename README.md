# Robotic Joint Characterization with Moteus

This repository implements a standard suite of characterization experiments for robotic manipulators using the [Moteus](https://mjbots.github.io/moteus/) motor controller. The protocols are derived from the *OpenWrist* exoskeleton validation and are designed to estimate the dynamic properties of a single degree-of-freedom (DOF) joint.

## Table of Contents
1. [Step Input Experiment: Inertia & Damping](#1-step-input-experiment-inertia--damping)
2. [Ramp Input Experiment: Static Friction](#2-ramp-input-experiment-static-friction)
3. [Schroeder Multisine Experiment: Bandwidth](#3-schroeder-multisine-experiment-bandwidth)
4. [Usage Guide](#usage-guide)

---

## 1. Step Input Experiment: Inertia & Damping
**Objective:** Identify the joint's total moment of inertia ($J$), viscous damping coefficient ($b$), and kinetic friction ($f_k$).

### Theory
The joint is modeled as a rotational mass-spring-damper system. By applying a step command with a pure Proportional (P) controller (setting $K_d=0$), the system response is dominated by its natural underdamped dynamics.

The equation of motion is:
$$J\ddot{\theta} + b\dot{\theta} + f_k \text{sign}(\dot{\theta}) = \tau$$

Where:
* $\tau = K_p(\theta_{cmd} - \theta)$ is the spring torque from the controller.
* $b$ represents the mechanical viscous damping ($b = b_{mech} + K_d$, where $K_d=0$)[cite: 646].
* $f_k$ represents the kinetic (Coulomb) friction[cite: 638].

### Analysis Method
We use the **Logarithmic Decrement** method on the decaying oscillations. By measuring the amplitude of successive peaks ($Y_i$) and valleys, we calculate the damping ratio ($\zeta$) and natural frequency ($\omega_n$).

1.  **Logarithmic Decrement ($\beta$):**
    Calculated using successive peaks and valleys[cite: 716]:
    $$\beta = -\frac{1}{\pi} \ln\left( -\frac{Y_{i+1} - Y_{i-1}}{Y_i - Y_{i-2}} \right)$$

2.  **Damping Ratio ($\zeta$):**
    Derived from $\beta$[cite: 718]:
    $$\zeta = \sqrt{\frac{\beta^2}{\beta^2 + 1}}$$

3.  **Parameter Extraction:**
    * **Inertia ($J$):** Derived from the stiffness $K_p$ and natural frequency[cite: 722].
        $$J = \frac{K_p}{\omega_n^2}$$
    * **Viscous Damping ($b$):** Calculated using the inertia and damping ratio[cite: 723].
        $$b = 2 J \zeta \omega_n$$
    * **Kinetic Friction ($f_k$):** Estimated from the offset of the oscillation center ($x_k$).
        $$f_k = x_k K_p$$

---

## 2. Ramp Input Experiment: Static Friction
**Objective:** Measure the Static Friction (Stiction) torque ($f_s$) required to initiate movement.

### Theory
Static friction is the torque threshold that must be overcome for the joint to transition from zero velocity to motion. This experiment uses a "quasi-static" approach where torque is increased linearly and slowly until the joint slips.

### Experimental Protocol
1.  **Controller:** Use a "soft" P-controller (low $K_p$, zero $K_d$).
2.  **Trajectory:** Command a slow ramp (e.g., 2.5°/s) followed by a pause. Repeat this "Ramp-and-Hold" pattern across the workspace.
3.  **Measurement:** Monitor the velocity. The instant the velocity becomes non-zero (breaks away), record the **commanded torque** from the previous time step.
    * *Note:* Commanded torque is used because measured torque can be noisy at zero velocity.

$$f_s = \tau_{cmd} \quad \text{at} \quad t = t_{breakaway} - \Delta t$$

---

## 3. Schroeder Multisine Experiment: Bandwidth
**Objective:** Determine the Closed-Loop Position Bandwidth of the system (the frequency at which tracking performance drops by -3dB).

### Theory
Bandwidth is a standard measure of a robot's dynamic performance and "transparency". To measure it efficiently, we use a **Schroeder Multisine** signal. This is a deterministic signal composed of a sum of sinusoids that excites a wide frequency range (e.g., 0.1 to 20 Hz) simultaneously.

The signal is constructed as:
$$u_m(t) = \sum_{k=1}^{N} A_m \cos(2\pi \omega_k t + \phi_k)$$

The phases ($\phi_k$) are specifically chosen to minimize the peak-to-peak amplitude of the signal, preventing actuator saturation while maximizing energy:
$$\phi_k = -\frac{k(k-1)\pi}{N}$$

### Analysis Method
1.  **Excitation:** Drive the joint with the Schroeder signal using a stiff PD controller (representing nominal operation).
2.  **Frequency Response:** Calculate the Transfer Function estimate ($H(f)$) between the Command Input ($U$) and Measured Position ($Y$) using the Cross-Spectral Density.
    $$H(f) = \frac{P_{xy}(f)}{P_{xx}(f)}$$
3.  **Bandwidth:** Find the cutoff frequency $f_c$ where the magnitude $|H(f)|$ drops below -3 dB ($ \approx 0.707$).

---

### Configuration
Update the `GEAR_RATIO` and `KP_JOINT` constants in each script to match your hardware. Ensure `KP` values are converted correctly between Joint Space (Nm/rad) and Motor Space (Nm/rev).