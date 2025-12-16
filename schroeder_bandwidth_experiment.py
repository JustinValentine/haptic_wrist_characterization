import asyncio
import math
import time
import moteus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- USER CONFIGURATION ---
GEAR_RATIO = 10.0         # Example: 10:1 reduction
KP_JOINT_NM_RAD = 450.0   # High Stiffness (Example from Table 3.5 for Locked Joint) 
                          # Note: For a single moving DOF, start lower (e.g., 20-50) if unsafe.
KD_JOINT_NMS_RAD = 0.75   # Damping (Example from Table 3.5)

# Signal Parameters [cite: 886, 897]
AMPLITUDE_DEG = 10.0      # Conditioned between -10 and 10 degrees
FREQ_MIN = 0.1            # Hz
FREQ_MAX = 20.0           # Hz (Testing beyond the ~10Hz bandwidth found in thesis)
DURATION = 40.0           # Seconds (Matches Fig 3.6 duration) [cite: 922]
SAMPLING_FREQ = 100.0     # Hz (Command update rate)

# --- HELPER: Schroeder Signal Generator ---
def generate_schroeder_signal(f_min, f_max, T, fs, amp_deg):
    """
    Implements Equations 3.12 - 3.15 from the OpenWrist Thesis [cite: 894-901]
    """
    dt = 1/fs
    N_samples = int(T * fs)
    t = np.linspace(0, T, N_samples)
    
    # Eq 3.13: Number of frequency components
    # eta (freq resolution) is 1/T for a periodic signal over T
    eta = 1.0 / T 
    N_components = int((f_max - f_min) / eta) + 1
    
    print(f"Generating Multisine: {N_components} sinusoids from {f_min} to {f_max} Hz...")
    
    # Pre-calculate phases (Eq 3.14) and frequencies (Eq 3.15)
    # k is 1-based index in the thesis
    k = np.arange(1, N_components + 1)
    
    # Eq 3.15
    w_k = f_min + eta * (k - 1) # Hz
    w_k_rad = 2 * np.pi * w_k   # rad/s
    
    # Eq 3.14: Schroeder Phases
    # phi_k = -k(k-1) * pi / N_components
    phi_k = -k * (k - 1) * np.pi / N_components
    
    # Summation (Eq 3.12)
    # This is vectorised for speed
    # Signal shape: (Samples,)
    # We sum over the 'components' axis
    
    # This matrix is [Samples x Components] -> can be large. 
    # For T=40, fs=100 (4000 samples) and ~800 components, it fits in memory easily.
    # Argument of cos: 2*pi*f*t + phi
    arg = w_k_rad[np.newaxis, :] * t[:, np.newaxis] + phi_k[np.newaxis, :]
    u_m = np.sum(np.cos(arg), axis=1)
    
    # Normalize to desired amplitude
    # Thesis says "conditioned between -10 and 10"
    scale_factor = amp_deg / np.max(np.abs(u_m))
    u_m_scaled = u_m * scale_factor
    
    return t, u_m_scaled

# Generate Signal Offline
t_ref, pos_ref_deg = generate_schroeder_signal(FREQ_MIN, FREQ_MAX, DURATION, SAMPLING_FREQ, AMPLITUDE_DEG)

# --- MOTEUS UNITS ---
KP_MOTEUS = KP_JOINT_NM_RAD * (2 * np.pi) / (GEAR_RATIO**2)
KD_MOTEUS = KD_JOINT_NMS_RAD * (2 * np.pi) / (GEAR_RATIO**2)

async def run_bandwidth_experiment():
    qr = moteus.QueryResolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    
    # 1. Initialization
    state = await c.set_position(position=math.nan, query=True)
    start_pos_rev = state.values[moteus.Register.POSITION]
    
    print(f"Starting Bandwidth Experiment.")
    print(f"Duration: {DURATION}s | Freq: {FREQ_MIN}-{FREQ_MAX} Hz")
    print(f"Controller: Kp={KP_JOINT_NM_RAD} Kd={KD_JOINT_NMS_RAD} (Joint Units)")
    
    # Move to neutral (0 relative to start) gently before starting oscillation
    print("Centering...")
    for _ in range(100):
        await c.set_position(position=start_pos_rev, kp_scale=1.0, kp=KP_MOTEUS, kd_scale=1.0, kd=KD_MOTEUS)
        await asyncio.sleep(0.01)
        
    data = []
    start_loop_time = time.time()
    
    # 2. Execution Loop
    for i in range(len(t_ref)):
        loop_start = time.time()
        
        # Get Command for this timestep
        cmd_deg = pos_ref_deg[i]
        
        # Convert to Motor Revolutions
        # Delta from start + Command
        cmd_rev_delta = (cmd_deg / 360.0) * GEAR_RATIO
        target_pos_rev = start_pos_rev + cmd_rev_delta
        
        # Send Command
        # Note: We send the position trajectory. 
        # Standard bandwidth tests measure the CLOSED LOOP response T(s) = Y(s)/R(s).
        state = await c.set_position(
            position=target_pos_rev,
            velocity=0.0, # Letting PD handle the dynamics (Standard Step/Bandwidth methodology)
            kp_scale=1.0,
            kp=KP_MOTEUS,
            kd_scale=1.0,
            kd=KD_MOTEUS,
            query=True
        )
        
        # Log
        data.append({
            'time': time.time() - start_loop_time,
            'cmd_pos_rev': target_pos_rev,
            'cmd_pos_deg': cmd_deg, # Relative command
            'motor_pos_rev': state.values[moteus.Register.POSITION],
            'motor_vel_rev_s': state.values[moteus.Register.VELOCITY],
            'motor_torque': state.values[moteus.Register.TORQUE]
        })
        
        # Maintain Sampling Rate
        elapsed = time.time() - loop_start
        sleep_time = (1.0 / SAMPLING_FREQ) - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
            
    await c.set_stop()
    
    # Save
    df = pd.DataFrame(data)
    df.to_csv('bandwidth_data.csv', index=False)
    print("Data saved to 'bandwidth_data.csv'")

if __name__ == '__main__':
    # Visual check of the signal before running
    plt.plot(t_ref, pos_ref_deg)
    plt.title("Generated Schroeder Multisine Input (Check this!)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (deg)")
    plt.show()
    
    input("Press Enter to run the motor...")
    asyncio.run(run_bandwidth_experiment())