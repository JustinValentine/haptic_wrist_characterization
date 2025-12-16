import asyncio
import math
import time
import moteus
import pandas as pd
import numpy as np

# --- USER CONFIGURATION ---
GEAR_RATIO = 10.0      # Example: 10:1 reduction
KP_JOINT_NM_RAD = 15.0 # Desired stiffness in Joint Space (Nm/rad) [cite: 650]
STEP_SIZE_DEG = 20.0   # Step size in degrees [cite: 654]
DURATION = 4.0         # Seconds per step (allow settling)

# Calculate Moteus Gains
# Kp_motor (Nm/rev) = Kp_joint (Nm/rad) * (2pi / N^2)
KP_MOTEUS = KP_JOINT_NM_RAD * (2 * np.pi) / (GEAR_RATIO**2)
KD_MOTEUS = 0.0        # KD MUST be 0 for this experiment 

async def run_experiment():
    qr = moteus.QueryResolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    
    # 1. Read initial position to avoid a violent jump at start
    state = await c.set_position(position=math.nan, query=True)
    start_pos_rev = state.values[moteus.Register.POSITION]
    
    # Calculate target positions in revolutions
    # Step size is peak-to-peak, so we move +/- half the step size
    delta_rev = (STEP_SIZE_DEG / 360.0) * GEAR_RATIO
    pos_low = start_pos_rev - (delta_rev / 2)
    pos_high = start_pos_rev + (delta_rev / 2)
    
    print(f"Starting Experiment.")
    print(f"Joint Kp: {KP_JOINT_NM_RAD} Nm/rad | Moteus Kp: {KP_MOTEUS:.4f} Nm/rev")
    print("WARNING: System is underdamped. Expect oscillations.")
    
    data = []
    start_time = time.time()
    
    # Sequence: Neutral -> Low -> High -> Low (collecting 2-3 cycles)
    targets = [start_pos_rev, pos_low, pos_high, pos_low, pos_high]
    
    for target in targets:
        step_start = time.time()
        while (time.time() - step_start) < DURATION:
            t_now = time.time() - start_time
            
            # Send Command: Pure P-control (Soft Spring)
            state = await c.set_position(
                position=target,
                kp_scale=1.0, 
                kp=KP_MOTEUS,
                kd_scale=0.0, # Explicitly disable damping
                ki_scale=0.0,
                query=True
            )
            
            # Log Data
            data.append({
                'time': t_now,
                'cmd_pos_rev': target,
                'motor_pos_rev': state.values[moteus.Register.POSITION],
                'motor_vel_rev_s': state.values[moteus.Register.VELOCITY],
                'motor_torque': state.values[moteus.Register.TORQUE]
            })
            
            await asyncio.sleep(0.002) # 500Hz logging

    await c.set_stop()
    
    # Save Data
    df = pd.DataFrame(data)
    df.to_csv('step_response_data.csv', index=False)
    print("Data saved to 'step_response_data.csv'")

if __name__ == '__main__':
    asyncio.run(run_experiment())