import asyncio
import math
import time
import moteus
import pandas as pd
import numpy as np

# --- USER CONFIGURATION ---
GEAR_RATIO = 10.0       # Example: 10:1 reduction
KP_JOINT_NM_RAD = 3.0   # Very Soft Kp (Thesis used ~2-5 for different joints)
RAMP_STEP_DEG = 5.0     # 5 degree increments
RAMP_DURATION = 2.0     # 2 seconds to move 5 degrees (2.5 deg/s)
PAUSE_DURATION = 2.0    # 2 second pause
WORKSPACE_LIMIT_DEG = 45.0 # How far to move in one direction before stopping

# Units Calculation
KP_MOTEUS = KP_JOINT_NM_RAD * (2 * np.pi) / (GEAR_RATIO**2)

async def run_ramp_experiment():
    qr = moteus.QueryResolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    
    # 1. Initialize
    state = await c.set_position(position=math.nan, query=True)
    start_pos_rev = state.values[moteus.Register.POSITION]
    
    print(f"Starting Static Friction Ramp Experiment.")
    print(f"Joint Kp: {KP_JOINT_NM_RAD} Nm/rad (Soft Spring)")
    print(f"Ramping {RAMP_STEP_DEG} deg over {RAMP_DURATION} s...")
    
    data = []
    start_time = time.time()
    
    # Generate Waypoints (Start -> Max -> Start -> Min -> Start)
    # This covers the workspace in both directions
    # Note: Thesis moved through full workspace. We will do a simple sweep here.
    waypoints_deg = np.arange(0, WORKSPACE_LIMIT_DEG + RAMP_STEP_DEG, RAMP_STEP_DEG)
    
    # Current theoretical target
    current_target_rev = start_pos_rev
    
    for i in range(len(waypoints_deg)):
        # Calculate next target position
        # Direction: We just move incrementally positive for this demo logic
        # You can expand this list to go negative/backwards if needed.
        delta_rev = (RAMP_STEP_DEG / 360.0) * GEAR_RATIO
        next_target_rev = current_target_rev + delta_rev
        
        # --- PHASE 1: RAMP (Move 5 deg slowly) ---
        ramp_start_time = time.time()
        while (time.time() - ramp_start_time) < RAMP_DURATION:
            t_local = time.time() - ramp_start_time
            progress = t_local / RAMP_DURATION # 0.0 to 1.0
            
            # Interpolate command
            cmd_pos = current_target_rev + (delta_rev * progress)
            
            # Send Soft Position Command
            state = await c.set_position(
                position=cmd_pos,
                kp_scale=1.0,
                kp=KP_MOTEUS,
                kd_scale=0.0, # Zero damping is critical 
                ki_scale=0.0,
                query=True
            )
            
            # Log Data
            data.append({
                'time': time.time() - start_time,
                'phase': 'ramp',
                'cmd_pos_rev': cmd_pos,
                'motor_pos_rev': state.values[moteus.Register.POSITION],
                'motor_vel_rev_s': state.values[moteus.Register.VELOCITY],
                'motor_torque': state.values[moteus.Register.TORQUE]
            })
            await asyncio.sleep(0.002) # 500Hz
            
        # --- PHASE 2: PAUSE (Hold 2s) ---
        # Update current baseline to the target we just reached
        current_target_rev = next_target_rev
        
        pause_start_time = time.time()
        while (time.time() - pause_start_time) < PAUSE_DURATION:
            # Hold the final position of the ramp
            state = await c.set_position(
                position=current_target_rev,
                kp_scale=1.0,
                kp=KP_MOTEUS,
                kd_scale=0.0,
                ki_scale=0.0,
                query=True
            )
            
            data.append({
                'time': time.time() - start_time,
                'phase': 'pause',
                'cmd_pos_rev': current_target_rev,
                'motor_pos_rev': state.values[moteus.Register.POSITION],
                'motor_vel_rev_s': state.values[moteus.Register.VELOCITY],
                'motor_torque': state.values[moteus.Register.TORQUE]
            })
            await asyncio.sleep(0.002)

    await c.set_stop()
    
    # Save
    df = pd.DataFrame(data)
    df.to_csv('friction_ramp_data.csv', index=False)
    print("Data saved to 'friction_ramp_data.csv'")

if __name__ == '__main__':
    asyncio.run(run_ramp_experiment())