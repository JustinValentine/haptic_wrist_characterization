import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
GEAR_RATIO = 10.0
KP_JOINT = 3.0  # Must match experiment (Nm/rad)
VELOCITY_THRESHOLD_DEG_S = 0.1 # Threshold to detect "movement initiation"

# Load Data
df = pd.read_csv('friction_ramp_data.csv')

# 1. Convert Units to Joint Space
df['joint_pos_rad'] = df['motor_pos_rev'] * (1/GEAR_RATIO) * (2*np.pi)
df['joint_cmd_rad'] = df['cmd_pos_rev'] * (1/GEAR_RATIO) * (2*np.pi)
df['joint_pos_deg'] = np.degrees(df['joint_pos_rad'])

# 2. Calculate "Commanded Torque" 
# Thesis: "inferred from the commanded torque when movement is initiated" 
# We use the theoretical spring force because measured torque is noisy at 0 velocity.
df['calc_torque'] = KP_JOINT * (df['joint_cmd_rad'] - df['joint_pos_rad'])

# 3. Calculate Backwards-Differentiated Velocity
# Thesis explicitly uses "backwards-differentiated velocity" 
# v[i] = (pos[i] - pos[i-1]) / dt
dt = df['time'].diff().fillna(0.002)
df['diff_vel_rad_s'] = df['joint_pos_rad'].diff() / dt
df['diff_vel_deg_s'] = np.degrees(df['diff_vel_rad_s'])
# Apply a slight smoothing filter to differentiation noise (optional but recommended)
df['diff_vel_deg_s'] = df['diff_vel_deg_s'].rolling(window=5, center=False).mean().fillna(0)

# 4. Detect Breakaway Events
static_friction_values = []
breakaway_indices = []

# We interpret the data segment by segment. 
# A "breakaway" happens when velocity jumps from near-zero to > threshold.
# We must ensure we are in a 'ramp' phase.
is_moving = False

for i in range(10, len(df)):
    # Only analyze during the RAMP phase
    if df['phase'].iloc[i] != 'ramp':
        is_moving = False # Reset flag during pauses
        continue
        
    current_vel = abs(df['diff_vel_deg_s'].iloc[i])
    
    # Check for initiation of movement
    if not is_moving and current_vel > VELOCITY_THRESHOLD_DEG_S:
        is_moving = True
        
        # Thesis: "one time step before... velocity becomes non-zero" 
        # We grab index i-1 or i-2
        idx_static = i - 2 
        
        breakaway_torque = abs(df['calc_torque'].iloc[idx_static])
        breakaway_pos = df['joint_pos_deg'].iloc[idx_static]
        
        static_friction_values.append({
            'pos_deg': breakaway_pos,
            'friction_nm': breakaway_torque
        })
        breakaway_indices.append(idx_static)

# 5. Output and Plotting
results = pd.DataFrame(static_friction_values)

if not results.empty:
    mean_friction = results['friction_nm'].mean()
    max_friction = results['friction_nm'].max()
    
    print("-" * 30)
    print("STATIC FRICTION RESULTS")
    print("-" * 30)
    print(f"Mean Static Friction: {mean_friction:.4f} Nm")
    print(f"Max Static Friction:  {max_friction:.4f} Nm")
    print(f"Points detected:      {len(results)}")
    
    # Plot 1: Friction vs Position (Reproducing Thesis Fig 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(results['pos_deg'], results['friction_nm'], color='red', label='Static Friction')
    plt.xlabel('Joint Position [deg]')
    plt.ylabel('Static Friction [Nm]')
    plt.title('Static Friction vs. Joint Position')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Time Series Verification
    # Show Torque and Velocity to visually confirm we picked the "breakaway" points correctly
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df['time'], df['calc_torque'], 'b-', label='Commanded Torque', alpha=0.6)
    ax1.set_ylabel('Torque [Nm]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['diff_vel_deg_s'], 'g-', label='Velocity', alpha=0.4)
    ax2.set_ylabel('Velocity [deg/s]', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Mark detected points
    breakaway_times = df['time'].iloc[breakaway_indices]
    breakaway_torques = df['calc_torque'].iloc[breakaway_indices]
    ax1.scatter(breakaway_times, breakaway_torques, c='r', marker='x', s=100, label='Detected Breakaway')
    
    plt.title("Breakaway Detection Verification")
    plt.show()

else:
    print("No breakaway events detected. Try lowering velocity threshold or checking data.")