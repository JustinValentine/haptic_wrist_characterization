import asyncio
from datetime import datetime, timezone
import time
import moteus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from experiment_common import (
    MOTEUS_OUTPUT_RATIO_NOTE,
    configure_position_pid,
    joint_degrees_to_motor_revs,
    joint_gains_to_moteus,
    make_query_resolution,
    motor_revs_to_joint_degrees,
    motor_torque_to_joint_torque,
    motor_velocity_to_joint_deg_s,
    state_register_value,
    state_value,
    write_json,
)

# --- USER CONFIGURATION ---
GEAR_RATIO = 10.0         # Cable reduction: motor revs per joint/load rev
KP_JOINT_NM_RAD = 450.0   # Load-side stiffness. Start lower if this is unsafe.
KD_JOINT_NMS_RAD = 0.75   # Load-side damping

# Signal Parameters
AMPLITUDE_DEG = 10.0      # Conditioned between -10 and 10 degrees
FREQ_MIN = 0.1            # Hz
FREQ_MAX = 20.0           # Hz (Testing beyond the ~10Hz bandwidth found in thesis)
DURATION = 40.0           # Seconds; frequency resolution is 1 / DURATION
SAMPLING_FREQ = 100.0     # Hz (Command update rate)
SEND_COMMAND_VELOCITY = True
PERSIST_MOTEUS_CONFIG = False
DATA_FILE = 'bandwidth_data.csv'
METADATA_FILE = 'bandwidth_collection_metadata.json'

# --- HELPER: Schroeder Signal Generator ---
def generate_schroeder_signal(f_min, f_max, T, fs, amp_deg, start_near_zero=True):
    """
    Generate a periodic Schroeder multisine and its analytic velocity.
    """
    N_samples = int(T * fs)
    t = np.arange(N_samples) / fs

    # Frequency components sit on DFT bins so the signal is periodic over T.
    eta = 1.0 / T
    N_components = int(round((f_max - f_min) / eta)) + 1

    print(f"Generating Multisine: {N_components} sinusoids from {f_min} to {f_max} Hz...")

    k = np.arange(1, N_components + 1)
    freqs_hz = f_min + eta * (k - 1)
    freqs_rad_s = 2 * np.pi * freqs_hz
    phi_k = -k * (k - 1) * np.pi / N_components

    arg = freqs_rad_s[np.newaxis, :] * t[:, np.newaxis] + phi_k[np.newaxis, :]
    position_deg = np.sum(np.cos(arg), axis=1)
    velocity_deg_s = np.sum(
        -freqs_rad_s[np.newaxis, :] * np.sin(arg),
        axis=1
    )

    position_deg -= np.mean(position_deg)

    # Normalize to desired amplitude
    scale_factor = amp_deg / np.max(np.abs(position_deg))
    position_deg *= scale_factor
    velocity_deg_s *= scale_factor

    if start_near_zero:
        velocity_scale = np.max(np.abs(velocity_deg_s))
        start_score = np.abs(position_deg) / amp_deg
        if velocity_scale > 0:
            start_score += np.abs(velocity_deg_s) / velocity_scale
        start_index = int(np.argmin(start_score))
        position_deg = np.roll(position_deg, -start_index)
        velocity_deg_s = np.roll(velocity_deg_s, -start_index)

    return t, position_deg, velocity_deg_s, freqs_hz

# Generate Signal Offline
t_ref, pos_ref_deg, vel_ref_deg_s, excitation_freqs_hz = generate_schroeder_signal(
    FREQ_MIN, FREQ_MAX, DURATION, SAMPLING_FREQ, AMPLITUDE_DEG
)

# --- MOTEUS UNITS ---
KP_MOTEUS, KD_MOTEUS = joint_gains_to_moteus(
    KP_JOINT_NM_RAD, KD_JOINT_NMS_RAD, GEAR_RATIO
)

async def run_bandwidth_experiment():
    qr = make_query_resolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    data = []
    verified = {}
    start_pos_rev = np.nan
    run_started_monotonic = time.monotonic()
    run_started_utc = datetime.now(timezone.utc).isoformat()

    try:
        verified = await configure_position_pid(
            c, KP_MOTEUS, KD_MOTEUS, ki=0.0, persist=PERSIST_MOTEUS_CONFIG
        )

        # Initialization
        state = await c.set_position(position=math.nan, query=True)
        start_pos_rev = state_value(state, moteus.Register.POSITION)

        print("Starting Bandwidth Experiment.")
        print(MOTEUS_OUTPUT_RATIO_NOTE)
        print(f"Duration: {DURATION}s | Freq: {FREQ_MIN}-{FREQ_MAX} Hz")
        print(f"Controller: Kp={KP_JOINT_NM_RAD} Kd={KD_JOINT_NMS_RAD} (load-side units)")
        print(f"Moteus PID: Kp={KP_MOTEUS:.6g} Nm/rev, Kd={KD_MOTEUS:.6g} Nm/(rev/s)")
        print(f"Verified config: {verified}")

        # Move to neutral gently before starting oscillation.
        print("Centering...")
        for _ in range(100):
            await c.set_position(
                position=start_pos_rev,
                velocity=0.0,
                kp_scale=1.0,
                kd_scale=1.0,
                query=False
            )
            await asyncio.sleep(0.01)

        start_loop_time = time.monotonic()

        for i, (cmd_deg, cmd_vel_deg_s) in enumerate(zip(pos_ref_deg, vel_ref_deg_s)):
            scheduled_time = start_loop_time + t_ref[i]
            sleep_time = scheduled_time - time.monotonic()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            target_pos_rev = start_pos_rev + joint_degrees_to_motor_revs(
                cmd_deg, GEAR_RATIO
            )
            cmd_vel_rev_s = joint_degrees_to_motor_revs(cmd_vel_deg_s, GEAR_RATIO)
            velocity_command = cmd_vel_rev_s if SEND_COMMAND_VELOCITY else 0.0

            state = await c.set_position(
                position=target_pos_rev,
                velocity=velocity_command,
                kp_scale=1.0,
                kd_scale=1.0,
                query=True
            )

            motor_pos_rev = state_value(state, moteus.Register.POSITION)
            motor_vel_rev_s = state_value(state, moteus.Register.VELOCITY)
            motor_torque_nm = state_value(state, moteus.Register.TORQUE)
            rel_motor_rev = motor_pos_rev - start_pos_rev

            data.append({
                'sample_index': i,
                'time': time.monotonic() - start_loop_time,
                't_ref': t_ref[i],
                'cmd_pos_rev': target_pos_rev,
                'cmd_pos_deg': cmd_deg,
                'cmd_vel_rev_s': cmd_vel_rev_s,
                'cmd_vel_deg_s': cmd_vel_deg_s,
                'sent_velocity_feedforward': SEND_COMMAND_VELOCITY,
                'motor_pos_rev': motor_pos_rev,
                'motor_pos_rel_rev': rel_motor_rev,
                'joint_pos_deg': motor_revs_to_joint_degrees(rel_motor_rev, GEAR_RATIO),
                'motor_vel_rev_s': motor_vel_rev_s,
                'joint_vel_deg_s': motor_velocity_to_joint_deg_s(motor_vel_rev_s, GEAR_RATIO),
                'motor_torque_nm': motor_torque_nm,
                'joint_torque_nm': motor_torque_to_joint_torque(motor_torque_nm, GEAR_RATIO),
                'gear_ratio': GEAR_RATIO,
                'kp_joint_nm_rad': KP_JOINT_NM_RAD,
                'kd_joint_nms_rad': KD_JOINT_NMS_RAD,
                'kp_moteus_nm_rev': KP_MOTEUS,
                'kd_moteus_nm_rev_s': KD_MOTEUS,
                'mode': state_register_value(state, moteus.Register, "MODE"),
                'fault': state_register_value(state, moteus.Register, "FAULT"),
            })

    finally:
        await c.set_stop()
        if data:
            df = pd.DataFrame(data)
            df.to_csv(DATA_FILE, index=False)
            write_json(METADATA_FILE, {
                'experiment': 'schroeder_bandwidth_experiment.py',
                'data_file': DATA_FILE,
                'run_started_utc': run_started_utc,
                'run_duration_s': time.monotonic() - run_started_monotonic,
                'num_samples': len(data),
                'measured_sample_frequency_hz': measured_sample_frequency(data),
                'moteus_controller_id': 1,
                'moteus_rotor_to_output_ratio_assumption': 1,
                'external_gear_ratio': GEAR_RATIO,
                'start_pos_rev': start_pos_rev,
                'kp_joint_nm_rad': KP_JOINT_NM_RAD,
                'kd_joint_nms_rad': KD_JOINT_NMS_RAD,
                'kp_moteus_nm_rev': KP_MOTEUS,
                'kd_moteus_nm_rev_s': KD_MOTEUS,
                'amplitude_deg': AMPLITUDE_DEG,
                'freq_min_hz': FREQ_MIN,
                'freq_max_hz': FREQ_MAX,
                'duration_s': DURATION,
                'sample_frequency_commanded_hz': SAMPLING_FREQ,
                'num_excitation_frequencies': len(excitation_freqs_hz),
                'send_command_velocity': SEND_COMMAND_VELOCITY,
                'persist_moteus_config': PERSIST_MOTEUS_CONFIG,
                'verified_moteus_pid_config': verified,
            })
            print(f"Data saved to '{DATA_FILE}'")
            print(f"Collection metadata saved to '{METADATA_FILE}'")


def measured_sample_frequency(data):
    if len(data) < 2:
        return np.nan
    duration = data[-1]['time'] - data[0]['time']
    if duration <= 0:
        return np.nan
    return (len(data) - 1) / duration

if __name__ == '__main__':
    # Visual check of the signal before running
    plt.plot(t_ref, pos_ref_deg)
    plt.title("Generated Schroeder Multisine Input (Check this!)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (deg)")
    plt.show()
    
    input("Press Enter to run the motor...")
    asyncio.run(run_bandwidth_experiment())
