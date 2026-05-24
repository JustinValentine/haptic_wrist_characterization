import asyncio
from datetime import datetime, timezone
import math
import time
import moteus
import pandas as pd

from experiment_common import (
    MOTEUS_OUTPUT_RATIO_NOTE,
    configure_position_pid,
    joint_gains_to_moteus,
    joint_degrees_to_motor_revs,
    make_query_resolution,
    motor_revs_to_joint_degrees,
    motor_torque_to_joint_torque,
    motor_velocity_to_joint_deg_s,
    state_register_value,
    state_value,
    write_json,
)

# --- USER CONFIGURATION ---
GEAR_RATIO = 10.0       # Cable reduction: motor revs per joint/load rev
KP_JOINT_NM_RAD = 15.0  # Desired load-side stiffness (Nm/rad)
STEP_SIZE_DEG = 20.0    # Peak-to-peak joint step size in degrees
DURATION = 4.0          # Seconds per step (allow settling)
SAMPLE_PERIOD = 0.002   # 500 Hz command/logging target
PERSIST_MOTEUS_CONFIG = False
DATA_FILE = 'step_response_data.csv'
METADATA_FILE = 'step_response_collection_metadata.json'

# Calculate Moteus Gains
# Kp_motor (Nm/motor-rev) = Kp_joint (Nm/rad) * (2pi / N^2)
KP_MOTEUS, KD_MOTEUS = joint_gains_to_moteus(KP_JOINT_NM_RAD, 0.0, GEAR_RATIO)

async def run_experiment():
    qr = make_query_resolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    data = []
    verified = {}
    start_pos_rev = math.nan
    run_started_monotonic = time.monotonic()
    run_started_utc = datetime.now(timezone.utc).isoformat()

    try:
        verified = await configure_position_pid(
            c, KP_MOTEUS, KD_MOTEUS, ki=0.0, persist=PERSIST_MOTEUS_CONFIG
        )

        # Read initial position to avoid a violent jump at start.
        state = await c.set_position(position=math.nan, query=True)
        start_pos_rev = state_value(state, moteus.Register.POSITION)

        half_step_rev = joint_degrees_to_motor_revs(STEP_SIZE_DEG / 2.0, GEAR_RATIO)
        pos_low = start_pos_rev - half_step_rev
        pos_high = start_pos_rev + half_step_rev

        print("Starting Step Input Experiment.")
        print(MOTEUS_OUTPUT_RATIO_NOTE)
        print(f"Joint Kp: {KP_JOINT_NM_RAD} Nm/rad")
        print(f"Moteus PID: Kp={KP_MOTEUS:.6g} Nm/rev, Kd={KD_MOTEUS:.6g} Nm/(rev/s)")
        print(f"Verified config: {verified}")
        print("WARNING: System is intentionally underdamped. Expect oscillations.")

        start_time = time.monotonic()

        # Sequence: Neutral -> Low -> High -> Low -> High.
        targets = [
            (0.0, start_pos_rev),
            (-STEP_SIZE_DEG / 2.0, pos_low),
            (STEP_SIZE_DEG / 2.0, pos_high),
            (-STEP_SIZE_DEG / 2.0, pos_low),
            (STEP_SIZE_DEG / 2.0, pos_high),
        ]

        sample_index = 0
        for segment, (target_joint_deg, target_motor_rev) in enumerate(targets):
            step_start = time.monotonic()
            while (time.monotonic() - step_start) < DURATION:
                t_now = time.monotonic() - start_time

                # Pure P-control: Kd is configured to zero and kd_scale is held at zero.
                state = await c.set_position(
                    position=target_motor_rev,
                    kp_scale=1.0,
                    kd_scale=0.0,
                    query=True
                )

                motor_pos_rev = state_value(state, moteus.Register.POSITION)
                motor_vel_rev_s = state_value(state, moteus.Register.VELOCITY)
                motor_torque_nm = state_value(state, moteus.Register.TORQUE)
                rel_motor_rev = motor_pos_rev - start_pos_rev

                data.append({
                    'sample_index': sample_index,
                    'time': t_now,
                    'segment': segment,
                    'cmd_pos_rev': target_motor_rev,
                    'cmd_joint_deg': target_joint_deg,
                    'motor_pos_rev': motor_pos_rev,
                    'motor_pos_rel_rev': rel_motor_rev,
                    'joint_pos_deg': motor_revs_to_joint_degrees(rel_motor_rev, GEAR_RATIO),
                    'motor_vel_rev_s': motor_vel_rev_s,
                    'joint_vel_deg_s': motor_velocity_to_joint_deg_s(motor_vel_rev_s, GEAR_RATIO),
                    'motor_torque_nm': motor_torque_nm,
                    'joint_torque_nm': motor_torque_to_joint_torque(motor_torque_nm, GEAR_RATIO),
                    'gear_ratio': GEAR_RATIO,
                    'kp_joint_nm_rad': KP_JOINT_NM_RAD,
                    'kp_moteus_nm_rev': KP_MOTEUS,
                    'kd_moteus_nm_rev_s': KD_MOTEUS,
                    'mode': state_register_value(state, moteus.Register, "MODE"),
                    'fault': state_register_value(state, moteus.Register, "FAULT"),
                })
                sample_index += 1

                await asyncio.sleep(SAMPLE_PERIOD)

    finally:
        await c.set_stop()
        if data:
            df = pd.DataFrame(data)
            df.to_csv(DATA_FILE, index=False)
            write_json(METADATA_FILE, {
                'experiment': 'step_input_experiment.py',
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
                'kd_joint_nms_rad': 0.0,
                'kp_moteus_nm_rev': KP_MOTEUS,
                'kd_moteus_nm_rev_s': KD_MOTEUS,
                'step_size_deg': STEP_SIZE_DEG,
                'duration_per_step_s': DURATION,
                'sample_period_commanded_s': SAMPLE_PERIOD,
                'persist_moteus_config': PERSIST_MOTEUS_CONFIG,
                'verified_moteus_pid_config': verified,
            })
            print(f"Data saved to '{DATA_FILE}'")
            print(f"Collection metadata saved to '{METADATA_FILE}'")


def measured_sample_frequency(data):
    if len(data) < 2:
        return math.nan
    duration = data[-1]['time'] - data[0]['time']
    if duration <= 0:
        return math.nan
    return (len(data) - 1) / duration

if __name__ == '__main__':
    asyncio.run(run_experiment())
