import asyncio
from datetime import datetime, timezone
import time
import moteus
import pandas as pd
import numpy as np

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
GEAR_RATIO = 10.0          # Cable reduction: motor revs per joint/load rev
KP_JOINT_NM_RAD = 3.0      # Very soft load-side stiffness (Nm/rad)
RAMP_STEP_DEG = 5.0        # Joint-space increment size
RAMP_DURATION = 2.0        # 2 seconds to move 5 degrees (2.5 deg/s)
PAUSE_DURATION = 2.0       # Pause between ramps so the joint can settle
WORKSPACE_LIMIT_DEG = 45.0 # Positive and negative travel from the start position
SAMPLE_PERIOD = 0.002      # 500 Hz command/logging target
PERSIST_MOTEUS_CONFIG = False
DATA_FILE = 'friction_ramp_data.csv'
METADATA_FILE = 'friction_ramp_collection_metadata.json'

# Units Calculation
KP_MOTEUS, KD_MOTEUS = joint_gains_to_moteus(KP_JOINT_NM_RAD, 0.0, GEAR_RATIO)


def build_ramp_targets(step_deg, workspace_limit_deg):
    targets = []

    def add_range(start, stop, step):
        value = start
        if step > 0:
            while value <= stop + 1e-9:
                targets.append(round(value, 10))
                value += step
        else:
            while value >= stop - 1e-9:
                targets.append(round(value, 10))
                value += step

    add_range(step_deg, workspace_limit_deg, step_deg)
    add_range(workspace_limit_deg - step_deg, 0.0, -step_deg)
    add_range(-step_deg, -workspace_limit_deg, -step_deg)
    add_range(-workspace_limit_deg + step_deg, 0.0, step_deg)

    return targets

async def run_ramp_experiment():
    qr = make_query_resolution()
    c = moteus.Controller(id=1, query_resolution=qr)
    await c.set_stop()
    data = []
    verified = {}
    start_pos_rev = np.nan
    targets_deg = []
    run_started_monotonic = time.monotonic()
    run_started_utc = datetime.now(timezone.utc).isoformat()

    try:
        verified = await configure_position_pid(
            c, KP_MOTEUS, KD_MOTEUS, ki=0.0, persist=PERSIST_MOTEUS_CONFIG
        )

        # Initialize from the measured motor position.
        state = await c.set_position(position=math.nan, query=True)
        start_pos_rev = state_value(state, moteus.Register.POSITION)

        targets_deg = build_ramp_targets(RAMP_STEP_DEG, WORKSPACE_LIMIT_DEG)

        print("Starting Static Friction Ramp Experiment.")
        print(MOTEUS_OUTPUT_RATIO_NOTE)
        print(f"Joint Kp: {KP_JOINT_NM_RAD} Nm/rad (soft spring)")
        print(f"Moteus PID: Kp={KP_MOTEUS:.6g} Nm/rev, Kd={KD_MOTEUS:.6g} Nm/(rev/s)")
        print(f"Verified config: {verified}")
        print(
            f"Sweeping {len(targets_deg)} ramp segments: "
            f"+/-{WORKSPACE_LIMIT_DEG} deg in {RAMP_STEP_DEG} deg steps."
        )

        start_time = time.monotonic()
        current_target_deg = 0.0
        sample_index = 0

        for segment, next_target_deg in enumerate(targets_deg):
            segment_start_deg = current_target_deg
            delta_deg = next_target_deg - segment_start_deg
            direction = int(np.sign(delta_deg))

            # Phase 1: ramp slowly to the next joint-space target.
            ramp_start_time = time.monotonic()
            while (time.monotonic() - ramp_start_time) < RAMP_DURATION:
                t_local = time.monotonic() - ramp_start_time
                progress = min(t_local / RAMP_DURATION, 1.0)
                cmd_joint_deg = segment_start_deg + (delta_deg * progress)
                cmd_pos_rev = start_pos_rev + joint_degrees_to_motor_revs(
                    cmd_joint_deg, GEAR_RATIO
                )

                state = await c.set_position(
                    position=cmd_pos_rev,
                    kp_scale=1.0,
                    kd_scale=0.0,
                    query=True
                )

                sample_index = append_sample(
                    data, sample_index, state, start_time, start_pos_rev,
                    cmd_pos_rev, cmd_joint_deg, segment, 'ramp', direction
                )
                await asyncio.sleep(SAMPLE_PERIOD)

            current_target_deg = next_target_deg
            hold_pos_rev = start_pos_rev + joint_degrees_to_motor_revs(
                current_target_deg, GEAR_RATIO
            )

            # Phase 2: pause at the new target before reversing/continuing.
            pause_start_time = time.monotonic()
            while (time.monotonic() - pause_start_time) < PAUSE_DURATION:
                state = await c.set_position(
                    position=hold_pos_rev,
                    kp_scale=1.0,
                    kd_scale=0.0,
                    query=True
                )

                sample_index = append_sample(
                    data, sample_index, state, start_time, start_pos_rev,
                    hold_pos_rev, current_target_deg, segment, 'pause', direction
                )
                await asyncio.sleep(SAMPLE_PERIOD)

    finally:
        await c.set_stop()
        if data:
            df = pd.DataFrame(data)
            df.to_csv(DATA_FILE, index=False)
            write_json(METADATA_FILE, {
                'experiment': 'ramp_input_experiment.py',
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
                'ramp_step_deg': RAMP_STEP_DEG,
                'ramp_duration_s': RAMP_DURATION,
                'pause_duration_s': PAUSE_DURATION,
                'workspace_limit_deg': WORKSPACE_LIMIT_DEG,
                'targets_deg': targets_deg,
                'sample_period_commanded_s': SAMPLE_PERIOD,
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


def append_sample(data, sample_index, state, start_time, start_pos_rev,
                  cmd_pos_rev, cmd_joint_deg, segment, phase, direction):
    motor_pos_rev = state_value(state, moteus.Register.POSITION)
    motor_vel_rev_s = state_value(state, moteus.Register.VELOCITY)
    motor_torque_nm = state_value(state, moteus.Register.TORQUE)
    rel_motor_rev = motor_pos_rev - start_pos_rev

    data.append({
        'sample_index': sample_index,
        'time': time.monotonic() - start_time,
        'segment': segment,
        'phase': phase,
        'direction': direction,
        'cmd_pos_rev': cmd_pos_rev,
        'cmd_joint_deg': cmd_joint_deg,
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
    return sample_index + 1

if __name__ == '__main__':
    asyncio.run(run_ramp_experiment())
