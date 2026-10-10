"""Device-level settings shared by every experiment and analysis script.

This branch drives the three-joint P4 wrist (PS on puck 6, FE/UR differential
on pucks 7 and 8) through the p4wrist library, on the bench: no WAM, no
passive J5, no safety board. Edit this file once for your wrist and bench.
Experiment-specific settings (gains, amplitudes, durations) live in the
Config dataclass at the top of each experiment script.

All joint quantities are load-side. Experiments characterize one joint
(JOINT) while the library holds the other two at their start angles. Joint
positions in the data are relative to the start pose of each run.
"""
import math

# --- Wrist and transmission -----------------------------------------------------
JOINT = 'FE'                       # Joint under test: 'PS', 'FE' or 'UR'
JOINT_NAMES = ('PS', 'FE', 'UR')
JOINT_INDEX = JOINT_NAMES.index(JOINT)
CAN_INTERFACE = 'can0'             # SocketCAN interface (1 Mbit/s)
NODES = (6, 7, 8)                  # Pucks of motors 6 (PS), 7 and 8 (FE/UR differential)

# Motor angle = J2MP * joint angle (rows motors 6-8, columns PS FE UR), from
# libbarrett-canopen config/wam8g.conf as commissioned on 2026-09-22:
#   m6 = -N2 PS,  m7 = N1 (FE + UR),  m8 = N1 (-FE + UR)
# FE is the difference of the differential motors, UR the sum. The positive
# directions of FE and UR have not been checked physically; flip a column's
# sign here (both motors of FE, say) if a joint turns the wrong way.
N1 = 7.5077
N2 = 8.7456
J2MP = ((-N2, 0.0, 0.0),
        (0.0, N1, N1),
        (0.0, -N1, N1))
POSITION_SIGN = (1, 1, 1)          # Per motor; +1 on the wrist pucks (wam8g.conf)
TORQUE_SIGN = (1, 1, 1)

# Effective reduction of the joint under test: the largest motor turn per
# joint turn. The analyses use it to express positions in motor turns, so
# each motor's cogging (periodic in its own angle) stays periodic. For FE
# and UR both differential motors turn N1 times per joint turn.
GEAR_RATIO = max(abs(row[JOINT_INDEX]) for row in J2MP)

# --- Zeroing ---------------------------------------------------------------------
# Home pose in joint angles (deg): where the wrist is placed to zero it. The
# WAM home was PS 0, FE -93.3, UR -1.5 (calibration_data/wam8g/zerocal.conf);
# on the bench, pick a repeatable pose and record it with zero_calibration.py.
HOME_DEG = (0.0, 0.0, 0.0)
ZERO_CALIBRATION_FILE = 'wrist_calibration.json'   # written by zero_calibration.py
ZERO_RECORD_FILE = 'data/p4wrist_zero_record.json'  # kept between runs (no reset since)

# --- Safety limits ------------------------------------------------------------------
# Enforced by the p4wrist loop (500 Hz), absolute joint angles in deg. These
# are placeholders: set them inside the wrist's hard stops for your home pose.
JOINT_RANGE_DEG = {'PS': (-90.0, 90.0), 'FE': (-60.0, 60.0), 'UR': (-30.0, 30.0)}
MAX_JOINT_VELOCITY_DEG_S = 1000.0   # Every joint; the loop brakes and disables above this
MAX_JOINT_TORQUE_NM = 3.0           # Cap on every joint's command
MAX_MOTOR_PERMILLE = 1000           # Per-motor torque target cap (per mille of rated)
WATCHDOG_TIMEOUT_S = 0.1            # The loop brakes if the host stops commanding
FAULT_BRAKE_S = 0.5                 # Brake (phases grounded) this long, then disable

# Checked by the host on every sample, relative to the start pose of a run.
JOINT_LIMITS_DEG = (-45.0, 45.0)
MAX_MOTOR_TEMPERATURE_C = 90.0      # Needs MOTOR_THERMISTOR below; ignored if absent
MAX_BOARD_TEMPERATURE_C = 75.0      # P4 amplifier temperature (0x3000:02)

# How the library holds the joints not under test (PD about their start angle).
# The WAM8g profile used kp 20/10/10 N m/rad, kd 0.1 N m s/rad on J6-J8.
HOLD_KP_NM_RAD = {'PS': 20.0, 'FE': 10.0, 'UR': 10.0}
HOLD_KD_NMS_RAD = {'PS': 0.1, 'FE': 0.1, 'UR': 0.1}
HOLD_MAX_TORQUE_NM = 2.0

# --- p4wrist loop -----------------------------------------------------------------
LOOP_PERIOD_S = 0.002              # One SYNC per 2 ms (500 Hz), as on the WAM. 0.001 (1 kHz)
                                   # halves the PC loop's delay and doubles the sampling; with
                                   # only 3 pucks the bus is ~55% loaded (drive loop: use 2 ms)
VELOCITY_FILTER_HZ = 100.0         # Filter on differenced positions (libbarrett: 180 rad/s,
                                   # ~29 Hz). The loop's ~3 ms delay caps stable PD stiffness near
                                   # 40 N m/rad at 30 Hz and ~80 at 100 Hz (p4_sim gain scan).
SYNC_LOSS_TIMEOUT_MS = 20          # Drives fault this long after SYNC stops (host death)
REALTIME_PRIORITY = 80             # SCHED_FIFO for the loop thread (needs CAP_SYS_NICE); 0 = off
# The P4 reports q-axis current as a phase peak and Kt per A rms (as the
# libbarrett-canopen notes read them), so torque = Kt x I / sqrt(2). Not yet
# checked on this wrist: torque_calibration_experiment.py measures the truth.
CURRENT_TORQUE_FACTOR = 1.0 / math.sqrt(2.0)
# Measure the command-to-torque delay at the start of every run (a small torque
# pulse; the drive's measured current shows when it arrives) and use it in the
# analyses instead of the nominal 2 SYNC periods.
MEASURE_LOOP_DELAY = True
LOOP_DELAY_PROBE_NM = 0.15
# What the pucks report about their motors (resistance, inductance, rotor
# inertia, Kt, current limits, loop gains), saved at every hardware connect;
# --sim runs build the simulated motors from it when it exists.
DRIVE_PARAMS_FILE = 'data/p4wrist_drive_params.json'
SIM_USES_DRIVE_PARAMS = True
# Motor thermistor (0x3010:03) to deg C. Its scale is unknown, so it is not
# used until this is set to a function of the raw reading.
MOTOR_THERMISTOR = None

# --- Optional sensors ------------------------------------------------------------
# External torque sensor used by the torque calibration and transparency
# experiments.
#   'manual'    - type the reading from a force gauge at each hold (calibration only)
#   'ati_netft' - ATI Net F/T over UDP (not hardware-tested in this repo)
#   None        - no sensor
EXTERNAL_TORQUE_SENSOR = {'type': 'manual', 'lever_arm_m': 0.10}
# EXTERNAL_TORQUE_SENSOR = {
#     'type': 'ati_netft',
#     'host': '192.168.1.1',
#     'axis': 'Tz',               # F/T axis aligned with the wrist axis
#     'counts_per_unit': 1e6,     # cfgcpt (torque) or cfgcpf (force) from netftapi2.xml
#     'lever_arm_m': None,        # Set when reading a force axis at a lever arm
#     'sign': 1,
# }

# --- Prior estimates used for planning and safety checks ----------------------
# Update these after running the identification experiments; they size the
# excitation signals and predict torque requirements before anything moves.
NOMINAL_JOINT_INERTIA_KG_M2 = 0.005
NOMINAL_JOINT_DAMPING_NMS_RAD = 0.02
NOMINAL_COULOMB_FRICTION_NM = 0.09
