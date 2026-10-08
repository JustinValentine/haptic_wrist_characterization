"""Device-level settings shared by every experiment and analysis script.

Edit this file once for your wrist. Experiment-specific settings (gains,
amplitudes, durations) live in the Config dataclass at the top of each
experiment script.

All joint quantities are load-side (after the cable reduction). Joint
positions are measured relative to the motor position at the start of each
experiment, so place the handle at its neutral pose before running.
"""

# --- Transmission and controller ----------------------------------------------
# Cable reductions (motor revs per joint rev) at the cable centreline: each
# drum's surface diameter plus one cable diameter (0.038 in = 0.965 mm).
#   PS: one stage, 18 -> 185 mm drums:             185.97 / 18.97
#   FE: two stages, 18 -> 36 mm then 15 -> 77 mm:  (36.97 / 18.97) * (77.97 / 15.97)
GEAR_RATIOS = {'PS': 9.806, 'FE': 9.518}
JOINT = 'PS'                      # Haptic joint under test: 'PS' or 'FE'
GEAR_RATIO = GEAR_RATIOS[JOINT]   # Cable reduction: motor revs per joint rev
CONTROLLER_ID = 1                 # moteus CAN ID of the joint under test

# --- Safety limits (checked on every sample; any violation stops the run) -----
JOINT_LIMITS_DEG = (-60.0, 60.0)    # Allowed travel relative to the start pose
MAX_JOINT_VELOCITY_DEG_S = 2000.0   # Abort above this joint speed
MAX_JOINT_TORQUE_NM = 10.0          # Torque cap sent with every command
MAX_MOTOR_TEMPERATURE_C = 90.0      # Needs a motor thermistor; ignored if absent
MAX_BOARD_TEMPERATURE_C = 75.0
WATCHDOG_TIMEOUT_S = 0.1            # moteus stops if the host goes quiet

# --- Optional sensors ----------------------------------------------------------
# Output-side (joint) encoder wired to moteus, e.g. configured as
# motor_position.sources.1 and reported through ENCODER_1_POSITION. Set to
# None when the joint has no output encoder.
OUTPUT_ENCODER = None
# OUTPUT_ENCODER = {
#     'register': 'ENCODER_1_POSITION',
#     'revs_per_joint_rev': 1.0,   # Encoder revs per joint rev
#     'sign': 1,                   # Flip to -1 if it counts opposite the motor
# }

# External torque sensor used by the torque calibration and transparency
# experiments.
#   'manual'        - type the reading from a force gauge at each hold (calibration only)
#   'moteus_analog' - load-cell amplifier on a moteus aux analog input
#   'ati_netft'     - ATI Net F/T over UDP (not hardware-tested in this repo)
#   None            - no sensor
EXTERNAL_TORQUE_SENSOR = {'type': 'manual', 'lever_arm_m': 0.10}
# EXTERNAL_TORQUE_SENSOR = {
#     'type': 'moteus_analog',
#     'register': 'AUX1_ANALOG_IN1',
#     'nm_per_unit': 25.0,    # Joint torque per register unit
#     'offset_units': 0.5,    # Register value at zero torque
# }
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
NOMINAL_JOINT_INERTIA_KG_M2 = 0.008
NOMINAL_JOINT_DAMPING_NMS_RAD = 0.02
NOMINAL_COULOMB_FRICTION_NM = 0.05
