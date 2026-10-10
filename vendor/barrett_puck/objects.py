"""Object dictionary entries, enums, and bit definitions for the Barrett P4.

Source: "Barrett Puck P4 CANopen Manual", Revision BA. The P4 implements
CiA DS301 (communication) and DS402 (drives and motion control). Standard
profile objects live at 0x6000-0x6FFF; Barrett manufacturer-specific objects
live at 0x2000-0x3FFF.
"""

import enum
from dataclasses import dataclass


@dataclass(frozen=True)
class Obj:
    """One object-dictionary entry: address plus wire encoding."""

    index: int
    sub: int
    size: int        # bytes: 1, 2, or 4
    signed: bool
    name: str = ""

    @property
    def addr(self) -> "tuple[int, int]":
        return (self.index, self.sub)

    @property
    def bits(self) -> int:
        return self.size * 8

    def __repr__(self):
        return f"Obj(0x{self.index:04X},{self.sub} {self.name})"


def _o(index, sub, size, signed, name):
    return Obj(index, sub, size, signed, name)


class OD:
    """Catalog of the P4 object dictionary entries this library uses."""

    # --- DS301 communication profile -------------------------------------
    DEVICE_TYPE          = _o(0x1000, 0, 4, False, "device type")
    ERROR_REGISTER       = _o(0x1001, 0, 1, False, "error register")
    MFG_STATUS           = _o(0x1002, 0, 4, False, "manufacturer status")
    ERROR_COUNT          = _o(0x1003, 0, 1, False, "error history count")
    SYNC_COB_ID          = _o(0x1005, 0, 4, False, "SYNC COB ID")
    SAVE_ALL             = _o(0x1010, 1, 4, False, "save all parameters")
    HEARTBEAT_PERIOD     = _o(0x1017, 0, 2, False, "heartbeat period (ms)")
    VENDOR_ID            = _o(0x1018, 1, 4, False, "vendor ID")
    PRODUCT_CODE         = _o(0x1018, 2, 4, False, "product code")
    REVISION             = _o(0x1018, 3, 4, False, "revision number")
    SERIAL               = _o(0x1018, 4, 4, False, "serial number")
    NMT_STARTUP          = _o(0x1F80, 0, 4, False, "NMT startup behavior")

    # --- DS402 states and modes ------------------------------------------
    CONTROLWORD          = _o(0x6040, 0, 2, False, "controlword")
    STATUSWORD           = _o(0x6041, 0, 2, False, "statusword")
    LAST_ERROR           = _o(0x603F, 0, 2, False, "last EMCY error code")
    MODE_OF_OPERATION    = _o(0x6060, 0, 1, True, "mode of operation")
    MODE_DISPLAY         = _o(0x6061, 0, 1, True, "mode of operation display")
    SUPPORTED_MODES      = _o(0x6502, 0, 4, False, "supported drive modes")

    # --- Position ----------------------------------------------------------
    ACTUAL_POSITION      = _o(0x6064, 0, 4, True, "actual position (cts)")
    POSITION_DEMAND      = _o(0x60FC, 0, 4, True, "position demand (cts)")
    FOLLOWING_ERROR      = _o(0x60F4, 0, 4, True, "following error (cts)")
    TARGET_POSITION      = _o(0x607A, 0, 4, True, "target position (cts)")
    POSITION_OFFSET      = _o(0x60B0, 0, 4, True, "position offset (cts)")
    POSITION_MIN         = _o(0x607D, 1, 4, True, "position minimum (cts)")
    POSITION_MAX         = _o(0x607D, 2, 4, True, "position maximum (cts)")
    PROFILE_VELOCITY     = _o(0x6081, 0, 4, False, "profile velocity (cts/s)")
    END_VELOCITY         = _o(0x6082, 0, 4, False, "endpoint velocity (cts/s)")
    ACCELERATION         = _o(0x6083, 0, 4, False, "acceleration (cts/s^2)")
    DECELERATION         = _o(0x6084, 0, 4, False, "deceleration (cts/s^2)")
    QUICK_STOP_DECEL     = _o(0x6085, 0, 4, False, "quick-stop decel (cts/s^2)")

    # --- Velocity ----------------------------------------------------------
    TARGET_VELOCITY      = _o(0x60FF, 0, 4, True, "target velocity (cts/s)")
    ACTUAL_VELOCITY      = _o(0x606C, 0, 4, True, "actual velocity (cts/s)")
    VELOCITY_DEMAND      = _o(0x606B, 0, 4, True, "velocity demand (cts/s)")
    VELOCITY_OFFSET      = _o(0x60B1, 0, 4, True, "velocity offset (cts/s)")
    MAX_VELOCITY         = _o(0x6080, 0, 4, False, "max velocity (cts/s)")

    # --- Torque / current ----------------------------------------------------
    TARGET_TORQUE        = _o(0x6071, 0, 2, True, "target torque (rated/1000)")
    TORQUE_DEMAND        = _o(0x6074, 0, 2, True, "torque demand (rated/1000)")
    TORQUE_OFFSET        = _o(0x60B2, 0, 2, True, "torque offset (rated/1000)")
    TORQUE_SLOPE         = _o(0x6087, 0, 4, False, "torque slope (rated/1000/s)")
    RATED_TORQUE         = _o(0x6076, 0, 4, False, "motor rated torque (mNm)")
    ACTUAL_CURRENT       = _o(0x6078, 0, 2, True, "actual current (peak/1000)")
    PEAK_CURRENT         = _o(0x3011, 9, 2, True, "peak current (mA)")
    CONTINUOUS_CURRENT   = _o(0x3011, 8, 2, True, "continuous current (mA)")

    # --- Homing -------------------------------------------------------------
    HOMING_METHOD        = _o(0x6098, 0, 1, True, "homing method")
    HOMING_OFFSET        = _o(0x607C, 0, 4, True, "homing offset (cts)")

    # --- Cyclic synchronous ---------------------------------------------------
    INTERP_PERIOD        = _o(0x60C2, 1, 1, False, "interpolation period value")
    INTERP_PERIOD_SCALE  = _o(0x60C2, 2, 1, True, "interpolation period 10^n")

    # --- Barrett manufacturer-specific ----------------------------------------
    BUS_VOLTAGE          = _o(0x3000, 1, 2, False, "bus voltage (0.1 V)")
    TEMPERATURE          = _o(0x3000, 2, 2, False, "amplifier temperature (C)")
    ENCODER_RESOLUTION   = _o(0x3013, 1, 4, False, "encoder resolution (cts/rev)")
    ENCODER_USER_ZERO    = _o(0x3013, 2, 4, False, "encoder user zero (cts)")
    RAW_ENCODER          = _o(0x3012, 1, 2, True, "raw encoder reading (cts)")
    MOTOR_TEMP_LIMIT     = _o(0x220A, 0, 1, False, "motor temp limit (C)")
    POSITION_KP          = _o(0x2382, 1, 2, False, "position Kp")
    POSITION_KVFF        = _o(0x2382, 2, 2, False, "velocity feed-forward")
    POSITION_KAFF        = _o(0x2382, 3, 2, False, "acceleration feed-forward")
    VELOCITY_KP          = _o(0x2381, 1, 4, False, "velocity Kp (float bits)")
    VELOCITY_KI          = _o(0x2381, 2, 4, False, "velocity Ki (float bits)")


#: Magic value written to 0x1010,n to persist parameters ("SAVE" in ASCII).
SAVE_MAGIC = 0x65766173


class OpMode(enum.IntEnum):
    """DS402 modes of operation (0x6060 / 0x6061)."""

    IDLE = 0                  # current controller off, phases tied to ground
    PROFILE_POSITION = 1      # PP
    PROFILE_VELOCITY = 3      # PV
    PROFILE_TORQUE = 4        # PT
    HOMING = 6                # HM (method 37, home on current position)
    CYCLIC_POSITION = 8       # CSP
    CYCLIC_VELOCITY = 9       # CSV
    CYCLIC_TORQUE = 10        # CST
    PHASE_VOLTAGE = 12        # PVCA, used during motor calibration


class DriveState(enum.Enum):
    """DS402 motion-control state, decoded from the statusword."""

    NOT_READY = "not ready to switch on"
    SWITCH_ON_DISABLED = "switch on disabled"
    READY_TO_SWITCH_ON = "ready to switch on"
    SWITCHED_ON = "switched on"
    OPERATION_ENABLED = "operation enabled"
    QUICK_STOP_ACTIVE = "quick stop active"
    FAULT_REACTION = "fault reaction active"
    FAULT = "fault"

    @staticmethod
    def from_statusword(sw: int) -> "DriveState":
        if sw & 0x4F == 0x00:
            return DriveState.NOT_READY
        if sw & 0x4F == 0x40:
            return DriveState.SWITCH_ON_DISABLED
        if sw & 0x6F == 0x21:
            return DriveState.READY_TO_SWITCH_ON
        if sw & 0x6F == 0x23:
            return DriveState.SWITCHED_ON
        if sw & 0x6F == 0x27:
            return DriveState.OPERATION_ENABLED
        if sw & 0x6F == 0x07:
            return DriveState.QUICK_STOP_ACTIVE
        if sw & 0x4F == 0x0F:
            return DriveState.FAULT_REACTION
        if sw & 0x4F == 0x08:
            return DriveState.FAULT
        return DriveState.NOT_READY


class CW:
    """Controlword (0x6040) command values and bits."""

    SHUTDOWN = 0x06           # -> Ready To Switch On
    SWITCH_ON = 0x07          # -> Switched On
    ENABLE_OPERATION = 0x0F   # -> Operation Enabled
    DISABLE_VOLTAGE = 0x00    # -> Switch On Disabled
    QUICK_STOP = 0x02
    FAULT_RESET = 0x80        # rising edge clears latched faults

    # Mode-specific bits.
    NEW_SETPOINT = 0x10       # PP: accept a new waypoint (rising edge)
    START_HOMING = 0x10       # HM: begin homing
    IMMEDIATE = 0x20          # PP: replace active profile (vs. buffer)
    RELATIVE = 0x40           # PP: target is relative to present position
    HALT = 0x100


class SW:
    """Statusword (0x6041) bits."""

    READY_TO_SWITCH_ON = 0x0001
    SWITCHED_ON = 0x0002
    OPERATION_ENABLED = 0x0004
    FAULT = 0x0008
    VOLTAGE_ENABLED = 0x0010
    QUICK_STOP = 0x0020
    SWITCH_ON_DISABLED = 0x0040
    WARNING = 0x0080
    TARGET_REACHED = 0x0400
    INTERNAL_LIMIT = 0x0800
    SETPOINT_ACK = 0x1000     # PP; also "homing attained" in HM
    FOLLOWING_ERROR = 0x2000  # PP


class NmtState(enum.IntEnum):
    """NMT states as reported in heartbeat messages."""

    BOOT_UP = 0x00
    STOPPED = 0x04
    OPERATIONAL = 0x05
    PRE_OPERATIONAL = 0x7F


class NmtCommand(enum.IntEnum):
    START = 0x01
    STOP = 0x02
    PRE_OPERATIONAL = 0x80
    RESET_NODE = 0x81
    RESET_COMMUNICATION = 0x82


#: EMCY error codes the P4 can emit (manual, Table 19).
EMCY_CODES = {
    0x0000: "error cleared",
    0x2310: "current limit active (i2t)",
    0x2320: "short circuit detected",
    0x3210: "bus over-voltage",
    0x3220: "bus under-voltage",
    0x4210: "amplifier over-temperature",
    0x4310: "motor over-temperature",
    0x6010: "watchdog reset",
    0x6320: "parameter error",
    0x7122: "motor phasing error",
    0x7320: "encoder feedback fault",
    0x7321: "encoder magnet distance fault",
    0x73A0: "position wrapped",
    0x8110: "CAN overrun (frames lost)",
    0x8140: "recovered from bus-off",
    0x8180: "SYNC loss",
    0x8411: "velocity tracking warning",
    0x8414: "velocity tracking fault",
    0x8418: "velocity limit active",
    0x8480: "acceleration limit active",
    0x8611: "position tracking warning",
    0x8613: "homing error",
    0x8614: "position tracking fault",
}
