import json
import math
from pathlib import Path


MOTEUS_OUTPUT_RATIO_NOTE = (
    "This project assumes moteus motor_position.rotor_to_output_ratio = 1. "
    "GEAR_RATIO is the external cable reduction from measured motor revs to "
    "the characterized joint/load."
)


def joint_degrees_to_motor_revs(joint_deg, gear_ratio):
    return (joint_deg / 360.0) * gear_ratio


def motor_revs_to_joint_degrees(motor_revs, gear_ratio):
    return (motor_revs / gear_ratio) * 360.0


def motor_revs_to_joint_radians(motor_revs, gear_ratio):
    return (motor_revs / gear_ratio) * (2.0 * math.pi)


def joint_radians_to_motor_revs(joint_rad, gear_ratio):
    return (joint_rad / (2.0 * math.pi)) * gear_ratio


def motor_velocity_to_joint_deg_s(motor_rev_s, gear_ratio):
    return motor_revs_to_joint_degrees(motor_rev_s, gear_ratio)


def motor_torque_to_joint_torque(motor_torque_nm, gear_ratio):
    return motor_torque_nm * gear_ratio


def joint_gains_to_moteus(kp_joint_nm_rad, kd_joint_nms_rad, gear_ratio):
    """Convert load-side joint gains to moteus gains in motor-rev units.

    With moteus configured at rotor_to_output_ratio=1, position is measured in
    motor revolutions and torque is motor-side Nm. A cable reduction of N means
    joint angle = motor angle / N and joint torque ~= motor torque * N, so both
    Kp and Kd scale by 2*pi/N^2.
    """
    scale = (2.0 * math.pi) / (gear_ratio ** 2)
    return kp_joint_nm_rad * scale, kd_joint_nms_rad * scale


def make_query_resolution():
    import moteus

    qr = moteus.QueryResolution()

    optional_fields = {
        "mode": "INT8",
        "fault": "INT8",
        "voltage": "F32",
        "temperature": "F32",
    }

    for field, resolution in optional_fields.items():
        if hasattr(qr, field) and hasattr(moteus, resolution):
            setattr(qr, field, getattr(moteus, resolution))

    return qr


def state_value(state, register, default=math.nan):
    if register is None:
        return default
    try:
        return state.values[register]
    except (AttributeError, KeyError):
        return default


def state_register_value(state, register_group, name, default=math.nan):
    return state_value(state, getattr(register_group, name, None), default)


async def configure_position_pid(controller, kp_nm_per_rev, kd_nm_per_rev_s, ki=0.0,
                                 persist=False):
    """Configure moteus position PID gains for this run.

    The Python position command supports kp_scale/kd_scale, but not arbitrary
    kp/kd values. The actual gains live in servo.pid_position, so experiments
    set them through the diagnostic configuration channel and then use scales
    per command.
    """
    import moteus

    if not hasattr(moteus, "Stream"):
        raise RuntimeError(
            "This moteus Python package does not expose moteus.Stream; configure "
            "servo.pid_position.kp/kd/ki manually before running."
        )

    stream = moteus.Stream(controller, verbose=False)
    requested = {
        "kp": kp_nm_per_rev,
        "kd": kd_nm_per_rev_s,
        "ki": ki,
    }

    for name, value in requested.items():
        await stream.command(
            f"conf set servo.pid_position.{name} {value:.12g}".encode("utf8")
        )

    if persist:
        await stream.command(b"conf write")

    verified = {}
    for name in requested:
        response = await stream.command(
            f"conf get servo.pid_position.{name}".encode("utf8")
        )
        verified[name] = _decode_stream_response(response)

    return verified


def _decode_stream_response(response):
    if response is None:
        return ""
    if isinstance(response, bytes):
        return response.decode("utf8", errors="replace").strip()
    return str(response).strip()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
