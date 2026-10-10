"""CANopen frame encoding/decoding for the Barrett P4.

Pure functions only -- no I/O. CANopen splits the 11-bit CAN ID into a 4-bit
function code and a 7-bit node ID. The P4 uses the standard pre-defined
connection set:

====================  ==============  =========================
Message               COB-ID          Direction
====================  ==============  =========================
NMT                   0x000           master -> all
SYNC                  0x080           master -> all
EMCY                  0x080 + node    device -> master
TPDO1..4              0x180/0x280/0x380/0x480 + node
RPDO1..4              0x200/0x300/0x400/0x500 + node
SDO request           0x600 + node    master -> device
SDO response          0x580 + node    device -> master
Heartbeat             0x700 + node    device -> master
====================  ==============  =========================

Only expedited SDO transfers (1-4 bytes) are supported, which matches the P4
(it does not implement segmented or block SDOs). All multi-byte data is
little-endian.
"""

import struct
from dataclasses import dataclass

from .exceptions import PuckProtocolError
from .objects import Obj

MAX_NODE = 127

COB_NMT = 0x000
COB_SYNC = 0x080
COB_EMCY = 0x080         # + node
COB_TPDO = (0x180, 0x280, 0x380, 0x480)   # + node
COB_RPDO = (0x200, 0x300, 0x400, 0x500)   # + node
COB_SDO_REQUEST = 0x600  # + node, master -> device
COB_SDO_RESPONSE = 0x580 # + node, device -> master
COB_HEARTBEAT = 0x700    # + node


def node_of(can_id: int) -> int:
    return can_id & 0x7F


def function_code(can_id: int) -> int:
    return can_id & 0x780


def check_node(node: int) -> int:
    if not 1 <= node <= MAX_NODE:
        raise ValueError(f"node ID {node} out of range 1-{MAX_NODE}")
    return node


# ---------------------------------------------------------------------------
# NMT / SYNC / heartbeat
# ---------------------------------------------------------------------------

def nmt(command: int, node: int = 0) -> "tuple[int, bytes]":
    """NMT frame; ``node=0`` addresses every device on the bus."""
    return COB_NMT, bytes([int(command), node])


def sync() -> "tuple[int, bytes]":
    return COB_SYNC, b""


def is_heartbeat(can_id: int, data: bytes) -> bool:
    return function_code(can_id) == COB_HEARTBEAT and len(data) == 1


def is_emcy(can_id: int, data: bytes) -> bool:
    return (function_code(can_id) == COB_EMCY and node_of(can_id) != 0
            and len(data) == 8)


@dataclass(frozen=True)
class Emergency:
    node: int
    code: int             # 16-bit CANopen error code (0x0000 = cleared)
    error_register: int   # copy of 0x1001,0
    data: bytes

    def __str__(self):
        from .objects import EMCY_CODES

        meaning = EMCY_CODES.get(self.code, "unknown")
        return (f"EMCY from node {self.node}: 0x{self.code:04X} ({meaning}), "
                f"error register 0x{self.error_register:02X}")


def decode_emcy(can_id: int, data: bytes) -> Emergency:
    if not is_emcy(can_id, data):
        raise PuckProtocolError(f"not an EMCY frame: 0x{can_id:03X} {data.hex()}")
    code = int.from_bytes(data[0:2], "little")
    return Emergency(node_of(can_id), code, data[2], bytes(data[3:]))


# ---------------------------------------------------------------------------
# Expedited SDO
# ---------------------------------------------------------------------------

_SDO_DOWNLOAD_CMD = {1: 0x2F, 2: 0x2B, 4: 0x23}   # bytes -> command specifier
_SDO_UPLOAD_RESP = {0x4F: 1, 0x4B: 2, 0x43: 4}


def encode_value(value: int, size: int, signed: bool) -> bytes:
    try:
        return int(value).to_bytes(size, "little", signed=signed)
    except OverflowError:
        kind = "signed" if signed else "unsigned"
        raise ValueError(
            f"value {value} does not fit in {size} {kind} byte(s)"
        ) from None


def sdo_download(node: int, obj: Obj, value: int) -> "tuple[int, bytes]":
    """Expedited SDO write of ``value`` to ``obj``."""
    if obj.size not in _SDO_DOWNLOAD_CMD:
        raise ValueError(f"expedited SDO size must be 1, 2, or 4, got {obj.size}")
    payload = (
        bytes([_SDO_DOWNLOAD_CMD[obj.size]])
        + struct.pack("<HB", obj.index, obj.sub)
        + encode_value(value, obj.size, obj.signed).ljust(4, b"\x00")
    )
    return COB_SDO_REQUEST + check_node(node), payload


def sdo_upload(node: int, index: int, sub: int) -> "tuple[int, bytes]":
    """SDO read request for object ``index,sub``."""
    return (COB_SDO_REQUEST + check_node(node),
            bytes([0x40]) + struct.pack("<HB", index, sub) + b"\x00" * 4)


@dataclass(frozen=True)
class SdoResponse:
    node: int
    index: int
    sub: int
    kind: str                  # "upload" | "download" | "abort"
    data: bytes = b""          # raw value bytes (upload)
    abort_code: int = 0

    def value(self, signed: bool) -> int:
        return int.from_bytes(self.data, "little", signed=signed)


def decode_sdo_response(can_id: int, data: bytes) -> SdoResponse:
    if function_code(can_id) != COB_SDO_RESPONSE or len(data) < 8:
        raise PuckProtocolError(
            f"not an SDO response: 0x{can_id:03X} {data.hex()}"
        )
    node = node_of(can_id)
    cmd = data[0]
    index, sub = struct.unpack_from("<HB", data, 1)
    if cmd == 0x80:
        code = int.from_bytes(data[4:8], "little")
        return SdoResponse(node, index, sub, "abort", abort_code=code)
    if cmd == 0x60:
        return SdoResponse(node, index, sub, "download")
    if cmd in _SDO_UPLOAD_RESP:
        size = _SDO_UPLOAD_RESP[cmd]
        return SdoResponse(node, index, sub, "upload", data=bytes(data[4:4 + size]))
    if cmd & 0xE3 == 0x42:  # expedited upload, size not indicated
        return SdoResponse(node, index, sub, "upload", data=bytes(data[4:8]))
    raise PuckProtocolError(f"unsupported SDO command specifier 0x{cmd:02X}")


def sdo_abort(node: int, index: int, sub: int, code: int) -> "tuple[int, bytes]":
    """Abort frame as sent by the device (used by the mock)."""
    return (COB_SDO_RESPONSE + check_node(node),
            bytes([0x80]) + struct.pack("<HB", index, sub)
            + code.to_bytes(4, "little"))
