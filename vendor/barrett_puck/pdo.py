"""PDO mapping: pack/unpack process data and configure it on the device.

A :class:`PdoMap` describes one RPDO or TPDO slot: which object-dictionary
entries it carries, in order. ``DEFAULT_MAPS`` mirrors the mapping shipped in
Barrett's DevKit configuration CSV (manual, Example 2):

* RPDO1: controlword (u16), mode of operation (i8), target torque (i16)
* RPDO2: target velocity (i32), target position (i32)
* TPDO1: statusword (u16), mode display (i8), actual position (i32)
* TPDO2: actual velocity (i32), actual current (i16)

All PDOs apply/transmit on every SYNC (type 0).
"""

from dataclasses import dataclass

from . import frames
from .exceptions import PuckProtocolError
from .objects import OD, Obj


@dataclass(frozen=True)
class PdoMap:
    slot: int                   # 1-4
    direction: str              # "rx" (master -> device) or "tx"
    entries: "tuple[Obj, ...]"
    transmission_type: int = 0  # 0 = apply/transmit on every SYNC

    def __post_init__(self):
        if self.direction not in ("rx", "tx"):
            raise ValueError("direction must be 'rx' or 'tx'")
        if not 1 <= self.slot <= 4:
            raise ValueError("PDO slot must be 1-4")
        if sum(e.size for e in self.entries) > 8:
            raise ValueError("mapped entries exceed the 8-byte CAN frame")
        if len(self.entries) > 4:
            raise ValueError("standard CAN PDOs carry at most 4 entries")

    def cob(self, node: int) -> int:
        base = frames.COB_RPDO if self.direction == "rx" else frames.COB_TPDO
        return base[self.slot - 1] + node

    @property
    def comm_index(self) -> int:
        # RPDO comm params at 0x1400+, TPDO at 0x1800+.
        return (0x1400 if self.direction == "rx" else 0x1800) + self.slot - 1

    @property
    def map_index(self) -> int:
        # RPDO mapping at 0x1600+, TPDO at 0x1A00+.
        return (0x1600 if self.direction == "rx" else 0x1A00) + self.slot - 1

    def encode(self, values: "dict[tuple[int, int], int]") -> bytes:
        """Pack ``{(index, sub): value}`` into a PDO payload.

        Entries missing from ``values`` are sent as zero (the P4 only acts
        on the target matching its present mode of operation, so zero-filled
        inactive targets are harmless).
        """
        out = b""
        for e in self.entries:
            out += frames.encode_value(values.get(e.addr, 0), e.size, e.signed)
        return out

    def decode(self, data: bytes) -> "dict[tuple[int, int], int]":
        expected = sum(e.size for e in self.entries)
        if len(data) != expected:
            raise PuckProtocolError(
                f"{self.direction.upper()}PDO{self.slot} payload is "
                f"{len(data)} bytes, mapping expects {expected}"
            )
        values, offset = {}, 0
        for e in self.entries:
            values[e.addr] = int.from_bytes(
                data[offset:offset + e.size], "little", signed=e.signed
            )
            offset += e.size
        return values

    def mapping_values(self) -> "list[int]":
        """The 0x1600/0x1A00 entry words: [index<<16 | sub<<8 | bit-length]."""
        return [(e.index << 16) | (e.sub << 8) | e.bits for e in self.entries]


DEFAULT_RPDO1 = PdoMap(1, "rx", (OD.CONTROLWORD, OD.MODE_OF_OPERATION,
                                 OD.TARGET_TORQUE))
DEFAULT_RPDO2 = PdoMap(2, "rx", (OD.TARGET_VELOCITY, OD.TARGET_POSITION))
DEFAULT_TPDO1 = PdoMap(1, "tx", (OD.STATUSWORD, OD.MODE_DISPLAY,
                                 OD.ACTUAL_POSITION))
DEFAULT_TPDO2 = PdoMap(2, "tx", (OD.ACTUAL_VELOCITY, OD.ACTUAL_CURRENT))

DEFAULT_MAPS = (DEFAULT_RPDO1, DEFAULT_RPDO2, DEFAULT_TPDO1, DEFAULT_TPDO2)
