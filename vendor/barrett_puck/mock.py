"""In-process simulated P4 drives for development and testing without
hardware.

::

    transport = MockTransport()
    transport.add_puck(MockP4(1))
    motor = Controller(transport, 1)
    motor.connect()
    motor.enable()

The simulation covers the protocol surface this library uses: NMT states,
expedited SDO with aborts, the DS402 motion-control state machine, modes of
operation, PDO mapping (read from the simulated object dictionary, so
reconfiguring PDOs over SDO works), SYNC-latched cyclic setpoints, and EMCY.
Motion is simplified for determinism: position moves complete instantly,
velocity/torque targets are reflected straight into the feedback objects.
"""

import collections

from . import frames
from .objects import CW, OD, SAVE_MAGIC, NmtState, OpMode, SW
from .pdo import DEFAULT_MAPS


class MockP4:
    def __init__(self, id: int, *, counts_per_rev: int = 4096,
                 rated_torque_mnm: int = 10000, peak_current_ma: int = 17152,
                 position_counts: int = 0):
        self.id = id
        self.nmt_state = NmtState.PRE_OPERATIONAL
        self.ds402 = "SWITCH_ON_DISABLED"
        self._prev_controlword = 0
        self._setpoint_ack = False
        self._homing_attained = False
        self._rpdo_buffer: "dict[int, bytes]" = {}   # slot -> latest payload
        self._sync_count = 0
        self.outbox: "list[tuple[int, bytes]]" = []

        # Object dictionary: (index, sub) -> value. Sizes/signedness for
        # SDO replies come from _TYPES below.
        od = self.od = collections.defaultdict(int)
        od[OD.DEVICE_TYPE.addr] = 0x00000192          # DS402 device
        od[OD.VENDOR_ID.addr] = 0x42                  # arbitrary
        od[OD.PRODUCT_CODE.addr] = 0x0437             # "P4-37"
        od[OD.SERIAL.addr] = 1000 + id
        od[OD.ENCODER_RESOLUTION.addr] = counts_per_rev
        od[OD.RATED_TORQUE.addr] = rated_torque_mnm
        od[OD.PEAK_CURRENT.addr] = peak_current_ma
        od[OD.SUPPORTED_MODES.addr] = 0b101110001101
        od[OD.ACTUAL_POSITION.addr] = position_counts
        od[OD.TEMPERATURE.addr] = 35
        od[OD.BUS_VOLTAGE.addr] = 240                 # 24.0 V
        od[OD.NMT_STARTUP.addr] = 4                   # boot to pre-op
        self._install_default_pdo_config()

    # -- convenience accessors for tests --------------------------------

    @property
    def position_counts(self) -> int:
        return self.od[OD.ACTUAL_POSITION.addr]

    @position_counts.setter
    def position_counts(self, v: int) -> None:
        self.od[OD.ACTUAL_POSITION.addr] = int(v)

    @property
    def mode(self) -> OpMode:
        return OpMode(self.od[OD.MODE_DISPLAY.addr])

    @property
    def torque_demand(self) -> int:
        return self.od[OD.TORQUE_DEMAND.addr]

    @property
    def operation_enabled(self) -> bool:
        return self.ds402 == "OPERATION_ENABLED"

    def _install_default_pdo_config(self):
        for m in DEFAULT_MAPS:
            self.od[(m.comm_index, 1)] = m.cob(self.id)
            self.od[(m.comm_index, 2)] = m.transmission_type
            for i, word in enumerate(m.mapping_values(), start=1):
                self.od[(m.map_index, i)] = word
            self.od[(m.map_index, 0)] = len(m.entries)

    # -- statusword ------------------------------------------------------

    _SW_BASE = {
        "NOT_READY": 0x00,
        "SWITCH_ON_DISABLED": 0x40,
        "READY_TO_SWITCH_ON": 0x31,
        "SWITCHED_ON": 0x33,
        "OPERATION_ENABLED": 0x37,
        "QUICK_STOP": 0x17,
        "FAULT": 0x08,
    }

    def statusword(self) -> int:
        sw = self._SW_BASE[self.ds402]
        if self.operation_enabled:
            sw |= SW.TARGET_REACHED
            if self._setpoint_ack:
                sw |= SW.SETPOINT_ACK
            if self.mode == OpMode.HOMING and self._homing_attained:
                sw |= SW.SETPOINT_ACK | SW.TARGET_REACHED
        return sw

    # -- DS402 state machine ----------------------------------------------

    def _apply_controlword(self, cw: int) -> None:
        prev, self._prev_controlword = self._prev_controlword, cw
        if self.ds402 == "FAULT":
            if cw & CW.FAULT_RESET and not prev & CW.FAULT_RESET:
                self.ds402 = "SWITCH_ON_DISABLED"
                self.od[OD.ERROR_REGISTER.addr] = 0
            return
        if cw & 0x02 == 0:  # disable voltage
            self.ds402 = "SWITCH_ON_DISABLED"
            return
        if cw & 0x87 == 0x06:  # shutdown
            if self.ds402 in ("SWITCH_ON_DISABLED", "SWITCHED_ON",
                              "OPERATION_ENABLED", "READY_TO_SWITCH_ON"):
                self.ds402 = "READY_TO_SWITCH_ON"
            return
        if cw & 0x8F == 0x07:  # switch on
            if self.ds402 in ("READY_TO_SWITCH_ON", "OPERATION_ENABLED"):
                self.ds402 = "SWITCHED_ON"
            return
        if cw & 0x8F == 0x0F:  # switch on + enable operation
            if self.ds402 in ("READY_TO_SWITCH_ON", "SWITCHED_ON",
                              "OPERATION_ENABLED"):
                entering = self.ds402 != "OPERATION_ENABLED"
                self.ds402 = "OPERATION_ENABLED"
                if entering:
                    # Manual: mode display cleared on entering Op Enabled.
                    self.od[OD.MODE_DISPLAY.addr] = 0
            if self.operation_enabled:
                self._mode_specific_bits(cw, prev)

    def _mode_specific_bits(self, cw: int, prev: int) -> None:
        rising = cw & ~prev
        if self.mode == OpMode.PROFILE_POSITION:
            self._setpoint_ack = bool(cw & CW.NEW_SETPOINT)
            if rising & CW.NEW_SETPOINT:
                target = self.od[OD.TARGET_POSITION.addr]
                if cw & CW.RELATIVE:
                    target += self.position_counts
                self.position_counts = target
        elif self.mode == OpMode.HOMING and rising & CW.START_HOMING:
            self.position_counts = self.od[OD.HOMING_OFFSET.addr]
            self._homing_attained = True

    # -- object dictionary writes ------------------------------------------

    _READ_ONLY = {
        OD.STATUSWORD.addr, OD.MODE_DISPLAY.addr, OD.ACTUAL_POSITION.addr,
        OD.ACTUAL_VELOCITY.addr, OD.ACTUAL_CURRENT.addr,
        OD.TORQUE_DEMAND.addr, OD.DEVICE_TYPE.addr, OD.VENDOR_ID.addr,
        OD.PRODUCT_CODE.addr, OD.REVISION.addr, OD.SERIAL.addr,
        OD.TEMPERATURE.addr, OD.BUS_VOLTAGE.addr, OD.LAST_ERROR.addr,
        OD.SUPPORTED_MODES.addr, OD.ERROR_REGISTER.addr,
    }

    def _write(self, addr: "tuple[int, int]", value: int) -> "int | None":
        """Apply a write; returns an SDO abort code or None on success."""
        if addr in self._READ_ONLY:
            return 0x06010002
        if addr == OD.SAVE_ALL.addr:
            if value != SAVE_MAGIC:
                return 0x08000020
            return None
        self.od[addr] = value
        if addr == OD.CONTROLWORD.addr:
            self._apply_controlword(value)
        elif addr == OD.MODE_OF_OPERATION.addr:
            if self.operation_enabled:
                self.od[OD.MODE_DISPLAY.addr] = value
                self._setpoint_ack = False
                self._homing_attained = False
                self._apply_targets()
        elif addr in (OD.TARGET_TORQUE.addr, OD.TARGET_VELOCITY.addr):
            self._apply_targets()
        return None

    def _apply_targets(self) -> None:
        """Reflect active-mode targets into feedback objects (instant
        'physics'). Profile and cyclic flavors behave the same here; cyclic
        targets are also re-applied on SYNC."""
        if not self.operation_enabled:
            return
        mode = self.mode
        if mode in (OpMode.PROFILE_TORQUE, OpMode.CYCLIC_TORQUE):
            t = self.od[OD.TARGET_TORQUE.addr]
            self.od[OD.TORQUE_DEMAND.addr] = t
            self.od[OD.ACTUAL_CURRENT.addr] = t  # 1:1 stand-in
        elif mode in (OpMode.PROFILE_VELOCITY, OpMode.CYCLIC_VELOCITY):
            self.od[OD.ACTUAL_VELOCITY.addr] = self.od[OD.TARGET_VELOCITY.addr]
        elif mode == OpMode.CYCLIC_POSITION:
            self.position_counts = self.od[OD.TARGET_POSITION.addr]
        elif mode == OpMode.IDLE:
            self.od[OD.TORQUE_DEMAND.addr] = 0
            self.od[OD.ACTUAL_CURRENT.addr] = 0
            self.od[OD.ACTUAL_VELOCITY.addr] = 0

    # -- fault injection ----------------------------------------------------

    def trigger_fault(self, emcy_code: int, error_register: int = 0x01) -> None:
        self.ds402 = "FAULT"
        self.od[OD.MODE_DISPLAY.addr] = int(OpMode.IDLE)
        self.od[OD.LAST_ERROR.addr] = emcy_code
        self.od[OD.ERROR_REGISTER.addr] = error_register
        self.outbox.append((
            frames.COB_EMCY + self.id,
            emcy_code.to_bytes(2, "little") + bytes([error_register])
            + b"\x00" * 5,
        ))

    # -- frame handling -------------------------------------------------------

    def handle(self, can_id: int, data: bytes) -> None:
        if can_id == frames.COB_NMT and len(data) == 2:
            if data[1] in (0, self.id):
                self._handle_nmt(data[0])
            return
        if self.nmt_state == NmtState.STOPPED:
            return
        if can_id == frames.COB_SDO_REQUEST + self.id:
            self._handle_sdo(data)
            return
        if self.nmt_state != NmtState.OPERATIONAL:
            return
        if can_id == frames.COB_SYNC and not data:
            self._handle_sync()
            return
        for slot in range(1, 5):
            if can_id == self.od[(0x1400 + slot - 1, 1)] and \
                    self.od[(0x1600 + slot - 1, 0)] > 0:
                self._rpdo_buffer[slot] = bytes(data)
                return

    def _handle_nmt(self, command: int) -> None:
        if command == 0x01:
            self.nmt_state = NmtState.OPERATIONAL
        elif command == 0x02:
            self.nmt_state = NmtState.STOPPED
        elif command == 0x80:
            self.nmt_state = NmtState.PRE_OPERATIONAL
        elif command in (0x81, 0x82):
            self.nmt_state = NmtState.PRE_OPERATIONAL
            self._rpdo_buffer.clear()
            self.outbox.append((frames.COB_HEARTBEAT + self.id,
                                bytes([NmtState.BOOT_UP])))

    # -- SDO server -----------------------------------------------------------

    _SIZES = {0x2F: 1, 0x2B: 2, 0x23: 4}

    def _handle_sdo(self, data: bytes) -> None:
        if len(data) < 8:
            return
        cmd = data[0]
        index = int.from_bytes(data[1:3], "little")
        sub = data[3]
        addr = (index, sub)
        if cmd == 0x40:  # upload
            obj = _TYPES.get(addr)
            if obj is None and addr not in self.od:
                self._abort(index, sub, 0x06020000)
                return
            size, signed = obj if obj else (4, False)
            if addr == OD.STATUSWORD.addr:
                value = self.statusword()
            else:
                value = self.od[addr]
            cmd_byte = {1: 0x4F, 2: 0x4B, 4: 0x43}[size]
            payload = (bytes([cmd_byte]) + index.to_bytes(2, "little")
                       + bytes([sub])
                       + (value & (2 ** (size * 8) - 1)).to_bytes(
                           size, "little").ljust(4, b"\x00"))
            self.outbox.append((frames.COB_SDO_RESPONSE + self.id, payload))
        elif cmd in self._SIZES:  # expedited download
            size = self._SIZES[cmd]
            known = _TYPES.get(addr)
            signed = known[1] if known else False
            value = int.from_bytes(data[4:4 + size], "little", signed=signed)
            abort = self._write(addr, value)
            if abort:
                self._abort(index, sub, abort)
            else:
                self.outbox.append((
                    frames.COB_SDO_RESPONSE + self.id,
                    bytes([0x60]) + data[1:4] + b"\x00" * 4,
                ))
        else:
            self._abort(index, sub, 0x05040001)

    def _abort(self, index: int, sub: int, code: int) -> None:
        self.outbox.append(frames.sdo_abort(self.id, index, sub, code))

    # -- SYNC / PDO engine -------------------------------------------------------

    def _handle_sync(self) -> None:
        self._sync_count += 1
        # Apply buffered RPDOs per the drive's own mapping configuration.
        for slot, payload in sorted(self._rpdo_buffer.items()):
            for addr, size, signed in self._mapping(0x1600 + slot - 1, payload):
                value = int.from_bytes(payload[:size], "little", signed=signed)
                payload = payload[size:]
                self._write(addr, value)
        self._rpdo_buffer.clear()
        self._apply_targets()
        # Emit TPDOs per mapping and transmission type.
        for slot in range(1, 5):
            count = self.od[(0x1A00 + slot - 1, 0)]
            if count <= 0:
                continue
            tx_type = self.od[(0x1800 + slot - 1, 2)]
            n = max(1, tx_type) if tx_type <= 240 else 0
            if n == 0 or self._sync_count % n != 0:
                continue
            payload = b""
            for addr, size, signed in self._mapping(0x1A00 + slot - 1):
                if addr == OD.STATUSWORD.addr:
                    value = self.statusword()
                else:
                    value = self.od[addr]
                payload += (value & (2 ** (size * 8) - 1)).to_bytes(size, "little")
            self.outbox.append((self.od[(0x1800 + slot - 1, 1)], payload))

    def _mapping(self, map_index: int, _payload=None):
        """Yield (addr, size, signed) for each mapped entry of a PDO."""
        for i in range(1, self.od[(map_index, 0)] + 1):
            word = self.od[(map_index, i)]
            addr = ((word >> 16) & 0xFFFF, (word >> 8) & 0xFF)
            size = (word & 0xFF) // 8
            known = _TYPES.get(addr)
            yield addr, size, known[1] if known else False


# (size, signed) for every OD entry the library knows about.
_TYPES = {obj.addr: (obj.size, obj.signed)
          for obj in vars(OD).values() if isinstance(obj, type(OD.STATUSWORD))}


class MockTransport:
    """Drop-in :class:`~barrett_puck.transport.Transport` with fake drives."""

    def __init__(self):
        self.pucks: "dict[int, MockP4]" = {}
        self.rx = collections.deque()
        self.sent_log: "list[tuple[int, bytes]]" = []

    def add_puck(self, puck: MockP4) -> MockP4:
        self.pucks[puck.id] = puck
        return puck

    def send(self, can_id: int, data: bytes) -> None:
        self.sent_log.append((can_id, bytes(data)))
        for puck in sorted(self.pucks.values(), key=lambda p: p.id):
            puck.handle(can_id, bytes(data))
            while puck.outbox:
                self.rx.append(puck.outbox.pop(0))

    def recv(self, timeout: float = 0):
        if self.rx:
            return self.rx.popleft()
        return None

    def recv_matching(self, predicate, timeout: float):
        while self.rx:
            frame = self.rx.popleft()
            if predicate(*frame):
                return frame
        return None

    def flush(self) -> None:
        self.rx.clear()

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
