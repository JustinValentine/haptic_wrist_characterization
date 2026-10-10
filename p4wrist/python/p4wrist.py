"""Python binding for the p4wrist C API (ctypes, no compiled extension).

    import p4wrist
    w = p4wrist.Wrist.socketcan('can0', p4wrist.default_config())
    kept, report = w.connect(record)        # record from a previous run, or None
    if not kept:
        w.zero(cal, exact=True)             # wrist at cal['home']
    w.start(); w.enable()
    w.set_command(p4wrist.hold(w.sample().q, kp=(2, 1, 1), kd=(0.05,) * 3))
    ...
    log = w.read_log()                      # numpy structured array, one row per cycle
    w.close()

The library is found at $P4WRIST_LIB, else ../build/libp4wrist.{so,dylib}
next to this file (build it with cmake -S p4wrist -B p4wrist/build).
Angles are rad, torques N m, joint order PS, FE, UR.
"""
import ctypes as C
import math
import os
import sys
from pathlib import Path

import numpy as np

JOINTS = ('PS', 'FE', 'UR')
PS, FE, UR = 0, 1, 2

STATE_STOPPED, STATE_DISABLED, STATE_ENABLING, STATE_ACTIVE = 0, 1, 2, 3
STATE_BRAKE, STATE_BRAKING, STATE_DISABLING, STATE_FAULT = 4, 5, 6, 7
STATE_NAMES = {0: 'stopped', 1: 'disabled', 2: 'enabling', 3: 'active', 4: 'brake',
               5: 'braking', 6: 'disabling', 7: 'fault'}
MODE_POSITION, MODE_STAY_WITHIN = 10, 13

FAULT_NAMES = {0: 'none', 1: 'position limit', 2: 'velocity limit', 3: 'host timeout',
               4: 'feedback lost', 5: 'drive fault', 6: 'puck reset', 7: 'heartbeat lost',
               8: 'drive dropped out', 9: 'enable timeout', 10: 'CAN bus', 11: 'SDO', 12: 'overrun'}

D3 = C.c_double * 3
I3 = C.c_int32 * 3


class Config(C.Structure):
    _fields_ = [
        ('node', I3),
        ('controlword_refresh_cycles', C.c_int32),
        ('j2mp', D3 * 3),
        ('position_sign', D3),
        ('torque_sign', D3),
        ('period_s', C.c_double),
        ('joint_min', D3),
        ('joint_max', D3),
        ('max_joint_velocity', D3),
        ('max_joint_torque', D3),
        ('velocity_filter_hz', C.c_double),
        ('command_timeout_s', C.c_double),
        ('brake_s', C.c_double),
        ('heartbeat_timeout_s', C.c_double),
        ('tpdo_wait_s', C.c_double),
        ('current_torque_factor', C.c_double),
        ('max_motor_permille', C.c_int32),
        ('sync_loss_timeout_ms', C.c_int32),
        ('puck_heartbeat_ms', C.c_int32),
        ('feedback_timeout_cycles', C.c_int32),
        ('overrun_limit_cycles', C.c_int32),
        ('telemetry_divider', C.c_int32),
        ('realtime_priority', C.c_int32),
        ('map_current', C.c_int32),
    ]

    def to_dict(self):
        out = {}
        for name, _ in self._fields_:
            value = getattr(self, name)
            if name == 'j2mp':
                value = [list(row) for row in value]
            elif hasattr(value, '__len__'):
                value = list(value)
            out[name] = value
        return out


class JointCommand(C.Structure):
    _fields_ = [('position', C.c_double), ('velocity', C.c_double), ('torque', C.c_double),
                ('kp', C.c_double), ('kd', C.c_double), ('max_torque', C.c_double),
                ('lower', C.c_double), ('upper', C.c_double),
                ('mode', C.c_int32), ('reserved', C.c_int32)]


class Command(C.Structure):
    _fields_ = [('joint', JointCommand * 3), ('t_host', C.c_double), ('seq', C.c_uint32),
                ('brake', C.c_int32)]


class Sample(C.Structure):
    _fields_ = [
        ('cycle', C.c_uint64),
        ('time', C.c_double),
        ('q', D3), ('qd', D3), ('setpoint', D3), ('setpoint_velocity', D3),
        ('tau_cmd', D3), ('tau_meas', D3), ('motor_angle', D3), ('current', D3),
        ('bus_voltage', D3), ('amp_temp', D3), ('motor_temp', D3),
        ('command_t_host', C.c_double),
        ('late_s', C.c_double),
        ('counts', I3), ('permille', I3), ('statusword', I3), ('mode_display', I3),
        ('joint_mode', I3),
        ('wrist_state', C.c_int32),
        ('fault', C.c_int32),
        ('fault_node', C.c_int32),
        ('emcy_code', C.c_int32),
        ('command_seq', C.c_uint32),
        ('missed_feedback', C.c_uint32),
        ('zeroed', C.c_int32),
        ('reserved', C.c_int32),
    ]


class ZeroCal(C.Structure):
    _fields_ = [('home', D3), ('single_turn_at_home', I3), ('single_turn_dir', I3),
                ('has_single_turn', C.c_int32), ('reserved', C.c_int32)]


class ZeroRecord(C.Structure):
    _fields_ = [('motor_offset', D3), ('counts', I3), ('single_turn', I3),
                ('valid', C.c_int32), ('reserved', C.c_int32)]


SEND_FN = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint8), C.c_int32)
RECV_FN = C.CFUNCTYPE(C.c_int32, C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(C.c_uint8),
                      C.POINTER(C.c_int32))


class P4WristError(RuntimeError):
    pass


def _find_library():
    env = os.environ.get('P4WRIST_LIB')
    if env:
        return env
    build = Path(__file__).resolve().parent.parent / 'build'
    name = 'libp4wrist.dylib' if sys.platform == 'darwin' else 'libp4wrist.so'
    path = build / name
    if not path.exists():
        raise P4WristError(f'{path} not found: build it with '
                           f'cmake -S {build.parent} -B {build} && cmake --build {build}')
    return str(path)


_lib = None


def lib():
    global _lib
    if _lib is not None:
        return _lib
    L = C.CDLL(_find_library())
    P = C.c_void_p
    L.p4w_last_error.restype = C.c_char_p
    L.p4w_abi_version.restype = C.c_int
    L.p4w_struct_sizes.argtypes = [C.POINTER(C.c_int32)] * 5
    L.p4w_default_config.argtypes = [C.POINTER(Config)]
    L.p4w_open_socketcan.argtypes = [C.POINTER(Config), C.c_char_p]
    L.p4w_open_socketcan.restype = P
    L.p4w_open_callback.argtypes = [C.POINTER(Config), SEND_FN, RECV_FN, P]
    L.p4w_open_callback.restype = P
    L.p4w_close.argtypes = [P]
    L.p4w_connect.argtypes = [P, C.POINTER(ZeroRecord), C.POINTER(C.c_int32), C.c_char_p, C.c_int32]
    L.p4w_read_encoders.argtypes = [P, I3, I3]
    L.p4w_zero.argtypes = [P, C.POINTER(ZeroCal), C.c_int32, D3]
    L.p4w_get_zero_record.argtypes = [P, C.POINTER(ZeroRecord)]
    L.p4w_single_turn_direction.argtypes = [C.c_int32, C.c_int32, C.c_int32]
    L.p4w_single_turn_direction.restype = C.c_int32
    for name in ('p4w_start', 'p4w_stop', 'p4w_clear_fault'):
        getattr(L, name).argtypes = [P]
    L.p4w_step.argtypes = [P, C.c_int32]
    L.p4w_now.argtypes = [P]
    L.p4w_now.restype = C.c_double
    L.p4w_wait_until.argtypes = [P, C.c_double]
    L.p4w_enable.argtypes = [P, C.c_double]
    L.p4w_disable.argtypes = [P, C.c_double]
    L.p4w_set_command.argtypes = [P, C.POINTER(Command)]
    L.p4w_get_sample.argtypes = [P, C.POINTER(Sample)]
    L.p4w_read_log.argtypes = [P, C.POINTER(Sample), C.c_int32, C.POINTER(C.c_uint32)]
    L.p4w_read_log.restype = C.c_int32
    L.p4w_fault_text.argtypes = [P, C.c_char_p, C.c_int32]
    L.p4w_drive_info.argtypes = [P, D3, D3, D3, I3]

    if L.p4w_abi_version() != 1:
        raise P4WristError('libp4wrist ABI version mismatch; rebuild the library')
    sizes = [C.c_int32() for _ in range(5)]
    L.p4w_struct_sizes(*[C.byref(s) for s in sizes])
    for (struct, size) in zip((Config, Command, Sample, ZeroCal, ZeroRecord), sizes):
        if C.sizeof(struct) != size.value:
            raise P4WristError(f'{struct.__name__} is {C.sizeof(struct)} bytes here but '
                               f'{size.value} in libp4wrist; the binding and header disagree')
    _lib = L
    return L


def default_config():
    config = Config()
    lib().p4w_default_config(C.byref(config))
    return config


def single_turn_direction(position_delta, single_turn_delta, counts_per_rev=4096):
    return lib().p4w_single_turn_direction(int(position_delta), int(single_turn_delta),
                                           int(counts_per_rev))


def joint_command(position=math.nan, velocity=0.0, torque=0.0, kp=0.0, kd=0.0,
                  max_torque=math.nan, lower=math.nan, upper=math.nan, mode=MODE_POSITION):
    return JointCommand(position, velocity, torque, kp, kd, max_torque, lower, upper, mode, 0)


def hold(q, kp, kd, max_torque=(math.nan,) * 3):
    """Command holding each joint at q with PD gains."""
    cmd = Command()
    for j in range(3):
        cmd.joint[j] = joint_command(position=q[j], kp=kp[j], kd=kd[j], max_torque=max_torque[j])
    return cmd


def zero_cal(home, single_turn_at_home=None, single_turn_dir=None):
    cal = ZeroCal()
    cal.home = D3(*home)
    if single_turn_at_home is not None and single_turn_dir is not None:
        cal.single_turn_at_home = I3(*single_turn_at_home)
        cal.single_turn_dir = I3(*single_turn_dir)
        cal.has_single_turn = 1
    return cal


def record_to_dict(record):
    return {'motor_offset': list(record.motor_offset), 'counts': list(record.counts),
            'single_turn': list(record.single_turn), 'valid': int(record.valid)}


def record_from_dict(d):
    r = ZeroRecord()
    r.motor_offset = D3(*d['motor_offset'])
    r.counts = I3(*d['counts'])
    r.single_turn = I3(*d['single_turn'])
    r.valid = int(d.get('valid', 1))
    return r


# A numpy view of p4w_sample, for read_log().
def _sample_dtype():
    fields = []
    for name, ctype in Sample._fields_:
        if hasattr(ctype, '_length_'):
            base = {C.c_double: np.float64, C.c_int32: np.int32}[ctype._type_]
            fields.append((name, base, (ctype._length_,)))
        else:
            base = {C.c_uint64: np.uint64, C.c_double: np.float64, C.c_int32: np.int32,
                    C.c_uint32: np.uint32}[ctype]
            fields.append((name, base))
    dtype = np.dtype(fields, align=True)
    assert dtype.itemsize == C.sizeof(Sample)
    return dtype


SAMPLE_DTYPE = _sample_dtype()


class Wrist:
    """One wrist. Use Wrist.socketcan() for hardware, Wrist.callback() for a
    simulated bus (an object with send(id, data) and recv() -> (id, data) or None)."""

    def __init__(self, handle, keepalive=None):
        self._h = handle
        self._keep = keepalive
        self._bus_error = None
        self._log_buf = (Sample * 4096)()

    @classmethod
    def socketcan(cls, interface='can0', config=None):
        config = config or default_config()
        h = lib().p4w_open_socketcan(C.byref(config), interface.encode())
        if not h:
            raise P4WristError(lib().p4w_last_error().decode())
        return cls(h)

    @classmethod
    def callback(cls, bus, config=None):
        config = config or default_config()
        holder = {}

        def send(_user, can_id, data, length):
            try:
                bus.send(int(can_id), bytes(data[:length]))
                return 0
            except BaseException as exc:   # never let an exception unwind through C
                holder['wrist']._bus_error = exc
                return -1

        def recv(_user, can_id, data, length):
            try:
                frame = bus.recv()
            except BaseException as exc:
                holder['wrist']._bus_error = exc
                return 0
            if frame is None:
                return 0
            fid, payload = frame
            can_id[0] = fid
            n = min(len(payload), 8)
            for i in range(n):
                data[i] = payload[i]
            length[0] = n
            return 1

        send_cb, recv_cb = SEND_FN(send), RECV_FN(recv)
        h = lib().p4w_open_callback(C.byref(config), send_cb, recv_cb, None)
        if not h:
            raise P4WristError(lib().p4w_last_error().decode())
        wrist = cls(h, keepalive=(send_cb, recv_cb, bus))
        holder['wrist'] = wrist
        return wrist

    # --- plumbing ---------------------------------------------------------------
    def _check(self, rc):
        if self._bus_error is not None:
            exc, self._bus_error = self._bus_error, None
            raise exc
        if rc != 0:
            raise P4WristError(lib().p4w_last_error().decode())

    def close(self):
        if self._h:
            lib().p4w_close(self._h)
            self._h = None
            self._check(0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- setup ------------------------------------------------------------------
    def connect(self, record=None):
        """Bring the pucks up; returns (kept_zero, report)."""
        kept = C.c_int32(0)
        text = C.create_string_buffer(512)
        rec = C.byref(record) if record is not None else None
        self._check(lib().p4w_connect(self._h, rec, C.byref(kept), text, len(text)))
        return bool(kept.value), text.value.decode()

    def read_encoders(self):
        counts, single = I3(), I3()
        self._check(lib().p4w_read_encoders(self._h, counts, single))
        return list(counts), list(single)

    def zero(self, cal, exact=True):
        out = D3()
        self._check(lib().p4w_zero(self._h, C.byref(cal), int(bool(exact)), out))
        return list(out)

    def zero_record(self):
        r = ZeroRecord()
        self._check(lib().p4w_get_zero_record(self._h, C.byref(r)))
        return r

    def drive_info(self):
        rated, kt, peak, cpr = D3(), D3(), D3(), I3()
        self._check(lib().p4w_drive_info(self._h, rated, kt, peak, cpr))
        return [{'rated_torque_mnm': rated[i], 'kt_mnm_per_a': kt[i],
                 'peak_current_ma': peak[i], 'counts_per_rev': cpr[i]} for i in range(3)]

    # --- loop -------------------------------------------------------------------
    def start(self):
        self._check(lib().p4w_start(self._h))

    def stop(self):
        self._check(lib().p4w_stop(self._h))

    def step(self, cycles=1):
        self._check(lib().p4w_step(self._h, int(cycles)))

    def now(self):
        return lib().p4w_now(self._h)

    def wait_until(self, t):
        self._check(lib().p4w_wait_until(self._h, float(t)))

    def enable(self, timeout_s=2.0):
        self._check(lib().p4w_enable(self._h, float(timeout_s)))

    def disable(self, timeout_s=1.0):
        self._check(lib().p4w_disable(self._h, float(timeout_s)))

    def clear_fault(self):
        self._check(lib().p4w_clear_fault(self._h))

    def set_command(self, command):
        self._check(lib().p4w_set_command(self._h, C.byref(command)))

    def sample(self):
        s = Sample()
        self._check(lib().p4w_get_sample(self._h, C.byref(s)))
        return s

    def read_log(self):
        """All samples logged since the last call, as a numpy structured array,
        and how many were dropped because the log was full."""
        chunks, dropped_total = [], 0
        while True:
            dropped = C.c_uint32(0)
            n = lib().p4w_read_log(self._h, self._log_buf, len(self._log_buf), C.byref(dropped))
            if n < 0:
                self._check(-1)
            dropped_total += dropped.value
            if n == 0:
                break
            chunks.append(np.frombuffer(self._log_buf, dtype=SAMPLE_DTYPE, count=n).copy())
            if n < len(self._log_buf):
                break
        log = np.concatenate(chunks) if chunks else np.zeros(0, dtype=SAMPLE_DTYPE)
        return log, dropped_total

    def fault_text(self):
        text = C.create_string_buffer(256)
        self._check(lib().p4w_fault_text(self._h, text, len(text)))
        return text.value.decode()
