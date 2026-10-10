"""Shared infrastructure for the P4 wrist characterization experiments.

Conventions
-----------
* The wrist is driven through the p4wrist library (p4wrist/): a 500 Hz
  CANopen SYNC loop that renders PD + feedforward per joint, maps joint
  torques through the PS/FE/UR differential, and supervises limits and
  liveness itself. device_config.JOINT is characterized; the loop holds the
  other two joints at their start angles.
* Joint positions are relative to the start pose of each run. The loop's
  absolute joint angles (after zeroing) are logged too (q_ps_deg, ...).
* motor_pos_rev is the joint angle in motor turns (GEAR_RATIO x joint turns):
  each differential motor turns N1 times per FE or UR turn, so cogging, which
  is periodic in each motor's angle, stays periodic in this coordinate.
* Experiments describe what to command as a list of Segments: functions of
  time (and of the last measurement, for host-side feedback) that return a
  joint-side Command. WristSession sends them at a fixed rate; every loop
  cycle (2 ms) is logged from the library's own log, tagged with the command
  it used. Safety limits are checked on every sample; on exit the wrist is
  always braked or disabled and the data saved.
* Every experiment runs against the simulated wrist in p4_sim.py with --sim:
  the same library, against emulated pucks and a plant.
"""
import argparse
import asyncio
import json
import math
import platform
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import device_config

DATA_ROOT = Path('data')

# Logged 'mode' column, numbered as moteus modes so old and new logs read alike.
MODE_STOPPED = 0
MODE_FAULT = 1
MODE_POSITION = 10
MODE_TIMEOUT = 11
MODE_STAY_WITHIN = 13
MODE_BRAKE = 15


# --- unit conversions ------------------------------------------------------------
def joint_degrees_to_motor_revs(joint_deg, gear_ratio):
    return (joint_deg / 360.0) * gear_ratio


def motor_revs_to_joint_degrees(motor_revs, gear_ratio):
    return (motor_revs / gear_ratio) * 360.0


# --- motion profiles ----------------------------------------------------------------
def min_jerk(s):
    """Minimum-jerk profile: position fraction and d/ds for s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5,
            30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4)


def min_jerk_duration(distance_deg, peak_speed_deg_s):
    """Duration of a minimum-jerk move whose peak speed is peak_speed_deg_s."""
    if peak_speed_deg_s <= 0:
        raise ValueError('peak_speed_deg_s must be positive')
    return max(1.875 * abs(distance_deg) / peak_speed_deg_s, 0.05)


# --- commands and segments ------------------------------------------------------
@dataclass
class Command:
    """Joint-side command. position_deg is relative to the start pose."""
    position_deg: float = math.nan   # nan holds the current setpoint, integrating velocity
    velocity_deg_s: float = 0.0      # Velocity feedforward (setpoint integrates it)
    torque_nm: float = 0.0           # Feedforward torque
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    max_torque_nm: Optional[float] = None   # Per-command cap; device cap if None
    # Stay-within: (lower_deg, upper_deg), nan for an open side. Inside the
    # bounds only torque_nm is applied; outside, the gains (x kp/kd_scale) push
    # back toward the violated bound: a unilateral wall rendered by the 500 Hz
    # loop. position_deg/velocity_deg_s are ignored.
    stay_within: Optional[tuple] = None


@dataclass
class Segment:
    """A phase of an experiment: command(t_local, last_sample) for duration_s."""
    phase: str
    duration_s: float
    command: Callable[[float, Optional[dict]], Command]
    info: dict = field(default_factory=dict)       # Constant columns logged with each row
    until: Optional[Callable[[float, dict], Optional[str]]] = None   # Early stop -> reason
    rate_hz: Optional[float] = None


# Columns every logged row already has; Segment.info must not overwrite them.
RESERVED_COLUMNS = frozenset({
    'time', 't_send', 'segment', 'phase', 't_local', 'cmd_joint_deg', 'cmd_joint_vel_deg_s',
    'cmd_ff_torque_nm', 'kp_scale', 'kd_scale', 'kp_joint_nm_rad', 'kd_joint_nms_rad',
    'max_torque_nm', 'motor_pos_rev', 'joint_pos_deg', 'joint_vel_deg_s', 'joint_torque_nm',
    'q_current_a', 'control_joint_deg', 'control_joint_vel_deg_s', 'control_joint_torque_nm',
    'voltage_v', 'board_temp_c', 'motor_temp_c', 'mode', 'fault', 'ext_torque_nm',
    'stay_lower_deg', 'stay_upper_deg', 'loop_cycle', 'wrist_state', 'late_ms', 'missed_feedback',
    'q_ps_deg', 'q_fe_deg', 'q_ur_deg', 'tau_cmd_ps_nm', 'tau_cmd_fe_nm', 'tau_cmd_ur_nm',
    'tau_meas_ps_nm', 'tau_meas_fe_nm', 'tau_meas_ur_nm', 'counts_6', 'counts_7', 'counts_8',
    'permille_6', 'permille_7', 'permille_8', 'current_6_a', 'current_7_a', 'current_8_a',
})


def check_info(info):
    clash = sorted(set(info or {}) & RESERVED_COLUMNS)
    if clash:
        raise ValueError(f'Segment info keys {clash} would overwrite logged columns; rename them.')


@dataclass
class SegmentResult:
    segment: int
    phase: str
    rows: list
    ended_by: Optional[str] = None


def hold(duration_s, position_deg=math.nan, phase='hold', info=None, **command_kwargs):
    command = Command(position_deg=position_deg, **command_kwargs)
    return Segment(phase, duration_s, lambda t, s: command, info or {})


def min_jerk_move(start_deg, end_deg, peak_speed_deg_s, phase='move', info=None,
                  **command_kwargs):
    """Smooth repositioning move with matching velocity feedforward."""
    duration = min_jerk_duration(end_deg - start_deg, peak_speed_deg_s)
    delta = end_deg - start_deg

    def command(t, _):
        frac, dfrac = min_jerk(t / duration)
        return Command(position_deg=start_deg + delta * frac,
                       velocity_deg_s=delta * dfrac / duration, **command_kwargs)

    return Segment(phase, duration, command, info or {})


def total_duration(segments):
    return sum(s.duration_s for s in segments)


def describe_plan(segments, max_torque_nm=None, points_per_segment=50):
    """Summary lines for the confirmation prompt (open-loop commands only)."""
    positions, torques = [], []
    for seg in segments:
        for i in range(points_per_segment + 1):
            try:
                cmd = seg.command(seg.duration_s * i / points_per_segment, None)
            except Exception:
                break   # Feedback segments need a measurement; skip them.
            if math.isfinite(cmd.position_deg):
                positions.append(cmd.position_deg)
            torques.append(abs(cmd.torque_nm))
    lines = [f'  Duration: {total_duration(segments):.0f} s in {len(segments)} segments']
    if positions:
        lines.append(f'  Commanded position: {min(positions):+.1f} to {max(positions):+.1f} deg')
    if torques and max(torques) > 0:
        lines.append(f'  Peak feedforward torque: {max(torques):.2f} N m')
    if max_torque_nm is not None:
        lines.append(f'  Torque cap: {max_torque_nm:.2f} N m')
    return lines


# --- safety ------------------------------------------------------------------------
class SafetyAbort(RuntimeError):
    """Raised when a safety limit trips; the session brakes and disables the wrist."""


@dataclass
class SafetyLimits:
    """Host-side checks on every logged sample (the p4wrist loop has its own)."""
    joint_min_deg: float = device_config.JOINT_LIMITS_DEG[0]
    joint_max_deg: float = device_config.JOINT_LIMITS_DEG[1]
    max_joint_velocity_deg_s: float = device_config.MAX_JOINT_VELOCITY_DEG_S
    max_motor_temperature_c: float = device_config.MAX_MOTOR_TEMPERATURE_C
    max_board_temperature_c: float = device_config.MAX_BOARD_TEMPERATURE_C

    def check(self, row):
        mode = row.get('mode')
        if mode == MODE_FAULT:
            return f"p4wrist fault: {row.get('fault_text') or row.get('fault')}"
        if mode == MODE_TIMEOUT:
            return 'p4wrist watchdog timeout (host stopped commanding in time)'
        pos = row.get('joint_pos_deg', math.nan)
        if math.isfinite(pos) and not (self.joint_min_deg <= pos <= self.joint_max_deg):
            return (f'joint position {pos:.1f} deg outside '
                    f'[{self.joint_min_deg}, {self.joint_max_deg}] deg')
        vel = row.get('joint_vel_deg_s', math.nan)
        if math.isfinite(vel) and abs(vel) > self.max_joint_velocity_deg_s:
            return f'joint velocity {vel:.0f} deg/s above {self.max_joint_velocity_deg_s} deg/s'
        temp = row.get('motor_temp_c', math.nan)
        if math.isfinite(temp) and temp > 0 and temp > self.max_motor_temperature_c:
            return f'motor temperature {temp:.1f} C above {self.max_motor_temperature_c} C'
        temp = row.get('board_temp_c', math.nan)
        if math.isfinite(temp) and temp > self.max_board_temperature_c:
            return f'board temperature {temp:.1f} C above {self.max_board_temperature_c} C'
        return None


# --- clocks --------------------------------------------------------------------------
class WristClock:
    """The p4wrist loop clock, with sub-millisecond waits (sleep, then yield-spin)."""

    def __init__(self, wrist):
        self.wrist = wrist

    def now(self):
        return self.wrist.now()

    async def sleep_until(self, t):
        while True:
            remaining = t - self.now()
            if remaining <= 0:
                return
            if remaining > 0.002:
                await asyncio.sleep(remaining - 0.0015)
            else:
                await asyncio.sleep(0)

    async def sleep(self, dt):
        await self.sleep_until(self.now() + dt)


class SimClock:
    """Simulated time: waiting steps the p4wrist loop (and so the plant)."""

    def __init__(self, wrist):
        self.wrist = wrist

    def now(self):
        return self.wrist.now()

    async def sleep_until(self, t):
        self.wrist.wait_until(t)

    async def sleep(self, dt):
        await self.sleep_until(self.now() + dt)


# --- external torque sensors ---------------------------------------------------------
class ExternalSensor:
    kind = 'none'
    continuous = False      # True if every sample carries ext_torque_nm

    def sample(self):
        return math.nan

    def close(self):
        pass

    def describe(self):
        return {'type': self.kind}


class ManualSensor(ExternalSensor):
    """Operator types a force-gauge or torque-wrench reading at each hold."""
    kind = 'manual'

    def __init__(self, config):
        self.lever_arm_m = config.get('lever_arm_m')

    def to_torque(self, text):
        value = float(text)
        if self.lever_arm_m:
            return value * self.lever_arm_m     # Reading in N at the lever arm
        return value                            # Reading already in N m

    def prompt_units(self):
        return f'force in N (lever arm {self.lever_arm_m} m)' if self.lever_arm_m else 'torque in N m'

    def describe(self):
        return {'type': self.kind, 'lever_arm_m': self.lever_arm_m}


class AtiNetFTSensor(ExternalSensor):
    """ATI Net F/T streaming over UDP (RDT protocol), latest value per sample.

    Not hardware-tested in this repository; verify the sign and scale against
    a known load before trusting it.
    """
    kind = 'ati_netft'
    continuous = True
    AXES = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

    def __init__(self, config):
        self.config = config
        self.axis = self.AXES.index(config.get('axis', 'Tz'))
        self.scale = config.get('sign', 1) / config.get('counts_per_unit', 1e6)
        if config.get('lever_arm_m'):
            self.scale *= config['lever_arm_m']
        self.latest = math.nan
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._address = (config['host'], 49152)
        self._running = True
        self._sock.sendto(struct.pack('!HHI', 0x1234, 0x0002, 0), self._address)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            try:
                packet = self._sock.recv(36)
            except OSError:
                continue
            if len(packet) == 36:
                values = struct.unpack('!3I6i', packet)
                self.latest = values[3 + self.axis] * self.scale

    def sample(self):
        return self.latest

    def close(self):
        self._running = False
        try:
            self._sock.sendto(struct.pack('!HHI', 0x1234, 0x0000, 0), self._address)
        except OSError:
            pass

    def describe(self):
        return dict(self.config)


class SimSensor(ExternalSensor):
    """Simulated sensor: the plant's true handle (or lock) torque plus noise,
    sampled once per loop cycle."""
    continuous = True

    def __init__(self, config, plant):
        self.kind = (config or {}).get('type', 'sim')
        self.config = config or {}
        self.plant = plant
        self.lever_arm_m = self.config.get('lever_arm_m') if self.kind == 'manual' else None

    def sample(self):
        return self.plant.external_torque()

    def describe(self):
        return {'type': f'sim({self.kind})'}


def make_sensor(config, plant=None):
    if plant is not None:
        return SimSensor(config, plant)
    if not config:
        return ExternalSensor()
    kind = config.get('type')
    if kind == 'manual':
        return ManualSensor(config)
    if kind == 'ati_netft':
        return AtiNetFTSensor(config)
    raise ValueError(f'Unknown EXTERNAL_TORQUE_SENSOR type {kind!r}')


# --- files and provenance ------------------------------------------------------------
def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write('\n')


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def git_provenance():
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()
        return {'git_commit': commit or None, 'git_dirty': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'git_commit': None, 'git_dirty': None}


def new_run_dir(experiment, sim, out=None):
    if out:
        path = Path(out)
    else:
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        path = DATA_ROOT / experiment / (stamp + ('_sim' if sim else ''))
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_is_simulated(run_dir, experiment):
    meta_path = Path(run_dir) / f'{experiment}_metadata.json'
    if meta_path.exists():
        try:
            return bool(json.loads(meta_path.read_text()).get('simulated'))
        except ValueError:
            pass
    return Path(run_dir).name.endswith('_sim')


def latest_summary(experiment, simulated=False):
    """(run_dir, summary dict) of the newest analyzed run, or None.

    Only runs whose simulated flag matches are considered, so a --sim dry run
    can never feed parameters into a hardware experiment (or vice versa).
    """
    root = DATA_ROOT / experiment
    if not root.is_dir():
        return None
    runs = sorted((d for d in root.iterdir() if (d / 'analysis' / 'summary.json').exists()),
                  key=lambda d: d.name)
    for run in reversed(runs):
        if run_is_simulated(run, experiment) == bool(simulated):
            return run, json.loads((run / 'analysis' / 'summary.json').read_text())
    return None


def device_config_snapshot():
    return {name: getattr(device_config, name) for name in dir(device_config)
            if name.isupper()}


# --- command line -------------------------------------------------------------------
def make_arg_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--sim', action='store_true',
                        help='run against the simulated wrist in p4_sim.py')
    parser.add_argument('--seed', type=int, default=0, help='simulator random seed')
    parser.add_argument('--out', help='output directory (default data/<experiment>/<timestamp>)')
    parser.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    parser.add_argument('--quick', action='store_true',
                        help='shortened protocol for dry runs and tests')
    parser.add_argument('--interface', default=device_config.CAN_INTERFACE,
                        help='SocketCAN interface of the wrist pucks')
    return parser


def confirm(args, lines):
    print('\n'.join(lines))
    if args.sim or args.yes:
        return
    try:
        input('Press Enter to start (Ctrl-C to abort)... ')
    except EOFError:
        raise SystemExit('No terminal for confirmation; rerun with --yes.')


def run_cli(coro_factory):
    """Run an experiment's async main, reporting aborts without a traceback."""
    try:
        asyncio.run(coro_factory())
    except SafetyAbort as exc:
        print(f'\nABORTED by safety check: {exc}', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nABORTED by user; wrist braked and disabled, partial data saved.', file=sys.stderr)
        raise SystemExit(130)


# --- p4wrist setup --------------------------------------------------------------------
JOINT_NAMES = device_config.JOINT_NAMES


def wrist_config(p4wrist, simulated):
    """p4wrist loop configuration from device_config."""
    c = p4wrist.default_config()
    c.node = p4wrist.I3(*device_config.NODES)
    for i in range(3):
        for j in range(3):
            c.j2mp[i][j] = device_config.J2MP[i][j]
    c.position_sign = p4wrist.D3(*device_config.POSITION_SIGN)
    c.torque_sign = p4wrist.D3(*device_config.TORQUE_SIGN)
    c.period_s = device_config.LOOP_PERIOD_S
    for j, name in enumerate(JOINT_NAMES):
        lo, hi = device_config.JOINT_RANGE_DEG[name]
        c.joint_min[j] = math.radians(lo)
        c.joint_max[j] = math.radians(hi)
        c.max_joint_velocity[j] = math.radians(device_config.MAX_JOINT_VELOCITY_DEG_S)
        c.max_joint_torque[j] = device_config.MAX_JOINT_TORQUE_NM
    c.velocity_filter_hz = device_config.VELOCITY_FILTER_HZ
    c.command_timeout_s = device_config.WATCHDOG_TIMEOUT_S
    c.brake_s = device_config.FAULT_BRAKE_S
    c.current_torque_factor = device_config.CURRENT_TORQUE_FACTOR
    c.max_motor_permille = device_config.MAX_MOTOR_PERMILLE
    c.sync_loss_timeout_ms = device_config.SYNC_LOSS_TIMEOUT_MS
    c.realtime_priority = 0 if simulated else device_config.REALTIME_PRIORITY
    return c


def load_zero_calibration():
    """wrist_calibration.json (from zero_calibration.py), or None."""
    path = Path(device_config.ZERO_CALIBRATION_FILE)
    if not path.exists():
        return None
    cal = json.loads(path.read_text())
    if any(abs(a - b) > 1e-9 for a, b in zip(cal['home_deg'], device_config.HOME_DEG)):
        raise SystemExit(f'{path} was recorded for home {cal["home_deg"]} deg but HOME_DEG is '
                         f'{device_config.HOME_DEG}; rerun zero_calibration.py.')
    return cal


def load_zero_record(p4wrist):
    path = Path(device_config.ZERO_RECORD_FILE)
    if not path.exists():
        return None
    try:
        return p4wrist.record_from_dict(json.loads(path.read_text()))
    except (ValueError, KeyError):
        return None


def save_zero_record(p4wrist, record):
    path = Path(device_config.ZERO_RECORD_FILE)
    if record.valid:
        write_json(path, p4wrist.record_to_dict(record))
    elif path.exists():
        path.unlink()


def placement_warnings(from_home_rad):
    """Rough placement is right only within half a motor turn of home."""
    lines = []
    m = [sum(device_config.J2MP[i][j] * from_home_rad[j] for j in range(3)) for i in range(3)]
    for i, node in enumerate(device_config.NODES):
        fraction = abs(m[i]) / math.pi
        if fraction > 0.6:
            lines.append(f'WARNING: motor {node} was {100 * fraction:.0f}% of the way to the '
                         'half-turn limit of rough placement; if the zero looks wrong, place the '
                         'wrist closer to home and restart.')
    return lines


# --- the session ---------------------------------------------------------------------
@dataclass
class _SegmentMeta:
    index: int
    phase: str
    info: dict
    t0: float
    log: bool


@dataclass
class _Sent:
    meta: _SegmentMeta
    command: Command
    t_send: float
    cap: float
    kp: float
    kd: float


class WristSession:
    """One experiment run: connection, zeroing, safety, timing, logging.

    Usage:
        async with WristSession('step_response', args, kp, kd, config=cfg) as s:
            await s.run(segments)
    """

    def __init__(self, experiment, args, kp_joint_nm_rad, kd_joint_nms_rad, *,
                 config=None, rate_hz=500.0, needs_sensor=False, sim_params=None,
                 limits: Optional[SafetyLimits] = None, max_torque_nm=None):
        self.experiment = experiment
        self.args = args
        self.config = config
        self.j = device_config.JOINT_INDEX
        self.joint = device_config.JOINT
        self.N = device_config.GEAR_RATIO
        self.loop_rate_hz = 1.0 / device_config.LOOP_PERIOD_S
        # Commands faster than the loop are never seen by it.
        self.rate_hz = min(rate_hz, self.loop_rate_hz)
        self.requested_rate_hz = rate_hz
        self.kp_joint = kp_joint_nm_rad
        self.kd_joint = kd_joint_nms_rad
        self.max_torque_nm = min(max_torque_nm or device_config.MAX_JOINT_TORQUE_NM,
                                 device_config.MAX_JOINT_TORQUE_NM)
        self.limits = limits or SafetyLimits()
        self.is_sim = bool(getattr(args, 'sim', False))
        self.needs_sensor = needs_sensor
        self.sim_params = sim_params
        self.rows = []
        self.events = []
        self.metadata = {}
        self.segment_index = -1
        self.last_sample = None
        self.last_command = Command()
        self.start_q = [0.0, 0.0, 0.0]
        self.abort_reason = None
        self.completed = False
        self.run_dir = None
        self.plant = None
        self.bus = None
        self.wrist = None
        self.sensor = ExternalSensor()
        self._t_started = None
        self._seq = 0
        self._sent = {}
        self._ext_by_cycle = []
        self._fault_text = ''
        self._p4 = None
        # Primary motor of the joint under test (reports its current and
        # temperatures) and the sign that makes its current follow joint torque.
        col = [device_config.J2MP[i][self.j] for i in range(3)]
        self._primary = max(range(3), key=lambda i: abs(col[i]))
        inv_t = _inverse_transpose(device_config.J2MP)
        self._current_sign = math.copysign(1.0, inv_t[self._primary][self.j])

    # --- lifecycle ---------------------------------------------------------------------
    async def __aenter__(self):
        from wrist_backend import p4wrist
        self._p4 = p4wrist
        self.run_dir = new_run_dir(self.experiment, self.is_sim, getattr(self.args, 'out', None))
        sensor_config = device_config.EXTERNAL_TORQUE_SENSOR if self.needs_sensor else None
        config = wrist_config(p4wrist, self.is_sim)
        if self.is_sim:
            import p4_sim
            params = self.sim_params or p4_sim.SimParams()
            params.seed = getattr(self.args, 'seed', params.seed)
            params.j2mp = device_config.J2MP
            params.initial_q = tuple(math.radians(x) for x in device_config.HOME_DEG)
            self.bus, self.plant = p4_sim.make_sim(params, nodes=device_config.NODES)
            self.plant.test_joint = self.j
            self.wrist = p4wrist.Wrist.callback(self.bus, config)
            self.clock = SimClock(self.wrist)
            if self.needs_sensor:
                self.sensor = make_sensor(sensor_config, plant=self.plant)
                self.bus.sync_hooks.append(lambda: self._ext_by_cycle.append(self.sensor.sample()))
        else:
            self.wrist = p4wrist.Wrist.socketcan(getattr(self.args, 'interface', None)
                                                 or device_config.CAN_INTERFACE, config)
            self.clock = WristClock(self.wrist)
            if self.needs_sensor:
                self.sensor = make_sensor(sensor_config)
        self.metadata['p4wrist_config'] = config.to_dict()
        try:
            self._connect_and_zero()
            self.wrist.start()
            await self.clock.sleep(0.05)   # positions, statuswords, SYNC-loss timer armed
            s = self.wrist.sample()
            if s.fault:
                self.log_event('fault_before_enable', fault=int(s.fault), text=self.wrist.fault_text())
                self.wrist.clear_fault()
            self.wrist.enable(3.0)
            self.start_q = list(self.wrist.sample().q)
            if self.is_sim:
                self.plant.human_origin = self.start_q[self.j]
            # Feed the watchdog at once: hold every joint where it is.
            self._send(Command(position_deg=0.0, kp_scale=0.0, kd_scale=0.0),
                       _SegmentMeta(-1, 'start', {}, self.clock.now(), False))
            self.metadata['drive_info'] = self.wrist.drive_info()
            self._limit_torque_to_drives()
        except BaseException:
            self._shutdown()
            raise
        self._t_started = datetime.now(timezone.utc).isoformat()
        return self

    def _limit_torque_to_drives(self):
        """Cap commands at what the motors can deliver to this joint: a pure
        joint torque needs j2mp^-T times it from every motor it couples to."""
        inv_t = _inverse_transpose(device_config.J2MP)
        limit = math.inf
        for i, drive in enumerate(self.metadata['drive_info']):
            share = abs(inv_t[i][self.j])
            if share > 0:
                motor_cap = device_config.MAX_MOTOR_PERMILLE * drive['rated_torque_mnm'] / 1e6
                limit = min(limit, motor_cap / share)
        self.metadata['motor_limited_torque_nm'] = limit
        if limit < self.max_torque_nm:
            self.log_event('torque_cap_lowered', from_nm=self.max_torque_nm, to_nm=limit)
            self.max_torque_nm = limit

    def _connect_and_zero(self):
        p4wrist = self._p4
        record = None if self.is_sim else load_zero_record(p4wrist)
        kept, report = self.wrist.connect(record)
        home = [math.radians(x) for x in device_config.HOME_DEG]
        zero = {'kept_previous_zero': kept, 'report': report}
        if not self.is_sim:
            print(report)
        if not kept:
            cal = None if self.is_sim else load_zero_calibration()
            if cal and cal.get('single_turn_dir') and all(cal['single_turn_dir']):
                cal_struct = p4wrist.zero_cal(home, cal['single_turn_at_home'], cal['single_turn_dir'])
                from_home = self.wrist.zero(cal_struct, exact=False)
                zero['method'] = 'single-turn encoders (rough placement)'
                print('Zeroed from the single-turn encoders; the wrist sat this far from home (deg): '
                      + ', '.join(f'{n} {math.degrees(d):+.2f}' for n, d in zip(JOINT_NAMES, from_home)))
                for line in placement_warnings(from_home):
                    print(line)
            else:
                if not self.is_sim:
                    confirm(self.args, ['', 'ZERO: place the wrist exactly at the home pose '
                                        f'{device_config.HOME_DEG} deg (PS, FE, UR).'])
                from_home = self.wrist.zero(p4wrist.zero_cal(home), exact=True)
                zero['method'] = 'placed at home'
            zero['from_home_deg'] = [math.degrees(d) for d in from_home]
            if not self.is_sim:
                save_zero_record(p4wrist, self.wrist.zero_record())
        self.metadata['zero'] = zero

    async def __aexit__(self, exc_type, exc, tb):
        if exc is None:
            self.completed = True
        elif self.abort_reason is None:
            self.abort_reason = f'{exc_type.__name__}: {exc}' if str(exc) else exc_type.__name__
        try:
            if exc is not None:
                # Brake first so a moving joint is damped, then disable.
                try:
                    await self.brake(0.3, phase='abort_brake')
                except Exception:
                    pass
            self.wrist.disable(1.0)
        except BaseException as cleanup_error:   # Keep going: data must still be saved.
            print(f'WARNING: cleanup error: {cleanup_error!r}', file=sys.stderr)
        finally:
            self._shutdown()
            self.save()
        return False

    def _shutdown(self):
        try:
            if self.wrist is not None:
                try:
                    self._drain()
                except Exception:
                    pass
                self.wrist.stop()
                if not self.is_sim:
                    save_zero_record(self._p4, self.wrist.zero_record())
                self.wrist.close()
        except BaseException as error:
            print(f'WARNING: shutdown error: {error!r}', file=sys.stderr)
        finally:
            self.sensor.close()

    # --- commanding --------------------------------------------------------------------
    def _send(self, cmd: Command, meta: _SegmentMeta, brake=False):
        p4wrist = self._p4
        self._seq += 1
        t = self.clock.now()
        cap = self.max_torque_nm if cmd.max_torque_nm is None else min(
            abs(cmd.max_torque_nm), self.max_torque_nm)
        kp = self.kp_joint * cmd.kp_scale
        kd = self.kd_joint * cmd.kd_scale
        out = p4wrist.Command()
        out.seq = self._seq
        out.t_host = t
        out.brake = 1 if brake else 0
        for j, name in enumerate(JOINT_NAMES):
            if j == self.j:
                continue
            out.joint[j] = p4wrist.joint_command(
                position=self.start_q[j], kp=device_config.HOLD_KP_NM_RAD[name],
                kd=device_config.HOLD_KD_NMS_RAD[name], max_torque=device_config.HOLD_MAX_TORQUE_NM)
        q0 = self.start_q[self.j]
        if cmd.stay_within is not None:
            def bound(deg):
                return q0 + math.radians(deg) if deg is not None and math.isfinite(deg) else math.nan
            out.joint[self.j] = p4wrist.joint_command(
                torque=cmd.torque_nm, kp=kp, kd=kd, max_torque=cap, lower=bound(cmd.stay_within[0]),
                upper=bound(cmd.stay_within[1]), mode=p4wrist.MODE_STAY_WITHIN)
        else:
            position = q0 + math.radians(cmd.position_deg) if math.isfinite(cmd.position_deg) else math.nan
            out.joint[self.j] = p4wrist.joint_command(
                position=position, velocity=math.radians(cmd.velocity_deg_s), torque=cmd.torque_nm,
                kp=kp, kd=kd, max_torque=cap)
        self.wrist.set_command(out)
        self._sent[self._seq] = _Sent(meta, cmd, t, cap, kp, kd)
        self.last_command = cmd

    def _drain(self):
        """Rows for every loop cycle since the last call, tagged with their commands."""
        log, dropped = self.wrist.read_log()
        if dropped:
            self.log_event('log_overflow', dropped=int(dropped))
        rows = []
        if len(log) and int(log['fault'][-1]) and not self._fault_text:
            self._fault_text = self.wrist.fault_text()
        ext = self.sensor.sample() if (self.sensor.continuous and not self.is_sim) else math.nan
        for rec in log:
            sent = self._sent.get(int(rec['command_seq']))
            if sent is None:
                continue   # cycles before this session's first command
            row = self._row(rec, sent)
            if self.sensor.continuous:
                if self.is_sim:
                    cycle = int(rec['cycle'])
                    row['ext_torque_nm'] = (self._ext_by_cycle[cycle] if cycle < len(self._ext_by_cycle)
                                            else math.nan)
                else:
                    row['ext_torque_nm'] = ext
            rows.append(row)
            if sent.meta.log:
                self.rows.append(row)
        if len(log):
            oldest = int(log['command_seq'][-1])
            for seq in [s for s in self._sent if s < oldest]:
                del self._sent[seq]
        return rows

    def _row(self, rec, sent):
        j, N, q0 = self.j, self.N, self.start_q
        p = self._primary
        q, qd = rec['q'], rec['qd']
        state, fault = int(rec['wrist_state']), int(rec['fault'])
        p4 = self._p4
        if state == p4.STATE_ACTIVE:
            mode = int(rec['joint_mode'][j])
        elif state == p4.STATE_BRAKE:
            mode = MODE_BRAKE
        elif state in (p4.STATE_BRAKING, p4.STATE_FAULT):
            mode = MODE_TIMEOUT if fault == 3 else MODE_FAULT
        else:
            mode = MODE_STOPPED
        cmd = sent.command
        setpoint = float(rec['setpoint'][j])
        motor_temp = float(rec['motor_temp'][p])
        if not self.is_sim:
            thermistor = device_config.MOTOR_THERMISTOR
            motor_temp = thermistor(motor_temp) if thermistor and math.isfinite(motor_temp) else math.nan
        t = float(rec['time'])
        row = {
            'time': t,
            't_send': sent.t_send,
            'loop_cycle': int(rec['cycle']),
            'segment': sent.meta.index,
            'phase': sent.meta.phase,
            't_local': t - sent.meta.t0,
            'cmd_joint_deg': cmd.position_deg,
            'cmd_joint_vel_deg_s': cmd.velocity_deg_s,
            'cmd_ff_torque_nm': cmd.torque_nm,
            'kp_scale': cmd.kp_scale,
            'kd_scale': cmd.kd_scale,
            'kp_joint_nm_rad': sent.kp,
            'kd_joint_nms_rad': sent.kd,
            'max_torque_nm': sent.cap,
            'motor_pos_rev': N * float(q[j]) / (2.0 * math.pi),
            'joint_pos_deg': math.degrees(float(q[j]) - q0[j]),
            'joint_vel_deg_s': math.degrees(float(qd[j])),
            'joint_torque_nm': float(rec['tau_meas'][j]),
            'q_current_a': self._current_sign * float(rec['current'][p]),
            'control_joint_deg': math.degrees(setpoint - q0[j]) if math.isfinite(setpoint) else math.nan,
            'control_joint_vel_deg_s': math.degrees(float(rec['setpoint_velocity'][j])),
            'control_joint_torque_nm': float(rec['tau_cmd'][j]),
            'voltage_v': float(rec['bus_voltage'][p]),
            'board_temp_c': float(rec['amp_temp'][p]),
            'motor_temp_c': motor_temp,
            'mode': mode,
            'fault': fault,
            'wrist_state': state,
            'late_ms': 1000.0 * float(rec['late_s']),
            'missed_feedback': int(rec['missed_feedback']),
        }
        for k, name in enumerate(('ps', 'fe', 'ur')):
            row[f'q_{name}_deg'] = math.degrees(float(q[k]))
            row[f'tau_cmd_{name}_nm'] = float(rec['tau_cmd'][k])
            row[f'tau_meas_{name}_nm'] = float(rec['tau_meas'][k])
        for i, node in enumerate(device_config.NODES):
            row[f'counts_{node}'] = int(rec['counts'][i])
            row[f'permille_{node}'] = int(rec['permille'][i])
            row[f'current_{node}_a'] = float(rec['current'][i])
        if cmd.stay_within is not None:
            row['stay_lower_deg'] = cmd.stay_within[0]
            row['stay_upper_deg'] = cmd.stay_within[1]
        if fault:
            row['fault_text'] = self._fault_text
        row.update(sent.meta.info)
        return row

    def _check(self, rows):
        for row in rows:
            reason = self.limits.check(row)
            if reason:
                # Keep the offending sample in the log for diagnosis.
                self.rows.append({**row, 'phase': 'safety_abort'})
                self.abort_reason = reason
                raise SafetyAbort(reason)

    async def _loop(self, meta, period, command_for, done, brake=False):
        """Send command_for(t_local, last) every period until done(t_local, rows) says stop.

        Returns this segment's rows."""
        rows = []
        t0 = meta.t0
        deadline = t0
        last = self.last_sample
        while True:
            t_local = self.clock.now() - t0
            if done(t_local, None):
                break
            self._send(command_for(t_local, last), meta, brake=brake)
            deadline += period
            now = self.clock.now()
            if deadline < now - period:
                deadline = now   # Fell behind; resynchronize instead of bursting.
            await self.clock.sleep_until(deadline)
            new = self._drain()
            mine = [r for r in new if r['segment'] == meta.index]
            rows.extend(mine)
            if new:
                last = new[-1]
                self.last_sample = last
            self._check(new)
            stop = False
            for row in mine:
                if done(row['t_local'], row):
                    stop = True
                    break
            if stop:
                break
        return rows

    async def command(self, cmd: Command):
        """Send one command, wait one loop cycle, and return the latest row."""
        meta = _SegmentMeta(self.segment_index, 'command', {}, self.clock.now(), True)
        self._send(cmd, meta)
        await self.clock.sleep(1.0 / self.loop_rate_hz)
        new = self._drain()
        self._check(new)
        if new:
            self.last_sample = new[-1]
        return self.last_sample

    async def run(self, segments, log=True):
        """Execute segments in order; returns a SegmentResult per segment."""
        results = []
        for segment in segments:
            results.append(await self.run_segment(segment, log=log))
        return results

    async def run_segment(self, segment: Segment, log=True):
        check_info(segment.info)
        self.segment_index += 1
        period = 1.0 / min(segment.rate_hz or self.rate_hz, self.loop_rate_hz)
        meta = _SegmentMeta(self.segment_index, segment.phase, dict(segment.info),
                            self.clock.now(), log)
        ended = {}

        def done(t_local, row):
            if row is None:
                return t_local >= segment.duration_s
            if t_local >= segment.duration_s:
                return True
            if segment.until is not None:
                reason = segment.until(t_local, row)
                if reason:
                    ended['by'] = reason
                    return True
            return False

        rows = await self._loop(meta, period, segment.command, done)
        return SegmentResult(self.segment_index, segment.phase, rows, ended.get('by'))

    async def hold_until(self, done: asyncio.Future, command: Command, phase='hold', info=None):
        """Keep commanding (and logging) a hold until `done` completes."""
        check_info(info)
        self.segment_index += 1
        meta = _SegmentMeta(self.segment_index, phase, dict(info or {}), self.clock.now(), True)
        await self._loop(meta, 1.0 / self.rate_hz, lambda t, s: command,
                         lambda t, row: row is None and done.done())
        return done.result()

    async def brake(self, duration_s, phase='brake', info=None):
        """Drives in IDLE with grounded phases (passive damping), logged at rate.

        Keeps commanding throughout, so the watchdog stays fed. The next
        position command re-captures the setpoint from wherever the joint is.
        """
        check_info(info)
        self.segment_index += 1
        meta = _SegmentMeta(self.segment_index, phase, dict(info or {}), self.clock.now(), True)
        stop = Command(position_deg=math.nan, kp_scale=0.0, kd_scale=0.0)
        await self._loop(meta, 1.0 / self.rate_hz, lambda t, s: stop,
                         lambda t, row: row is None and t >= duration_s, brake=True)
        self.last_command = Command()

    async def prompt_while_holding(self, message, command: Command, phase='hold', info=None):
        """Ask the operator for input while the loop keeps holding."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, input, message)
        return await self.hold_until(future, command, phase, info)

    async def set_gains(self, kp_joint_nm_rad, kd_joint_nms_rad):
        """Joint-side PD gains for subsequent commands (applied by the loop every cycle)."""
        self.kp_joint = kp_joint_nm_rad
        self.kd_joint = kd_joint_nms_rad
        self.log_event('set_gains', kp_joint_nm_rad=kp_joint_nm_rad,
                       kd_joint_nms_rad=kd_joint_nms_rad)

    def current_position_deg(self):
        if self.last_sample is not None:
            return self.last_sample['joint_pos_deg']
        return 0.0

    def current_setpoint_deg(self):
        if math.isfinite(self.last_command.position_deg):
            return self.last_command.position_deg
        return self.current_position_deg()

    # --- operator interaction ------------------------------------------------------------
    def require_locked_output(self, note=''):
        """The output must be clamped to the frame (simulated: lock the plant)."""
        if self.is_sim:
            self.plant.lock_output()
            return
        lines = ['', f'LOCK THE OUTPUT: clamp the {self.joint} output link rigidly to the frame.']
        if note:
            lines.append(note)
        confirm(self.args, lines)

    def log_event(self, event, **fields):
        t = self.clock.now() if self.wrist is not None else math.nan
        self.events.append({'time': t, 'event': event, **fields})

    # --- saving ------------------------------------------------------------------------
    def save(self):
        import pandas as pd
        if self.run_dir is None:
            return
        data_path = self.run_dir / f'{self.experiment}_data.csv'
        if self.rows:
            pd.DataFrame(self.rows).to_csv(data_path, index=False)
        times = [r['time'] for r in self.rows]
        rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else math.nan
        metadata = {
            'experiment': self.experiment,
            'script': Path(sys.argv[0]).name,
            'data_file': data_path.name,
            'completed': self.completed,
            'abort_reason': self.abort_reason,
            'simulated': self.is_sim,
            'run_started_utc': self._t_started,
            'num_samples': len(self.rows),
            'target_rate_hz': self.loop_rate_hz,
            'host_rate_hz': self.rate_hz,
            'requested_rate_hz': self.requested_rate_hz,
            'measured_rate_hz': rate,
            'joint': self.joint,
            'gear_ratio': self.N,
            'start_q_deg': [math.degrees(x) for x in self.start_q],
            'kp_joint_nm_rad': self.kp_joint,
            'kd_joint_nms_rad': self.kd_joint,
            'max_torque_nm': self.max_torque_nm,
            # The CST ramp applies a torque 2 SYNC periods after the position it
            # came from (plus the drive's current loop, not included here).
            'loop_delay_s': 2.0 * device_config.LOOP_PERIOD_S,
            'velocity_filter_hz': device_config.VELOCITY_FILTER_HZ,
            'device_config': device_config_snapshot(),
            'experiment_config': self.config,
            'external_sensor': self.sensor.describe(),
            'events': self.events,
            'host': {'python': sys.version.split()[0], 'platform': platform.platform()},
            **git_provenance(),
            **self.metadata,
        }
        if self.is_sim and self.plant is not None:
            metadata['sim_truth'] = self.plant.truth(self.j)
            metadata['sim_params'] = asdict(self.plant.p)
        write_json(self.run_dir / f'{self.experiment}_metadata.json', metadata)
        status = 'complete' if self.completed else f'INCOMPLETE ({self.abort_reason})'
        print(f'Saved {len(self.rows)} samples to {data_path} [{status}]')


def _inverse_transpose(m):
    import numpy as np
    return np.linalg.inv(np.array(m, dtype=float)).T.tolist()
