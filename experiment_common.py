"""Shared infrastructure for the moteus characterization experiments.

Conventions
-----------
* moteus is configured with motor_position.rotor_to_output_ratio = 1, so
  moteus positions are motor revolutions and torques are motor-side N m.
  device_config.GEAR_RATIO converts to the joint: joint rev = motor rev / N,
  joint torque ~= N * motor torque (ideal, before cable losses).
* Joint positions are relative to the motor position at experiment start.
* Experiments describe what to command as a list of Segments: functions of
  time (and of the last measurement, for host-side feedback) that return a
  joint-side Command. MoteusSession sends them at a fixed rate, checks safety
  limits on every sample, logs everything, and always stops the motor,
  restores the moteus configuration, and saves the data on exit.
* Every experiment runs against the simulator in moteus_sim.py with --sim.
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


MOTEUS_OUTPUT_RATIO_NOTE = (
    "This project assumes moteus motor_position.rotor_to_output_ratio = 1. "
    "GEAR_RATIO is the external cable reduction from measured motor revs to "
    "the characterized joint/load."
)

DATA_ROOT = Path('data')

# moteus modes and the fault code reported with them.
MODE_STOPPED = 0
MODE_FAULT = 1
MODE_POSITION = 10
MODE_TIMEOUT = 11

# Read (best effort) at startup and stored with every run.
CONFIG_SNAPSHOT_KEYS = [
    'servo.pid_position.kp',
    'servo.pid_position.ki',
    'servo.pid_position.kd',
    'servo.pid_position.ilimit',
    'servo.max_current_A',
    'servo.max_velocity',
    'servo.max_voltage',
    'servo.default_velocity_limit',
    'servo.default_accel_limit',
    'servo.default_timeout_s',
    'servo.timeout_max_torque',
    'servo.max_position_slip',
    'servopos.position_min',
    'servopos.position_max',
    'servo.pwm_rate_hz',
    'motor_position.rotor_to_output_ratio',
    'motor.Kv',
]


# --- unit conversions ------------------------------------------------------------
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


def joint_torque_to_motor_torque(joint_torque_nm, gear_ratio):
    return joint_torque_nm / gear_ratio


def joint_gains_to_moteus(kp_joint_nm_rad, kd_joint_nms_rad, gear_ratio):
    """Convert load-side joint gains to moteus gains in motor-rev units.

    With moteus configured at rotor_to_output_ratio=1, position is measured in
    motor revolutions and torque is motor-side Nm. A cable reduction of N means
    joint angle = motor angle / N and joint torque ~= motor torque * N, so both
    Kp and Kd scale by 2*pi/N^2.
    """
    scale = (2.0 * math.pi) / (gear_ratio ** 2)
    return kp_joint_nm_rad * scale, kd_joint_nms_rad * scale


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
    position_deg: float = math.nan   # nan holds the current setpoint (moteus semantics)
    velocity_deg_s: float = 0.0      # Velocity feedforward (setpoint integrates it)
    torque_nm: float = 0.0           # Feedforward torque
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    max_torque_nm: Optional[float] = None   # Per-command cap; device cap if None
    # moteus stay-within mode: (lower_deg, upper_deg), nan for an open side.
    # Inside the bounds only torque_nm is applied; outside, the position gains
    # (x kp/kd_scale) push back toward the violated bound: a unilateral wall
    # rendered at the firmware rate. position_deg/velocity_deg_s are ignored.
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
    'voltage_v', 'board_temp_c', 'motor_temp_c', 'mode', 'fault', 'output_enc_rev',
    'output_joint_raw_deg', 'ext_torque_nm', 'stay_lower_deg', 'stay_upper_deg',
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
    """Raised when a safety limit trips; the session brakes and stops the motor."""


@dataclass
class SafetyLimits:
    joint_min_deg: float = device_config.JOINT_LIMITS_DEG[0]
    joint_max_deg: float = device_config.JOINT_LIMITS_DEG[1]
    max_joint_velocity_deg_s: float = device_config.MAX_JOINT_VELOCITY_DEG_S
    max_motor_temperature_c: float = device_config.MAX_MOTOR_TEMPERATURE_C
    max_board_temperature_c: float = device_config.MAX_BOARD_TEMPERATURE_C

    def check(self, row):
        mode = row.get('mode')
        if mode == MODE_FAULT:
            return f"moteus fault (code {row.get('fault')})"
        if mode == MODE_TIMEOUT:
            return 'moteus watchdog timeout (host stopped commanding in time)'
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
class HardwareClock:
    """Monotonic clock with sub-millisecond waits (sleep, then yield-spin)."""

    def __init__(self):
        self._t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self._t0

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


# --- external torque sensors ---------------------------------------------------------
class ExternalSensor:
    kind = 'none'
    continuous = False      # True if every sample carries ext_torque_nm

    def sample(self, registers):
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


class MoteusAnalogSensor(ExternalSensor):
    """Load-cell amplifier read through a moteus aux analog input register."""
    kind = 'moteus_analog'
    continuous = True

    def __init__(self, config):
        import moteus
        self.config = config
        self.register = int(getattr(moteus.Register, config['register']))

    def sample(self, registers):
        raw = registers.get(self.register, math.nan)
        return (raw - self.config.get('offset_units', 0.0)) * self.config['nm_per_unit']

    def describe(self):
        return dict(self.config)


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

    def sample(self, registers):
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
    """Simulated sensor: reads the plant's true output torque plus noise."""
    continuous = True

    def __init__(self, config, plant):
        self.kind = (config or {}).get('type', 'sim')
        self.config = config or {}
        self.plant = plant
        self.lever_arm_m = self.config.get('lever_arm_m') if self.kind == 'manual' else None

    def sample(self, registers):
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
    if kind == 'moteus_analog':
        return MoteusAnalogSensor(config)
    if kind == 'ati_netft':
        return AtiNetFTSensor(config)
    raise ValueError(f'Unknown EXTERNAL_TORQUE_SENSOR type {kind!r}')


# --- moteus helpers ------------------------------------------------------------------
def make_query_resolution(output_encoder=None, analog_register=None):
    """Query position/velocity/torque/current/temperatures plus control state."""
    import moteus
    mp = moteus.multiplex
    qr = moteus.QueryResolution()
    qr.mode = mp.INT8
    qr.position = mp.F32
    qr.velocity = mp.F32
    qr.torque = mp.F32
    qr.q_current = mp.F32
    qr.motor_temperature = mp.F32
    qr.voltage = mp.INT8
    qr.temperature = mp.INT8
    qr.fault = mp.INT8
    extra = {
        moteus.Register.CONTROL_POSITION: mp.F32,
        moteus.Register.CONTROL_VELOCITY: mp.F32,
        moteus.Register.CONTROL_TORQUE: mp.F32,
    }
    if output_encoder:
        extra[getattr(moteus.Register, output_encoder['register'])] = mp.F32
    if analog_register:
        extra[getattr(moteus.Register, analog_register)] = mp.F32
    qr._extra = extra   # Instance attribute; the class default dict is shared.
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


def _register_ids():
    try:
        import moteus
        R = moteus.Register
        return {name: int(getattr(R, name)) for name in [
            'MODE', 'POSITION', 'VELOCITY', 'TORQUE', 'Q_CURRENT', 'VOLTAGE',
            'TEMPERATURE', 'MOTOR_TEMPERATURE', 'FAULT', 'CONTROL_POSITION',
            'CONTROL_VELOCITY', 'CONTROL_TORQUE']}
    except ImportError:
        return {}


def _format_conf_value(value):
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'nan'
    return f'{value:.9g}'


async def configure_position_pid(controller, kp_nm_per_rev, kd_nm_per_rev_s, ki=0.0,
                                 persist=False):
    """Set servo.pid_position gains (kept for scripts that use the old API)."""
    import moteus
    stream = moteus.Stream(controller, verbose=False)
    for name, value in {'kp': kp_nm_per_rev, 'kd': kd_nm_per_rev_s, 'ki': ki}.items():
        await stream.command(f'conf set servo.pid_position.{name} {value:.9g}'.encode('latin1'))
    if persist:
        await stream.command(b'conf write')
    verified = {}
    for name in ('kp', 'kd', 'ki'):
        response = await stream.command(
            f'conf get servo.pid_position.{name}'.encode('latin1'), allow_any_response=True)
        verified[name] = _decode(response)
    return verified


def _decode(response):
    if response is None:
        return ''
    if isinstance(response, bytes):
        return response.decode('latin1', errors='replace').strip()
    return str(response).strip()


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
                        help='run against the simulated wrist in moteus_sim.py')
    parser.add_argument('--seed', type=int, default=0, help='simulator random seed')
    parser.add_argument('--out', help='output directory (default data/<experiment>/<timestamp>)')
    parser.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    parser.add_argument('--quick', action='store_true',
                        help='shortened protocol for dry runs and tests')
    parser.add_argument('--controller-id', type=int, default=device_config.CONTROLLER_ID)
    try:
        import moteus
        moteus.make_transport_args(parser)
    except ImportError:
        pass
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
        print('\nABORTED by user; motor stopped and partial data saved.', file=sys.stderr)
        raise SystemExit(130)


# --- the session ---------------------------------------------------------------------
class MoteusSession:
    """One experiment run: connection, configuration, safety, timing, logging.

    Usage:
        async with MoteusSession('step_response', args, kp, kd, config=cfg) as s:
            await s.run(segments)
    """

    def __init__(self, experiment, args, kp_joint_nm_rad, kd_joint_nms_rad, *,
                 config=None, rate_hz=1000.0, needs_sensor=False, sim_params=None,
                 limits: Optional[SafetyLimits] = None, max_torque_nm=None):
        self.experiment = experiment
        self.args = args
        self.config = config
        self.rate_hz = rate_hz
        self.N = device_config.GEAR_RATIO
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
        self.start_rev = math.nan
        self.original_config = {}
        self.config_snapshot = {}
        self.changed_keys = []
        self.abort_reason = None
        self.completed = False
        self.run_dir = None
        self.plant = None
        self.sensor = ExternalSensor()
        self._reg = _register_ids()
        self._output_encoder = device_config.OUTPUT_ENCODER
        self._encoder_reg = None
        self._t_started = None

    # --- lifecycle ---------------------------------------------------------------------
    async def __aenter__(self):
        self.run_dir = new_run_dir(self.experiment, self.is_sim, getattr(self.args, 'out', None))
        sensor_config = device_config.EXTERNAL_TORQUE_SENSOR if self.needs_sensor else None
        analog_register = (sensor_config.get('register')
                           if sensor_config and sensor_config.get('type') == 'moteus_analog'
                           else None)
        if self.is_sim:
            import moteus_sim
            params = self.sim_params or moteus_sim.SimParams()
            params.seed = getattr(self.args, 'seed', params.seed)
            params.gear_ratio = self.N
            analog = sensor_config if analog_register else None
            self.controller, self.clock, stream_factory, self.plant = moteus_sim.make_sim(
                params, output_encoder=self._output_encoder, analog_sensor=analog)
            self.stream = stream_factory(self.controller)
            if self.needs_sensor:
                self.sensor = make_sensor(sensor_config, plant=self.plant)
        else:
            import moteus
            qr = make_query_resolution(self._output_encoder, analog_register)
            transport = moteus.get_singleton_transport(self.args)
            self.controller = moteus.Controller(
                id=self.args.controller_id, query_resolution=qr, transport=transport)
            self.clock = HardwareClock()
            self.stream = moteus.Stream(self.controller, verbose=False)
            if self.needs_sensor:
                self.sensor = make_sensor(sensor_config)
        if self._output_encoder:
            import moteus
            self._encoder_reg = int(getattr(moteus.Register, self._output_encoder['register']))

        try:
            await self.controller.set_stop()
            await self._stop_telemetry()
            await self._snapshot_config()
            self._check_units()
            await self._apply_config()

            state = await self.controller.query()
            self.start_rev = state_value(state, self._reg['POSITION'])
            if not math.isfinite(self.start_rev):
                raise RuntimeError('Could not read the motor position from moteus.')
            mode = state_value(state, self._reg['MODE'])
            if mode == MODE_FAULT:
                raise RuntimeError(f"moteus is in fault {state_value(state, self._reg['FAULT'])}; "
                                   'clear it before running.')
        except BaseException:
            try:
                await self.controller.set_stop()
                await self._restore_config()
            finally:
                self.sensor.close()
            raise
        self._t_started = datetime.now(timezone.utc).isoformat()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc is None:
            self.completed = True
        elif self.abort_reason is None:
            self.abort_reason = f'{exc_type.__name__}: {exc}' if str(exc) else exc_type.__name__
        try:
            if exc is not None:
                # Brake first so a moving joint is damped, then leave it stopped.
                try:
                    await self.controller.set_brake()
                    await self.clock.sleep(0.3)
                except Exception:
                    pass
            await self.controller.set_stop()
            await self._restore_config()
        except BaseException as cleanup_error:   # Keep going: data must still be saved.
            print(f'WARNING: cleanup error: {cleanup_error!r}', file=sys.stderr)
        finally:
            self.sensor.close()
            self.save()
        return False

    # --- moteus configuration --------------------------------------------------------
    async def _stop_telemetry(self):
        try:
            await self.stream.write_message(b'tel stop')
            await self.stream.flush_read()
        except Exception as exc:
            print(f'WARNING: could not stop telemetry: {exc!r}', file=sys.stderr)

    async def conf_get(self, key, timeout=1.0):
        try:
            response = await asyncio.wait_for(
                self.stream.command(f'conf get {key}'.encode('latin1'), allow_any_response=True),
                timeout)
        except Exception:
            await self._recover_stream()
            return None
        text = _decode(response)
        return None if text.startswith('ERR') or text == '' else text

    async def conf_set(self, key, value, remember=True, timeout=1.0):
        if remember and key not in self.original_config:
            original = self.config_snapshot.get(key)
            if original is None:
                original = await self.conf_get(key)
            self.original_config[key] = original
        text = _format_conf_value(value)
        await asyncio.wait_for(
            self.stream.command(f'conf set {key} {text}'.encode('latin1')), timeout)
        if key not in self.changed_keys:
            self.changed_keys.append(key)

    async def _recover_stream(self):
        try:
            await self.stream.flush_read()
        except Exception:
            pass

    async def _snapshot_config(self):
        for key in CONFIG_SNAPSHOT_KEYS:
            value = await self.conf_get(key)
            if value is not None:
                self.config_snapshot[key] = value

    def _check_units(self):
        ratio = self.config_snapshot.get('motor_position.rotor_to_output_ratio')
        if ratio is not None and abs(float(ratio) - 1.0) > 1e-6:
            raise RuntimeError(
                f'motor_position.rotor_to_output_ratio is {ratio}, but these scripts assume 1 '
                '(positions in motor revs, GEAR_RATIO applied on the host).')

    async def _apply_config(self):
        """Gains for this run, no trajectory limiting, no integrator."""
        await self.set_gains(self.kp_joint, self.kd_joint)
        await self.conf_set('servo.pid_position.ki', 0.0)
        # A non-nan default velocity/accel limit makes moteus turn steps into
        # trajectories, which would invalidate step and frequency responses.
        # A finite max_position_slip drags the setpoint toward the measured
        # position, which shrinks large steps and distorts tracking tests.
        for key in ('servo.default_velocity_limit', 'servo.default_accel_limit',
                    'servo.max_position_slip'):
            if key not in self.config_snapshot:
                print(f'NOTE: {key} not reported by this firmware; assuming it is disabled.')
                continue
            await self.conf_set(key, math.nan)
            value = await self.conf_get(key)
            if value is not None and value.lower() != 'nan':
                raise RuntimeError(f'{key} reads back {value}; expected nan.')
        verified = {}
        for name in ('kp', 'kd', 'ki'):
            verified[name] = await self.conf_get(f'servo.pid_position.{name}')
        self.metadata['verified_moteus_pid_config'] = verified

    async def set_gains(self, kp_joint_nm_rad, kd_joint_nms_rad):
        """Set the moteus position-loop gains from joint-side values."""
        kp_m, kd_m = joint_gains_to_moteus(kp_joint_nm_rad, kd_joint_nms_rad, self.N)
        await self.conf_set('servo.pid_position.kp', kp_m)
        await self.conf_set('servo.pid_position.kd', kd_m)
        self.kp_joint = kp_joint_nm_rad
        self.kd_joint = kd_joint_nms_rad
        self.events.append({'time': self.clock.now(), 'event': 'set_gains',
                            'kp_joint_nm_rad': kp_joint_nm_rad,
                            'kd_joint_nms_rad': kd_joint_nms_rad})

    async def _restore_config(self):
        for key in reversed(self.changed_keys):
            original = self.original_config.get(key)
            if original is None:
                continue
            try:
                await self.conf_set(key, original, remember=False)
            except Exception as exc:
                print(f'WARNING: could not restore {key}={original}: {exc!r}', file=sys.stderr)

    # --- commanding --------------------------------------------------------------------
    async def command(self, cmd: Command):
        """Send one command, read the reply, check safety, and return the row."""
        N = self.N
        if math.isfinite(cmd.position_deg):
            position = self.start_rev + joint_degrees_to_motor_revs(cmd.position_deg, N)
        else:
            position = math.nan
        cap = self.max_torque_nm if cmd.max_torque_nm is None else min(
            abs(cmd.max_torque_nm), self.max_torque_nm)
        t_send = self.clock.now()
        if cmd.stay_within is not None:
            def bound(deg):
                if deg is None or not math.isfinite(deg):
                    return math.nan
                return self.start_rev + joint_degrees_to_motor_revs(deg, N)

            state = await self.controller.set_stay_within(
                lower_bound=bound(cmd.stay_within[0]),
                upper_bound=bound(cmd.stay_within[1]),
                feedforward_torque=cmd.torque_nm / N,
                kp_scale=cmd.kp_scale,
                kd_scale=cmd.kd_scale,
                maximum_torque=cap / N,
                watchdog_timeout=device_config.WATCHDOG_TIMEOUT_S,
                query=True,
            )
        else:
            state = await self.controller.set_position(
                position=position,
                velocity=joint_degrees_to_motor_revs(cmd.velocity_deg_s, N),
                feedforward_torque=cmd.torque_nm / N,
                kp_scale=cmd.kp_scale,
                kd_scale=cmd.kd_scale,
                maximum_torque=cap / N,
                watchdog_timeout=device_config.WATCHDOG_TIMEOUT_S,
                query=True,
            )
        t_reply = self.clock.now()
        row = self._parse(state)
        if cmd.stay_within is not None:
            row['stay_lower_deg'] = cmd.stay_within[0]
            row['stay_upper_deg'] = cmd.stay_within[1]
        row.update({
            'time': t_reply,
            't_send': t_send,
            'cmd_joint_deg': cmd.position_deg,
            'cmd_joint_vel_deg_s': cmd.velocity_deg_s,
            'cmd_ff_torque_nm': cmd.torque_nm,
            'kp_scale': cmd.kp_scale,
            'kd_scale': cmd.kd_scale,
            'kp_joint_nm_rad': self.kp_joint * cmd.kp_scale,
            'kd_joint_nms_rad': self.kd_joint * cmd.kd_scale,
            'max_torque_nm': cap,
        })
        self.last_command = cmd
        reason = self.limits.check(row)
        if reason:
            # Keep the offending sample in the log for diagnosis.
            self.rows.append({**row, 'phase': 'safety_abort', 'segment': self.segment_index})
            self.abort_reason = reason
            raise SafetyAbort(reason)
        return row

    def _parse(self, state):
        values = getattr(state, 'values', {}) if state is not None else {}
        R, N = self._reg, self.N

        def get(name):
            return values.get(R[name], math.nan)

        pos = get('POSITION')
        control_pos = get('CONTROL_POSITION')
        row = {
            'motor_pos_rev': pos,
            'joint_pos_deg': motor_revs_to_joint_degrees(pos - self.start_rev, N),
            'joint_vel_deg_s': motor_velocity_to_joint_deg_s(get('VELOCITY'), N),
            'joint_torque_nm': motor_torque_to_joint_torque(get('TORQUE'), N),
            'q_current_a': get('Q_CURRENT'),
            'control_joint_deg': motor_revs_to_joint_degrees(control_pos - self.start_rev, N),
            'control_joint_vel_deg_s': motor_velocity_to_joint_deg_s(get('CONTROL_VELOCITY'), N),
            'control_joint_torque_nm': motor_torque_to_joint_torque(get('CONTROL_TORQUE'), N),
            'voltage_v': get('VOLTAGE'),
            'board_temp_c': get('TEMPERATURE'),
            'motor_temp_c': get('MOTOR_TEMPERATURE'),
            'mode': values.get(R['MODE'], -1),
            'fault': values.get(R['FAULT'], 0),
        }
        if self._encoder_reg is not None:
            enc = values.get(self._encoder_reg, math.nan)
            scale = self._output_encoder.get('revs_per_joint_rev', 1.0) * self._output_encoder.get('sign', 1)
            row['output_enc_rev'] = enc
            row['output_joint_raw_deg'] = enc / scale * 360.0
        if self.sensor.continuous:
            row['ext_torque_nm'] = self.sensor.sample(values)
        return row

    async def run(self, segments, log=True):
        """Execute segments in order; returns a SegmentResult per segment."""
        results = []
        for segment in segments:
            results.append(await self.run_segment(segment, log=log))
        return results

    async def run_segment(self, segment: Segment, log=True):
        check_info(segment.info)
        self.segment_index += 1
        period = 1.0 / (segment.rate_hz or self.rate_hz)
        rows = []
        ended_by = None
        t0 = self.clock.now()
        deadline = t0
        last = self.last_sample
        while True:
            t_local = self.clock.now() - t0
            if t_local >= segment.duration_s:
                break
            row = await self.command(segment.command(t_local, last))
            row.update({'segment': self.segment_index, 'phase': segment.phase,
                        't_local': t_local})
            row.update(segment.info)
            rows.append(row)
            if log:
                self.rows.append(row)
            last = row
            self.last_sample = row
            if segment.until is not None:
                ended_by = segment.until(t_local, row)
                if ended_by:
                    break
            deadline += period
            now = self.clock.now()
            if deadline < now - period:
                deadline = now   # Fell behind; resynchronize instead of bursting.
            await self.clock.sleep_until(deadline)
        return SegmentResult(self.segment_index, segment.phase, rows, ended_by)

    async def hold_until(self, done: asyncio.Future, command: Command, phase='hold', info=None):
        """Keep commanding (and logging) a hold until `done` completes."""
        check_info(info)
        period = 1.0 / self.rate_hz
        self.segment_index += 1
        t0 = self.clock.now()
        deadline = t0
        while not done.done():
            row = await self.command(command)
            row.update({'segment': self.segment_index, 'phase': phase,
                        't_local': self.clock.now() - t0})
            row.update(info or {})
            self.rows.append(row)
            self.last_sample = row
            deadline += period
            await self.clock.sleep_until(max(deadline, self.clock.now()))
        return done.result()

    async def brake(self, duration_s, phase='brake', info=None):
        """moteus brake mode (shorted windings: passive damping), logged at rate.

        Keeps commanding throughout, so the run never goes quiet. The next
        position command re-captures the setpoint from wherever the joint is.
        """
        check_info(info)
        period = 1.0 / self.rate_hz
        self.segment_index += 1
        t0 = self.clock.now()
        deadline = t0
        while self.clock.now() - t0 < duration_s:
            t_send = self.clock.now()
            state = await self.controller.set_brake(query=True)
            row = self._parse(state)
            row.update({'time': self.clock.now(), 't_send': t_send, 'cmd_joint_deg': math.nan,
                        'cmd_joint_vel_deg_s': 0.0, 'cmd_ff_torque_nm': 0.0, 'kp_scale': 0.0,
                        'kd_scale': 0.0, 'kp_joint_nm_rad': 0.0, 'kd_joint_nms_rad': 0.0,
                        'max_torque_nm': math.nan, 'segment': self.segment_index,
                        'phase': phase, 't_local': t_send - t0})
            row.update(info or {})
            self.rows.append(row)
            self.last_sample = row
            reason = self.limits.check(row)
            if reason:
                self.abort_reason = reason
                raise SafetyAbort(reason)
            deadline += period
            await self.clock.sleep_until(max(deadline, self.clock.now()))
        self.last_command = Command()

    async def prompt_while_holding(self, message, command: Command, phase='hold', info=None):
        """Ask the operator for input while the controller keeps holding."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, input, message)
        return await self.hold_until(future, command, phase, info)

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
        lines = ['', 'LOCK THE OUTPUT: clamp the handle/output link rigidly to the frame.']
        if note:
            lines.append(note)
        confirm(self.args, lines)

    def log_event(self, event, **fields):
        self.events.append({'time': self.clock.now(), 'event': event, **fields})

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
            'target_rate_hz': self.rate_hz,
            'measured_rate_hz': rate,
            'gear_ratio': self.N,
            'start_pos_rev': self.start_rev,
            'kp_joint_nm_rad': self.kp_joint,
            'kd_joint_nms_rad': self.kd_joint,
            'max_torque_nm': self.max_torque_nm,
            'moteus_rotor_to_output_ratio_assumption': 1,
            'moteus_config_snapshot': self.config_snapshot,
            'moteus_config_changed_then_restored': self.original_config,
            'device_config': device_config_snapshot(),
            'experiment_config': self.config,
            'external_sensor': self.sensor.describe(),
            'events': self.events,
            'host': {'python': sys.version.split()[0], 'platform': platform.platform()},
            **git_provenance(),
            **self.metadata,
        }
        if self.is_sim and self.plant is not None:
            metadata['sim_truth'] = self.plant.truth()
        write_json(self.run_dir / f'{self.experiment}_metadata.json', metadata)
        status = 'complete' if self.completed else f'INCOMPLETE ({self.abort_reason})'
        print(f'Saved {len(self.rows)} samples to {data_path} [{status}]')
