"""Simulated moteus controller driving a 1-DOF cable-driven wrist.

The simulator implements the subset of the moteus Python API used by these
experiments (set_position, set_stop, set_brake, query, and the diagnostic
"conf get/set" stream) on top of a two-mass plant:

    motor side  Jm wm' = tau_act - tau_t - b wm - friction(wm) - cogging(thm)
    output side Jo wo' = tau_t - bo wo - gravity(tho) + tau_human
    cable       tau_t  = k (thm - tho) + c (wm - wo)

with Karnopp stick-slip friction (static, Coulomb, Stribeck, viscous), the
moteus position/velocity/feedforward control law, torque saturation and
back-EMF derating, current-loop lag, encoder noise, a first-order winding
thermal model, and optional locked output, external torque sensor, and a
human operator. All quantities are joint-referred (after the cable reduction)
unless the name says motor.

Every experiment accepts --sim to run against this model, which is useful for
dry-running a protocol before touching hardware and for checking that each
analysis recovers the known parameters below (see tests/).
"""
import math
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

try:
    import moteus
    _R = moteus.Register
    _MODE = {m.name: int(m.value) for m in moteus.Mode}
except ImportError:  # Allows analysis-only installs to import this module.
    moteus = None
    _R = None
    _MODE = {'STOPPED': 0, 'FAULT': 1, 'POSITION': 10, 'TIMEOUT': 11, 'STAY_WITHIN': 13,
             'BRAKE': 15}


@dataclass
class SimParams:
    """Ground-truth plant parameters (joint-referred SI units unless named motor).

    Motor defaults are the CubeMars GL60 II KV28 at a 36 V supply. Properties
    give the joint-referred cogging and torque-speed limits that follow from them.
    """
    gear_ratio: float = 9.806            # MoteusSession sets this from device_config
    motor_inertia: float = 0.004         # Rotor (GL60 II: 401 g cm^2 x N^2) + capstan (kg m^2)
    output_inertia: float = 0.004        # Handle and link (kg m^2)
    transmission_stiffness: float = 2000.0   # Cable stiffness (N m/rad); math.inf for rigid
    transmission_damping: float = 0.4    # Cable damping (N m s/rad)
    backlash_deg: float = 0.0            # Total transmission deadband (deg)
    viscous: float = 0.02                # Motor-side viscous friction (N m s/rad)
    coulomb: float = 0.05                # Kinetic friction (N m)
    static: float = 0.07                 # Breakaway friction (N m)
    stribeck_velocity: float = 0.05      # Stribeck velocity (rad/s)
    output_viscous: float = 0.002        # Output bearing damping (N m s/rad)
    cogging_amplitude_motor: float = 0.015   # GL60 II: 1.5 cN m, taken as the amplitude (motor N m)
    cogging_periods_per_motor_rev: float = 168.0   # LCM of the GL60 II's 24 slots and 28 poles
    gravity_torque: float = 0.0          # m g l of the handle (N m), acts as -m g l sin(tho)
    max_torque: float = 12.0             # Controller current limit (N m)
    motor_kv: float = 28.0               # Speed constant (rpm/V); no-load speed = kv x bus_voltage
    motor_stall_torque_per_volt: float = 0.0625   # Voltage-limited stall torque (motor N m/V):
                                                  # GL60 II peaks 1.0 N m at 16 V, 1.5 at 24 V
    current_loop_time_constant: float = 1e-4   # First-order torque lag (s)
    torque_constant_ratio: float = 1.0   # Actual / moteus-reported torque (Kt error)
    motor_torque_constant: float = 0.08  # Motor-side N m/A, sets reported q current
    velocity_filter_hz: float = 400.0    # Bandwidth of moteus' velocity estimate
    position_noise_rev: float = 2e-5     # Motor encoder noise, 1 sigma (motor revs)
    velocity_noise_rev_s: float = 2e-3   # Reported velocity noise (motor rev/s)
    torque_noise: float = 0.002          # Reported torque noise (motor N m)
    output_encoder_noise_rev: float = 1e-5
    external_sensor_noise: float = 0.01  # External torque sensor noise (N m)
    initial_motor_rev: float = 3.3712    # Absolute moteus position at start (tests offsets)
    winding_resistance: float = 0.12     # Ohm
    thermal_resistance: float = 4.0      # Winding to ambient (C/W)
    thermal_time_constant: float = 90.0  # Winding (s); real motors are slower
    board_thermal_resistance: float = 1.0
    board_time_constant: float = 30.0
    ambient_temperature: float = 25.0
    bus_voltage: float = 36.0            # Supply (V); sets the back-EMF torque-speed line
    brake_damping: float = 0.5           # Joint damping in brake/timeout mode (N m s/rad)
    round_trip_s: float = 4e-4           # Host <-> moteus command/reply latency
    round_trip_jitter_s: float = 1e-4
    dt: float = 1e-4                     # Integration step (s)
    seed: int = 0

    @property
    def cogging_amplitude(self):
        """Cogging torque amplitude at the joint (N m)."""
        return self.cogging_amplitude_motor * self.gear_ratio

    @property
    def cogging_period_deg(self):
        """Cogging period in joint degrees."""
        return 360.0 / (self.cogging_periods_per_motor_rev * self.gear_ratio)

    @property
    def stall_torque(self):
        """Voltage-limited stall torque at the joint (N m)."""
        return self.motor_stall_torque_per_volt * self.bus_voltage * self.gear_ratio

    @property
    def no_load_speed(self):
        """Joint speed at which the motoring torque reaches zero (rad/s)."""
        return self.motor_kv * self.bus_voltage * (2.0 * math.pi / 60.0) / self.gear_ratio

    def derived(self):
        return {'cogging_amplitude': self.cogging_amplitude,
                'cogging_period_deg': self.cogging_period_deg,
                'stall_torque': self.stall_torque, 'no_load_speed': self.no_load_speed}


@dataclass
class HumanModel:
    """Operator who drags the handle along a reference through a spring-damper."""
    motion: Callable[[float], tuple]      # t -> (theta_rad, omega_rad_s), joint frame
    stiffness: float = 40.0               # Arm/hand stiffness (N m/rad)
    damping: float = 0.8                  # (N m s/rad)


_DEFAULT_CONFIG = {
    'servo.pid_position.kp': '4.0',
    'servo.pid_position.ki': '0.0',
    'servo.pid_position.kd': '0.05',
    'servo.pid_position.ilimit': '0.0',
    'servo.max_current_A': '40.0',
    'servo.max_velocity': '500.0',
    'servo.max_voltage': '46.0',
    'servo.default_velocity_limit': 'nan',
    'servo.default_accel_limit': 'nan',
    'servo.default_timeout_s': '0.1',
    'servo.timeout_max_torque': '5.0',
    'servo.max_position_slip': 'nan',
    'servopos.position_min': 'nan',
    'servopos.position_max': 'nan',
    'servo.pwm_rate_hz': '30000',
    'motor_position.rotor_to_output_ratio': '1.0',
    'motor.Kv': '120.0',
}


class CommandError(RuntimeError):
    pass


class SimPlant:
    """Two-mass plant plus the moteus position controller, integrated at dt."""

    def __init__(self, params: SimParams, output_encoder=None, analog_sensor=None):
        self.p = params
        self.rng = np.random.default_rng(params.seed)
        self.config = dict(_DEFAULT_CONFIG)
        self.output_encoder = output_encoder
        self.analog_sensor = analog_sensor
        self.human: Optional[HumanModel] = None
        self.output_locked = False
        self.lock_angle = 0.0

        self.t = 0.0
        self.thm = 0.0
        self.wm = 0.0
        self.tho = 0.0
        self.wo = 0.0
        self.stuck = True
        self.tau_act = 0.0           # Joint-side torque command after the current loop
        self.tau_cur = 0.0           # ... after back-EMF limiting (what moteus reports)
        self.vel_est = 0.0
        self.tau_t = 0.0
        self.tau_h = 0.0
        self.temp_motor = params.ambient_temperature
        self.temp_board = params.ambient_temperature

        self.mode = _MODE['STOPPED']
        self.fault = 0
        self.control_pos_rev = math.nan
        self.control_vel_rev_s = 0.0
        self.target_pos_rev = math.nan
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.ff_motor = 0.0
        self.max_torque_motor = math.inf
        self.velocity_limit = math.nan
        self.stay_lower_rev = math.nan
        self.stay_upper_rev = math.nan
        self.last_command_t = 0.0
        self.watchdog = math.inf
        self.control_torque_motor = 0.0

    # --- configuration ---------------------------------------------------------
    def conf_float(self, key):
        return float(self.config.get(key, 'nan'))

    def lock_output(self, angle_rad=None):
        if not math.isfinite(self.p.transmission_stiffness):
            raise ValueError('A locked output needs finite transmission_stiffness.')
        self.output_locked = True
        self.lock_angle = self.tho if angle_rad is None else angle_rad
        self.tho = self.lock_angle
        self.wo = 0.0

    def unlock_output(self):
        self.output_locked = False

    # --- moteus firmware emulation ---------------------------------------------
    def motor_rev(self):
        return self.p.initial_motor_rev + self.thm * self.p.gear_ratio / (2.0 * math.pi)

    def apply_position_command(self, kw):
        position = kw.get('position')
        velocity = kw.get('velocity')
        if self.mode != _MODE['POSITION']:
            self.control_pos_rev = self.motor_rev()
            self.control_vel_rev_s = 0.0
        self.mode = _MODE['POSITION']
        self.fault = 0
        self.control_vel_rev_s = 0.0 if velocity is None else float(velocity)
        if position is not None and math.isfinite(position):
            self.target_pos_rev = float(position)
            vlim = kw.get('velocity_limit')
            if vlim is None:
                vlim = self.conf_float('servo.default_velocity_limit')
            self.velocity_limit = vlim if vlim is not None else math.nan
            if not math.isfinite(self.velocity_limit):
                self.control_pos_rev = self.target_pos_rev
        else:
            self.target_pos_rev = math.nan
        self._apply_common(kw)

    def apply_stay_within_command(self, kw):
        """Stay-within mode: feedforward torque only inside [lower, upper];
        outside, the position gains push toward the violated bound with zero
        target velocity (a unilateral wall at the firmware rate)."""
        def bound(value):
            return float(value) if value is not None and math.isfinite(value) else math.nan

        self.mode = _MODE['STAY_WITHIN']
        self.fault = 0
        self.control_pos_rev = math.nan
        self.control_vel_rev_s = 0.0
        self.stay_lower_rev = bound(kw.get('lower_bound'))
        self.stay_upper_rev = bound(kw.get('upper_bound'))
        self._apply_common(kw)

    def _apply_common(self, kw):
        self.kp_scale = 1.0 if kw.get('kp_scale') is None else float(kw['kp_scale'])
        self.kd_scale = 1.0 if kw.get('kd_scale') is None else float(kw['kd_scale'])
        self.ff_motor = 0.0 if kw.get('feedforward_torque') is None else float(kw['feedforward_torque'])
        mt = kw.get('maximum_torque')
        self.max_torque_motor = math.inf if mt is None or not math.isfinite(mt) else abs(float(mt))
        wd = kw.get('watchdog_timeout')
        if wd is None:
            wd = self.conf_float('servo.default_timeout_s')
        self.watchdog = wd if (wd is not None and math.isfinite(wd)) else math.inf
        self.last_command_t = self.t

    def set_mode(self, name):
        self.mode = _MODE[name]
        self.control_pos_rev = math.nan
        self.last_command_t = self.t
        self.watchdog = math.inf
        if name == 'STOPPED':
            self.fault = 0

    # --- integration -------------------------------------------------------------
    def advance_to(self, t_end):
        """Integrate the plant and controller from self.t to t_end."""
        p = self.p
        if t_end <= self.t:
            return
        dt = p.dt
        n = int(math.ceil((t_end - self.t) / dt - 1e-9))
        if n <= 0:
            return
        dt = (t_end - self.t) / n

        N = p.gear_ratio
        two_pi = 2.0 * math.pi
        rev_per_rad = N / two_pi
        kp = self.conf_float('servo.pid_position.kp') * self.kp_scale
        kd = self.conf_float('servo.pid_position.kd') * self.kd_scale
        rigid = not math.isfinite(p.transmission_stiffness)
        Jm = p.motor_inertia + (p.output_inertia if rigid else 0.0)
        Jo = p.output_inertia
        k, c = p.transmission_stiffness, p.transmission_damping
        half_bl = math.radians(p.backlash_deg) / 2.0
        b, bo = p.viscous, p.output_viscous
        fc, fs, vs = p.coulomb, p.static, p.stribeck_velocity
        cog_a = p.cogging_amplitude
        cog_w = N * p.cogging_periods_per_motor_rev
        mgl = p.gravity_torque
        tau_hw = p.max_torque
        tau_stall = p.stall_torque
        w_nl = p.no_load_speed
        kt_ratio = p.torque_constant_ratio
        alpha = 1.0 - math.exp(-dt / p.current_loop_time_constant)
        beta = 1.0 - math.exp(-two_pi * p.velocity_filter_hz * dt)
        mode = self.mode
        position_mode = mode == _MODE['POSITION']
        stay_mode = mode == _MODE['STAY_WITHIN']
        stay_lo, stay_hi = self.stay_lower_rev, self.stay_upper_rev
        braking = mode in (_MODE['BRAKE'], _MODE['TIMEOUT'])
        locked = self.output_locked and not rigid
        human = self.human

        thm, wm, tho, wo = self.thm, self.wm, self.tho, self.wo
        stuck, tau_act, vel_est = self.stuck, self.tau_act, self.vel_est
        tau_cur = self.tau_cur
        cp, cv = self.control_pos_rev, self.control_vel_rev_s
        target, vlim = self.target_pos_rev, self.velocity_limit
        ff, tmax_cmd = self.ff_motor, self.max_torque_motor
        t = self.t
        sum_i2 = 0.0
        tau_t = self.tau_t
        tau_h = 0.0
        ctrl_m = self.control_torque_motor
        rev0 = p.initial_motor_rev
        watchdog_deadline = self.last_command_t + self.watchdog

        for _ in range(n):
            t += dt
            # Watchdog: moteus drops into timeout mode if commands stop arriving.
            if (position_mode or stay_mode) and t > watchdog_deadline:
                position_mode = stay_mode = False
                braking = True
                self.mode = mode = _MODE['TIMEOUT']

            # Controller (motor-side units, like the firmware).
            if position_mode:
                pos_rev = rev0 + thm * rev_per_rad
                tau_m = (kp * (cp - pos_rev)
                         + kd * (cv - vel_est * rev_per_rad) + ff)
                limit_m = min(tmax_cmd, tau_hw / N)
                tau_m = max(-limit_m, min(limit_m, tau_m))
                ctrl_m = tau_m
                tau_cmd = tau_m * N
                # Advance the setpoint after using it, so the error compares
                # setpoint and position at the same instant whatever dt is.
                if math.isfinite(vlim) and math.isfinite(target):
                    step = vlim * dt
                    cp += max(-step, min(step, target - cp))
                else:
                    cp += cv * dt
            elif stay_mode:
                pos_rev = rev0 + thm * rev_per_rad
                if pos_rev > stay_hi:
                    tau_m = kp * (stay_hi - pos_rev) - kd * vel_est * rev_per_rad + ff
                elif pos_rev < stay_lo:
                    tau_m = kp * (stay_lo - pos_rev) - kd * vel_est * rev_per_rad + ff
                else:
                    tau_m = ff
                limit_m = min(tmax_cmd, tau_hw / N)
                tau_m = max(-limit_m, min(limit_m, tau_m))
                ctrl_m = tau_m
                tau_cmd = tau_m * N
            elif braking:
                tau_cmd = -p.brake_damping * wm
                ctrl_m = tau_cmd / N
            else:
                tau_cmd = 0.0
                ctrl_m = 0.0
            tau_act += alpha * (tau_cmd - tau_act)
            # Back-EMF limits the current (and so the reported torque) while
            # motoring: the supply allows the stall torque at rest, falling
            # linearly to zero at the no-load speed, and the current limit caps
            # it below that. The Kt error then scales the physical torque.
            tau_cur = tau_act
            if tau_cur * wm > 0.0:
                avail = min(tau_hw, tau_stall * max(0.0, 1.0 - abs(wm) / w_nl))
                if abs(tau_cur) > avail:
                    tau_cur = math.copysign(avail, tau_cur)
            sum_i2 += (tau_cur / N / p.motor_torque_constant) ** 2
            tau_phys = tau_cur * kt_ratio

            cog = cog_a * math.sin(cog_w * thm) if cog_a else 0.0

            if rigid:
                tau_t = 0.0
                ext = tau_phys - b * wm - cog
                if mgl:
                    ext -= mgl * math.sin(thm)
                if human is not None:
                    th_h, w_h = human.motion(t)
                    tau_h = human.stiffness * (th_h - thm) + human.damping * (w_h - wm)
                    ext += tau_h
            else:
                d = thm - tho
                if half_bl > 0.0:
                    if d > half_bl:
                        tau_t = k * (d - half_bl) + c * (wm - wo)
                    elif d < -half_bl:
                        tau_t = k * (d + half_bl) + c * (wm - wo)
                    else:
                        tau_t = 0.0
                else:
                    tau_t = k * d + c * (wm - wo)
                ext = tau_phys - tau_t - b * wm - cog

            # Karnopp stick-slip friction on the motor side.
            if stuck or wm == 0.0:
                if abs(ext) <= fs:
                    wm_new = 0.0
                    stuck = True
                else:
                    stuck = False
                    wm_new = wm + dt * (ext - math.copysign(fs, ext)) / Jm
            else:
                fr = fc + (fs - fc) * math.exp(-(wm / vs) ** 2)
                wm_new = wm + dt * (ext - math.copysign(fr, wm)) / Jm
                if wm_new * wm < 0.0:
                    wm_new = 0.0
                    stuck = True
            thm += dt * wm_new
            wm = wm_new
            vel_est += beta * (wm - vel_est)

            if rigid:
                tho, wo = thm, wm
            elif locked:
                tho, wo = self.lock_angle, 0.0
            else:
                ext_o = tau_t - bo * wo
                if mgl:
                    ext_o -= mgl * math.sin(tho)
                if human is not None:
                    th_h, w_h = human.motion(t)
                    tau_h = human.stiffness * (th_h - tho) + human.damping * (w_h - wo)
                    ext_o += tau_h
                wo += dt * ext_o / Jo
                tho += dt * wo

        elapsed = t - self.t
        self.t = t
        self.thm, self.wm, self.tho, self.wo = thm, wm, tho, wo
        self.stuck, self.tau_act, self.vel_est = stuck, tau_act, vel_est
        self.tau_cur = tau_cur
        self.control_pos_rev = cp
        self.tau_t, self.tau_h = tau_t, tau_h
        self.control_torque_motor = ctrl_m

        # Thermal model, updated with the mean copper loss over this interval.
        power = p.winding_resistance * sum_i2 / n
        t_ss = p.ambient_temperature + power * p.thermal_resistance
        self.temp_motor += (t_ss - self.temp_motor) * (1.0 - math.exp(-elapsed / p.thermal_time_constant))
        b_ss = p.ambient_temperature + power * p.board_thermal_resistance
        self.temp_board += (b_ss - self.temp_board) * (1.0 - math.exp(-elapsed / p.board_time_constant))
        if self.temp_motor > 130.0 or self.temp_board > 100.0:
            self.mode = _MODE['FAULT']
            self.fault = 38

    # --- measurement -----------------------------------------------------------
    def external_torque(self):
        """Torque an external sensor would read (handle or lock), joint N m."""
        if self.output_locked:
            value = self.tau_t
        else:
            value = self.tau_h
        return value + self.rng.normal(0.0, self.p.external_sensor_noise)

    def registers(self):
        p = self.p
        N = p.gear_ratio
        rng = self.rng
        tau_m = self.tau_cur / N
        values = {
            int(_R.MODE): self.mode,
            int(_R.POSITION): self.motor_rev() + rng.normal(0.0, p.position_noise_rev),
            int(_R.VELOCITY): self.vel_est * N / (2.0 * math.pi) + rng.normal(0.0, p.velocity_noise_rev_s),
            int(_R.TORQUE): tau_m + rng.normal(0.0, p.torque_noise),
            int(_R.Q_CURRENT): tau_m / p.motor_torque_constant,
            int(_R.VOLTAGE): p.bus_voltage,
            int(_R.TEMPERATURE): self.temp_board,
            int(_R.MOTOR_TEMPERATURE): self.temp_motor,
            int(_R.FAULT): self.fault,
            int(_R.CONTROL_POSITION): self.control_pos_rev,
            int(_R.CONTROL_VELOCITY): self.control_vel_rev_s,
            int(_R.CONTROL_TORQUE): self.control_torque_motor,
        }
        if self.output_encoder:
            reg = int(getattr(_R, self.output_encoder['register']))
            scale = self.output_encoder.get('revs_per_joint_rev', 1.0) * self.output_encoder.get('sign', 1)
            values[reg] = self.tho / (2.0 * math.pi) * scale + rng.normal(0.0, p.output_encoder_noise_rev)
        if self.analog_sensor:
            reg = int(getattr(_R, self.analog_sensor['register']))
            values[reg] = (self.analog_sensor.get('offset_units', 0.0)
                           + self.external_torque() / self.analog_sensor['nm_per_unit'])
        return values

    def truth(self):
        """Ground-truth parameters for metadata and tests."""
        return {**asdict(self.p), **self.p.derived()}


class _Result:
    def __init__(self, values):
        self.values = values
        self.id = 1

    def __repr__(self):
        return f'SimResult({self.values})'


class SimClock:
    """Virtual clock: sleeping integrates the plant instead of waiting."""

    def __init__(self, plant: SimPlant):
        self.plant = plant
        self.t = 0.0

    def now(self):
        return self.t

    async def sleep_until(self, t):
        if t > self.t:
            self.plant.advance_to(t)
            self.t = t

    async def sleep(self, dt):
        await self.sleep_until(self.t + dt)


class SimController:
    """Drop-in for moteus.Controller in these experiments."""

    def __init__(self, plant: SimPlant, clock: SimClock):
        self.plant = plant
        self.clock = clock
        self.rng = np.random.default_rng(plant.p.seed + 1)

    def _exchange(self, apply):
        p = self.plant.p
        rt = max(5e-5, p.round_trip_s + self.rng.normal(0.0, p.round_trip_jitter_s))
        arrive = self.clock.t + rt / 2.0
        self.plant.advance_to(arrive)
        apply()
        values = self.plant.registers()
        self.clock.t = max(self.clock.t + rt, self.plant.t)
        return _Result(values)

    async def set_position(self, *, query=False, **kwargs):
        result = self._exchange(lambda: self.plant.apply_position_command(kwargs))
        return result if query else None

    async def set_stop(self, *, query=False, **_):
        result = self._exchange(lambda: self.plant.set_mode('STOPPED'))
        return result if query else None

    async def set_stay_within(self, *, query=False, **kwargs):
        result = self._exchange(lambda: self.plant.apply_stay_within_command(kwargs))
        return result if query else None

    async def set_brake(self, *, query=False, **_):
        result = self._exchange(lambda: self.plant.set_mode('BRAKE'))
        return result if query else None

    async def query(self, **_):
        return self._exchange(lambda: None)


class SimStream:
    """Diagnostic channel: supports conf get/set/write and tel stop."""

    def __init__(self, controller: SimController):
        self.plant = controller.plant

    async def write_message(self, data):
        return None

    async def flush_read(self, timeout=0.2):
        return None

    async def command(self, data, allow_any_response=False):
        text = data.decode('latin1').strip() if isinstance(data, bytes) else str(data).strip()
        parts = text.split()
        if parts[:2] == ['conf', 'get'] and len(parts) == 3:
            if parts[2] not in self.plant.config:
                if allow_any_response:
                    return b'ERR error reading'
                raise CommandError(f'ERR unknown config {parts[2]}')
            return self.plant.config[parts[2]].encode('latin1')
        if parts[:2] == ['conf', 'set'] and len(parts) == 4:
            if parts[2] not in self.plant.config:
                raise CommandError(f'ERR unknown config {parts[2]}')
            float(parts[3])  # Rejects malformed numbers like the firmware.
            self.plant.config[parts[2]] = parts[3]
            return b''
        if parts in (['conf', 'write'], ['tel', 'stop']):
            return b''
        raise CommandError(f'ERR unknown command {text}')


def make_sim(params: Optional[SimParams] = None, output_encoder=None, analog_sensor=None):
    """Return (controller, clock, stream_factory, plant) for a simulated run."""
    plant = SimPlant(params or SimParams(), output_encoder=output_encoder,
                     analog_sensor=analog_sensor)
    clock = SimClock(plant)
    controller = SimController(plant, clock)
    return controller, clock, (lambda c: SimStream(c)), plant
