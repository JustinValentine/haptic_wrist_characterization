"""Simulated P4 wrist: emulated pucks on a virtual CAN bus, driving a 3-joint
cable-differential plant. The real p4wrist library runs against it unchanged
(through a callback bus), so --sim exercises the same setup, PDO layout, state
machine, supervision and control law as the hardware.

Pucks: SimP4 extends barrett-puck's MockP4 (vendor/barrett_puck) with what the
library depends on and MockP4 leaves out, following the P4 CANopen manual
(Rev BA) and firmware 4.3.2 behaviour recorded in libbarrett-canopen's notes:
  * physics: 0x6064 counts the plant's motor angle, restarting at 0 on reset;
    0x3012:01 is an absolute single-turn reading; q-axis current is reported
  * CST: the torque target latched at a SYNC is reached by ramping over the
    next SYNC period; IDLE while enabled grounds the phases (a brake)
  * reset: NMT reset restores power-on objects (the saved heartbeat period
    included), sends one last heartbeat, then the boot-up
  * heartbeats on the 0x1017 period, SYNC-loss faults (0x2406), and the
    self-disable after ~55 SYNCs without a controlword
  * RPDO type 255 applies on receipt; TPDO2 goes out every 100 ms whatever its
    type; invalid PDOs stay silent; modes are accepted only while Operation
    Enabled; PDO mapping refuses objects the drive cannot map
Simulated time moves one SYNC period per SYNC frame (and with silence()).

Plant, per joint j and motor i (motor angles m = j2mp q, as in p4wrist):
    rotor     Jr_i m_i'' = tau_i - b_i m_i' - friction_i(m_i') - cogging_i(m_i) - (j2mp^-T tau_t)_i
    output    Jo_j q_j'' = tau_t,j - bo_j q_j' - gravity_j(q_j) + tau_human,j
    cable     tau_t,j    = k_j (qt_j - q_j) + c_j (qt_j' - q_j'),  qt = j2mp^-1 m
with Karnopp stick-slip friction per motor, a first-order current loop, a
back-EMF torque-speed line, and first-order winding and amplifier thermal
models. Motor parameters are placeholders of the right order for the wrist's
ILM50 motors, not datasheet values.
"""
import collections
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from vendor.barrett_puck.mock import MockP4
from vendor.barrett_puck.objects import OD, NmtState

TWO_PI = 2.0 * math.pi
N1 = 7.5077     # differential (FE, UR)
N2 = 8.7456     # PS roll
DEFAULT_J2MP = ((-N2, 0.0, 0.0), (0.0, N1, N1), (0.0, -N1, N1))
JOINTS = ('PS', 'FE', 'UR')
CURRENT_TORQUE_FACTOR = 1.0 / math.sqrt(2.0)   # same convention as p4wrist's default

DRIVE_OFF, DRIVE_TORQUE, DRIVE_BRAKE = 0, 1, 2


def _three(x):
    return [float(v) for v in x] if hasattr(x, '__len__') else [float(x)] * 3


@dataclass
class SimParams:
    """Ground truth. Motor quantities are motor-side; joint quantities per joint."""
    j2mp: tuple = DEFAULT_J2MP
    rotor_inertia: object = 3.0e-5          # kg m^2 per motor (ILM50-class rotor + capstan)
    motor_viscous: object = 1.5e-4          # N m s/rad
    motor_coulomb: object = 0.0057          # N m kinetic
    motor_static: object = 0.008            # N m breakaway
    motor_stribeck_velocity: object = 0.44  # rad/s
    cogging_amplitude_motor: object = 0.002  # N m
    cogging_periods_per_motor_rev: float = 60.0   # assumes 20 poles, 12 slots (unverified)
    cogging_phase: object = (0.0, 1.1, 2.3)  # rad of cogging phase, per motor
    output_inertia: object = (0.003, 0.002, 0.002)   # kg m^2 (handle, links)
    transmission_stiffness: object = 1500.0  # N m/rad per joint
    transmission_damping: object = 0.3
    backlash_deg: object = 0.0
    output_viscous: object = 0.002
    gravity_torque: object = 0.0            # m g l per joint, acts as -m g l sin(q)
    current_loop_time_constant: float = 2e-4
    torque_constant_ratio: float = 1.0      # physical / drive-assumed torque
    back_emf_constant: float = 0.058        # V s/rad motor (ILM50-class)
    winding_resistance: float = 0.5         # Ohm, phase
    bus_voltage: float = 48.0
    brake_damping_motor: Optional[float] = None   # phases grounded; default Ke Kt / R
    thermal_resistance: float = 3.0         # C/W winding to ambient
    thermal_time_constant: float = 60.0
    board_thermal_resistance: float = 1.0
    board_time_constant: float = 30.0
    ambient_temperature: float = 25.0
    external_sensor_noise: float = 0.01     # N m
    initial_q: object = (0.0, 0.0, 0.0)     # joint angles at start (rad)
    initial_motor_offset: object = (0.37, -1.9, 2.6)  # rad added to j2mp q (where counters land)
    rated_torque_mnm: object = 250.0        # 0x6076
    kt_mnm_per_a: object = 58.0             # 0x3011:04
    peak_current_ma: object = 23471.0       # 0x3011:09 (the WAM wrist's setting)
    counts_per_rev: int = 4096
    single_turn_offset: object = (812, 3010, 145)   # counts at motor angle 0
    single_turn_dir: object = (1, -1, 1)
    drive_dir: object = (1, 1, 1)
    current_mappable: bool = True
    motor_temp_mappable: bool = True
    substeps: int = 20
    seed: int = 0

    def arr(self, name):
        return _three(getattr(self, name))

    def kt_si(self):
        """Motor N m per A of reported q current."""
        return [k / 1000.0 * CURRENT_TORQUE_FACTOR for k in self.arr('kt_mnm_per_a')]

    def brake_damping(self):
        if self.brake_damping_motor is not None:
            return self.brake_damping_motor
        return self.back_emf_constant * self.kt_si()[0] / self.winding_resistance


@dataclass
class HumanModel:
    """Operator who drags one joint along a reference through a spring-damper.

    motion(t) gives (theta, omega) in rad relative to plant.human_origin; the
    joint defaults to plant.test_joint (both set by the experiment session)."""
    motion: Callable[[float], tuple]
    stiffness: float = 40.0
    damping: float = 0.8
    joint: Optional[int] = None


class WristPlant:
    """Three motors, three cable-coupled outputs, integrated at period/substeps."""

    def __init__(self, params: SimParams):
        p = self.p = params
        self.rng = np.random.default_rng(p.seed)
        self.T = [list(row) for row in p.j2mp]
        self.Ti = _inv3(self.T)                 # qt = Ti m
        self.TiT = _transpose(self.Ti)          # motor reaction = Ti^T tau_t
        q0 = _three(p.initial_q)
        off = _three(p.initial_motor_offset)
        m0 = _mul(self.T, q0)
        self.m = [m0[i] + off[i] for i in range(3)]
        self.qt_offset = _mul(self.Ti, off)     # so qt = q at rest
        self.wm = [0.0] * 3
        self.q = list(q0)
        self.wq = [0.0] * 3
        self.stuck = [True] * 3
        self.tau_act = [0.0] * 3               # after the current loop (motor, plant frame)
        self.tau_t = [0.0] * 3
        self.tau_h = [0.0] * 3
        self.t = 0.0
        self.temp_motor = [p.ambient_temperature] * 3
        self.temp_board = [p.ambient_temperature] * 3
        self.locked = [None] * 3               # locked output angle per joint
        self.human: Optional[HumanModel] = None
        self.test_joint = 0                    # joint under test (set by the session)
        self.human_origin = 0.0                # rad, where the human's motion is centred

    # --- operator interaction ----------------------------------------------
    def lock_output(self, joint=None, angle=None):
        joint = self.test_joint if joint is None else joint
        self.locked[joint] = self.q[joint] if angle is None else angle
        self.q[joint] = self.locked[joint]
        self.wq[joint] = 0.0

    def unlock_output(self, joint=None):
        self.locked[self.test_joint if joint is None else joint] = None

    def external_torque(self, joint=None):
        """What an external sensor at the handle (or the lock) reads, N m."""
        joint = self.test_joint if joint is None else joint
        value = self.tau_t[joint] if self.locked[joint] is not None else self.tau_h[joint]
        return value + self.rng.normal(0.0, self.p.external_sensor_noise)

    def transmission_angles(self):
        qt = _mul(self.Ti, self.m)
        return [qt[j] - self.qt_offset[j] for j in range(3)]

    # --- integration -----------------------------------------------------------
    def advance(self, dt, tau_start, tau_end, modes):
        p = self.p
        if dt <= 0:
            return list(self.m)
        n = max(1, int(p.substeps))
        h = dt / n
        Jr, b = p.arr('rotor_inertia'), p.arr('motor_viscous')
        fc, fs, vs = p.arr('motor_coulomb'), p.arr('motor_static'), p.arr('motor_stribeck_velocity')
        cog_a, cog_ph = p.arr('cogging_amplitude_motor'), p.arr('cogging_phase')
        cog_w = p.cogging_periods_per_motor_rev
        Jo, k, c = p.arr('output_inertia'), p.arr('transmission_stiffness'), p.arr('transmission_damping')
        half_bl = [math.radians(x) / 2.0 for x in p.arr('backlash_deg')]
        bo, mgl = p.arr('output_viscous'), p.arr('gravity_torque')
        alpha = 1.0 - math.exp(-h / p.current_loop_time_constant)
        kt_ratio = p.torque_constant_ratio
        ke, R, V = p.back_emf_constant, p.winding_resistance, p.bus_voltage
        kt = p.kt_si()
        b_brake = p.brake_damping()
        Ti, TiT, off = self.Ti, self.TiT, self.qt_offset
        m, wm, q, wq = self.m, self.wm, self.q, self.wq
        stuck, tau_act = self.stuck, self.tau_act
        human = self.human
        human_joint = (self.test_joint if human is None or human.joint is None else human.joint)
        origin = self.human_origin
        tau_t = self.tau_t
        tau_h = [0.0, 0.0, 0.0]
        sum_i2 = [0.0, 0.0, 0.0]
        t = self.t
        for s in range(n):
            t += h
            frac = (s + 1) / n
            # Cable torques from the transmission-side joint angles.
            qt = [Ti[j][0] * m[0] + Ti[j][1] * m[1] + Ti[j][2] * m[2] - off[j] for j in range(3)]
            wqt = [Ti[j][0] * wm[0] + Ti[j][1] * wm[1] + Ti[j][2] * wm[2] for j in range(3)]
            for j in range(3):
                d = qt[j] - q[j]
                if half_bl[j] > 0.0:
                    d = d - half_bl[j] if d > half_bl[j] else (d + half_bl[j] if d < -half_bl[j] else 0.0)
                    tau_t[j] = k[j] * d + (c[j] * (wqt[j] - wq[j]) if d != 0.0 else 0.0)
                else:
                    tau_t[j] = k[j] * d + c[j] * (wqt[j] - wq[j])
            react = [TiT[i][0] * tau_t[0] + TiT[i][1] * tau_t[1] + TiT[i][2] * tau_t[2] for i in range(3)]
            for i in range(3):
                if modes[i] == DRIVE_TORQUE:
                    cmd = tau_start[i] + (tau_end[i] - tau_start[i]) * frac
                else:
                    cmd = 0.0
                tau_act[i] += alpha * (cmd - tau_act[i])
                tau = tau_act[i]
                if modes[i] == DRIVE_TORQUE and tau * wm[i] > 0.0:
                    # Back-EMF: the supply runs out of voltage as speed rises.
                    stall = V / R * kt[i] if R > 0 else math.inf
                    avail = max(0.0, stall * (1.0 - abs(wm[i]) * ke / V))
                    if abs(tau) > avail:
                        tau = math.copysign(avail, tau)
                if modes[i] != DRIVE_TORQUE:
                    tau_act[i] = 0.0
                    tau = -b_brake * wm[i] if modes[i] == DRIVE_BRAKE else 0.0
                else:
                    sum_i2[i] += (tau / kt[i]) ** 2
                    tau *= kt_ratio
                cog = cog_a[i] * math.sin(cog_w * m[i] + cog_ph[i]) if cog_a[i] else 0.0
                ext = tau - b[i] * wm[i] - cog - react[i]
                w = wm[i]
                if stuck[i] or w == 0.0:
                    if abs(ext) <= fs[i]:
                        w_new = 0.0
                        stuck[i] = True
                    else:
                        stuck[i] = False
                        w_new = w + h * (ext - math.copysign(fs[i], ext)) / Jr[i]
                else:
                    fr = fc[i] + (fs[i] - fc[i]) * math.exp(-(w / vs[i]) ** 2)
                    w_new = w + h * (ext - math.copysign(fr, w)) / Jr[i]
                    if w_new * w < 0.0:
                        w_new = 0.0
                        stuck[i] = True
                m[i] += h * w_new
                wm[i] = w_new
            for j in range(3):
                if self.locked[j] is not None:
                    q[j], wq[j] = self.locked[j], 0.0
                    continue
                ext_o = tau_t[j] - bo[j] * wq[j]
                if mgl[j]:
                    ext_o -= mgl[j] * math.sin(q[j])
                if human is not None and human_joint == j:
                    th, om = human.motion(t)
                    tau_h[j] = human.stiffness * (origin + th - q[j]) + human.damping * (om - wq[j])
                    ext_o += tau_h[j]
                wq[j] += h * ext_o / Jo[j]
                q[j] += h * wq[j]
        self.t = t
        self.tau_h = tau_h
        # Thermal models, with the mean copper loss over this interval.
        for i in range(3):
            power = R * sum_i2[i] / n
            ss = p.ambient_temperature + power * p.thermal_resistance
            self.temp_motor[i] += (ss - self.temp_motor[i]) * (1.0 - math.exp(-dt / p.thermal_time_constant))
            bs = p.ambient_temperature + power * p.board_thermal_resistance
            self.temp_board[i] += (bs - self.temp_board[i]) * (1.0 - math.exp(-dt / p.board_time_constant))
        return list(m)

    # --- ground truth -----------------------------------------------------------
    def truth(self, joint, others_held=True):
        """Joint-referred parameters seen when only `joint` moves (the other
        joints held), in the names the analyses expect."""
        p = self.p
        T = self.T
        col = [T[i][joint] for i in range(3)]
        movers = [i for i in range(3) if col[i] != 0.0]
        n_eff = max(abs(col[i]) for i in movers)
        Jr, b = p.arr('rotor_inertia'), p.arr('motor_viscous')
        fc, fs = p.arr('motor_coulomb'), p.arr('motor_static')
        vs = p.arr('motor_stribeck_velocity')
        jt = _transpose(self.Ti)         # tau_m = Ti^T tau_q  (= j2mp^-T)
        primary = max(movers, key=lambda i: abs(col[i]))
        kt = p.kt_si()[primary]
        tau_per_amp = kt / abs(jt[primary][joint])
        # Cogging seen at this joint: sum of the moving motors' ripples, which
        # share one spatial period; their phases depend on the held joints.
        cog = 0j
        for i in movers:
            phase = p.cogging_periods_per_motor_rev * (self.m[i] - col[i] * self.q[joint]) + p.arr('cogging_phase')[i]
            cog += abs(col[i]) * p.arr('cogging_amplitude_motor')[i] * np.exp(1j * np.sign(col[i]) * phase)
        stall_motor = p.bus_voltage / p.winding_resistance * kt
        no_load_motor = p.bus_voltage / p.back_emf_constant
        out = {
            'joint': JOINTS[joint],
            'gear_ratio': n_eff,
            'motor_inertia': sum(col[i] ** 2 * Jr[i] for i in movers),
            'output_inertia': p.arr('output_inertia')[joint],
            'transmission_stiffness': p.arr('transmission_stiffness')[joint],
            'transmission_damping': p.arr('transmission_damping')[joint],
            'backlash_deg': p.arr('backlash_deg')[joint],
            'viscous': sum(col[i] ** 2 * b[i] for i in movers),
            'coulomb': sum(abs(col[i]) * fc[i] for i in movers),
            'static': sum(abs(col[i]) * fs[i] for i in movers),
            'stribeck_velocity': vs[primary] / abs(col[primary]),
            'output_viscous': p.arr('output_viscous')[joint],
            'gravity_torque': p.arr('gravity_torque')[joint],
            'cogging_amplitude': float(abs(cog)),
            'cogging_period_deg': 360.0 / (p.cogging_periods_per_motor_rev * n_eff),
            'cogging_periods_per_motor_rev': p.cogging_periods_per_motor_rev,
            'current_loop_time_constant': p.current_loop_time_constant,
            'torque_constant_ratio': p.torque_constant_ratio,
            # gear_ratio x motor_torque_constant = joint torque per amp of the
            # primary motor's reported current
            'motor_torque_constant': tau_per_amp / n_eff,
            'winding_resistance': p.winding_resistance,
            'thermal_resistance': p.thermal_resistance,
            'thermal_time_constant': p.thermal_time_constant,
            'stall_torque': min(stall_motor, self._drive_limit(primary)) / abs(jt[primary][joint]),
            'no_load_speed': no_load_motor / abs(col[primary]),
            'max_torque': self._drive_limit(primary) / abs(jt[primary][joint]),
            'brake_damping': sum(col[i] ** 2 for i in movers) * p.brake_damping(),
            'primary_motor': primary,
        }
        return out

    def _drive_limit(self, motor):
        return self.p.arr('peak_current_ma')[motor] / 1000.0 * self.p.kt_si()[motor]


class SimP4(MockP4):
    """MockP4 plus physics and the firmware behaviour the library relies on."""

    MAPPABLE_RX = {0x6040, 0x6060, 0x6071, 0x60FF, 0x607A}
    MAPPABLE_TX = {0x6041, 0x6061, 0x6064, 0x606C, 0x6078, (0x3000, 1), (0x3000, 2), (0x3010, 3)}

    def __init__(self, node, motor, params: SimParams, bus):
        self.node_id = node
        self.motor = motor
        self.params = params
        self.bus = bus
        self.dropped = False
        self.angle = 0.0                    # drive-frame motor angle (rad)
        self._init_od()

    def _init_od(self):
        p, i = self.params, self.motor
        MockP4.__init__(self, self.node_id,
                        counts_per_rev=p.counts_per_rev,
                        rated_torque_mnm=int(p.arr('rated_torque_mnm')[i]),
                        peak_current_ma=int(p.arr('peak_current_ma')[i]))
        od = self.od
        od[OD.HEARTBEAT_PERIOD.addr] = 2000     # the saved value a reset restores
        od[(0x3011, 4)] = int(p.arr('kt_mnm_per_a')[i])
        od[(0x2406, 0)] = 500
        od[(0x1006, 0)] = 0
        od[(0x3010, 3)] = int(round(p.ambient_temperature))
        od[OD.BUS_VOLTAGE.addr] = int(round(p.bus_voltage * 10))
        od[OD.TEMPERATURE.addr] = int(round(p.ambient_temperature))
        for slot in range(4):   # factory transmission types, none invalid
            od[(0x1800 + slot, 2)] = od.get((0x1800 + slot, 2), 0)
        self.counter_origin = self.angle
        self.ramp = (0.0, 0.0)
        self.ramp_mode = DRIVE_OFF
        self.syncs_since_cw = 0
        self.last_sync = None
        self.next_heartbeat = self.bus.t
        self._in_pdo = False

    # --- sensors -------------------------------------------------------------
    def _refresh_sensors(self):
        p = self.params
        cpr = p.counts_per_rev
        od = self.od
        od[OD.ACTUAL_POSITION.addr] = int(math.floor((self.angle - self.counter_origin) * cpr / TWO_PI + 0.5))
        st = p.arr('single_turn_offset')[self.motor] + _three(p.single_turn_dir)[self.motor] * self.angle * cpr / TWO_PI
        od[OD.RAW_ENCODER.addr] = int(math.floor(st + 0.5)) % cpr
        applied = self.ramp[1]   # torque now: where the ramp in progress (or just finished) ends
        kt = p.kt_si()[self.motor]
        peak = p.arr('peak_current_ma')[self.motor] / 1000.0
        od[OD.ACTUAL_CURRENT.addr] = int(round(applied / kt / peak * 1000.0)) if self.ramp_mode == DRIVE_TORQUE else 0
        rated = p.arr('rated_torque_mnm')[self.motor] / 1000.0
        od[OD.TORQUE_DEMAND.addr] = int(round(applied / rated * 1000.0)) if self.ramp_mode == DRIVE_TORQUE else 0
        plant = self.bus.plant
        od[(0x3010, 3)] = int(round(plant.temp_motor[self.motor]))
        od[OD.TEMPERATURE.addr] = int(round(plant.temp_board[self.motor]))

    # --- frames ----------------------------------------------------------------
    def handle(self, can_id, data):
        if self.dropped:
            return
        if can_id == 0x000 and len(data) == 2 and data[1] in (0, self.id):
            self._nmt(data[0])
            return
        if self.nmt_state == NmtState.OPERATIONAL and can_id != 0x080:
            # RPDOs with transmission type 254/255 apply on receipt.
            for slot in range(1, 5):
                cob = self.od[(0x1400 + slot - 1, 1)]
                if can_id == cob and self.od[(0x1600 + slot - 1, 0)] > 0 \
                        and self.od[(0x1400 + slot - 1, 2)] >= 254:
                    self._apply_rpdo(slot, bytes(data))
                    return
        if can_id == 0x600 + self.id:
            self._refresh_sensors()
        super().handle(can_id, data)

    def _apply_rpdo(self, slot, payload):
        self._in_pdo = True
        try:
            for addr, size, signed in self._mapping(0x1600 + slot - 1):
                value = int.from_bytes(payload[:size], 'little', signed=signed)
                payload = payload[size:]
                if addr[0] < 0x1000:
                    continue   # dummy entry
                self._write(addr, value)
        finally:
            self._in_pdo = False

    def _nmt(self, command):
        if command == 0x01:
            self.nmt_state = NmtState.OPERATIONAL
        elif command == 0x02:
            self.nmt_state = NmtState.STOPPED
        elif command == 0x80:
            self.nmt_state = NmtState.PRE_OPERATIONAL
        elif command in (0x81, 0x82):
            # A P4 sends one last heartbeat in its old state, then boots.
            last = (0x700 + self.id, bytes([int(self.nmt_state)]))
            angle = self.angle
            self._init_od()
            self.angle = angle
            self.counter_origin = angle
            self.outbox.append(last)
            self.outbox.append((0x700 + self.id, bytes([NmtState.BOOT_UP])))

    def _write(self, addr, value):
        if addr == OD.MODE_OF_OPERATION.addr and not self.operation_enabled:
            # fw 4.3.2: mode writes only land while Operation Enabled.
            return None if self._in_pdo else 0x08000022
        if addr[0] in (0x1600, 0x1601, 0x1602, 0x1603, 0x1A00, 0x1A01, 0x1A02, 0x1A03) and addr[1] >= 1:
            index, sub = (value >> 16) & 0xFFFF, (value >> 8) & 0xFF
            tx = addr[0] >= 0x1A00
            allowed = self.MAPPABLE_TX if tx else self.MAPPABLE_RX
            dummy = not tx and index < 0x1000
            ok = dummy or index in allowed or (index, sub) in allowed
            if index == 0x6078 and not self.params.current_mappable:
                ok = False
            if (index, sub) == (0x3010, 3) and not self.params.motor_temp_mappable:
                ok = False
            if not ok:
                return 0x06040041   # object cannot be mapped to the PDO
        if addr == OD.CONTROLWORD.addr:
            self.syncs_since_cw = 0
        return super()._write(addr, value)

    def _apply_targets(self):
        pass   # physics, not MockP4's instant feedback

    def _handle_sync(self):
        """SYNC while operational: latch RPDOs, start the next CST ramp, send TPDOs."""
        self._sync_count += 1
        self._in_pdo = True
        try:
            for slot, payload in sorted(self._rpdo_buffer.items()):
                for addr, size, signed in self._mapping(0x1600 + slot - 1):
                    value = int.from_bytes(payload[:size], 'little', signed=signed)
                    payload = payload[size:]
                    if addr[0] >= 0x1000:
                        self._write(addr, value)
        finally:
            self._in_pdo = False
        self._rpdo_buffer.clear()
        self.syncs_since_cw += 1
        if self.operation_enabled and self.syncs_since_cw > 55:
            self.ds402 = 'SWITCH_ON_DISABLED'   # fw 4.3.2 self-disable
        self._refresh_sensors()
        self._next_ramp()
        for slot in range(1, 5):
            count = self.od[(0x1A00 + slot - 1, 0)]
            cob = self.od[(0x1800 + slot - 1, 1)]
            if count <= 0 or cob & 0x80000000:
                continue
            tx_type = self.od[(0x1800 + slot - 1, 2)]
            if slot == 2:
                due = self._sync_count % 50 == 0   # every 100 ms whatever the type (bench)
            elif tx_type == 0:
                due = True
            elif 1 <= tx_type <= 240:
                due = self._sync_count % tx_type == 0
            else:
                due = False
            if not due:
                continue
            payload = b''
            for addr, size, signed in self._mapping(0x1A00 + slot - 1):
                value = self.statusword() if addr == OD.STATUSWORD.addr else self.od[addr]
                payload += (int(value) & (2 ** (size * 8) - 1)).to_bytes(size, 'little')
            self.outbox.append((cob, payload))

    def _next_ramp(self):
        p, i = self.params, self.motor
        mode_display = self.od[OD.MODE_DISPLAY.addr]
        if self.ds402 == 'FAULT':
            mode, target = DRIVE_BRAKE, 0.0
        elif self.operation_enabled and mode_display == 10:
            rated = p.arr('rated_torque_mnm')[i] / 1000.0
            target = self.od[OD.TARGET_TORQUE.addr] / 1000.0 * rated
            limit = p.arr('peak_current_ma')[i] / 1000.0 * p.kt_si()[i]
            target = max(-limit, min(limit, target))
            mode = DRIVE_TORQUE
        elif self.operation_enabled and mode_display == 0:
            mode, target = DRIVE_BRAKE, 0.0
        else:
            mode, target = DRIVE_OFF, 0.0
        start = self.ramp[1] if (mode == DRIVE_TORQUE and self.ramp_mode == DRIVE_TORQUE) else 0.0
        self.ramp = (start, target)
        self.ramp_mode = mode

    def drive_input(self):
        """(start, end, mode) for the plant over the coming period, plant frame."""
        d = _three(self.params.drive_dir)[self.motor]
        if self.ramp_mode != DRIVE_TORQUE:
            return 0.0, 0.0, self.ramp_mode
        return d * self.ramp[0], d * self.ramp[1], DRIVE_TORQUE

    def hold_torque(self):
        """When SYNC stops, the drive keeps its last target."""
        self.ramp = (self.ramp[1], self.ramp[1])

    def sync_lost(self):
        self.trigger_fault(0x8180)
        self.ramp, self.ramp_mode = (0.0, 0.0), DRIVE_BRAKE


class SimBus:
    """The virtual CAN bus: frames from the library go to every puck at once."""

    def __init__(self, plant: WristPlant, nodes=(6, 7, 8)):
        self.plant = plant
        self.nodes = list(nodes)
        self.t = 0.0
        self.rx = collections.deque()
        self.pucks = []
        d = _three(plant.p.drive_dir)
        for i in range(3):
            puck = SimP4(self.nodes[i], i, plant.p, self)
            puck.angle = d[i] * plant.m[i]
            puck.counter_origin = puck.angle
            self.pucks.append(puck)
        self.sent = collections.Counter()
        self.sync_hooks = []        # called after the plant advances, once per SYNC

    def period(self):
        us = self.pucks[0].od[(0x1006, 0)]
        return us * 1e-6 if us > 0 else 0.002

    def send(self, can_id, data):
        self.sent[can_id] += 1
        if can_id == 0x080 and not data:
            self._sync()
            return
        for puck in self.pucks:
            puck.handle(can_id, data)
            self._collect(puck)

    def recv(self):
        return self.rx.popleft() if self.rx else None

    def _collect(self, puck):
        while puck.outbox:
            self.rx.append(puck.outbox.pop(0))

    def _advance(self, dt):
        starts, ends, modes = [], [], []
        for puck in self.pucks:
            s, e, m = puck.drive_input()
            starts.append(s)
            ends.append(e)
            modes.append(m)
        angles = self.plant.advance(dt, starts, ends, modes)
        d = _three(self.plant.p.drive_dir)
        for i, puck in enumerate(self.pucks):
            puck.angle = d[i] * angles[i]
        self.t += dt

    def _sync(self):
        self._advance(self.period())
        for hook in self.sync_hooks:
            hook()
        for puck in self.pucks:
            if puck.dropped:
                continue
            puck.last_sync = self.t
            puck.handle(0x080, b'')
            self._collect(puck)
        self._heartbeats()

    def _heartbeats(self):
        for puck in self.pucks:
            hb = puck.od[OD.HEARTBEAT_PERIOD.addr]
            if puck.dropped or hb <= 0:
                continue
            if self.t >= puck.next_heartbeat:
                puck.outbox.append((0x700 + puck.id, bytes([int(puck.nmt_state)])))
                puck.next_heartbeat = self.t + hb / 1000.0
            self._collect(puck)

    def silence(self, seconds, step=0.001):
        """Time passes without SYNC: drives hold their last torque until their
        SYNC-loss timeout (0x2406) faults them."""
        for puck in self.pucks:
            puck.hold_torque()
        end = self.t + seconds
        while self.t < end - 1e-12:
            self._advance(min(step, end - self.t))
            for puck in self.pucks:
                timeout = puck.od[(0x2406, 0)] / 1000.0
                if (puck.last_sync is not None and timeout > 0 and puck.ds402 != 'FAULT'
                        and self.t - puck.last_sync > timeout):
                    puck.sync_lost()
                    if not puck.dropped:
                        self._collect(puck)
            self._heartbeats()

    # Test hooks.
    def drop(self, motor, dropped=True):
        self.pucks[motor].dropped = dropped

    def fault(self, motor, code):
        self.pucks[motor].trigger_fault(code)
        self._collect(self.pucks[motor])

    def reset(self, motor):
        self.pucks[motor]._nmt(0x81)
        self._collect(self.pucks[motor])


def make_sim(params: Optional[SimParams] = None, nodes=(6, 7, 8)):
    """(bus, plant) for p4wrist.Wrist.callback(bus, config)."""
    plant = WristPlant(params or SimParams())
    return SimBus(plant, nodes), plant


# --- 3x3 helpers -----------------------------------------------------------------------
def _mul(A, x):
    return [A[i][0] * x[0] + A[i][1] * x[1] + A[i][2] * x[2] for i in range(3)]


def _transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def _inv3(A):
    return np.linalg.inv(np.array(A, dtype=float)).tolist()
