"""The p4wrist library against emulated P4 pucks (p4_sim.py).

These run the real C++ library: CANopen setup, the SYNC loop, the
differential transmission, supervision and the control law. Only the bus
and the plant are simulated.
"""
import math
import unittest

import numpy as np

import p4_sim
from wrist_backend import p4wrist

KP = (20.0, 20.0, 20.0)
KD = (0.3, 0.3, 0.3)


def frictionless(**overrides):
    params = p4_sim.SimParams(motor_coulomb=0.0, motor_static=0.0, cogging_amplitude_motor=0.0,
                              **overrides)
    return params


class Rig:
    """A simulated wrist with the library connected, zeroed at home (0, 0, 0)."""

    def __init__(self, params=None, config=None, connect=True, zero=True):
        self.bus, self.plant = p4_sim.make_sim(params or p4_sim.SimParams())
        if config is None:
            # Steps at these gains run faster than the default 360 deg/s limit.
            config = p4wrist.default_config()
            config.max_joint_velocity = p4wrist.D3(20.0, 20.0, 20.0)
        self.config = config
        self.wrist = p4wrist.Wrist.callback(self.bus, self.config)
        self.seq = 0
        if connect:
            self.kept, self.report = self.wrist.connect(None)
        if zero:
            self.wrist.zero(p4wrist.zero_cal((0.0, 0.0, 0.0)), exact=True)

    def run(self, command, seconds):
        cycles = int(round(seconds / self.config.period_s))
        for _ in range(cycles):
            self.seq += 1
            command.seq = self.seq
            self.wrist.set_command(command)
            self.wrist.step(1)
        return self.wrist.sample()

    def active(self):
        self.wrist.start()
        self.wrist.step(5)
        self.wrist.enable(2.0)
        return self

    def close(self):
        self.wrist.close()


def torque_command(tau):
    cmd = p4wrist.Command()
    for j in range(3):
        cmd.joint[j] = p4wrist.joint_command(torque=tau[j])
    return cmd


class SetupTest(unittest.TestCase):
    def test_cold_connect_maps_the_wam_bus_layout(self):
        rig = Rig()
        self.assertFalse(rig.kept)
        for i, puck in enumerate(rig.bus.pucks):
            od = puck.od
            self.assertEqual(od[(0x1400, 1)], 0x205)                    # torque quad B
            self.assertEqual(od[(0x1600, i + 2)], 0x60710010)           # slot motor+1
            self.assertEqual(od[(0x1601, 1)], 0x60400010)
            self.assertEqual(od[(0x1401, 1)], 0x28A)                    # controlword broadcast
            self.assertEqual(od[(0x1401, 2)], 255)
            self.assertEqual(od[(0x1A00, 1)], 0x60640020)               # position ...
            self.assertEqual(od[(0x1A00, 2)], 0x60780010)               # ... and q current
            self.assertEqual(od[(0x1800, 1)], 0x180 + puck.id)
            self.assertEqual(od[(0x1017, 0)], 250)
            self.assertEqual(od[(0x1006, 0)], 2000)
            self.assertTrue(od[(0x1402, 1)] & 0x80000000)               # unused PDOs off
            self.assertTrue(od[(0x1803, 1)] & 0x80000000)
        info = rig.wrist.drive_info()
        self.assertEqual(info[1]['counts_per_rev'], 4096)
        self.assertEqual(info[2]['rated_torque_mnm'], 250.0)
        rig.close()

    def test_unmappable_current_falls_back_to_position_only(self):
        rig = Rig(p4_sim.SimParams(current_mappable=False)).active()
        s = rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.2)
        self.assertEqual(rig.bus.pucks[0].od[(0x1A00, 0)], 1)
        self.assertTrue(all(math.isnan(x) for x in s.tau_meas))
        self.assertEqual(s.wrist_state, p4wrist.STATE_ACTIVE)
        rig.close()

    def test_enable_needs_a_zero(self):
        rig = Rig(zero=False)
        rig.wrist.start()
        with self.assertRaisesRegex(p4wrist.P4WristError, 'zero'):
            rig.wrist.enable(1.0)
        rig.close()

    def test_keeps_zero_across_runs_until_a_puck_resets(self):
        rig = Rig(frictionless()).active()
        rig.run(p4wrist.hold((0.3, -0.2, 0.1), KP, KD), 1.0)
        before = list(rig.wrist.sample().q)
        record = rig.wrist.zero_record()
        rig.wrist.close()

        # A new program on the same (unreset) pucks keeps the zero.
        again = p4wrist.Wrist.callback(rig.bus, rig.config)
        kept, report = again.connect(record)
        self.assertTrue(kept, report)
        again.start()
        again.step(3)
        np.testing.assert_allclose(list(again.sample().q), before, atol=2e-3)
        again.close()

        # After a puck reset its counter restarted: the zero must not be kept.
        rig.bus.reset(1)
        third = p4wrist.Wrist.callback(rig.bus, rig.config)
        kept, report = third.connect(record)
        self.assertFalse(kept)
        self.assertIn('reset', report)
        third.close()

    def test_rough_placement_zero_from_single_turn_encoders(self):
        # Calibrate: read the encoders at home, then move each joint by hand.
        rig = Rig(zero=False)
        counts0, st0 = rig.wrist.read_encoders()
        rig.plant.m = [m + d for m, d in zip(rig.plant.m, (0.9, -1.3, 0.7))]
        for i, puck in enumerate(rig.bus.pucks):
            puck.angle = rig.plant.m[i]
        counts1, st1 = rig.wrist.read_encoders()
        dirs = [p4wrist.single_turn_direction(c1 - c0, s1 - s0) for c0, c1, s0, s1 in
                zip(counts0, counts1, st0, st1)]
        self.assertEqual(dirs, [1, -1, 1])   # the simulator's encoder directions
        cal = p4wrist.zero_cal((0.0, 0.0, 0.0), st0, dirs)
        rig.close()

        # Power cycle elsewhere: counters restart, wrist placed roughly near home.
        q0 = (0.05, -0.04, 0.03)
        params = p4_sim.SimParams(initial_q=q0)
        rig2 = Rig(params, zero=False)
        from_home = rig2.wrist.zero(cal, exact=False)
        np.testing.assert_allclose(from_home, q0, atol=3e-4)
        rig2.close()


class ControlTest(unittest.TestCase):
    def test_holds_and_tracks_each_joint(self):
        rig = Rig(frictionless()).active()
        for target in [(0.2, 0.0, 0.0), (0.0, -0.15, 0.0), (0.0, 0.0, 0.1), (0.1, 0.1, -0.1)]:
            s = rig.run(p4wrist.hold(target, KP, KD), 1.0)
            np.testing.assert_allclose(list(s.q), target, atol=2e-3)
            np.testing.assert_allclose(rig.plant.q, target, atol=2e-3)
        rig.close()

    def test_differential_torque_signs(self):
        # Pure FE torque accelerates FE only; pure UR torque UR only.
        for joint in (1, 2, 0):
            rig = Rig(frictionless()).active()
            tau = [0.0, 0.0, 0.0]
            tau[joint] = 0.05
            s = rig.run(torque_command(tau), 0.1)
            moved = np.array(s.q)
            self.assertGreater(moved[joint], 1e-3, f'joint {joint}')
            others = [k for k in range(3) if k != joint]
            self.assertLess(np.abs(moved[others]).max(), 0.02 * moved[joint])
            rig.close()

    def test_measured_torque_matches_command(self):
        rig = Rig().active()
        rig.plant.lock_output(1)
        s = rig.run(torque_command((0.0, 0.3, 0.0)), 0.1)
        self.assertAlmostEqual(s.tau_cmd[1], 0.3, places=6)
        self.assertAlmostEqual(s.tau_meas[1], 0.3, delta=0.01)
        self.assertAlmostEqual(s.tau_meas[2], 0.0, delta=0.01)
        rig.close()

    def test_saturation_keeps_differential_direction(self):
        config = p4wrist.default_config()
        config.max_joint_velocity = p4wrist.D3(20.0, 20.0, 20.0)
        config.max_motor_permille = 400              # 0.1 N m per motor
        config.max_joint_torque = p4wrist.D3(10, 10, 10)
        rig = Rig(config=config).active()
        rig.plant.lock_output(1)
        rig.plant.lock_output(2)
        s = rig.run(torque_command((0.0, 5.0, 0.0)), 0.05)
        self.assertLessEqual(max(abs(p) for p in s.permille), 400)
        self.assertAlmostEqual(s.tau_cmd[2], 0.0, places=9)
        self.assertGreater(s.tau_cmd[1], 0.0)
        rig.close()

    def test_velocity_setpoint_integrates_with_nan_position(self):
        rig = Rig(frictionless()).active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.2)
        cmd = p4wrist.hold((math.nan, 0.0, 0.0), KP, KD)
        cmd.joint[0].velocity = 0.5
        s = rig.run(cmd, 1.0)
        self.assertAlmostEqual(s.setpoint[0], 0.5, delta=0.01)
        self.assertAlmostEqual(s.q[0], 0.5, delta=0.03)
        rig.close()

    def test_stay_within_is_one_sided(self):
        # Outside a bound: PD toward it plus feedforward; inside: feedforward
        # only. FE's output is locked so the law is seen at a fixed angle.
        rig = Rig().active()
        rig.plant.lock_output(1)
        q0 = rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1).q[1]
        cases = [('upper', q0 - 0.05, 5.0 * -0.05 + 0.02),   # beyond the upper bound
                 ('upper', q0 + 0.05, 0.02),                 # inside
                 ('lower', q0 + 0.05, 5.0 * 0.05 + 0.02),    # beyond the lower bound
                 ('lower', q0 - 0.05, 0.02)]
        for bound, wall, expected in cases:
            cmd = p4wrist.hold((0, 0, 0), KP, KD)
            cmd.joint[1] = p4wrist.joint_command(torque=0.02, kp=5.0, kd=0.0,
                                                 mode=p4wrist.MODE_STAY_WITHIN, **{bound: wall})
            s = rig.run(cmd, 0.01)
            self.assertEqual(s.joint_mode[1], p4wrist.MODE_STAY_WITHIN)
            self.assertAlmostEqual(s.tau_cmd[1], expected, delta=0.005, msg=(bound, wall))
        rig.close()

    def test_brake_grounds_phases_then_resumes(self):
        rig = Rig(frictionless()).active()
        rig.run(torque_command((0.1, 0.0, 0.0)), 0.15)
        moving = rig.wrist.sample().qd[0]
        self.assertGreater(moving, 1.0)
        cmd = torque_command((0.0, 0.0, 0.0))
        cmd.brake = 1
        s = rig.run(cmd, 0.5)
        self.assertEqual(s.wrist_state, p4wrist.STATE_BRAKE)
        self.assertEqual(rig.bus.pucks[0].ramp_mode, p4_sim.DRIVE_BRAKE)
        self.assertLess(abs(s.qd[0]), 0.1 * moving)
        s = rig.run(p4wrist.hold((0.2, 0, 0), KP, KD), 1.0)
        self.assertEqual(s.wrist_state, p4wrist.STATE_ACTIVE)
        self.assertAlmostEqual(s.q[0], 0.2, delta=2e-3)
        rig.close()


class SafetyTest(unittest.TestCase):
    def assertFault(self, rig, code):
        s = rig.wrist.sample()
        self.assertEqual(p4wrist.FAULT_NAMES[s.fault], p4wrist.FAULT_NAMES[code], rig.wrist.fault_text())
        return s

    def test_position_limit_brakes_then_disables(self):
        config = p4wrist.default_config()
        config.max_joint_velocity = p4wrist.D3(20.0, 20.0, 20.0)
        config.joint_max = p4wrist.D3(math.pi, 0.2, math.pi)
        rig = Rig(frictionless(), config=config).active()
        s = rig.run(p4wrist.hold((0, 0.5, 0), KP, KD), 0.4)
        s = self.assertFault(rig, 1)
        self.assertIn(s.wrist_state, (p4wrist.STATE_BRAKING, p4wrist.STATE_FAULT))
        rig.run(p4wrist.hold((0, 0.5, 0), KP, KD), config.brake_s + 0.3)
        s = rig.wrist.sample()
        self.assertEqual(s.wrist_state, p4wrist.STATE_FAULT)
        self.assertTrue(all(p == 0 for p in s.permille))
        self.assertFalse(any(p.operation_enabled for p in rig.bus.pucks))
        # Latched until cleared, and enabling outside the limits is refused.
        with self.assertRaisesRegex(p4wrist.P4WristError, 'clear the fault'):
            rig.wrist.enable(0.5)
        rig.wrist.clear_fault()
        with self.assertRaisesRegex(p4wrist.P4WristError, 'outside the joint limits'):
            rig.wrist.enable(0.5)
        rig.close()

    def test_velocity_limit(self):
        config = p4wrist.default_config()
        config.max_joint_velocity = p4wrist.D3(1.0, 1.0, 1.0)
        rig = Rig(frictionless(), config=config).active()
        rig.run(torque_command((0.5, 0, 0)), 0.5)
        self.assertFault(rig, 2)
        rig.close()

    def test_host_timeout(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1)
        rig.wrist.step(int(0.15 / 0.002))   # no new commands
        self.assertFault(rig, 3)
        rig.close()

    def test_lost_puck(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1)
        rig.bus.drop(2)
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.05)
        s = self.assertFault(rig, 4)
        self.assertEqual(s.fault_node, 8)
        self.assertEqual(s.wrist_state, p4wrist.STATE_FAULT)
        # The dropped puck no longer hears SYNC: its own SYNC-loss fault stops it.
        rig.bus.silence(0.05)
        self.assertEqual(rig.bus.pucks[2].ds402, 'FAULT')
        rig.close()

    def test_drive_fault_from_emcy(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1)
        rig.bus.fault(0, 0x3220)
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.01)
        self.assertFault(rig, 5)
        self.assertIn('under-voltage', rig.wrist.fault_text())
        self.assertEqual(rig.wrist.sample().emcy_code, 0x3220)
        rig.close()

    def test_puck_reset_loses_the_zero(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1)
        rig.bus.reset(1)
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.01)
        s = self.assertFault(rig, 6)
        self.assertEqual(s.zeroed, 0)
        self.assertEqual(rig.wrist.zero_record().valid, 0)
        rig.close()

    def test_drives_fault_when_the_host_dies(self):
        rig = Rig().active()
        rig.run(torque_command((0.2, 0.0, 0.0)), 0.05)
        rig.bus.silence(0.05)    # no SYNC: 20 ms SYNC-loss timeout written at start
        for puck in rig.bus.pucks:
            self.assertEqual(puck.ds402, 'FAULT')
            self.assertEqual(puck.od[(0x2406, 0)], 20)
        rig.close()

    def test_clean_stop_disarms_sync_loss(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 0.1)
        rig.wrist.stop()
        rig.bus.silence(1.0)
        for puck in rig.bus.pucks:
            self.assertNotEqual(puck.ds402, 'FAULT')
            self.assertEqual(puck.od[(0x2406, 0)], 0)
            self.assertFalse(puck.operation_enabled)
        rig.close()

    def test_controlword_refresh_keeps_drives_enabled(self):
        rig = Rig().active()
        rig.run(p4wrist.hold((0, 0, 0), KP, KD), 2.0)   # far beyond 55 SYNCs
        self.assertEqual(rig.wrist.sample().wrist_state, p4wrist.STATE_ACTIVE)
        self.assertTrue(all(p.operation_enabled for p in rig.bus.pucks))
        rig.close()


if __name__ == '__main__':
    unittest.main()
