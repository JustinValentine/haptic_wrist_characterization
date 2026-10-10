import json
import math
import unittest

import pandas as pd

from tests.helpers import frictionless, p4_loop_step, rel_err, run_analysis, run_sim_experiment, scratch_dir

import closed_loop_step_experiment
import closed_loop_step_experiment_analysis


class ClosedLoopStepTest(unittest.TestCase):
    def test_step_metrics_and_acceleration_capability(self):
        # Without Coulomb friction the small step is a linear loop, which the
        # reference model of the p4wrist loop (delay included) must reproduce.
        closed_loop_step_experiment.SIM_PARAMS = frictionless()
        try:
            with scratch_dir() as tmp:
                run_dir = run_sim_experiment(closed_loop_step_experiment, tmp / 'run')
                meta = json.loads((run_dir / 'closed_loop_step_metadata.json').read_text())
                self.assertTrue(meta['completed'])
                truth = meta['sim_truth']
                summary = run_analysis(closed_loop_step_experiment_analysis, run_dir)
                per_size = pd.read_csv(run_dir / 'analysis' / 'per_size.csv')
        finally:
            closed_loop_step_experiment.SIM_PARAMS = None

        kp, kd = summary['kp_joint_nm_rad'], summary['kd_joint_nms_rad']
        J = truth['motor_inertia'] + truth['output_inertia']
        b = truth['viscous'] + truth['output_viscous']
        step = math.radians(summary['linear_step_deg'])
        _, q = p4_loop_step(J, b, kp, kd, step, filter_hz=meta['velocity_filter_hz'])
        overshoot = 100.0 * (q.max() / step - 1.0)
        # A delay-free loop with these gains would overshoot ~2%; the 4 ms loop
        # delay makes it ~27%.
        self.assertGreater(overshoot, 15.0)

        # The smallest step stays below the torque cap: second-order behavior.
        self.assertEqual(summary['linear_step_deg'], min(per_size['step_size_deg']))
        self.assertAlmostEqual(summary['overshoot_percent'], overshoot, delta=5.0)
        self.assertTrue(2.0 < summary['rise_time_ms'] < 30.0)
        self.assertTrue(math.isfinite(summary['settling_time_ms']))

        # Small steps are linear, the largest saturate the cap.
        by_size = per_size.set_index('step_size_deg')
        self.assertEqual(by_size['saturated_steps'].iloc[0], 0)
        self.assertEqual(by_size['saturated_steps'].iloc[-1], by_size['num_steps'].iloc[-1])

        # Acceleration per delivered torque gives cap / J even though back-EMF derates the
        # torque at speed; with no friction-sweep analysis here friction is not removed.
        capability = summary['torque_cap_nm'] * truth['torque_constant_ratio'] / J
        self.assertLess(rel_err(summary['saturated_acceleration_rad_s2'], capability), 0.10)
        self.assertGreater(summary['peak_velocity_deg_s'], 500.0)
        # The joint cannot outrun the motor's no-load speed at the 48 V supply.
        self.assertLess(summary['peak_velocity_deg_s'], math.degrees(truth['no_load_speed']))


if __name__ == '__main__':
    unittest.main()
