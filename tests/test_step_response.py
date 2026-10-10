import json
import unittest

from tests.helpers import rel_err, run_analysis, run_sim_experiment, scratch_dir

import step_input_experiment
import step_input_experiment_analysis


class StepResponseTest(unittest.TestCase):
    def test_recovers_inertia_damping_and_coulomb_friction(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(step_input_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'step_response_metadata.json').read_text())
            self.assertTrue(meta['completed'])
            truth = meta['sim_truth']
            summary = run_analysis(step_input_experiment_analysis, run_dir)

        J = truth['motor_inertia'] + truth['output_inertia']
        b = truth['viscous'] + truth['output_viscous']
        # Primary: the rigid-body fit of the measured torque, which needs no model
        # of the loop's gains or delay.
        self.assertEqual(summary['primary_method'], 'measured torque (rigid-body fit)')
        self.assertLess(rel_err(summary['J_kg_m2'], J), 0.02)
        self.assertLess(rel_err(summary['b_nms_rad'], b), 0.05)
        self.assertLess(rel_err(summary['fc_nm'], truth['coulomb']), 0.02)
        # Cross-check: the ring-down, corrected for Kd and the loop delay.
        self.assertLess(rel_err(summary['ringdown_J_kg_m2'], J), 0.03)
        self.assertLess(rel_err(summary['ringdown_b_nms_rad'], b), 0.10)
        self.assertLess(rel_err(summary['ringdown_fc_nm'], truth['coulomb']), 0.05)
        self.assertGreater(summary['model_fit_percent_median'], 90.0)
        # The run measured its own loop delay at the start.
        self.assertEqual(meta['loop_delay_probe']['cycles'], 2)
        self.assertAlmostEqual(meta['loop_delay_probe']['measured_over_commanded_torque'], 1.0,
                               delta=0.02)


if __name__ == '__main__':
    unittest.main()
