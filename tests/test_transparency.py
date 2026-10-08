import json
import unittest

from tests.helpers import rel_err, run_analysis, run_sim_experiment, scratch_dir

import transparency_experiment as experiment
import transparency_experiment_analysis as analysis


class TransparencyTest(unittest.TestCase):
    def test_zero_torque_impedance_and_compensation_benefit(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(experiment, tmp / 'run')
            meta = json.loads((run_dir / 'transparency_metadata.json').read_text())
            self.assertTrue(meta['completed'])
            truth = meta['sim_truth']
            summary = run_analysis(analysis, run_dir)
        self.assertEqual(summary['torque_source'], 'measured')
        inertia = truth['motor_inertia'] + truth['output_inertia']
        self.assertLess(rel_err(summary['zero_torque_J_app_kg_m2'], inertia), 0.20)
        self.assertLess(rel_err(summary['zero_torque_f_app_nm'], truth['coulomb']), 0.25)
        self.assertLess(summary['compensated_f_app_nm'], 0.5 * summary['zero_torque_f_app_nm'])
        self.assertLess(summary['compensated_rms_torque_nm'],
                        summary['zero_torque_rms_torque_nm'])


if __name__ == '__main__':
    unittest.main()
