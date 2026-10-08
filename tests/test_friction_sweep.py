import json
import unittest

from tests.helpers import rel_err, run_analysis, run_sim_experiment, scratch_dir

import friction_sweep_experiment
import friction_sweep_experiment_analysis


class FrictionSweepTest(unittest.TestCase):
    def test_recovers_stribeck_friction_model(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(friction_sweep_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'friction_sweep_metadata.json').read_text())
            self.assertTrue(meta['completed'])
            truth = meta['sim_truth']
            summary = run_analysis(friction_sweep_experiment_analysis, run_dir)

        self.assertLess(rel_err(summary['coulomb_friction_nm'], truth['coulomb']), 0.10)
        self.assertLess(rel_err(summary['static_friction_stribeck_nm'], truth['static']), 0.15)
        self.assertLess(rel_err(summary['viscous_damping_nms_rad'],
                                truth['viscous'] + truth['output_viscous']), 0.20)
        self.assertGreater(summary['fit_r2'], 0.95)
        # No gravity in the default plant: the conservative torque is ~zero.
        self.assertLess(abs(summary['conservative_offset_nm']), 0.005)
        # The slowest pass resolves the GL60 II's cogging: 168 cycles per motor revolution.
        self.assertTrue(summary['cogging_detected'])
        self.assertLess(rel_err(summary['cogging_amplitude_nm'], truth['cogging_amplitude']), 0.10)
        self.assertLess(rel_err(summary['cogging_period_deg'], truth['cogging_period_deg']), 0.01)
        self.assertAlmostEqual(summary['cogging_cycles_per_motor_rev'],
                               truth['cogging_periods_per_motor_rev'], delta=0.5)
        # Whole-period bins keep cogging out of the remaining ripple.
        self.assertLess(summary['ripple_std_corrected_nm'], 0.1 * truth['cogging_amplitude'])


if __name__ == '__main__':
    unittest.main()
