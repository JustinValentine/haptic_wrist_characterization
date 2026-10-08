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

        self.assertLess(rel_err(summary['J_kg_m2'],
                                truth['motor_inertia'] + truth['output_inertia']), 0.03)
        self.assertLess(rel_err(summary['fc_nm'], truth['coulomb']), 0.10)
        # The simulated current-loop lag removes ~Kp*tau of damping, so allow 15%.
        self.assertLess(rel_err(summary['b_nms_rad'],
                                truth['viscous'] + truth['output_viscous']), 0.15)
        self.assertGreater(summary['model_fit_percent_median'], 90.0)


if __name__ == '__main__':
    unittest.main()
