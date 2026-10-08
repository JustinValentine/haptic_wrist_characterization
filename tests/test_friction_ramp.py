import json
import unittest

import pandas as pd

from tests.helpers import cogging_free, rel_err, run_analysis, run_sim_experiment, scratch_dir

import ramp_input_experiment
import ramp_input_experiment_analysis


class FrictionRampTest(unittest.TestCase):
    def setUp(self):
        ramp_input_experiment.SIM_PARAMS = cogging_free()

    def tearDown(self):
        ramp_input_experiment.SIM_PARAMS = None

    def test_recovers_static_friction_in_both_directions(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(ramp_input_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'friction_ramp_metadata.json').read_text())
            self.assertTrue(meta['completed'])
            truth = meta['sim_truth']
            summary = run_analysis(ramp_input_experiment_analysis, run_dir)

            # Old scripts wrote one segment per ramp + pause and no metadata.
            data = pd.read_csv(run_dir / 'friction_ramp_data.csv')
            data = data[data['ramp_id'].notna()].copy()
            data['segment'] = data['ramp_id'].astype(int)
            legacy_dir = tmp / 'legacy'
            legacy_dir.mkdir()
            legacy_csv = legacy_dir / 'friction_ramp_data.csv'
            data[['time', 'segment', 'phase', 'direction', 'cmd_joint_deg', 'joint_pos_deg',
                  'kp_joint_nm_rad']].to_csv(legacy_csv, index=False)
            legacy = run_analysis(ramp_input_experiment_analysis, legacy_csv)

        self.assertLess(rel_err(summary['static_friction_nm'], truth['static']), 0.10)
        self.assertGreaterEqual(summary['num_breakaways'], 0.8 * summary['num_ramps'])
        self.assertGreater(summary['num_breakaways_pos'], 0)
        self.assertGreater(summary['num_breakaways_neg'], 0)
        self.assertLess(abs(summary['direction_asymmetry_nm']), 0.1 * truth['static'])
        self.assertAlmostEqual(legacy['static_friction_nm'], summary['static_friction_nm'],
                               delta=0.02 * summary['static_friction_nm'])


if __name__ == '__main__':
    unittest.main()
