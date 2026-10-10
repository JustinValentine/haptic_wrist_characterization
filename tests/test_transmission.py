import json
import unittest

from tests.helpers import cogging_free, rel_err, run_analysis, run_sim_experiment, scratch_dir

import transmission_stiffness_experiment
import transmission_stiffness_experiment_analysis


def run_with(params):
    transmission_stiffness_experiment.SIM_PARAMS = params
    try:
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(transmission_stiffness_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'transmission_metadata.json').read_text())
            summary = run_analysis(transmission_stiffness_experiment_analysis, run_dir)
    finally:
        transmission_stiffness_experiment.SIM_PARAMS = None
    return meta, summary


class TransmissionTest(unittest.TestCase):
    def test_recovers_stiffness(self):
        # Static loops only: the multisine phase is off by default on the P4 wrist,
        # whose 4096-count encoders cannot resolve its motion (see the experiment).
        meta, summary = run_with(cogging_free())
        self.assertTrue(meta['completed'])
        truth = meta['sim_truth']
        # The encoders step FE by 0.006-0.012 deg (two motors), ~0.15-0.3 N m of cable
        # torque, so the loops span only ~8-16 steps: quantization limits the
        # stiffness to ~15%.
        self.assertLess(rel_err(summary['stiffness_nm_rad'], truth['transmission_stiffness']), 0.15)
        self.assertLess(abs(summary['lost_motion_deg']), 0.03)
        self.assertNotIn('dynamic_stiffness_nm_rad', summary)

    def test_detects_backlash(self):
        meta, summary = run_with(cogging_free(backlash_deg=0.2))
        self.assertTrue(meta['completed'])
        self.assertAlmostEqual(summary['lost_motion_deg'], 0.2, delta=0.04)
        self.assertGreater(summary['loop_width_at_zero_deg'], 0.15)
        # Backlash must not bias the stiffness of the engaged cable (beyond the
        # encoder resolution, see above).
        self.assertLess(rel_err(summary['stiffness_nm_rad'],
                                meta['sim_truth']['transmission_stiffness']), 0.15)


if __name__ == '__main__':
    unittest.main()
