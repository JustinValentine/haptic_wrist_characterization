import json
import math
import unittest

from tests.helpers import run_analysis, run_sim_experiment, scratch_dir

import zwidth_experiment
import zwidth_experiment_analysis


class ZWidthTest(unittest.TestCase):
    def test_boundary_ordering_and_containment(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(zwidth_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'zwidth_metadata.json').read_text())
            summary = run_analysis(zwidth_experiment_analysis, run_dir)

        # Unstable trials were contained by the trial stops; no hard safety abort.
        self.assertTrue(meta['completed'])
        self.assertIsNone(meta['abort_reason'])
        trials = meta['trials']
        self.assertGreater(len(trials), 0)
        self.assertTrue(all(isinstance(t['stable'], bool) and t['reason'] for t in trials))
        self.assertGreater(summary['zwidth_num_unstable'], 0)
        self.assertGreaterEqual(summary['zwidth_online_offline_agreement'], 0.95)

        boundary = {(b['mode'], b['virtual_b_nms_rad']): b for b in meta['boundary']}
        b_values = sorted({b for _, b in boundary})
        host_low = boundary[('host_1000', b_values[0])]['k_last_stable_nm_rad']
        host_high = boundary[('host_1000', b_values[-1])]['k_last_stable_nm_rad']
        moteus_low = boundary[('moteus', b_values[0])]['k_last_stable_nm_rad']
        # Virtual damping extends the host loop's stable stiffness range, and the
        # firmware loop (no host delay) renders far stiffer springs.
        self.assertIsNotNone(host_low)
        self.assertGreater(host_high, host_low)
        self.assertGreater(moteus_low, 5 * host_low)
        self.assertFalse(math.isnan(summary['zwidth_k_max_host_1000_b0_nm_rad']))
        self.assertAlmostEqual(summary['zwidth_k_max_host_1000_b0_nm_rad'], host_low)


if __name__ == '__main__':
    unittest.main()
