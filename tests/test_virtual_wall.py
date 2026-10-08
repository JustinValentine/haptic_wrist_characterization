import json
import math
import unittest

import pandas as pd

from tests.helpers import run_analysis, run_sim_experiment, scratch_dir

import virtual_wall_experiment
import virtual_wall_experiment_analysis


class VirtualWallTest(unittest.TestCase):
    def test_firmware_wall_is_passive_and_host_wall_injects_energy(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(virtual_wall_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'virtual_wall_metadata.json').read_text())
            summary = run_analysis(virtual_wall_experiment_analysis, run_dir)
            impacts = pd.read_csv(run_dir / 'analysis' / 'impacts.csv')

        # Every impact ran to completion and was classified; no hard safety abort.
        self.assertTrue(meta['completed'])
        self.assertIsNone(meta['abort_reason'])
        online = {r['trial_id']: r for r in meta['impacts']}
        self.assertEqual(len(online), len(impacts))
        self.assertTrue(all(r['ended_by'] and isinstance(r['unstable'], bool)
                            for r in online.values()))
        self.assertTrue(impacts['contact'].all())

        # The analysis re-measures exactly what the experiment measured online.
        for _, row in impacts.iterrows():
            self.assertAlmostEqual(row['restitution'], online[row['trial_id']]['restitution'],
                                   places=9)

        e = impacts.groupby(['mode', 'virtual_k_nm_rad'])['restitution'].mean()
        ks = sorted(impacts['virtual_k_nm_rad'].unique())
        # The firmware wall (no host delay) is passive (nearly lossless when very
        # stiff, hence the 2% margin); the 1 kHz host loop's delay makes stiff
        # walls launch the handle back faster than it came.
        for k in ks:
            self.assertLess(e[('moteus', k)], 1.0 if k <= 100 else 1.02)
            if k >= 100:
                self.assertLess(e[('moteus', k)], e[('host_1000', k)])
        host = [e[('host_1000', k)] for k in ks]
        self.assertEqual(host, sorted(host))
        self.assertGreater(host[-1], 1.0)

        # Rate-hardness of a delay-free spring tracks the commanded stiffness.
        moteus = impacts[impacts['mode'] == 'moteus']
        for _, row in moteus.iterrows():
            self.assertLess(abs(row['rate_hardness_nm_rad'] / row['virtual_k_nm_rad'] - 1.0), 0.3)

        self.assertGreaterEqual(summary['max_passive_k_moteus_nm_rad'], 100.0)
        self.assertTrue(math.isfinite(summary['first_active_k_host_1000_nm_rad']))
        self.assertGreater(summary['torque_delay_host_1000_median_ms'],
                           summary['torque_delay_moteus_median_ms'] + 1.0)


if __name__ == '__main__':
    unittest.main()
