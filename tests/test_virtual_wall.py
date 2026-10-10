import json
import unittest

import pandas as pd

from tests.helpers import run_analysis, run_sim_experiment, scratch_dir

import virtual_wall_experiment
import virtual_wall_experiment_analysis


class VirtualWallTest(unittest.TestCase):
    def test_wall_energy_follows_loop_delay(self):
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
        # Both loops are delayed, so stiffer walls give back more energy; the 250 Hz
        # host loop's longer delay gives back more than the 500 Hz loop at every K.
        # With b ~ 0.02 N m s/rad the loop's wall is passive only below ~b/(T/2) =
        # 10 N m/rad by Colgate's bound; friction keeps K = 20 passive in practice.
        rt = [e[('rt', k)] for k in ks]
        self.assertEqual(rt, sorted(rt))
        self.assertLess(e[('rt', ks[0])], 1.0)
        self.assertGreater(rt[-1], 1.0)
        for k in ks:
            if ('host_250', k) in e:
                self.assertGreater(e[('host_250', k)], e[('rt', k)])
        self.assertLessEqual(summary['first_active_k_host_250_nm_rad'],
                             summary['first_active_k_rt_nm_rad'])
        self.assertGreaterEqual(summary['max_passive_k_rt_nm_rad'], 20.0)

        # The loop's measured torque delay is its CST delay (2 SYNC periods). The host
        # loop's adds 0-4 ms depending on where its 250 Hz update falls after contact
        # (one impact per K here), so it is only guaranteed to be longer.
        self.assertAlmostEqual(summary['torque_delay_rt_median_ms'],
                               1000.0 * meta['loop_delay_s'], delta=1.0)
        self.assertGreater(summary['torque_delay_host_250_median_ms'],
                           summary['torque_delay_rt_median_ms'])


if __name__ == '__main__':
    unittest.main()
