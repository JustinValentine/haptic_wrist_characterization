import json
import math
import unittest

import pandas as pd

from tests.helpers import cogging_free, rel_err, run_analysis, run_sim_experiment, scratch_dir

import impedance_fidelity_experiment
import impedance_fidelity_experiment_analysis
import zwidth_experiment
import zwidth_experiment_analysis


class ImpedanceFidelityTest(unittest.TestCase):
    def test_rendered_impedance_and_loop_delay(self):
        impedance_fidelity_experiment.SIM_PARAMS = cogging_free()
        try:
            with scratch_dir() as tmp:
                run_dir = run_sim_experiment(impedance_fidelity_experiment, tmp / 'run')
                meta = json.loads((run_dir / 'impedance_metadata.json').read_text())
                modes = pd.read_csv(run_dir / 'impedance_data.csv')['mode']
                summary = run_analysis(impedance_fidelity_experiment_analysis, run_dir)
        finally:
            impedance_fidelity_experiment.SIM_PARAMS = None
        self._check_rendering(meta, modes, summary)

    def _check_rendering(self, meta, modes, summary):
        self.assertTrue(meta['completed'])
        self.assertIsNone(meta['abort_reason'])
        # The logged mode column is the moteus mode register, never segment info.
        self.assertTrue(pd.api.types.is_numeric_dtype(modes))
        truth = meta['sim_truth']
        targets = {(t['mode'], t['k_nm_rad'], t['b_nms_rad']): t['target_id']
                   for t in meta['targets']}
        firmware = targets[('moteus', 40.0, 0.2)]
        host_stiff = targets[('host_1000', 40.0, 0.2)]
        host_soft = targets[('host_1000', 10.0, 0.0)]

        # Free-space inertia from the near-free target.
        self.assertLess(rel_err(summary['impedance_free_J_kg_m2'],
                                truth['motor_inertia'] + truth['output_inertia']), 0.10)
        # The firmware loop renders the commanded stiffness and damping.
        self.assertLess(abs(summary[f't{firmware}_stiffness_error_percent']), 5.0)
        self.assertLess(abs(summary[f't{firmware}_damping_shift_nms_rad']), 0.03)
        # The host loop's delay (~1.5 samples + current lag) turns stiffness into
        # negative damping, about -K T.
        delay_ms = summary['impedance_delay_host_1000_ms']
        expected_ms = 1000.0 * (1.5 / 1000.0 + truth['current_loop_time_constant'])
        self.assertLess(rel_err(delay_ms, expected_ms), 0.2)
        shift = summary[f't{host_stiff}_damping_shift_nms_rad']
        self.assertLess(shift, -0.02)
        self.assertLess(rel_err(shift, -40.0 * delay_ms / 1000.0), 0.3)
        self.assertLess(summary[f't{host_soft}_damping_shift_nms_rad'], 0.0)
        # Linear stability limit ~ b / T.
        k_crit = summary['impedance_predicted_k_crit_host_1000_b0_nm_rad']
        b = summary['impedance_b_linear_nms_rad']
        self.assertLess(rel_err(k_crit, b / (delay_ms / 1000.0)), 0.2)
        # Firmware rendering stays faithful at least as far up as host rendering.
        self.assertGreaterEqual(summary['impedance_fidelity_bandwidth_moteus_hz'],
                                summary['impedance_fidelity_bandwidth_host_1000_hz'])
        self.assertNotIn('impedance_predicted_k_crit_moteus_b0_nm_rad', summary)

    def test_compares_with_zwidth(self):
        with scratch_dir() as tmp:
            zw_dir = run_sim_experiment(zwidth_experiment, tmp / 'data' / 'zwidth' / 'zw_sim')
            run_analysis(zwidth_experiment_analysis, zw_dir)
            run_dir = run_sim_experiment(impedance_fidelity_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'impedance_metadata.json').read_text())
            summary = run_analysis(impedance_fidelity_experiment_analysis, run_dir)

        self.assertIsNotNone(meta['impedance_protocol']['zwidth_source'])
        first_unstable = summary['impedance_zwidth_k_first_unstable_host_1000_b0_nm_rad']
        predicted = summary['impedance_predicted_k_crit_host_1000_b0_nm_rad']
        self.assertTrue(math.isfinite(first_unstable))
        # A pulse-based Z-width is friction-quenched, so it exceeds the linear limit.
        self.assertGreater(summary['impedance_zwidth_over_predicted_host_1000'], 1.0)
        self.assertAlmostEqual(summary['impedance_zwidth_over_predicted_host_1000'],
                               first_unstable / predicted)


if __name__ == '__main__':
    unittest.main()
