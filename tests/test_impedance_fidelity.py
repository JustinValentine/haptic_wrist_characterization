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
        # The logged mode column is numeric (moteus-style mode codes), never segment info.
        self.assertTrue(pd.api.types.is_numeric_dtype(modes))
        truth = meta['sim_truth']
        targets = {(t['mode'], t['k_nm_rad'], t['b_nms_rad']): t['target_id']
                   for t in meta['targets']}
        loop_stiff = targets[('rt', 20.0, 0.2)]
        host_stiff = targets[('host_250', 20.0, 0.2)]
        host_soft = targets[('host_250', 5.0, 0.0)]
        period = meta['p4wrist_config']['period_s']
        tau_c = truth['current_loop_time_constant']

        # Free-space inertia from the near-free target (its spring removed at the
        # loop delay).
        self.assertLess(rel_err(summary['impedance_free_J_kg_m2'],
                                truth['motor_inertia'] + truth['output_inertia']), 0.10)
        # The 500 Hz loop renders the commanded stiffness (Coulomb friction at these
        # ~2 deg motions biases the line-by-line subtraction by up to ~10%; without
        # friction the error is ~2%).
        self.assertLess(abs(summary[f't{loop_stiff}_stiffness_error_percent']), 15.0)
        # Its delay lies between the spring's path (2 SYNC periods, the CST ramp)
        # and the damping's (half a period more for differencing, plus the filter).
        loop_ms = summary['impedance_delay_rt_ms']
        filt = 1.0 / (2.0 * math.pi * meta['velocity_filter_hz'])
        self.assertGreater(loop_ms, 0.9e3 * (meta['loop_delay_s'] + tau_c))
        self.assertLess(loop_ms, 1.1e3 * (meta['loop_delay_s'] + period / 2 + filt + tau_c))
        # The host loop adds the sample's age and half its own period; its delay
        # turns stiffness into negative damping, about -K T.
        delay_ms = summary['impedance_delay_host_250_ms']
        expected_ms = 1000.0 * (meta['loop_delay_s'] + period + 0.5 / 250.0 + tau_c)
        self.assertLess(rel_err(delay_ms, expected_ms), 0.25)
        self.assertGreater(delay_ms, loop_ms)
        shift = summary[f't{host_stiff}_damping_shift_nms_rad']
        self.assertLess(shift, -0.05)
        self.assertLess(rel_err(shift, -20.0 * delay_ms / 1000.0), 0.3)
        self.assertLess(summary[f't{host_soft}_damping_shift_nms_rad'], 0.0)
        # Linear stability limit ~ b / T, lower for the slower loop.
        k_crit = summary['impedance_predicted_k_crit_host_250_b0_nm_rad']
        b = summary['impedance_b_linear_nms_rad']
        self.assertLess(rel_err(k_crit, b / (delay_ms / 1000.0)), 0.2)
        self.assertLess(k_crit, summary['impedance_predicted_k_crit_rt_b0_nm_rad'])
        # The 500 Hz loop stays faithful at least as far up as the host loop (when
        # friction leaves both with a defined fidelity bandwidth).
        rt_bw = summary.get('impedance_fidelity_bandwidth_rt_hz')
        host_bw = summary.get('impedance_fidelity_bandwidth_host_250_hz')
        if rt_bw is not None and host_bw is not None:
            self.assertGreaterEqual(rt_bw, host_bw)

    def test_compares_with_zwidth(self):
        with scratch_dir() as tmp:
            zw_dir = run_sim_experiment(zwidth_experiment, tmp / 'data' / 'zwidth' / 'zw_sim')
            run_analysis(zwidth_experiment_analysis, zw_dir)
            run_dir = run_sim_experiment(impedance_fidelity_experiment, tmp / 'run')
            meta = json.loads((run_dir / 'impedance_metadata.json').read_text())
            summary = run_analysis(impedance_fidelity_experiment_analysis, run_dir)

        self.assertIsNotNone(meta['impedance_protocol']['zwidth_source'])
        first_unstable = summary['impedance_zwidth_k_first_unstable_host_250_b0_nm_rad']
        predicted = summary['impedance_predicted_k_crit_host_250_b0_nm_rad']
        self.assertTrue(math.isfinite(first_unstable))
        # A pulse-based Z-width is friction-quenched, so it exceeds the linear limit.
        self.assertGreater(summary['impedance_zwidth_over_predicted_host_250'], 1.0)
        self.assertAlmostEqual(summary['impedance_zwidth_over_predicted_host_250'],
                               first_unstable / predicted)


if __name__ == '__main__':
    unittest.main()
