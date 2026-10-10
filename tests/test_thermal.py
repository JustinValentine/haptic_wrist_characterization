import json
import math
import unittest

from tests.helpers import rel_err, run_analysis, run_sim_experiment, scratch_dir

import p4_sim
import thermal_experiment
import thermal_experiment_analysis


class ThermalTest(unittest.TestCase):
    def test_recovers_thermal_model_and_continuous_torque(self):
        # A fast thermal plant so the quick protocol reaches steady state.
        thermal_experiment.SIM_PARAMS = p4_sim.SimParams(thermal_time_constant=15.0)
        try:
            with scratch_dir() as tmp:
                run_dir = run_sim_experiment(thermal_experiment, tmp / 'run')
                meta = json.loads((run_dir / 'thermal_metadata.json').read_text())
                summary = run_analysis(thermal_experiment_analysis, run_dir)
        finally:
            thermal_experiment.SIM_PARAMS = None

        self.assertTrue(meta['completed'])
        self.assertEqual(summary['temperature_channel'], 'motor_temp_c')
        truth = meta['sim_truth']
        c = truth['thermal_resistance'] * truth['winding_resistance'] / (
            truth['gear_ratio'] * truth['motor_torque_constant']) ** 2
        allowed = summary['temperature_limit_c'] - summary['start_temperature_c']
        self.assertLess(rel_err(summary['rise_per_torque_sq_c_per_nm2'], c), 0.10)
        self.assertLess(rel_err(summary['thermal_time_constant_s'],
                                truth['thermal_time_constant']), 0.10)
        self.assertLess(rel_err(summary['continuous_torque_nm'], math.sqrt(allowed / c)), 0.10)
        self.assertEqual(summary['num_levels'], 3)


if __name__ == '__main__':
    unittest.main()
