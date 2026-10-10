"""Cogging on the simulated wrist: the friction sweep identifies it, and the
analyses that hold the motor within a few cogging periods subtract its model."""
import json
import unittest

from tests.helpers import rel_err, run_analysis, run_cogging_sweep, run_sim_experiment, scratch_dir

import ramp_input_experiment
import ramp_input_experiment_analysis
import torque_calibration_experiment
import torque_calibration_experiment_analysis
import transmission_stiffness_experiment
import transmission_stiffness_experiment_analysis


class CoggingPipelineTest(unittest.TestCase):
    def test_sweep_model_is_removed_downstream(self):
        with scratch_dir() as tmp:
            sweep_dir, sweep = run_cogging_sweep(tmp)
            truth = json.loads((sweep_dir / 'friction_sweep_metadata.json').read_text())['sim_truth']
            runs = {}
            for name, experiment, analysis in (
                    ('ramp', ramp_input_experiment, ramp_input_experiment_analysis),
                    ('transmission', transmission_stiffness_experiment,
                     transmission_stiffness_experiment_analysis),
                    ('calibration', torque_calibration_experiment,
                     torque_calibration_experiment_analysis)):
                run_dir = run_sim_experiment(experiment, tmp / name)
                runs[name] = run_analysis(analysis, run_dir)

        self.assertTrue(sweep['cogging_detected'])
        # Read low through the loop's finite stiffness (see test_friction_sweep).
        self.assertLess(rel_err(sweep['cogging_amplitude_nm'], truth['cogging_amplitude']), 0.30)

        # Breakaway torque = static friction + cogging at the stuck position: the model
        # collapses the spread the cogging puts into individual breakaways.
        ramp = runs['ramp']
        self.assertTrue(ramp['cogging_removed'])
        self.assertLess(rel_err(ramp['static_friction_nm'], truth['static']), 0.10)
        # (This wrist's cogging is small next to its friction, so the spread it adds is too.)
        self.assertGreater(ramp['breakaway_torque_std_nm'], 1.5 * ramp['static_friction_std_nm'])

        # Cogging's local stiffness would otherwise add to the cable's. The P4's
        # 4096-count encoders resolve the cable's ~0.06 deg deflection in ~5 counts.
        transmission = runs['transmission']
        self.assertTrue(transmission['cogging_removed'])
        self.assertLess(rel_err(transmission['stiffness_nm_rad'],
                                truth['transmission_stiffness']), 0.10)

        calibration = runs['calibration']
        self.assertTrue(calibration['cogging_removed'])
        self.assertLess(rel_err(calibration['torque_scale_factor'], 1.0), 0.02)
        self.assertLess(rel_err(calibration['hysteresis_at_zero_nm'], 2 * truth['static']), 0.25)


if __name__ == '__main__':
    unittest.main()
