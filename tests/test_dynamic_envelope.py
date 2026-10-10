import json
import math
import unittest

import numpy as np
import pandas as pd

from tests.helpers import frictionless, rel_err, run_analysis, run_sim_experiment, scratch_dir

import dynamic_envelope_experiment as experiment
import dynamic_envelope_experiment_analysis as analysis


class DynamicEnvelopeTest(unittest.TestCase):
    def test_envelope_follows_torque_limit_at_high_frequency(self):
        # Without Coulomb friction the small-signal reference is linear, so the
        # high-frequency limit is the actuator's (with friction on this wrist the
        # reference gain itself depends on amplitude).
        experiment.SIM_PARAMS = frictionless()
        try:
            with scratch_dir() as tmp:
                run_dir = run_sim_experiment(experiment, tmp / 'run')
                meta = json.loads((run_dir / 'dynamic_envelope_metadata.json').read_text())
                self.assertTrue(meta['completed'])
                self.assertIsNone(meta['abort_reason'])
                truth = meta['sim_truth']
                summary = run_analysis(analysis, run_dir)
                envelope = pd.read_csv(run_dir / 'analysis' / 'envelope.csv')
                trials = pd.read_csv(run_dir / 'analysis' / 'trials.csv')
        finally:
            experiment.SIM_PARAMS = None

        # Every trial is classified, online and offline agree.
        self.assertTrue(trials['reason'].notna().all())
        self.assertTrue((trials['reason'].astype(str).str.len() > 0).all())
        self.assertEqual(summary['online_offline_agreement'], 1.0)

        env = envelope.sort_values('frequency_hz').set_index('frequency_hz')
        amax = env['amax_joint_deg']
        # Low frequency: the range-of-motion cap binds first (a lower bound).
        self.assertTrue(bool(env['lower_bound'].iloc[0]))
        self.assertIn('range', env['limited_by'].iloc[0])
        # High frequencies: decreasing envelope, set by the torque cap and, while motoring,
        # by the back-EMF torque-speed line.
        self.assertTrue((np.diff(amax.to_numpy()) < 0).all())
        J = truth['motor_inertia'] + truth['output_inertia']
        cap = summary['torque_cap_nm'] * truth['torque_constant_ratio']
        for f in env.index[-2:]:
            limit = analysis.actuator_limit_deg(f, J, cap, truth['stall_torque'],
                                                truth['no_load_speed'])
            self.assertFalse(bool(env.loc[f, 'lower_bound']))
            self.assertIn(env.loc[f, 'limited_by'], ('torque saturation', 'voltage limit'))
            self.assertLess(rel_err(amax.loc[f], limit), 0.15)
        self.assertGreater(summary['peak_power_w'], 10.0)

    def test_metrics_and_classification_on_synthetic_trial(self):
        f, ramp, steady = 5.0, 0.4, 1.0
        t = np.arange(0.0, 2 * ramp + steady, 1e-3)
        w = 2 * math.pi * f
        command = 10.0 * np.sin(w * t)
        joint = 9.0 * np.sin(w * t - 0.2) + 0.9 * np.sin(2 * w * t)   # gain 0.9, THD 0.1
        velocity = np.gradient(joint, t)
        control = np.where(np.abs(np.sin(w * t)) > 0.99, 10.0, 5.0)  # ~9% of samples at the cap
        cap = np.full_like(t, 10.0)
        metrics = experiment.trial_metrics(t, joint, command, velocity, control, control, cap,
                                           f, ramp, steady)
        self.assertAlmostEqual(metrics['gain'], 0.9, places=3)
        self.assertAlmostEqual(metrics['phase_deg'], math.degrees(-0.2), places=2)
        self.assertAlmostEqual(metrics['thd'], 0.1, places=3)
        self.assertGreater(metrics['saturation_fraction'], 0.02)

        reference = dict(metrics, G_real=0.9 * math.cos(-0.2), G_imag=0.9 * math.sin(-0.2))
        thresholds = experiment.thresholds_from({})
        verdict = experiment.classify(metrics, reference, thresholds)
        self.assertFalse(verdict['linear'])
        self.assertEqual(verdict['reason'], 'torque saturation')
        self.assertIn('distortion', verdict['all_reasons'])
        # A trial stopped before its steady window ends has no metrics.
        self.assertIsNone(experiment.trial_metrics(t[:900], joint[:900], command[:900],
                                                   velocity[:900], control[:900],
                                                   control[:900], cap[:900], f, ramp, steady))


if __name__ == '__main__':
    unittest.main()
