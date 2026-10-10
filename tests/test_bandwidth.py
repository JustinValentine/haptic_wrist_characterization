import json
import math
import unittest

import numpy as np
import pandas as pd
from tests.helpers import (frictionless, loop_velocity_path, rel_err, run_analysis,
                           run_sim_experiment, scratch_dir)

import schroeder_bandwidth_experiment
import schroeder_bandwidth_experiment_analysis as analysis


def true_model(freqs, truth, kp, kd, meta, ff=1.0):
    """The rigid joint under the p4wrist loop: torque 2 SYNC periods (plus the
    current loop) after its sample, damping on a differenced, filtered velocity."""
    J = truth['motor_inertia'] + truth['output_inertia']
    b = truth['viscous'] + truth['output_viscous']
    tau = meta['loop_delay_s'] + truth['current_loop_time_constant']
    return analysis.model_response(freqs, J, b, tau, kp, kd, ff, loop_velocity_path(meta))


def write_legacy_run(directory, kp=450.0, kd=0.75, J=0.008, b=0.022):
    """Data in the format of the old single-period, 100 Hz Schroeder script."""
    fs, period, f_min, f_max, amplitude = 100.0, 40.0, 0.1, 20.0, 10.0
    t = np.arange(int(period * fs)) / fs
    n = int(round((f_max - f_min) * period)) + 1
    k = np.arange(1, n + 1)
    freqs = f_min + (k - 1) / period
    phases = -k * (k - 1) * np.pi / n
    arg = 2 * np.pi * np.outer(t, freqs) + phases
    scale = amplitude / np.abs(np.cos(arg).sum(axis=1)).max()
    u = scale * np.cos(arg).sum(axis=1)
    # Exact periodic steady state of the continuous closed loop (velocity
    # feedforward on), so the test isolates the legacy analysis path.
    H = analysis.model_response(freqs, J, b, 0.0, kp, kd, 1.0)
    y = scale * (np.abs(H) * np.cos(arg + np.angle(H))).sum(axis=1)
    directory.mkdir(parents=True)
    pd.DataFrame({'sample_index': np.arange(len(t)), 'time': t, 't_ref': t,
                  'cmd_pos_deg': u, 'joint_pos_deg': y,
                  'kp_joint_nm_rad': kp, 'kd_joint_nms_rad': kd,
                  'sent_velocity_feedforward': True, 'gear_ratio': 10.0}).to_csv(
        directory / 'bandwidth_data.csv', index=False)
    (directory / 'bandwidth_collection_metadata.json').write_text(json.dumps({
        'experiment': 'schroeder_bandwidth_experiment.py', 'amplitude_deg': amplitude,
        'freq_min_hz': f_min, 'freq_max_hz': f_max, 'duration_s': period,
        'sample_frequency_commanded_hz': fs, 'kp_joint_nm_rad': kp, 'kd_joint_nms_rad': kd,
        'send_command_velocity': True}))
    return directory / 'bandwidth_data.csv'


class BandwidthTest(unittest.TestCase):
    def test_frequency_response_matches_plant(self):
        # Coulomb friction makes small-amplitude responses non-linear; without it
        # the measured response must match the linear model of plant and loop.
        schroeder_bandwidth_experiment.SIM_PARAMS = frictionless()
        try:
            with scratch_dir() as tmp:
                run_dir = run_sim_experiment(schroeder_bandwidth_experiment, tmp / 'run')
                meta = json.loads((run_dir / 'bandwidth_metadata.json').read_text())
                self.assertTrue(meta['completed'])
                truth = meta['sim_truth']
                summary = run_analysis(analysis, run_dir)
                response = pd.read_csv(run_dir / 'analysis' / 'frequency_response.csv')
        finally:
            schroeder_bandwidth_experiment.SIM_PARAMS = None

        kp, kd = summary['kp_joint_nm_rad'], summary['kd_joint_nms_rad']
        J_true = truth['motor_inertia'] + truth['output_inertia']
        # From the measured torque, with no loop model: J to ~1%.
        self.assertLess(rel_err(summary['plant_J_kg_m2'], J_true), 0.02)
        # The closed-loop fit's J and b trade off when Kd dominates the loop damping.
        self.assertLess(rel_err(summary['fit_J_kg_m2'], J_true), 0.15)
        self.assertTrue(summary['fit_delay_fixed'])
        self.assertEqual(summary['num_levels_valid'], summary['num_levels'])
        # With velocity feedforward the -3 dB point lies beyond the 30 Hz band.
        self.assertTrue(summary['bandwidth_above_range'])
        self.assertGreater(summary['fit_bandwidth_hz'], 30.0)

        valid = response[response['line_valid'] & response['level_valid']]
        H = valid['H_real'].to_numpy() + 1j * valid['H_imag'].to_numpy()
        error_db = 20 * np.log10(np.abs(H) / np.abs(
            true_model(valid['frequency_hz'].to_numpy(), truth, kp, kd, meta)))
        self.assertLess(np.max(np.abs(error_db)), 1.0)

    def test_legacy_single_period_data(self):
        with scratch_dir() as tmp:
            csv = write_legacy_run(tmp / 'legacy')
            summary = run_analysis(analysis, csv)
            response = pd.read_csv(tmp / 'legacy' / 'bandwidth_data_analysis' /
                                   'frequency_response.csv')
        self.assertTrue(summary['legacy'])
        low = response[response['frequency_hz'] <= 10.0]
        H = low['H_real'].to_numpy() + 1j * low['H_imag'].to_numpy()
        expected = analysis.model_response(low['frequency_hz'].to_numpy(), 0.008, 0.022, 0.0,
                                           450.0, 0.75, 1.0)
        error_db = 20 * np.log10(np.abs(H) / np.abs(expected))
        self.assertLess(np.max(np.abs(error_db)), 1.0)
        self.assertLess(rel_err(summary['fit_J_kg_m2'], 0.008), 0.15)
        self.assertTrue(math.isfinite(summary['fit_bandwidth_hz']))


if __name__ == '__main__':
    unittest.main()
