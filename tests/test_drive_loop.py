"""The pucks' own position loops (CSP): calibration, then rendering against the PC loop."""
import json
import math
import unittest

import numpy as np
import pandas as pd

from tests.helpers import (frictionless, loop_velocity_path, rel_err, run_analysis,
                           run_sim_experiment, scratch_dir)

import drive_gain_calibration_experiment
import drive_gain_calibration_experiment_analysis
import impedance_fidelity_experiment
import impedance_fidelity_experiment_analysis
import schroeder_bandwidth_experiment
import schroeder_bandwidth_experiment_analysis
import zwidth_experiment
import zwidth_experiment_analysis


def calibrate(tmp, params=None):
    drive_gain_calibration_experiment.SIM_PARAMS = params
    try:
        run_dir = run_sim_experiment(drive_gain_calibration_experiment,
                                     tmp / 'data' / 'drive_gain_calibration' / '00000000-000000_sim')
        return run_analysis(drive_gain_calibration_experiment_analysis, run_dir)
    finally:
        drive_gain_calibration_experiment.SIM_PARAMS = None


class DriveGainCalibrationTest(unittest.TestCase):
    def test_recovers_the_drive_units(self):
        with scratch_dir() as tmp:
            summary = calibrate(tmp)
        # p4_sim's drive law: demand (per mille) = kv (kp kout/100 error - velocity),
        # positions in counts: both units are 1 there.
        self.assertLess(rel_err(summary['k_unit_permille_per_count'], 1.0), 0.02)
        self.assertLess(rel_err(summary['b_unit_permille_per_count_s'], 1.0), 0.02)
        self.assertTrue(summary['linear_in_gains'])
        self.assertTrue(summary['torque_offset_works'])
        self.assertEqual(summary['loop_delay_cycles'], 2)

    def test_detects_an_ignored_torque_offset(self):
        import p4_sim
        with scratch_dir() as tmp:
            summary = calibrate(tmp, p4_sim.SimParams(torque_offset_in_csp=False))
            self.assertFalse(summary['torque_offset_works'])
            self.assertLess(abs(summary['torque_offset_coefficient']), 0.1)
            # Experiments then leave the drive loop out instead of rendering wrongly.
            zwidth_experiment.QUICK = {**ZW_QUICK, 'modes': ('rt', 'drive')}
            try:
                run_dir = run_sim_experiment(zwidth_experiment, tmp / 'zw')
            finally:
                zwidth_experiment.QUICK = ZW_QUICK
            meta = json.loads((run_dir / 'zwidth_metadata.json').read_text())
        self.assertIn('torque offset', meta['drive_loop_skipped'])
        self.assertFalse(any(t['mode'] == 'drive' for t in meta['trials']))


ZW_QUICK = dict(zwidth_experiment.QUICK)


class DriveLoopRenderingTest(unittest.TestCase):
    def test_drive_loop_renders_stiffer_and_faster_than_the_pc_loop(self):
        with scratch_dir() as tmp:
            calibrate(tmp)
            zw_dir = run_sim_experiment(zwidth_experiment, tmp / 'zw')
            run_analysis(zwidth_experiment_analysis, zw_dir)
            zw = json.loads((zw_dir / 'zwidth_metadata.json').read_text())
            imp_dir = run_sim_experiment(impedance_fidelity_experiment, tmp / 'imp')
            imp = run_analysis(impedance_fidelity_experiment_analysis, imp_dir)
            imp_meta = json.loads((imp_dir / 'impedance_metadata.json').read_text())

        # Z-width: at the same damping the drive loop (no CAN round trip) holds a
        # much stiffer spring than the 500 Hz PC loop.
        boundary = {(b['mode'], b['virtual_b_nms_rad']): b for b in zw['boundary']}
        b = max(B for m, B in boundary if m == 'drive')
        drive_k = boundary[('drive', b)]['k_last_stable_nm_rad']
        pc_k = boundary[('rt', b)]['k_last_stable_nm_rad']
        self.assertGreater(drive_k, 4 * pc_k)

        # Impedance: the drive target renders its stiffness with no measurable delay.
        target = next(t['target_id'] for t in imp_meta['targets'] if t['mode'] == 'drive')
        self.assertLess(abs(imp[f't{target}_stiffness_error_percent']), 5.0)
        self.assertLess(abs(imp['impedance_delay_drive_ms']), 1.5)
        self.assertGreater(imp['impedance_delay_rt_ms'], 3.5)
        self.assertGreaterEqual(imp['impedance_fidelity_bandwidth_drive_hz'],
                                imp.get('impedance_fidelity_bandwidth_rt_hz', 0.0))

    def test_drive_loop_bandwidth_matches_its_model(self):
        schroeder_bandwidth_experiment.SIM_PARAMS = frictionless()
        quick = schroeder_bandwidth_experiment.QUICK
        schroeder_bandwidth_experiment.QUICK = {**quick, 'loop': 'drive'}
        try:
            with scratch_dir() as tmp:
                calibrate(tmp)
                run_dir = run_sim_experiment(schroeder_bandwidth_experiment, tmp / 'bw')
                meta = json.loads((run_dir / 'bandwidth_metadata.json').read_text())
                summary = run_analysis(schroeder_bandwidth_experiment_analysis, run_dir)
                response = pd.read_csv(run_dir / 'analysis' / 'frequency_response.csv')
        finally:
            schroeder_bandwidth_experiment.SIM_PARAMS = None
            schroeder_bandwidth_experiment.QUICK = quick
        truth = meta['sim_truth']
        J = truth['motor_inertia'] + truth['output_inertia']
        b = truth['viscous'] + truth['output_viscous']
        valid = response[response['line_valid'] & response['level_valid']]
        H = valid['H_real'].to_numpy() + 1j * valid['H_imag'].to_numpy()
        # The rigid joint under the drives' PD, the commands arriving 2 SYNC periods late.
        model = schroeder_bandwidth_experiment_analysis.model_response(
            valid['frequency_hz'].to_numpy(), J, b, truth['current_loop_time_constant'],
            summary['kp_joint_nm_rad'], summary['kd_joint_nms_rad'], 1.0, None,
            meta['loop_delay_s'])
        self.assertLess(np.max(np.abs(20 * np.log10(np.abs(H) / np.abs(model)))), 1.0)
        self.assertEqual(summary['loop'], 'drive')
        self.assertNotIn('plant_J_kg_m2', summary)   # identified in the PC loop instead
        self.assertTrue(math.isfinite(summary['fit_bandwidth_hz']))


if __name__ == '__main__':
    unittest.main()
