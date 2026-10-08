import asyncio
import json
import unittest
from unittest import mock

import pandas as pd

from tests.helpers import cogging_free, rel_err, run_analysis, run_sim_experiment, scratch_dir

import experiment_common
import torque_calibration_experiment as experiment
import torque_calibration_experiment_analysis as analysis


class StubSession:
    """Answers operator prompts from a script (manual force-gauge path)."""

    def __init__(self, replies, lever_arm_m=0.1):
        self.replies = list(replies)
        self.sensor = experiment_common.ManualSensor({'lever_arm_m': lever_arm_m})

    async def prompt_while_holding(self, message, command, phase='hold', info=None):
        return self.replies.pop(0)


class TorqueCalibrationTest(unittest.TestCase):
    def setUp(self):
        self._sim_params = experiment.SIM_PARAMS

    def tearDown(self):
        experiment.SIM_PARAMS = self._sim_params

    def test_recovers_torque_scale_factor_and_friction_hysteresis(self):
        experiment.SIM_PARAMS = cogging_free(torque_constant_ratio=0.93)
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(experiment, tmp / 'run')
            meta = json.loads((run_dir / 'torque_calibration_metadata.json').read_text())
            self.assertTrue(meta['completed'])
            summary = run_analysis(analysis, run_dir)
        self.assertLess(rel_err(summary['torque_scale_factor'], 0.93), 0.02)
        # With a locked output, friction between motor and cable shows up as a
        # loop about twice the breakaway torque wide.
        self.assertLess(rel_err(summary['hysteresis_at_zero_nm'], 2 * meta['sim_truth']['static']),
                        0.25)
        self.assertLess(summary['nonlinearity_percent_fs'], 1.0)

    def test_manual_readings_drive_the_fit(self):
        with scratch_dir() as tmp:
            run_dir = run_sim_experiment(experiment, tmp / 'run')
            data_path = run_dir / 'torque_calibration_data.csv'
            meta_path = run_dir / 'torque_calibration_metadata.json'
            df = pd.read_csv(data_path)
            meta = json.loads(meta_path.read_text())
            measure = df[df['phase'] == 'measure']
            meta['manual_readings'] = [
                {'level_id': int(level_id), 'torque_nm': float(g['ext_torque_nm'].mean())}
                for level_id, g in measure.groupby('level_id')]
            meta['sensor_mode'] = 'manual'
            df.drop(columns=['ext_torque_nm']).to_csv(data_path, index=False)
            meta_path.write_text(json.dumps(meta))
            summary = run_analysis(analysis, run_dir)
        self.assertEqual(summary['sensor_mode'], 'manual')
        self.assertLess(rel_err(summary['torque_scale_factor'], 1.0), 0.02)

    def test_manual_prompt_parses_retries_and_applies_command_sign(self):
        cfg = experiment.TorqueCalibrationConfig()
        level = {'level_id': 3, 'level_nm': -2.0, 'branch': 'down'}
        session = StubSession(['not a number', '15'])
        with mock.patch('builtins.print'):
            reading = asyncio.run(experiment.manual_reading(session, cfg, level, 9))
        self.assertAlmostEqual(reading['torque_nm'], -1.5)   # 15 N x 0.1 m, commanded sign

    def test_unlocked_output_aborts_at_lock_check(self):
        with scratch_dir() as tmp, mock.patch.object(
                experiment_common.MoteusSession, 'require_locked_output',
                lambda self, note='': None):
            with self.assertRaises(experiment_common.SafetyAbort):
                run_sim_experiment(experiment, tmp / 'run')
            meta = json.loads((tmp / 'run' / 'torque_calibration_metadata.json').read_text())
        self.assertFalse(meta['completed'])
        self.assertTrue(meta['abort_reason'])


if __name__ == '__main__':
    unittest.main()
