"""Run an experiment against the simulator and analyze it, in a temp dir."""
import asyncio
import contextlib
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment_common import make_arg_parser  # noqa: E402


@contextlib.contextmanager
def scratch_dir():
    """Temporary working directory so tests never see or touch real data."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(old)


def run_sim_experiment(module, out_dir, extra=()):
    args = make_arg_parser('test').parse_args(
        ['--sim', '--quick', '--yes', '--out', str(out_dir), *extra])
    asyncio.run(module.main(args))
    return Path(out_dir)


def run_analysis(module, run_dir):
    return module.main([str(run_dir), '--no-show', '--paper-dir', ''])


def rel_err(estimate, truth):
    return abs(estimate - truth) / abs(truth)


def cogging_free(**overrides):
    """Simulator parameters without cogging, for tests of estimators that should be
    checked on their own; tests/test_cogging_pipeline.py covers the cogging path."""
    import p4_sim
    return p4_sim.SimParams(cogging_amplitude_motor=0.0, **overrides)


def run_cogging_sweep(tmp):
    """Quick friction sweep + analysis where later analyses look up the cogging model."""
    import friction_sweep_experiment
    import friction_sweep_experiment_analysis
    run_dir = run_sim_experiment(friction_sweep_experiment,
                                 Path(tmp) / 'data' / 'friction_sweep' / '00000000-000000_sim')
    return run_dir, run_analysis(friction_sweep_experiment_analysis, run_dir)


def frictionless(**overrides):
    """Simulator parameters without Coulomb/static friction or cogging, for checks of
    estimators against linear models (viscous damping remains)."""
    import p4_sim
    return p4_sim.SimParams(cogging_amplitude_motor=0.0, motor_coulomb=0.0, motor_static=0.0,
                            **overrides)


def loop_velocity_path(meta):
    """The p4wrist loop's velocity estimate, as schroeder_bandwidth's model_response takes it."""
    return {'lag_s': meta['p4wrist_config']['period_s'] / 2.0,
            'filter_hz': meta['velocity_filter_hz']}


def p4_loop_step(J, b, kp, kd, step, period=0.002, filter_hz=100.0, duration=0.6,
                 substeps=40):
    """Rigid joint under the p4wrist loop: positions sampled each period, PD on a
    filtered differenced velocity, torque reached by a linear ramp over the period
    after the next SYNC (the CST scheme). Returns (t, q)."""
    import numpy as np
    h = period / substeps
    alpha = 1.0 - math.exp(-2.0 * math.pi * filter_hz * period)
    q = w = 0.0
    q_prev = v = 0.0
    u_prev2 = u_prev = 0.0     # targets latched at the last two SYNCs
    ts, qs = [], []
    for k in range(int(round(duration / period))):
        raw = (q - q_prev) / period if k else 0.0
        v += alpha * (raw - v)
        q_prev = q
        u = kp * (step - q) - kd * v
        ts.append(k * period)
        qs.append(q)
        # Over this period the drive ramps between the two targets latched before.
        for s in range(substeps):
            tau = u_prev2 + (u_prev - u_prev2) * (s + 1) / substeps
            w += h * (tau - b * w) / J
            q += h * w
        u_prev2, u_prev = u_prev, u
    return np.array(ts), np.array(qs)


import math  # noqa: E402  (used by the helpers above)
