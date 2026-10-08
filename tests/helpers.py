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
    import moteus_sim
    return moteus_sim.SimParams(cogging_amplitude_motor=0.0, **overrides)


def run_cogging_sweep(tmp):
    """Quick friction sweep + analysis where later analyses look up the cogging model."""
    import friction_sweep_experiment
    import friction_sweep_experiment_analysis
    run_dir = run_sim_experiment(friction_sweep_experiment,
                                 Path(tmp) / 'data' / 'friction_sweep' / '00000000-000000_sim')
    return run_dir, run_analysis(friction_sweep_experiment_analysis, run_dir)
