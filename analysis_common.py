"""Shared helpers for the analysis scripts: loading runs, signal processing,
statistics, and exporting results and figures."""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from experiment_common import DATA_ROOT, latest_summary, write_json
from ieee_plot import FIGURE_DIR, save_ieee_figure


@dataclass
class Run:
    df: pd.DataFrame
    meta: dict
    out_dir: Path
    source: Path
    legacy: bool

    @property
    def simulated(self):
        return bool(self.meta.get('simulated'))

    def meta_value(self, key, default=None, column=None):
        """Value from metadata, else the first non-null entry of a data column."""
        if key in self.meta and self.meta[key] is not None:
            return self.meta[key]
        config = self.meta.get('experiment_config') or {}
        if key in config and config[key] is not None:
            return config[key]
        column = column or key
        if column in self.df.columns and not self.df[column].dropna().empty:
            return self.df[column].dropna().iloc[0]
        return default


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('data', nargs='?',
                        help='run directory or data CSV (default: latest run of this experiment)')
    parser.add_argument('--no-show', action='store_true', help='do not open figure windows')
    parser.add_argument('--paper-dir', default=str(FIGURE_DIR),
                        help='also export figures here; pass "" to skip')
    return parser


def latest_run_dir(experiment):
    root = DATA_ROOT / experiment
    if not root.is_dir():
        return None
    runs = [d for d in root.iterdir() if d.is_dir() and (d / f'{experiment}_data.csv').exists()]
    return max(runs, key=lambda d: d.name) if runs else None


def load_run(experiment, path=None, legacy_names=()):
    """Load data + metadata from a run directory, a CSV, or the latest run.

    legacy_names: (csv_name, metadata_name) pairs from older versions of the
    scripts that wrote fixed file names into the working directory.
    """
    legacy = False
    if path is None:
        run_dir = latest_run_dir(experiment)
        if run_dir is not None:
            path = run_dir
        else:
            for csv_name, _ in legacy_names:
                if Path(csv_name).exists():
                    path = Path(csv_name)
                    break
        if path is None:
            raise SystemExit(f'No {experiment} data found under {DATA_ROOT / experiment}. '
                             'Run the experiment first or pass a path.')
    path = Path(path)
    if path.is_dir():
        csv = path / f'{experiment}_data.csv'
        meta_candidates = [path / f'{experiment}_metadata.json']
        out_dir = path / 'analysis'
    else:
        csv = path
        meta_candidates = [csv.with_name(csv.name.replace('_data.csv', '_metadata.json'))]
        meta_candidates += [csv.with_name(m) for _, m in legacy_names]
        out_dir = csv.parent / f'{csv.stem}_analysis'
    if not csv.exists():
        raise SystemExit(f'Data file not found: {csv}')
    meta = {}
    for candidate in meta_candidates:
        if candidate.exists():
            meta = json.loads(candidate.read_text())
            break
    df = pd.read_csv(csv)
    if 'experiment' not in meta or 'device_config' not in meta:
        legacy = True
    out_dir.mkdir(parents=True, exist_ok=True)
    return Run(df=df, meta=meta, out_dir=out_dir, source=csv, legacy=legacy)


def sim_truth(run):
    return run.meta.get('sim_truth') if run.simulated else None


def print_header(title):
    print('-' * 60)
    print(title)
    print('-' * 60)


def print_truth_comparison(rows):
    """rows: list of (label, estimate, truth, unit). Prints % error."""
    print('Simulated run: estimate vs ground truth')
    for label, estimate, truth, unit in rows:
        if truth is None or estimate is None or not np.isfinite(estimate):
            print(f'  {label:<28} {estimate!s:>12} (truth {truth})')
            continue
        err = 100.0 * (estimate - truth) / truth if truth else math.nan
        print(f'  {label:<28} {estimate:12.5g} truth {truth:10.5g} {unit:<10} ({err:+.1f}%)')


def export_figure(fig, run, basename, metadata, paper_dir=str(FIGURE_DIR)):
    out_dirs = [run.out_dir / 'figures']
    if paper_dir:
        out_dirs.append(Path(paper_dir))
    return save_ieee_figure(fig, basename, metadata=metadata, out_dirs=out_dirs)


def save_outputs(run, tables=None, summary=None, metadata=None, prefix=''):
    paths = {}
    for name, table in (tables or {}).items():
        path = run.out_dir / f'{prefix}{name}.csv'
        table.to_csv(path, index=False)
        paths[name] = str(path)
    if summary is not None:
        path = run.out_dir / f'{prefix}summary.csv'
        pd.DataFrame([summary]).to_csv(path, index=False)
        write_json(run.out_dir / f'{prefix}summary.json', summary)
        paths['summary'] = str(path)
    if metadata is not None:
        write_json(run.out_dir / f'{prefix}analysis_metadata.json', metadata)
    return paths


def loop_delays(run):
    """Delays of the p4wrist loop's PD terms (s): {'position': ..., 'velocity': ...}.

    The position term reaches the motor 2 SYNC periods after its sample (the CST
    ramp); the velocity term also waits half a period (differencing) and the
    first-order velocity filter. Zero for runs without these fields (moteus).
    """
    delay = float(run.meta.get('loop_delay_s') or 0.0)
    if not delay:
        return {'position': 0.0, 'velocity': 0.0}
    period = float((run.meta.get('p4wrist_config') or {}).get('period_s', delay / 2.0))
    fv = run.meta.get('velocity_filter_hz')
    filt = 1.0 / (2.0 * math.pi * float(fv)) if fv else 0.0
    return {'position': delay, 'velocity': delay + period / 2.0 + filt}


def loop_timing(run):
    """{'delay_s', 'period_s'} of the p4wrist loop that produced the run, else None."""
    if not run.meta.get('loop_delay_s'):
        return None
    return {'delay_s': float(run.meta['loop_delay_s']),
            'period_s': float((run.meta.get('p4wrist_config') or {}).get(
                'period_s', run.meta['loop_delay_s'] / 2.0))}


def pwl_fourier(t, y, freqs, t_start, t_end):
    """Fourier coefficients (2/(t_end - t_start)) * integral of y(t) exp(-j w t) over
    [t_start, t_end] for y piecewise linear through the points (t, y): exact for a
    P4's applied torque, which ramps to each cycle's target over a SYNC period."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = (t > t_start) & (t < t_end)
    tt = np.concatenate([[t_start], t[inside], [t_end]])
    yy = np.concatenate([[np.interp(t_start, t, y)], y[inside], [np.interp(t_end, t, y)]])
    a, b = tt[:-1], tt[1:]
    ya, yb = yy[:-1], yy[1:]
    h = b - a
    keep = h > 0
    a, b, ya, yb, h = a[keep], b[keep], ya[keep], yb[keep], h[keep]
    jw = 1j * 2.0 * np.pi * np.asarray(freqs, dtype=float)
    Ea = np.exp(-np.outer(a, jw))
    Eb = np.exp(-np.outer(b, jw))
    total = ((ya[:, None] * Ea - yb[:, None] * Eb) / jw
             + ((yb - ya) / h)[:, None] * (Ea - Eb) / jw ** 2)
    return (2.0 / (t_end - t_start)) * total.sum(axis=0)


def applied_torque_times(time, loop):
    """When each loop cycle's torque target is fully applied: the CST ramp ends
    loop_delay after the cycle (2 SYNC periods), in the cycle's own time base."""
    return np.asarray(time, dtype=float) + loop['delay_s']


def correct_for_loop(J_app, b_app, kp, kd, delays):
    """Plant J and b from those seen through a delayed PD loop.

    (kp + kd s) with delays Tp, Td adds, to second order in s,
    -kd Td + kp Tp^2/2 to the apparent inertia and kd - kp Tp to the damping."""
    Tp, Td = delays['position'], delays['velocity']
    return J_app + kd * Td - kp * Tp ** 2 / 2.0, b_app - kd + kp * Tp


# --- signal processing ------------------------------------------------------------
def uniform_grid(t, fs=None):
    t = np.asarray(t, dtype=float)
    if fs is None:
        fs = 1.0 / np.median(np.diff(t))
    return np.arange(t[0], t[-1], 1.0 / fs), fs


def resample_uniform(t, y, fs=None):
    tu, fs = uniform_grid(t, fs)
    return tu, np.interp(tu, t, y), fs


def savgol_derivative(t, y, window_s, polyorder=3, deriv=1):
    """Savitzky-Golay smoothing/derivative of irregularly sampled data.

    Resamples onto a uniform grid at the median rate, filters, and
    interpolates back to the original sample times.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tu, yu, fs = resample_uniform(t, y)
    window = max(polyorder + 2, int(round(window_s * fs)) | 1)
    if window >= len(yu):
        window = (len(yu) - 1) | 1
        if window <= polyorder:
            return np.full_like(y, np.nan)
    d = signal.savgol_filter(yu, window, polyorder, deriv=deriv, delta=1.0 / fs)
    return np.interp(t, tu, d)


def fit_sinusoids(t, y, freqs_hz, trend=True):
    """Least-squares complex amplitudes at given frequencies (any sampling).

    Returns Y such that y(t) ~= offset (+ slope t) + sum Re(Y_k exp(j 2 pi f_k t)),
    and the residual RMS. Exact for non-uniform timestamps.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    tc = t - t.mean()
    w = 2.0 * np.pi * np.outer(tc, freqs_hz)
    columns = [np.cos(w), np.sin(w), np.ones((len(t), 1))]
    if trend:
        columns.append(tc[:, None])
    A = np.hstack(columns)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    n = len(freqs_hz)
    Y = coef[:n] - 1j * coef[n:2 * n]
    # Rotate so the phase refers to t (not t - mean(t)).
    Y = Y * np.exp(-1j * 2.0 * np.pi * freqs_hz * t.mean())
    residual = y - A @ coef
    return Y, float(np.sqrt(np.mean(residual ** 2)))


def robust_sigma(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def bootstrap_ci(groups, statistic, n_boot=1000, alpha=0.05, seed=0, keys=None):
    """Percentile CIs from resampling whole groups (e.g. steps, trials).

    statistic(list_of_groups) returns a number, or a dict when keys is given;
    the result is then {key: (low, high)}.
    """
    rng = np.random.default_rng(seed)
    groups = list(groups)
    empty = (math.nan, math.nan)
    if len(groups) < 2:
        return {k: empty for k in keys} if keys else empty
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        try:
            value = statistic([groups[i] for i in idx])
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if value is not None:
            samples.append(value)

    def interval(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        if len(values) < 20:
            return empty
        return (float(np.percentile(values, 100 * alpha / 2)),
                float(np.percentile(values, 100 * (1 - alpha / 2))))

    if keys:
        return {k: interval([s.get(k) for s in samples if isinstance(s, dict)]) for k in keys}
    return interval(samples)


def mean_std_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': math.nan, 'std': math.nan, 'n': 0, 'ci95': math.nan}
    std = float(np.std(x, ddof=1)) if x.size > 1 else math.nan
    from scipy import stats
    half = (float(stats.t.ppf(0.975, x.size - 1) * std / math.sqrt(x.size))
            if x.size > 1 else math.nan)
    return {'mean': float(np.mean(x)), 'std': std, 'n': int(x.size), 'ci95': half}


def segments(df, phase=None):
    """Iterate (segment_id, frame) in time order, optionally for one phase."""
    data = df if phase is None else df[df['phase'] == phase]
    for seg_id, frame in data.groupby('segment', sort=True):
        yield seg_id, frame


def saturation_fraction(df, cap_column='max_torque_nm', torque_column='control_joint_torque_nm',
                        threshold=0.98):
    """Fraction of samples where the controller torque sits at the cap."""
    if cap_column not in df or torque_column not in df:
        return math.nan
    torque = df[torque_column].abs()
    cap = df[cap_column].abs()
    valid = torque.notna() & cap.notna() & (cap > 0)
    if not valid.any():
        return math.nan
    return float((torque[valid] >= threshold * cap[valid]).mean())


# --- cogging -------------------------------------------------------------------------
def cogging_torque(model, motor_rev):
    """Cogging torque (joint N m) that a cogging model predicts at absolute motor positions.

    model: {'periods_per_motor_rev': P, 'harmonics': [[k, a_k, b_k], ...]}, as saved by
    friction_sweep_experiment_analysis.py, so that
        torque = sum_k a_k cos(2 pi k P m) + b_k sin(2 pi k P m),  m in motor revolutions.
    The model is tied to the controller's motor position: re-run the friction sweep after
    re-zeroing the encoder.
    """
    m = np.asarray(motor_rev, dtype=float)
    out = np.zeros_like(m)
    for k, a, b in model['harmonics']:
        arg = 2.0 * math.pi * k * model['periods_per_motor_rev'] * m
        out += a * np.cos(arg) + b * np.sin(arg)
    return out


def latest_cogging_model(simulated):
    """Cogging model from the newest friction-sweep analysis of the same kind, or None."""
    found = latest_summary('friction_sweep', simulated)
    if found is None:
        return None
    run_dir, summary = found
    model = summary.get('cogging_model')
    if not summary.get('cogging_detected') or not model:
        return None
    return {**model, 'source': run_dir.name}
