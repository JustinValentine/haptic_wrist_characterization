"""Matplotlib styling and export for IEEE (RA-L) figures.

Design rules used by every figure in this repository:
* single-column width (3.5 in), 8 pt labels / 7 pt ticks in Times, vector
  PDF/EPS with embedded TrueType fonts plus a 600 dpi PNG preview;
* no in-plot titles (captions go in the paper) and never a second y-axis;
* categorical series use at most three validated colors (CVD-safe for every
  pair on white) and always pair the color with a marker and line style, so
  figures stay readable in grayscale print;
* ordered series (amplitudes, gains) use a single-hue light-to-dark ramp;
* measured data in ink, identified models dashed, thresholds muted;
* solid hairline grid, axes and ticks one step quieter than the data.
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path('figures') / 'ieee'
PNG_DPI = 600
SINGLE_COLUMN_WIDTH_IN = 3.5
DOUBLE_COLUMN_WIDTH_IN = 7.16
MAX_FIGURE_HEIGHT_IN = 8.8

INK = '#0b0b0b'           # Measured data, text
INK_SECONDARY = '#52514e' # Axes, ticks
MUTED = '#898781'         # Thresholds, reference lines, raw traces in the background
GRID = '#e1e0d9'          # Hairline grid
BASELINE = '#c3c2b7'      # Zero baselines
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')   # Categorical slots 1-3 (fixed order)
MARKERS = ('o', 's', '^')
LINESTYLES = ('-', '--', '-.')
ORDINAL_RAMP = {           # Single-hue blue ramp, light to dark
    2: ('#86b6ef', '#1c5cab'),
    3: ('#86b6ef', '#2a78d6', '#104281'),
    4: ('#86b6ef', '#3987e5', '#1c5cab', '#0d366b'),
    5: ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281'),
}
TORQUE_UNIT = 'N·m'


def configure_ieee_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'text.color': INK,
        'axes.labelcolor': INK,
        'axes.edgecolor': INK_SECONDARY,
        'axes.linewidth': 0.6,
        'axes.prop_cycle': plt.cycler(color=list(SERIES)),
        'xtick.color': INK_SECONDARY,
        'ytick.color': INK_SECONDARY,
        'xtick.labelcolor': INK,
        'ytick.labelcolor': INK,
        'grid.color': GRID,
        'grid.linestyle': '-',
        'grid.linewidth': 0.4,
        'lines.linewidth': 1.0,
        'lines.markersize': 3.5,
        'lines.markeredgewidth': 0.7,
        'patch.linewidth': 0.6,
        'errorbar.capsize': 1.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 1.6,
        'ytick.minor.size': 1.6,
        'legend.frameon': False,
        'legend.handlelength': 2.2,
        'legend.borderaxespad': 0.3,
        'figure.dpi': 150,
        'savefig.dpi': PNG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def figure_size(width='single', height=2.25):
    if width == 'double':
        figure_width = DOUBLE_COLUMN_WIDTH_IN
    elif width == 'single':
        figure_width = SINGLE_COLUMN_WIDTH_IN
    else:
        figure_width = float(width)
    return figure_width, min(float(height), MAX_FIGURE_HEIGHT_IN)


def series_style(i, line=True, marker=True, filled=False):
    """Color + marker + line style for categorical slot i (0-2)."""
    if i >= len(SERIES):
        raise ValueError('More than three categorical series: facet the figure instead.')
    style = {'color': SERIES[i]}
    style['linestyle'] = LINESTYLES[i] if line else 'None'
    if marker:
        style['marker'] = MARKERS[i]
        style['markerfacecolor'] = SERIES[i] if filled else 'white'
        style['markeredgecolor'] = SERIES[i]
    return style


def ordinal_colors(n):
    """Light-to-dark single-hue colors for n ordered series (1-5)."""
    if n == 1:
        return ['#2a78d6']
    if n not in ORDINAL_RAMP:
        raise ValueError('Ordinal ramp supports 1-5 levels; facet larger sets.')
    return list(ORDINAL_RAMP[n])


def style_axis(ax, grid=True, minor=True):
    ax.tick_params(direction='in', which='both', top=True, right=True)
    if minor:
        ax.minorticks_on()
    if grid:
        ax.grid(True, which='major')
        ax.set_axisbelow(True)
    return ax


def add_panel_label(ax, label, x=-0.02, y=1.0):
    """Bold (a)/(b) label just outside the top-left corner, clear of the data."""
    ax.text(x, y, label, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, fontweight='bold', color=INK)


def reference_line(ax, value, orientation='h', label=None, **kwargs):
    style = dict(color=MUTED, linestyle='--', linewidth=0.8, zorder=1)
    style.update(kwargs)
    if orientation == 'h':
        return ax.axhline(value, label=label, **style)
    return ax.axvline(value, label=label, **style)


def zero_line(ax, orientation='h'):
    """Solid hairline baseline at zero (dashes are reserved for thresholds/models)."""
    style = dict(color=BASELINE, linestyle='-', linewidth=0.6, zorder=1)
    return ax.axhline(0.0, **style) if orientation == 'h' else ax.axvline(0.0, **style)


def pad_top_tick(ax, fraction=0.14):
    """Keep the highest y tick clear of the top edge (and of an outside panel label)."""
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    if ticks and hi - max(ticks) < fraction * (hi - lo):
        ax.set_ylim(lo, max(ticks) + fraction * (hi - lo))


def direct_label(ax, x, y, text, dx=3, dy=0, ha='left', va='center', **kwargs):
    """Small label next to a mark, offset in points, in ink (never series color)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va, fontsize=7, color=INK, **kwargs)


def save_ieee_figure(fig, basename, metadata=None, formats=('pdf', 'eps', 'png'),
                     out_dirs=None):
    """Save fig to each directory in out_dirs (default figures/ieee)."""
    out_dirs = [Path(d) for d in (out_dirs or [FIGURE_DIR]) if d]
    paths = {}
    for i, out_dir in enumerate(out_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = out_dir / f'{basename}.{fmt}'
            if i == 0:
                fig.savefig(path, dpi=PNG_DPI if fmt == 'png' else None)
                first = path
            else:
                shutil.copyfile(out_dirs[0] / f'{basename}.{fmt}', path)
            paths[f'{fmt}' if i == 0 else f'{fmt}_{i}'] = str(path)
        if metadata is not None:
            metadata_path = out_dir / f'{basename}.json'
            with metadata_path.open('w', encoding='utf8') as f:
                json.dump(metadata, f, indent=2, sort_keys=True, default=_json_default)
                f.write('\n')
            paths['metadata' if i == 0 else f'metadata_{i}'] = str(metadata_path)
    return paths


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8') as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write('\n')


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def log_ticks_hz(ax, axis='x'):
    """Plain-number tick labels (0.1, 1, 10) instead of 10^n on log axes."""
    from matplotlib.ticker import FuncFormatter

    def fmt(value, _):
        if value <= 0:
            return ''
        return f'{value:g}'

    target = ax.xaxis if axis == 'x' else ax.yaxis
    target.set_major_formatter(FuncFormatter(fmt))


def nice_limits(values, pad=0.08, include_zero=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = values.min(), values.max()
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span
