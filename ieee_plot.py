import json
from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_DIR = Path("figures") / "ieee"
PNG_DPI = 600
SINGLE_COLUMN_WIDTH_IN = 3.5
DOUBLE_COLUMN_WIDTH_IN = 7.16
MAX_FIGURE_HEIGHT_IN = 8.8


def configure_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.35,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": PNG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def figure_size(width="single", height=2.25):
    if width == "double":
        figure_width = DOUBLE_COLUMN_WIDTH_IN
    elif width == "single":
        figure_width = SINGLE_COLUMN_WIDTH_IN
    else:
        figure_width = float(width)

    return figure_width, min(float(height), MAX_FIGURE_HEIGHT_IN)


def style_axis(ax, grid=True):
    ax.tick_params(direction="in", top=True, right=True)
    if grid:
        ax.grid(True, which="major", color="0.86", linestyle="-")
        ax.grid(True, which="minor", color="0.92", linestyle=":")
    ax.minorticks_on()
    return ax


def add_panel_label(ax, label):
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )


def save_ieee_figure(fig, basename, metadata=None, formats=("pdf", "eps", "png")):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    for fmt in formats:
        path = FIGURE_DIR / f"{basename}.{fmt}"
        if fmt == "png":
            fig.savefig(path, dpi=PNG_DPI)
        else:
            fig.savefig(path)
        paths[fmt] = str(path)

    if metadata is not None:
        metadata_path = FIGURE_DIR / f"{basename}.json"
        with metadata_path.open("w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True, default=_json_default)
            f.write("\n")
        paths["metadata"] = str(metadata_path)

    return paths


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
