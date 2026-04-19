"""Shared plotting style for the optimal transport project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

METHOD_STYLES: dict[str, dict[str, str | float]] = {
    "simplex": {
        "label": "Simplex (HiGHS-DS)",
        "color": "#1d3557",
        "marker": "o",
        "linestyle": "-",
    },
    "ipm": {
        "label": "IPM (HiGHS-IPM)",
        "color": "#2a9d8f",
        "marker": "s",
        "linestyle": "-",
    },
    "pdlp": {
        "label": "PDLP (OR-Tools)",
        "color": "#e76f51",
        "marker": "D",
        "linestyle": "-",
    },
    "pdhg_numpy": {
        "label": "PDHG (NumPy)",
        "color": "#6c757d",
        "marker": "^",
        "linestyle": "--",
    },
    "admm": {
        "label": "ADMM (custom)",
        "color": "#6d597a",
        "marker": "o",
        "linestyle": "-",
    },
    "quadratic_ipm": {
        "label": "IPM (Clarabel)",
        "color": "#264653",
        "marker": "s",
        "linestyle": "-",
    },
    "sinkhorn": {
        "label": "Sinkhorn (POT)",
        "color": "#2a9d8f",
        "marker": "D",
        "linestyle": "-",
    },
    "sinkhorn_numpy": {
        "label": "Sinkhorn (NumPy)",
        "color": "#6d597a",
        "marker": "^",
        "linestyle": "--",
    },
    "entropic_ipm": {
        "label": "Conic baseline",
        "color": "#264653",
        "marker": "s",
        "linestyle": "-",
    },
}


def apply_project_style() -> None:
    """Apply a consistent, report-ready plotting style."""

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.color": "#cfd8dc",
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )


def apply_axis_style(ax: Axes) -> None:
    """Make one axis match the project-wide visual style."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.15)


def configure_log_x_ticks(ax: Axes, x_values: list[int]) -> None:
    """Show exact problem sizes on a log-scale x-axis."""

    ax.set_xscale("log", base=10)
    ax.set_xticks(x_values)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())


def configure_log_y_ticks(ax: Axes) -> None:
    """Use log scale on the y-axis with clean formatting."""

    ax.set_yscale("log", base=10)


def save_figure(fig: Figure, output_stem: Path) -> None:
    """Save a figure to both PNG and PDF."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
