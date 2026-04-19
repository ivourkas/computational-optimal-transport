"""Visualization helpers for transport couplings and point-cloud figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from .plotting import apply_axis_style

Array = npt.NDArray[np.float64]

SOURCE_COLOR = "#1d3557"
TARGET_COLOR = "#e76f51"
EDGE_COLOR = "#4f5d75"


@dataclass(frozen=True)
class CouplingDisplayStats:
    """Summary of the transport edges shown in one coupling panel."""

    shown_edges: int
    shown_mass: float
    shown_mass_fraction: float
    threshold: float


def shared_point_cloud_limits(
    source_points: Array,
    target_points: Array,
    *,
    padding_fraction: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return common x/y limits for side-by-side point-cloud panels."""

    stacked = np.vstack([np.asarray(source_points, dtype=float), np.asarray(target_points, dtype=float)])
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)

    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = padding_fraction * max(x_span, 1e-8)
    y_pad = padding_fraction * max(y_span, 1e-8)

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def select_coupling_edges(
    plan: Array,
    *,
    min_mass: float = 1e-4,
    max_edges: int = 2_000,
) -> tuple[Array, Array, Array]:
    """Select transport edges that are visually worth drawing."""

    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError("plan must be a 2D array.")

    row_indices, col_indices = np.nonzero(plan >= min_mass)
    masses = plan[row_indices, col_indices]

    if masses.size == 0:
        flat_index = int(np.argmax(plan))
        row_indices = np.array([flat_index // plan.shape[1]], dtype=int)
        col_indices = np.array([flat_index % plan.shape[1]], dtype=int)
        masses = np.array([plan[row_indices[0], col_indices[0]]], dtype=float)

    if masses.size > max_edges:
        keep = np.argsort(masses)[-max_edges:]
        row_indices = row_indices[keep]
        col_indices = col_indices[keep]
        masses = masses[keep]

    draw_order = np.argsort(masses)
    return row_indices[draw_order], col_indices[draw_order], masses[draw_order]


def draw_coupling_panel(
    ax: Axes,
    *,
    source_points: Array,
    target_points: Array,
    plan: Array,
    title: str,
    transport_cost: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    min_mass: float = 1e-4,
    max_edges: int = 2_000,
) -> CouplingDisplayStats:
    """Draw a 2D coupling panel with weighted transport edges."""

    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    plan = np.asarray(plan, dtype=float)

    row_indices, col_indices, masses = select_coupling_edges(
        plan,
        min_mass=min_mass,
        max_edges=max_edges,
    )

    max_mass = float(masses.max()) if masses.size else 1.0
    for row_index, col_index, mass in zip(row_indices, col_indices, masses, strict=True):
        normalized_mass = np.sqrt(max(mass / max_mass, 1e-12))
        ax.plot(
            [source_points[row_index, 0], target_points[col_index, 0]],
            [source_points[row_index, 1], target_points[col_index, 1]],
            color=EDGE_COLOR,
            alpha=0.05 + 0.65 * normalized_mass,
            linewidth=0.35 + 2.2 * normalized_mass,
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        s=36,
        color=SOURCE_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Source",
        zorder=3,
    )
    ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        s=42,
        marker="s",
        color=TARGET_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Target",
        zorder=4,
    )

    shown_mass = float(masses.sum())
    shown_mass_fraction = shown_mass / float(plan.sum())
    annotation = (
        rf"$\langle C,P\rangle = {transport_cost:.4f}$" "\n"
        rf"shown edges: {masses.size}" "\n"
        rf"shown mass: {shown_mass_fraction:.1%}"
    )

    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.92},
    )

    ax.set_title(title)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    apply_axis_style(ax)

    return CouplingDisplayStats(
        shown_edges=int(masses.size),
        shown_mass=shown_mass,
        shown_mass_fraction=float(shown_mass_fraction),
        threshold=min_mass,
    )
