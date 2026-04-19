"""Create the Section 7 side-by-side coupling visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ot_lab import (  # noqa: E402
    OTSinkhornConfig,
    default_gaussian_specs,
    draw_coupling_panel,
    generate_ot_instance,
    shared_point_cloud_limits,
    solve_entropic_ot_sinkhorn_pot,
    solve_ot_lp_ipm,
    solve_quadratic_ot_ipm,
)
from ot_lab.plotting import apply_project_style, save_figure  # noqa: E402
from ot_lab.visualization import EDGE_COLOR, SOURCE_COLOR, TARGET_COLOR  # noqa: E402

VIS_N = 200
VIS_SEED = 1200
QUADRATIC_LAMBDA = 0.1
ENTROPIC_LAMBDAS = (1.0, 0.01)
DISPLAY_THRESHOLD = 1e-4
MAX_DRAWN_EDGES = 2_000


def build_visualization_instance():
    """Construct the shared point cloud used throughout the visualization section."""

    source_spec, target_spec = default_gaussian_specs()
    return generate_ot_instance(
        n=VIS_N,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=VIS_SEED,
    )


def _sinkhorn_config(regularization_strength: float) -> OTSinkhornConfig:
    """Choose a practical Sinkhorn configuration for the visualization panels."""

    if regularization_strength >= 0.5:
        return OTSinkhornConfig(max_iterations=5_000, tolerance=1e-6, record_every=10)
    return OTSinkhornConfig(max_iterations=50_000, tolerance=1e-6, record_every=25)


def run_visualization() -> pd.DataFrame:
    """Solve the four couplings required by Section 7 and render them side by side."""

    apply_project_style()
    results_dir = PROJECT_ROOT / "results" / "section7_visualization"
    results_dir.mkdir(parents=True, exist_ok=True)

    instance = build_visualization_instance()
    x_limits, y_limits = shared_point_cloud_limits(
        instance.source_points,
        instance.target_points,
    )

    lp_solution = solve_ot_lp_ipm(instance)
    quadratic_solution = solve_quadratic_ot_ipm(
        instance,
        QUADRATIC_LAMBDA,
        solver="clarabel",
    )
    sinkhorn_large = solve_entropic_ot_sinkhorn_pot(
        instance,
        ENTROPIC_LAMBDAS[0],
        config=_sinkhorn_config(ENTROPIC_LAMBDAS[0]),
    )
    sinkhorn_small = solve_entropic_ot_sinkhorn_pot(
        instance,
        ENTROPIC_LAMBDAS[1],
        config=_sinkhorn_config(ENTROPIC_LAMBDAS[1]),
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    panels = [
        (
            axes[0, 0],
            "LP coupling",
            lp_solution.plan,
            lp_solution.objective_value,
        ),
        (
            axes[0, 1],
            r"Quadratic coupling ($\lambda=0.1$)",
            quadratic_solution.plan,
            quadratic_solution.transport_cost,
        ),
        (
            axes[1, 0],
            r"Sinkhorn coupling ($\lambda=1$)",
            sinkhorn_large.plan,
            sinkhorn_large.transport_cost,
        ),
        (
            axes[1, 1],
            r"Sinkhorn coupling ($\lambda=0.01$)",
            sinkhorn_small.plan,
            sinkhorn_small.transport_cost,
        ),
    ]

    summary_rows = []
    for axis, title, plan, transport_cost in panels:
        if plan is None or transport_cost is None:
            raise RuntimeError(f"Missing plan for panel: {title}")

        stats = draw_coupling_panel(
            axis,
            source_points=instance.source_points,
            target_points=instance.target_points,
            plan=plan,
            title=title,
            transport_cost=float(transport_cost),
            x_limits=x_limits,
            y_limits=y_limits,
            min_mass=DISPLAY_THRESHOLD,
            max_edges=MAX_DRAWN_EDGES,
        )
        summary_rows.append(
            {
                "panel": title,
                "transport_cost": float(transport_cost),
                "shown_edges": stats.shown_edges,
                "shown_mass": stats.shown_mass,
                "shown_mass_fraction": stats.shown_mass_fraction,
                "display_threshold": stats.threshold,
            }
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLOR, markeredgecolor="white", markersize=8, label="Source points"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=TARGET_COLOR, markeredgecolor="white", markersize=8, label="Target points"),
        Line2D([0], [0], color=EDGE_COLOR, linewidth=2.0, label=rf"Transport edges with mass $\geq {DISPLAY_THRESHOLD}$"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        "Section 7: Coupling Visualization on a Shared n=200 Point Cloud",
        y=1.06,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, results_dir / "section7_coupling_panels")
    plt.close(figure)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "section7_coupling_summary.csv", index=False)
    _write_markdown_table(
        summary,
        results_dir / "section7_coupling_summary.md",
        title="Section 7 Visualization Summary",
        float_columns={
            "transport_cost": "{:.8f}",
            "shown_mass": "{:.6f}",
            "shown_mass_fraction": "{:.1%}",
            "display_threshold": "{:.0e}",
        },
    )

    return summary


def _write_markdown_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    float_columns: dict[str, str] | None = None,
) -> None:
    """Write a compact markdown table without extra dependencies."""

    float_columns = float_columns or {}
    table = dataframe.copy()

    for column, formatter in float_columns.items():
        if column in table.columns:
            table[column] = table[column].map(
                lambda value: formatter.format(value) if pd.notna(value) else ""
            )

    headers = list(table.columns)
    rows = [[str(value) for value in row] for row in table.to_numpy()]

    separator = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [
        f"# {title}",
        "",
        "|" + "|".join(headers) + "|",
        separator,
    ]
    lines.extend("|" + "|".join(row) + "|" for row in rows)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = run_visualization()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
