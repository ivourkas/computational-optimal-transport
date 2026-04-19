"""Run Part 4 Gaussian-OT experiments and save report-ready outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ot_lab import (  # noqa: E402
    METHOD_STYLES,
    covariance_ellipse_patch,
    default_gaussian_specs,
    gaussian_w2_squared,
    generate_ot_instance,
    solve_ot_lp_ipm,
)
from ot_lab.plotting import apply_axis_style, apply_project_style, save_figure  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "part4"
SAMPLE_SIZES = (50, 100, 200, 500)
SEEDS = tuple(range(1200, 1210))
VISUALIZATION_N = 200
VISUALIZATION_SEED = 1200
SOURCE_COLOR = "#1d3557"
TARGET_COLOR = "#e76f51"
ARROW_COLOR = "#4f5d75"


def run_part4a_closed_form() -> tuple[pd.DataFrame, tuple]:
    """Compute the Bures-Wasserstein closed form for the chosen Gaussian pair."""

    source_spec, target_spec = default_gaussian_specs()
    closed_form = gaussian_w2_squared(source_spec, target_spec)
    result = pd.DataFrame(
        [
            {
                "dimension": source_spec.dimension,
                "mean_term": closed_form.mean_term,
                "covariance_term": closed_form.covariance_term,
                "w2_squared": closed_form.w2_squared,
            }
        ]
    )
    return result, (source_spec, target_spec)


def run_part4bc_trials(source_spec, target_spec) -> pd.DataFrame:
    """Solve the discrete OT problem across sample sizes and random seeds."""

    rows = []
    for n in SAMPLE_SIZES:
        for seed in SEEDS:
            print(f"Running Part 4(b,c) for n={n}, seed={seed}...")
            instance = generate_ot_instance(
                n=n,
                source_spec=source_spec,
                target_spec=target_spec,
                seed=seed,
            )
            solution = solve_ot_lp_ipm(instance)
            if not solution.success or solution.plan is None or solution.objective_value is None:
                raise RuntimeError(
                    f"LP IPM failed for n={n}, seed={seed}: {solution.status}"
                )

            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "method": "ipm",
                    "success": solution.success,
                    "status": solution.status,
                    "transport_cost": solution.objective_value,
                    "marginal_violation": solution.marginal_violation,
                    "runtime_seconds": solution.runtime_seconds,
                    "iterations": solution.iterations,
                }
            )

    return pd.DataFrame(rows)


def summarize_part4bc_trials(trial_results: pd.DataFrame, w2_squared: float) -> pd.DataFrame:
    """Aggregate the discrete OT costs into mean and standard deviation by n."""

    grouped = (
        trial_results.groupby("n", as_index=False)
        .agg(
            mean_transport_cost=("transport_cost", "mean"),
            std_transport_cost=("transport_cost", "std"),
            mean_marginal_violation=("marginal_violation", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values("n")
    )
    grouped["std_transport_cost"] = grouped["std_transport_cost"].fillna(0.0)
    grouped["w2_squared"] = float(w2_squared)
    grouped["mean_gap_vs_w2"] = grouped["mean_transport_cost"] - float(w2_squared)
    return grouped


def plot_part4bc_convergence(summary_results: pd.DataFrame, w2_squared: float, output_stem: Path) -> None:
    """Plot the discrete mean transport cost with one-standard-deviation bands."""

    style = METHOD_STYLES["ipm"]
    x_values = summary_results["n"].to_numpy(dtype=float)
    mean_costs = summary_results["mean_transport_cost"].to_numpy(dtype=float)
    std_costs = summary_results["std_transport_cost"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.fill_between(
        x_values,
        mean_costs - std_costs,
        mean_costs + std_costs,
        color=style["color"],
        alpha=0.18,
        linewidth=0.0,
        label=r"Mean $\pm$ one std. dev.",
    )
    ax.plot(
        x_values,
        mean_costs,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        label="Discrete OT mean",
    )
    ax.axhline(
        w2_squared,
        color="#222222",
        linestyle=":",
        linewidth=2.0,
        label=r"Closed-form $W_2^2$",
    )

    ax.set_title("Part 4(b,c): Discrete OT Convergence to Gaussian $W_2^2$")
    ax.set_xlabel("Sample size n")
    ax.set_ylabel(r"Transport cost $\langle C, P \rangle$")
    ax.set_xticks(list(SAMPLE_SIZES))
    apply_axis_style(ax)
    ax.legend(loc="best")
    save_figure(fig, output_stem)
    plt.close(fig)


def run_part4d_visualization(source_spec, target_spec) -> None:
    """Visualize matched source-target arrows on top of Gaussian contour ellipses."""

    instance = generate_ot_instance(
        n=VISUALIZATION_N,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=VISUALIZATION_SEED,
    )
    solution = solve_ot_lp_ipm(instance)
    if not solution.success or solution.plan is None or solution.objective_value is None:
        raise RuntimeError(f"LP IPM failed for the Part 4(d) instance: {solution.status}")

    plan = solution.plan
    matched_targets = np.argmax(plan, axis=1)
    matched_masses = plan[np.arange(instance.n_source), matched_targets]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for n_std, linestyle, alpha in ((2.0, "--", 0.8), (1.0, "-", 0.95)):
        ax.add_patch(
            covariance_ellipse_patch(
                instance.source_spec.mean,
                instance.source_spec.covariance,
                n_std=n_std,
                edgecolor=SOURCE_COLOR,
                linestyle=linestyle,
                alpha=alpha,
            )
        )
        ax.add_patch(
            covariance_ellipse_patch(
                instance.target_spec.mean,
                instance.target_spec.covariance,
                n_std=n_std,
                edgecolor=TARGET_COLOR,
                linestyle=linestyle,
                alpha=alpha,
            )
        )

    max_matched_mass = float(matched_masses.max())
    for row_index, col_index, matched_mass in zip(
        np.arange(instance.n_source),
        matched_targets,
        matched_masses,
        strict=True,
    ):
        normalized_mass = np.sqrt(max(matched_mass / max_matched_mass, 1e-12))
        start = instance.source_points[row_index]
        end = instance.target_points[col_index]
        delta = end - start
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            width=0.0030,
            head_width=0.065,
            head_length=0.085,
            length_includes_head=True,
            color=ARROW_COLOR,
            alpha=0.18 + 0.42 * normalized_mass,
            linewidth=0.2,
            zorder=1,
        )

    ax.scatter(
        instance.source_points[:, 0],
        instance.source_points[:, 1],
        s=30,
        color=SOURCE_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Source samples",
        zorder=3,
    )
    ax.scatter(
        instance.target_points[:, 0],
        instance.target_points[:, 1],
        s=36,
        marker="s",
        color=TARGET_COLOR,
        edgecolors="white",
        linewidths=0.4,
        label="Target samples",
        zorder=4,
    )

    x_values = np.hstack([instance.source_points[:, 0], instance.target_points[:, 0]])
    y_values = np.hstack([instance.source_points[:, 1], instance.target_points[:, 1]])
    x_pad = 0.08 * max(x_values.max() - x_values.min(), 1e-8)
    y_pad = 0.08 * max(y_values.max() - y_values.min(), 1e-8)
    ax.set_xlim(x_values.min() - x_pad, x_values.max() + x_pad)
    ax.set_ylim(y_values.min() - y_pad, y_values.max() + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Part 4(d): Discrete Matches over Gaussian Contours")
    ax.set_xlabel("x-coordinate")
    ax.set_ylabel("y-coordinate")
    apply_axis_style(ax)
    ax.legend(loc="upper left")
    save_figure(fig, RESULTS_DIR / "part4d_geometry")
    plt.close(fig)

def save_part4_tables(
    closed_form_results: pd.DataFrame,
    trial_results: pd.DataFrame,
    summary_results: pd.DataFrame,
) -> None:
    """Write the report-facing Part 4 tables."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    closed_form_results.to_csv(RESULTS_DIR / "part4a_closed_form.csv", index=False)
    trial_results.to_csv(RESULTS_DIR / "part4bc_discrete_trials.csv", index=False)
    summary_results.to_csv(RESULTS_DIR / "part4bc_summary.csv", index=False)

    _write_markdown_table(
        closed_form_results,
        RESULTS_DIR / "part4a_closed_form.md",
        title="Part 4(a): Closed-Form Gaussian Wasserstein Distance",
        float_columns={
            "mean_term": "{:.8f}",
            "covariance_term": "{:.8f}",
            "w2_squared": "{:.8f}",
        },
    )
    _write_markdown_table(
        summary_results,
        RESULTS_DIR / "part4bc_summary.md",
        title="Part 4(b,c): Discrete OT Convergence Summary",
        float_columns={
            "mean_transport_cost": "{:.8f}",
            "std_transport_cost": "{:.8f}",
            "mean_marginal_violation": "{:.2e}",
            "mean_runtime_seconds": "{:.6f}",
            "w2_squared": "{:.8f}",
            "mean_gap_vs_w2": "{:+.2e}",
        },
    )


def _write_markdown_table(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    float_columns: dict[str, str],
) -> None:
    """Write a compact markdown table with column-specific formatting."""

    formatted = frame.copy()
    for column_name, format_string in float_columns.items():
        if column_name in formatted.columns:
            formatted[column_name] = formatted[column_name].map(lambda value: format_string.format(value))

    header = "| " + " | ".join(formatted.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    row_lines = []
    for row in formatted.itertuples(index=False, name=None):
        row_lines.append("| " + " | ".join(str(value) for value in row) + " |")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write(header + "\n")
        handle.write(separator + "\n")
        handle.write("\n".join(row_lines))
        handle.write("\n")


def main() -> None:
    """Run the full Part 4 experiment suite."""

    apply_project_style()

    closed_form_results, (source_spec, target_spec) = run_part4a_closed_form()
    w2_squared = float(closed_form_results.loc[0, "w2_squared"])
    trial_results = run_part4bc_trials(source_spec, target_spec)
    summary_results = summarize_part4bc_trials(trial_results, w2_squared)
    run_part4d_visualization(source_spec, target_spec)

    save_part4_tables(
        closed_form_results,
        trial_results,
        summary_results,
    )
    plot_part4bc_convergence(summary_results, w2_squared, RESULTS_DIR / "part4bc_convergence")


if __name__ == "__main__":
    main()
