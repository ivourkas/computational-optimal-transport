"""Optional Part 1 extra-credit experiments for the custom NumPy PDHG solver."""

from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ot_lab import (  # noqa: E402
    OTPDHGConfig,
    default_gaussian_specs,
    generate_ot_instance,
    recommended_ot_pdhg_config,
    solve_ot_lp_pdhg_numpy,
    solve_ot_lp_pdlp,
)
from ot_lab.plotting import (  # noqa: E402
    METHOD_STYLES,
    apply_axis_style,
    apply_project_style,
    configure_log_x_ticks,
    configure_log_y_ticks,
    save_figure,
)

MAIN_CASES = (50, 100, 200, 500)
SEEDS = {
    50: 1050,
    100: 1100,
    200: 1200,
    500: 1500,
}
# We tune the custom PDHG solver on one representative medium-size case with m = n = 100.
STEP_SWEEP_SIZE = 100
STEP_MULTIPLIERS = (0.8, 1.0, 1.13, 1.25, 1.4, 1.6)
PRIMAL_WEIGHTS = (1.0, 2.0, 4.0, 8.0)


def run_optional_pdhg_comparison() -> pd.DataFrame:
    """Compare the tuned NumPy PDHG solver against OR-Tools PDLP."""

    source_spec, target_spec = default_gaussian_specs()
    rows = []

    for n in MAIN_CASES:
        instance = generate_ot_instance(
            n=n,
            source_spec=source_spec,
            target_spec=target_spec,
            seed=SEEDS[n],
        )
        print(f"Running optional PDHG comparison for n={n}...")
        pdlp_solution = solve_ot_lp_pdlp(instance)
        pdhg_solution = solve_ot_lp_pdhg_numpy(
            instance,
            config=recommended_ot_pdhg_config(
                n_source=n,
                n_target=n,
                max_iterations=30_000,
                tolerance=1e-6,
            ),
        )

        for solver_name, solution in (("pdlp", pdlp_solution), ("pdhg_numpy", pdhg_solution)):
            rows.append(
                {
                    "n": n,
                    "seed": SEEDS[n],
                    "solver": solver_name,
                    "success": solution.success,
                    "status": solution.status,
                    "runtime_seconds": solution.runtime_seconds,
                    "objective_value": solution.objective_value,
                    "marginal_violation": solution.marginal_violation,
                    "iterations": solution.iterations,
                }
            )

    results = pd.DataFrame(rows)
    pdlp_baseline = (
        results.loc[results["solver"] == "pdlp", ["n", "objective_value"]]
        .rename(columns={"objective_value": "pdlp_objective"})
        .copy()
    )
    results = results.merge(pdlp_baseline, on="n", how="left")
    results["objective_gap_vs_pdlp"] = results["objective_value"] - results["pdlp_objective"]
    return results


def run_step_sweep() -> pd.DataFrame:
    """Study how step size and primal weight affect the custom PDHG solver."""

    source_spec, target_spec = default_gaussian_specs()
    instance = generate_ot_instance(
        n=STEP_SWEEP_SIZE,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=SEEDS[STEP_SWEEP_SIZE],
    )
    pdlp_solution = solve_ot_lp_pdlp(instance)
    rows = []

    for primal_weight in PRIMAL_WEIGHTS:
        for multiplier in STEP_MULTIPLIERS:
            print(
                "Running PDHG step sweep for "
                f"n={STEP_SWEEP_SIZE}, primal_weight={primal_weight}, multiplier={multiplier}..."
            )
            config = OTPDHGConfig(
                max_iterations=12_000,
                tolerance=1e-6,
                primal_weight=primal_weight,
                step_size=multiplier / sqrt(2 * STEP_SWEEP_SIZE),
            )
            solution = solve_ot_lp_pdhg_numpy(instance, config=config)
            rows.append(
                {
                    "n": STEP_SWEEP_SIZE,
                    "primal_weight": primal_weight,
                    "step_multiplier": multiplier,
                    "success": solution.success,
                    "status": solution.status,
                    "runtime_seconds": solution.runtime_seconds,
                    "objective_value": solution.objective_value,
                    "objective_gap_vs_pdlp": solution.objective_value - pdlp_solution.objective_value,
                    "marginal_violation": solution.marginal_violation,
                    "iterations": solution.iterations,
                }
            )

    return pd.DataFrame(rows)


def save_optional_tables(comparison_results: pd.DataFrame, sweep_results: pd.DataFrame, results_dir: Path) -> None:
    """Save optional PDHG tables in CSV and Markdown form."""

    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_results.to_csv(results_dir / "optional_pdhg_vs_pdlp.csv", index=False)
    sweep_results.to_csv(results_dir / "optional_pdhg_step_sweep.csv", index=False)

    _write_markdown_table(
        comparison_results.sort_values(["n", "solver"]),
        results_dir / "optional_pdhg_vs_pdlp.md",
        title="Optional PDHG vs PDLP Comparison",
        float_columns={
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "objective_gap_vs_pdlp": "{:+.2e}",
            "marginal_violation": "{:.2e}",
        },
    )
    _write_markdown_table(
        sweep_results.sort_values(["primal_weight", "step_multiplier"]),
        results_dir / "optional_pdhg_step_sweep.md",
        title="Optional PDHG Step-Size Sweep",
        float_columns={
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "objective_gap_vs_pdlp": "{:+.2e}",
            "marginal_violation": "{:.2e}",
        },
    )


def plot_optional_comparison(comparison_results: pd.DataFrame, output_stem: Path) -> None:
    """Plot custom PDHG against PDLP with emphasis on quality and competitiveness."""

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_time, ax_violation = axes

    for solver_name in ("pdlp", "pdhg_numpy"):
        subset = comparison_results.loc[comparison_results["solver"] == solver_name].sort_values("n")
        style = METHOD_STYLES["pdlp" if solver_name == "pdlp" else "pdhg_numpy"]
        label = "PDLP (OR-Tools)" if solver_name == "pdlp" else "PDHG (NumPy, custom)"

        ax_time.plot(
            subset["n"],
            subset["runtime_seconds"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=label,
        )
        ax_violation.plot(
            subset["n"],
            subset["marginal_violation"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=label,
        )

        failed = subset.loc[~subset["success"]]
        if not failed.empty:
            ax_time.scatter(
                failed["n"],
                failed["runtime_seconds"],
                facecolors="none",
                edgecolors=style["color"],
                s=110,
                linewidths=2.0,
                zorder=5,
            )
            ax_violation.scatter(
                failed["n"],
                failed["marginal_violation"],
                facecolors="none",
                edgecolors=style["color"],
                s=110,
                linewidths=2.0,
                zorder=5,
            )

    ax_time.set_title("Runtime")
    ax_time.set_xlabel("Problem size n")
    ax_time.set_ylabel("Solve time (seconds)")
    configure_log_x_ticks(ax_time, list(MAIN_CASES))
    configure_log_y_ticks(ax_time)
    apply_axis_style(ax_time)

    ax_violation.set_title("Marginal Violation")
    ax_violation.set_xlabel("Problem size n")
    ax_violation.set_ylabel(r"$\|P1-a\|_1 + \|P^\top 1-b\|_1$")
    configure_log_x_ticks(ax_violation, list(MAIN_CASES))
    configure_log_y_ticks(ax_violation)
    apply_axis_style(ax_violation)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Optional PDHG Extra Credit: Custom PDHG vs PDLP", y=1.12, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_step_sweep(sweep_results: pd.DataFrame, output_stem: Path) -> None:
    """Plot what works and what does not for the custom PDHG step-size sweep."""

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_gap, ax_iters = axes

    color_cycle = ["#1d3557", "#2a9d8f", "#e76f51", "#6d597a"]
    for color, primal_weight in zip(color_cycle, PRIMAL_WEIGHTS, strict=True):
        subset = sweep_results.loc[sweep_results["primal_weight"] == primal_weight].sort_values("step_multiplier")
        label = rf"$\omega={primal_weight:g}$"
        ax_gap.plot(
            subset["step_multiplier"],
            subset["objective_gap_vs_pdlp"].abs().clip(lower=1e-12),
            color=color,
            marker="o",
            linestyle="-",
            label=label,
        )
        ax_iters.plot(
            subset["step_multiplier"],
            subset["iterations"],
            color=color,
            marker="o",
            linestyle="-",
            label=label,
        )

        failed = subset.loc[~subset["success"]]
        if not failed.empty:
            ax_gap.scatter(
                failed["step_multiplier"],
                failed["objective_gap_vs_pdlp"].abs().clip(lower=1e-12),
                facecolors="none",
                edgecolors=color,
                s=95,
                linewidths=2.0,
                zorder=5,
            )
            ax_iters.scatter(
                failed["step_multiplier"],
                failed["iterations"],
                facecolors="none",
                edgecolors=color,
                s=95,
                linewidths=2.0,
                zorder=5,
            )

    ax_gap.set_title(f"Objective Gap vs Step Multiplier (fixed m=n={STEP_SWEEP_SIZE})")
    ax_gap.set_xlabel(r"Step multiplier $\alpha$ in $\eta = \alpha / \sqrt{m+n}$")
    ax_gap.set_ylabel(r"$|\langle C,P\rangle - \langle C,P_{\mathrm{PDLP}}\rangle|$")
    configure_log_y_ticks(ax_gap)
    apply_axis_style(ax_gap)

    ax_iters.set_title(f"Iterations vs Step Multiplier (fixed m=n={STEP_SWEEP_SIZE})")
    ax_iters.set_xlabel(r"Step multiplier $\alpha$ in $\eta = \alpha / \sqrt{m+n}$")
    ax_iters.set_ylabel("Iterations")
    configure_log_y_ticks(ax_iters)
    apply_axis_style(ax_iters)

    handles, labels = ax_gap.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Optional PDHG Extra Credit: Step-Size and Primal-Weight Sweep", y=1.12, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def _write_markdown_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    float_columns: dict[str, str] | None = None,
) -> None:
    """Write a small markdown table without extra dependencies."""

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
    apply_project_style()
    results_dir = PROJECT_ROOT / "results" / "part1_optional_pdhg"

    comparison_results = run_optional_pdhg_comparison()
    sweep_results = run_step_sweep()

    save_optional_tables(comparison_results, sweep_results, results_dir)
    plot_optional_comparison(comparison_results, results_dir / "optional_pdhg_vs_pdlp")
    plot_step_sweep(sweep_results, results_dir / "optional_pdhg_step_sweep")

    print(f"Saved optional PDHG outputs to: {results_dir}")


if __name__ == "__main__":
    main()
