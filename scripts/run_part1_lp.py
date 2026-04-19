"""Run Part 1 LP experiments and produce report-ready outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ot_lab import (  # noqa: E402
    default_gaussian_specs,
    generate_ot_instance,
    solve_ot_lp,
)
from ot_lab.plotting import (  # noqa: E402
    METHOD_STYLES,
    apply_axis_style,
    apply_project_style,
    configure_log_x_ticks,
    configure_log_y_ticks,
    save_figure,
)

PART1_METHODS = ("simplex", "ipm", "pdlp")
SMALL_CASES = (3, 4)
MAIN_CASES = (50, 100, 200, 500)
SEEDS = {
    3: 11,
    4: 17,
    50: 1050,
    100: 1100,
    200: 1200,
    500: 1500,
}


def run_part1_experiments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the tiny verification cases and the main timing cases."""

    source_spec, target_spec = default_gaussian_specs()
    small_results = []
    main_results = []

    for n in SMALL_CASES + MAIN_CASES:
        instance = generate_ot_instance(
            n=n,
            source_spec=source_spec,
            target_spec=target_spec,
            seed=SEEDS[n],
        )
        for method in PART1_METHODS:
            print(f"Solving Part 1 LP for n={n}, method={method}...")
            solution = solve_ot_lp(instance, method=method)
            row = {
                "n": n,
                "seed": SEEDS[n],
                "method": method,
                "success": solution.success,
                "status": solution.status,
                "runtime_seconds": solution.runtime_seconds,
                "objective_value": solution.objective_value,
                "marginal_violation": solution.marginal_violation,
                "iterations": solution.iterations,
                "solver_message": solution.solver_message,
            }
            if n in SMALL_CASES:
                small_results.append(row)
            else:
                main_results.append(row)

    return pd.DataFrame(small_results), pd.DataFrame(main_results)


def add_objective_gap(results: pd.DataFrame) -> pd.DataFrame:
    """Add objective gaps relative to the simplex solution at each n."""

    simplex_baseline = (
        results.loc[results["method"] == "simplex", ["n", "objective_value"]]
        .rename(columns={"objective_value": "simplex_objective"})
        .copy()
    )
    merged = results.merge(simplex_baseline, on="n", how="left")
    merged["objective_gap_vs_simplex"] = merged["objective_value"] - merged["simplex_objective"]
    return merged


def save_tables(small_results: pd.DataFrame, main_results: pd.DataFrame, results_dir: Path) -> None:
    """Write raw and report-friendly tables for Part 1."""

    results_dir.mkdir(parents=True, exist_ok=True)
    method_order = list(PART1_METHODS)

    small_results.to_csv(results_dir / "part1_small_case_verification.csv", index=False)

    ordered_main = main_results.copy()
    ordered_main["method"] = pd.Categorical(ordered_main["method"], categories=method_order, ordered=True)
    report_table = (
        ordered_main[
            [
                "n",
                "method",
                "runtime_seconds",
                "objective_value",
                "objective_gap_vs_simplex",
                "marginal_violation",
                "iterations",
            ]
        ]
        .copy()
        .sort_values(["n", "method"])
        .rename(columns={"objective_value": "transport_cost"})
    )
    report_table.to_csv(results_dir / "part1b_results_table.csv", index=False)

    _write_markdown_table(
        report_table,
        results_dir / "part1b_results_table.md",
        title="Part 1(b): Timing and Accuracy Table",
        float_columns={
            "runtime_seconds": "{:.6f}",
            "transport_cost": "{:.8f}",
            "objective_gap_vs_simplex": "{:+.2e}",
            "marginal_violation": "{:.2e}",
        },
    )
    _write_markdown_table(
        small_results.assign(
            method=pd.Categorical(small_results["method"], categories=method_order, ordered=True)
        ).sort_values(["n", "method"]),
        results_dir / "part1_small_case_verification.md",
        title="Small-Case Verification (n = 3, 4)",
        float_columns={
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "marginal_violation": "{:.2e}",
        },
    )


def plot_part1_timing(main_results: pd.DataFrame, output_stem: Path) -> None:
    """Create the required log-log timing plot for Part 1(a)."""

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for method in PART1_METHODS:
        subset = main_results.loc[main_results["method"] == method].sort_values("n")
        style = METHOD_STYLES[method]
        ax.plot(
            subset["n"],
            subset["runtime_seconds"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
        )

    ax.set_title("Part 1(a): Exact OT LP Runtime by Solver Paradigm")
    ax.set_xlabel("Problem size n")
    ax.set_ylabel("Solve time (seconds)")
    configure_log_x_ticks(ax, list(MAIN_CASES))
    configure_log_y_ticks(ax)
    apply_axis_style(ax)
    ax.legend(loc="upper left")
    save_figure(fig, output_stem)
    plt.close(fig)


def _write_markdown_table(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
    float_columns: dict[str, str] | None = None,
) -> None:
    """Write a small markdown table without requiring extra dependencies."""

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
    results_dir = PROJECT_ROOT / "results" / "part1"

    small_results, main_results = run_part1_experiments()
    main_results = add_objective_gap(main_results)
    small_results = add_objective_gap(small_results)

    save_tables(small_results=small_results, main_results=main_results, results_dir=results_dir)
    plot_part1_timing(main_results, results_dir / "part1a_timing_loglog")

    print(f"Saved Part 1 outputs to: {results_dir}")


if __name__ == "__main__":
    main()
