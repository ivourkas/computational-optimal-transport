"""Run Part 3 entropic-regularization experiments and save report-ready outputs."""

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
    OTSinkhornConfig,
    default_gaussian_specs,
    generate_ot_instance,
    solve_entropic_ot_ipm,
    solve_entropic_ot_sinkhorn_numpy,
    solve_entropic_ot_sinkhorn_pot,
    solve_ot_lp_ipm,
)
from ot_lab.plotting import apply_axis_style, apply_project_style, save_figure  # noqa: E402

MAIN_N = 200
MAIN_SEED = 1200
COMPARISON_LAMBDAS = (1.0, 0.1, 0.01)
PATH_LAMBDAS = np.logspace(1.0, -2.0, num=15)


def build_main_instance():
    """Construct the shared n=200 entropic-OT test case for Part 3."""

    source_spec, target_spec = default_gaussian_specs()
    return generate_ot_instance(
        n=MAIN_N,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=MAIN_SEED,
    )


def sinkhorn_config_for_lambda(regularization_strength: float) -> OTSinkhornConfig:
    """Choose a practical Sinkhorn configuration based on the regularization scale."""

    if regularization_strength >= 0.5:
        return OTSinkhornConfig(max_iterations=5_000, tolerance=1e-6, record_every=10)
    if regularization_strength >= 0.05:
        return OTSinkhornConfig(max_iterations=20_000, tolerance=1e-6, record_every=20)
    return OTSinkhornConfig(max_iterations=50_000, tolerance=1e-6, record_every=25)


def run_part3ab_comparison(instance) -> pd.DataFrame:
    """Compare POT Sinkhorn, custom Sinkhorn, and a conic baseline."""

    rows = []
    for regularization_strength in COMPARISON_LAMBDAS:
        config = sinkhorn_config_for_lambda(regularization_strength)
        print(f"Running Part 3(a,b) for lambda={regularization_strength:.4g}...")

        pot_solution = solve_entropic_ot_sinkhorn_pot(
            instance,
            regularization_strength,
            config=config,
        )
        numpy_solution = solve_entropic_ot_sinkhorn_numpy(
            instance,
            regularization_strength,
            config=config,
        )
        ipm_solution = solve_entropic_ot_ipm(
            instance,
            regularization_strength,
            solver="clarabel",
        )

        for method_name, solution in (
            ("sinkhorn", pot_solution),
            ("sinkhorn_numpy", numpy_solution),
            ("ipm", ipm_solution),
        ):
            rows.append(
                {
                    "n": MAIN_N,
                    "seed": MAIN_SEED,
                    "lambda": regularization_strength,
                    "method": method_name,
                    "success": solution.success,
                    "status": solution.status,
                    "runtime_seconds": solution.runtime_seconds,
                    "objective_value": solution.objective_value,
                    "transport_cost": solution.transport_cost,
                    "marginal_violation": solution.marginal_violation,
                    "iterations": solution.iterations,
                }
            )

    return pd.DataFrame(rows)


def run_part3c_convergence(instance) -> pd.DataFrame:
    """Record marginal violation traces for the custom Sinkhorn loop."""

    rows = []
    for regularization_strength in COMPARISON_LAMBDAS:
        config = sinkhorn_config_for_lambda(regularization_strength)
        solution = solve_entropic_ot_sinkhorn_numpy(
            instance,
            regularization_strength,
            config=config,
        )

        if solution.history is None:
            continue

        for iteration, marginal_error in zip(
            solution.history.iterations,
            solution.history.marginal_violations,
            strict=True,
        ):
            rows.append(
                {
                    "n": MAIN_N,
                    "seed": MAIN_SEED,
                    "lambda": regularization_strength,
                    "iteration": iteration,
                    "marginal_violation": marginal_error,
                }
            )

    return pd.DataFrame(rows)


def run_part3d_regularization_path(instance, wlp: float) -> pd.DataFrame:
    """Trace <C, P_lambda^*> versus lambda for Sinkhorn."""

    rows = []
    for regularization_strength in PATH_LAMBDAS:
        config = sinkhorn_config_for_lambda(float(regularization_strength))
        print(f"Running Part 3(d) for lambda={regularization_strength:.4g}...")

        pot_solution = solve_entropic_ot_sinkhorn_pot(
            instance,
            float(regularization_strength),
            config=config,
        )
        rows.append(
            {
                "n": MAIN_N,
                "seed": MAIN_SEED,
                "lambda": float(regularization_strength),
                "method": "sinkhorn",
                "success": pot_solution.success,
                "status": pot_solution.status,
                "runtime_seconds": pot_solution.runtime_seconds,
                "objective_value": pot_solution.objective_value,
                "transport_cost": pot_solution.transport_cost,
                "transport_gap_vs_wlp": pot_solution.transport_cost - wlp,
                "marginal_violation": pot_solution.marginal_violation,
                "iterations": pot_solution.iterations,
            }
        )

    return pd.DataFrame(rows)


def save_part3_tables(
    comparison_results: pd.DataFrame,
    convergence_results: pd.DataFrame,
    path_results: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Write the report-facing Part 3 tables."""

    results_dir.mkdir(parents=True, exist_ok=True)

    comparison_results.to_csv(results_dir / "part3ab_comparison_table.csv", index=False)
    convergence_results.to_csv(results_dir / "part3c_convergence_trace.csv", index=False)
    path_results.to_csv(results_dir / "part3d_regularization_path.csv", index=False)

    _write_markdown_table(
        comparison_results.sort_values(["lambda", "method"], ascending=[False, True]),
        results_dir / "part3ab_comparison_table.md",
        title="Part 3(a,b): Sinkhorn and Conic Comparison",
        float_columns={
            "lambda": "{:.4g}",
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "transport_cost": "{:.8f}",
            "marginal_violation": "{:.2e}",
        },
    )
    _write_markdown_table(
        path_results.sort_values(["lambda", "method"], ascending=[False, True]),
        results_dir / "part3d_regularization_path.md",
        title="Part 3(d): Entropic Regularization Path",
        float_columns={
            "lambda": "{:.4g}",
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "transport_cost": "{:.8f}",
            "transport_gap_vs_wlp": "{:+.2e}",
            "marginal_violation": "{:.2e}",
        },
    )


def plot_part3c_convergence(convergence_results: pd.DataFrame, output_stem: Path) -> None:
    """Plot marginal violation versus iteration for the custom Sinkhorn loop."""

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {1.0: "#1d3557", 0.1: "#2a9d8f", 0.01: "#e76f51"}
    markers = {1.0: "o", 0.1: "s", 0.01: "D"}

    for regularization_strength in COMPARISON_LAMBDAS:
        subset = convergence_results.loc[
            convergence_results["lambda"] == regularization_strength
        ].sort_values("iteration")
        markevery = max(1, len(subset) // 10)
        ax.plot(
            subset["iteration"],
            subset["marginal_violation"].clip(lower=1e-12),
            color=colors[regularization_strength],
            linewidth=2.2,
            marker=markers[regularization_strength],
            markersize=5.5,
            markevery=markevery,
            label=rf"$\lambda={regularization_strength:g}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Part 3(c): Sinkhorn Marginal Violation vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|P1-a\|_1 + \|P^\top 1-b\|_1$")
    apply_axis_style(ax)
    ax.legend(loc="best")
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_part3d_regularization_path(path_results: pd.DataFrame, wlp: float, output_stem: Path) -> None:
    """Plot <C, P_lambda^*> versus lambda for Sinkhorn."""

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    style = METHOD_STYLES["sinkhorn"]
    ordered = path_results.sort_values("lambda")
    ax.plot(
        ordered["lambda"],
        ordered["transport_cost"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        label=style["label"],
    )

    ax.axhline(
        wlp,
        color="#222222",
        linestyle=":",
        linewidth=2.0,
        label=r"LP optimum $W_{LP}$",
    )

    ax.set_xscale("log")
    ax.set_title("Part 3(d): Entropic Regularization Path")
    ax.set_xlabel(r"Regularization strength $\lambda$")
    ax.set_ylabel(r"Transport cost $\langle C, P_\lambda^* \rangle$")
    apply_axis_style(ax)
    ax.legend(loc="best")
    save_figure(fig, output_stem)
    plt.close(fig)


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
    apply_project_style()
    results_dir = PROJECT_ROOT / "results" / "part3"

    instance = build_main_instance()
    lp_solution = solve_ot_lp_ipm(instance)
    if not lp_solution.success or lp_solution.objective_value is None:
        raise RuntimeError("Failed to compute the LP baseline W_LP for Part 3.")
    wlp = lp_solution.objective_value

    comparison_results = run_part3ab_comparison(instance)
    convergence_results = run_part3c_convergence(instance)
    path_results = run_part3d_regularization_path(instance, wlp=wlp)

    save_part3_tables(comparison_results, convergence_results, path_results, results_dir)
    plot_part3c_convergence(convergence_results, results_dir / "part3c_convergence")
    plot_part3d_regularization_path(path_results, wlp, results_dir / "part3d_regularization_path")

    print(f"Saved Part 3 outputs to: {results_dir}")


if __name__ == "__main__":
    main()
