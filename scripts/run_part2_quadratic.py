"""Run Part 2 quadratic-regularization experiments and save report-ready outputs."""

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
    OTADMMConfig,
    default_gaussian_specs,
    generate_ot_instance,
    recommended_admm_rho,
    solve_ot_lp_ipm,
    solve_quadratic_ot_admm,
    solve_quadratic_ot_ipm,
)
from ot_lab.plotting import apply_axis_style, apply_project_style, save_figure  # noqa: E402

MAIN_N = 200
MAIN_SEED = 1200
PART2A_LAMBDA = 0.1
PART2C_LAMBDA = 0.1
PATH_LAMBDAS = np.logspace(1.0, -3.0, num=10)
RHO_MULTIPLIERS = (0.1, 0.2, 0.5, 1.0, 2.0, 4.0)


def build_main_instance():
    """Construct the shared n=200 quadratic-OT test case for Part 2."""

    source_spec, target_spec = default_gaussian_specs()
    return generate_ot_instance(
        n=MAIN_N,
        source_spec=source_spec,
        target_spec=target_spec,
        seed=MAIN_SEED,
    )


def run_part2a_runtime_comparison(instance, rho: float) -> pd.DataFrame:
    """Compare ADMM and IPM on the representative n=200, lambda=0.1 case."""

    admm_solution = solve_quadratic_ot_admm(
        instance,
        PART2A_LAMBDA,
        config=OTADMMConfig(
            rho=rho,
            max_iterations=50_000,
            abs_tolerance=1e-6,
            rel_tolerance=1e-6,
            record_every=100,
        ),
    )
    ipm_solution = solve_quadratic_ot_ipm(
        instance,
        PART2A_LAMBDA,
        solver="clarabel",
    )

    rows = []
    for method_name, solution in (("admm", admm_solution), ("ipm", ipm_solution)):
        rows.append(
            {
                "n": MAIN_N,
                "seed": MAIN_SEED,
                "lambda": PART2A_LAMBDA,
                "method": method_name,
                "success": solution.success,
                "status": solution.status,
                "rho": solution.rho,
                "runtime_seconds": solution.runtime_seconds,
                "objective_value": solution.objective_value,
                "transport_cost": solution.transport_cost,
                "marginal_violation": solution.marginal_violation,
                "iterations": solution.iterations,
            }
        )

    return pd.DataFrame(rows)


def run_part2b_regularization_path(instance, rho: float, wlp: float) -> pd.DataFrame:
    """Trace the transport cost <C, P_lambda^*> as lambda varies."""

    rows = []
    for regularization_strength in PATH_LAMBDAS:
        print(f"Running Part 2(b) for lambda={regularization_strength:.4g}...")

        admm_solution = solve_quadratic_ot_admm(
            instance,
            float(regularization_strength),
            config=OTADMMConfig(
                rho=rho,
                max_iterations=50_000,
                abs_tolerance=1e-6,
                rel_tolerance=1e-6,
                record_every=100,
            ),
        )
        ipm_solution = solve_quadratic_ot_ipm(
            instance,
            float(regularization_strength),
            solver="clarabel",
        )

        for method_name, solution in (("admm", admm_solution), ("ipm", ipm_solution)):
            rows.append(
                {
                    "n": MAIN_N,
                    "seed": MAIN_SEED,
                    "lambda": float(regularization_strength),
                    "method": method_name,
                    "success": solution.success,
                    "status": solution.status,
                    "rho": solution.rho,
                    "runtime_seconds": solution.runtime_seconds,
                    "objective_value": solution.objective_value,
                    "transport_cost": solution.transport_cost,
                    "transport_gap_vs_wlp": solution.transport_cost - wlp,
                    "marginal_violation": solution.marginal_violation,
                    "iterations": solution.iterations,
                }
            )

    return pd.DataFrame(rows)


def run_part2c_rho_sweep(instance, baseline_transport_cost: float, baseline_objective: float) -> pd.DataFrame:
    """Study how the ADMM penalty rho changes convergence speed at fixed lambda."""

    base_rho = recommended_admm_rho(instance.cost_matrix)
    rows = []

    for multiplier in RHO_MULTIPLIERS:
        rho = base_rho * multiplier
        print(f"Running Part 2(c) with rho={rho:.4g}...")
        solution = solve_quadratic_ot_admm(
            instance,
            PART2C_LAMBDA,
            config=OTADMMConfig(
                rho=rho,
                max_iterations=50_000,
                abs_tolerance=1e-6,
                rel_tolerance=1e-6,
                record_every=100,
            ),
        )
        rows.append(
            {
                "n": MAIN_N,
                "seed": MAIN_SEED,
                "lambda": PART2C_LAMBDA,
                "rho": rho,
                "rho_multiplier": multiplier,
                "success": solution.success,
                "status": solution.status,
                "runtime_seconds": solution.runtime_seconds,
                "iterations": solution.iterations,
                "objective_value": solution.objective_value,
                "objective_gap_vs_ipm": solution.objective_value - baseline_objective,
                "transport_cost": solution.transport_cost,
                "transport_gap_vs_ipm": solution.transport_cost - baseline_transport_cost,
                "marginal_violation": solution.marginal_violation,
                "primal_residual": solution.primal_residual,
                "dual_residual": solution.dual_residual,
            }
        )

    return pd.DataFrame(rows)


def save_part2_tables(
    runtime_results: pd.DataFrame,
    path_results: pd.DataFrame,
    rho_results: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Write the report-facing Part 2 tables."""

    results_dir.mkdir(parents=True, exist_ok=True)

    runtime_results.to_csv(results_dir / "part2a_runtime_table.csv", index=False)
    path_results.to_csv(results_dir / "part2b_regularization_path.csv", index=False)
    rho_results.to_csv(results_dir / "part2c_rho_sensitivity.csv", index=False)

    _write_markdown_table(
        runtime_results.sort_values("method"),
        results_dir / "part2a_runtime_table.md",
        title="Part 2(a): Runtime Comparison at n = 200, lambda = 0.1",
        float_columns={
            "rho": "{:.2f}",
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "transport_cost": "{:.8f}",
            "marginal_violation": "{:.2e}",
        },
    )
    _write_markdown_table(
        path_results.sort_values(["lambda", "method"], ascending=[False, True]),
        results_dir / "part2b_regularization_path.md",
        title="Part 2(b): Regularization Path",
        float_columns={
            "lambda": "{:.4g}",
            "rho": "{:.2f}",
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "transport_cost": "{:.8f}",
            "transport_gap_vs_wlp": "{:+.2e}",
            "marginal_violation": "{:.2e}",
        },
    )
    _write_markdown_table(
        rho_results.sort_values("rho"),
        results_dir / "part2c_rho_sensitivity.md",
        title="Part 2(c): ADMM Penalty Sensitivity at lambda = 0.1",
        float_columns={
            "rho": "{:.2f}",
            "rho_multiplier": "{:.2f}",
            "runtime_seconds": "{:.6f}",
            "objective_value": "{:.8f}",
            "objective_gap_vs_ipm": "{:+.2e}",
            "transport_cost": "{:.8f}",
            "transport_gap_vs_ipm": "{:+.2e}",
            "marginal_violation": "{:.2e}",
            "primal_residual": "{:.2e}",
            "dual_residual": "{:.2e}",
        },
    )


def plot_regularization_path(path_results: pd.DataFrame, wlp: float, output_stem: Path) -> None:
    """Plot <C, P_lambda^*> vs lambda for ADMM and IPM."""

    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for method_name, style_key in (("admm", "admm"), ("ipm", "quadratic_ipm")):
        subset = path_results.loc[path_results["method"] == method_name].sort_values("lambda")
        style = METHOD_STYLES[style_key]
        ax.plot(
            subset["lambda"],
            subset["transport_cost"],
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
    ax.set_title("Part 2(b): Quadratic Regularization Path")
    ax.set_xlabel(r"Regularization strength $\lambda$")
    ax.set_ylabel(r"Transport cost $\langle C, P_\lambda^* \rangle$")
    apply_axis_style(ax)
    ax.legend(loc="best")
    save_figure(fig, output_stem)
    plt.close(fig)


def plot_rho_sensitivity(rho_results: pd.DataFrame, output_stem: Path) -> None:
    """Plot how rho changes ADMM runtime and iteration count."""

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax_runtime, ax_iterations = axes

    style = METHOD_STYLES["admm"]
    ordered = rho_results.sort_values("rho")

    ax_runtime.plot(
        ordered["rho"],
        ordered["runtime_seconds"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
    )
    ax_iterations.plot(
        ordered["rho"],
        ordered["iterations"],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
    )

    failed = ordered.loc[~ordered["success"]]
    if not failed.empty:
        for axis, column in ((ax_runtime, "runtime_seconds"), (ax_iterations, "iterations")):
            axis.scatter(
                failed["rho"],
                failed[column],
                facecolors="none",
                edgecolors=style["color"],
                s=105,
                linewidths=2.0,
                zorder=5,
            )

    for axis in axes:
        axis.set_xscale("log")
        apply_axis_style(axis)

    ax_runtime.set_title(r"Runtime vs ADMM Penalty $\rho$")
    ax_runtime.set_xlabel(r"ADMM penalty $\rho$")
    ax_runtime.set_ylabel("Solve time (seconds)")

    ax_iterations.set_title(r"Iterations vs ADMM Penalty $\rho$")
    ax_iterations.set_xlabel(r"ADMM penalty $\rho$")
    ax_iterations.set_ylabel("Iterations")

    fig.suptitle(r"Part 2(c): Effect of the ADMM Penalty $\rho$", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
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
    results_dir = PROJECT_ROOT / "results" / "part2"

    instance = build_main_instance()
    base_rho = recommended_admm_rho(instance.cost_matrix)
    lp_solution = solve_ot_lp_ipm(instance)
    if not lp_solution.success or lp_solution.objective_value is None:
        raise RuntimeError("Failed to compute the LP baseline W_LP for Part 2.")
    wlp = lp_solution.objective_value

    runtime_results = run_part2a_runtime_comparison(instance, rho=base_rho)
    baseline_ipm_row = runtime_results.loc[runtime_results["method"] == "ipm"].iloc[0]
    rho_results = run_part2c_rho_sweep(
        instance,
        baseline_transport_cost=float(baseline_ipm_row["transport_cost"]),
        baseline_objective=float(baseline_ipm_row["objective_value"]),
    )
    path_results = run_part2b_regularization_path(instance, rho=base_rho, wlp=wlp)

    save_part2_tables(runtime_results, path_results, rho_results, results_dir)
    plot_regularization_path(path_results, wlp, results_dir / "part2b_regularization_path")
    plot_rho_sensitivity(rho_results, results_dir / "part2c_rho_sensitivity")

    print(f"Saved Part 2 outputs to: {results_dir}")


if __name__ == "__main__":
    main()
