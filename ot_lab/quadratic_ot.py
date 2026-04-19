"""Quadratically regularized optimal transport solvers for Part 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import cvxpy as cp
import numpy as np
import numpy.typing as npt

from .lp import marginal_violation, transport_cost
from .problem_setup import OTInstance, validate_ot_instance

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OTADMMConfig:
    """Configuration for the custom ADMM solver for quadratic OT."""

    rho: float | None = None
    max_iterations: int = 50_000
    abs_tolerance: float = 1e-6
    rel_tolerance: float = 1e-6
    record_every: int = 50
    initial_plan: Array | None = None


@dataclass
class OTQuadraticHistory:
    """Iteration traces for the quadratic OT solvers."""

    iterations: list[int] = field(default_factory=list)
    objective_values: list[float] = field(default_factory=list)
    transport_costs: list[float] = field(default_factory=list)
    marginal_violations: list[float] = field(default_factory=list)
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    primal_thresholds: list[float] = field(default_factory=list)
    dual_thresholds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class OTQuadraticSolution:
    """Solver output for the quadratically regularized OT problem."""

    method: str
    status: str
    success: bool
    plan: Array | None
    transport_cost: float | None
    objective_value: float | None
    marginal_violation: float | None
    runtime_seconds: float
    regularization_strength: float
    iterations: int | None = None
    rho: float | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None
    solver_message: str | None = None
    history: OTQuadraticHistory | None = None


def quadratic_transport_objective(
    cost_matrix: Array,
    plan: Array,
    regularization_strength: float,
) -> float:
    """Compute <C, P> + (lambda / 2) * ||P||_F^2."""

    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    plan = np.asarray(plan, dtype=float)
    return float(
        transport_cost(cost_matrix, plan)
        + 0.5 * regularization_strength * np.sum(plan * plan)
    )


def recommended_admm_rho(cost_matrix: Array) -> float:
    """Choose a practical ADMM penalty based on the scale of the transport costs."""

    cost_matrix = np.asarray(cost_matrix, dtype=float)
    mean_cost = float(np.mean(cost_matrix))
    return max(1.0, 10.0 * mean_cost)


def project_onto_marginal_affine_set(
    matrix: Array,
    source_weights: Array,
    target_weights: Array,
) -> Array:
    """Project onto {Q : Q1 = a, Q^T 1 = b} under the Frobenius norm.

    The nonnegativity constraint is enforced in the P-update, so the Q-update only needs
    the exact row and column sums. This keeps the ADMM step closed form.
    """

    matrix = np.asarray(matrix, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")

    n_source, n_target = matrix.shape
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total_mass_gap = float(source_weights.sum() - row_sums.sum())

    row_correction = ((source_weights - row_sums) / n_target)[:, None]
    col_correction = ((target_weights - col_sums) / n_source)[None, :]
    shared_correction = total_mass_gap / (n_source * n_target)

    return matrix + row_correction + col_correction - shared_correction


def solve_quadratic_ot(
    instance: OTInstance,
    regularization_strength: float,
    method: str,
    *,
    admm_config: OTADMMConfig | None = None,
    ipm_solver: str = "clarabel",
    ipm_options: dict[str, float | int | bool] | None = None,
) -> OTQuadraticSolution:
    """Dispatch to one of the Part 2 quadratic OT solvers."""

    normalized_method = method.lower()
    if normalized_method == "admm":
        return solve_quadratic_ot_admm(
            instance,
            regularization_strength,
            config=admm_config,
        )
    if normalized_method == "ipm":
        return solve_quadratic_ot_ipm(
            instance,
            regularization_strength,
            solver=ipm_solver,
            solver_options=ipm_options,
        )
    raise ValueError(f"Unsupported quadratic OT method: {method}")


def solve_quadratic_ot_admm(
    instance: OTInstance,
    regularization_strength: float,
    *,
    config: OTADMMConfig | None = None,
) -> OTQuadraticSolution:
    """Solve the quadratic OT problem with a custom ADMM implementation.

    We use the equivalent split:
        P handles the objective and nonnegativity,
        Q handles the exact marginal constraints.
    This gives a closed-form P-update and a closed-form Euclidean projection for Q.
    """

    validate_ot_instance(instance)
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    config = config or OTADMMConfig(rho=recommended_admm_rho(instance.cost_matrix))
    rho = config.rho or recommended_admm_rho(instance.cost_matrix)
    if rho <= 0:
        raise ValueError("rho must be positive.")

    cost_matrix = np.asarray(instance.cost_matrix, dtype=float)
    source_weights = np.asarray(instance.source_weights, dtype=float)
    target_weights = np.asarray(instance.target_weights, dtype=float)

    plan = _initialize_admm_plan(instance, config.initial_plan)
    projected_plan = project_onto_marginal_affine_set(plan, source_weights, target_weights)
    scaled_dual = np.zeros_like(plan)

    history = OTQuadraticHistory()
    current_iteration = 0
    primal_residual = np.inf
    dual_residual = np.inf
    primal_threshold = np.inf
    dual_threshold = np.inf
    marginal_threshold = 10.0 * config.abs_tolerance

    start_time = perf_counter()
    for iteration in range(1, config.max_iterations + 1):
        current_iteration = iteration
        plan_next = np.maximum(
            0.0,
            (rho * (projected_plan - scaled_dual) - cost_matrix) / (regularization_strength + rho),
        )

        projected_plan_previous = projected_plan
        projected_plan = project_onto_marginal_affine_set(
            plan_next + scaled_dual,
            source_weights,
            target_weights,
        )
        scaled_dual = scaled_dual + plan_next - projected_plan

        primal_residual = float(np.linalg.norm(plan_next - projected_plan))
        dual_residual = float(rho * np.linalg.norm(projected_plan - projected_plan_previous))
        primal_threshold, dual_threshold = _admm_thresholds(
            plan=plan_next,
            projected_plan=projected_plan,
            scaled_dual=scaled_dual,
            rho=rho,
            abs_tolerance=config.abs_tolerance,
            rel_tolerance=config.rel_tolerance,
        )

        current_transport_cost = transport_cost(cost_matrix, plan_next)
        current_objective_value = quadratic_transport_objective(
            cost_matrix,
            plan_next,
            regularization_strength,
        )
        current_marginal_violation = marginal_violation(
            plan_next,
            source_weights,
            target_weights,
        )

        if iteration % config.record_every == 0 or iteration == 1:
            history.iterations.append(iteration)
            history.transport_costs.append(current_transport_cost)
            history.objective_values.append(current_objective_value)
            history.marginal_violations.append(current_marginal_violation)
            history.primal_residuals.append(primal_residual)
            history.dual_residuals.append(dual_residual)
            history.primal_thresholds.append(primal_threshold)
            history.dual_thresholds.append(dual_threshold)

        plan = plan_next

        if (
            primal_residual <= primal_threshold
            and dual_residual <= dual_threshold
            and current_marginal_violation <= marginal_threshold
        ):
            runtime_seconds = perf_counter() - start_time
            return OTQuadraticSolution(
                method="admm",
                status="converged",
                success=True,
                plan=plan,
                transport_cost=current_transport_cost,
                objective_value=current_objective_value,
                marginal_violation=current_marginal_violation,
                runtime_seconds=runtime_seconds,
                regularization_strength=regularization_strength,
                iterations=iteration,
                rho=rho,
                primal_residual=primal_residual,
                dual_residual=dual_residual,
                solver_message="ADMM met the primal and dual residual tolerances.",
                history=history,
            )

    runtime_seconds = perf_counter() - start_time
    final_transport_cost = transport_cost(cost_matrix, plan)
    final_objective_value = quadratic_transport_objective(
        cost_matrix,
        plan,
        regularization_strength,
    )
    final_marginal_violation = marginal_violation(plan, source_weights, target_weights)
    return OTQuadraticSolution(
        method="admm",
        status="max_iterations",
        success=False,
        plan=plan,
        transport_cost=final_transport_cost,
        objective_value=final_objective_value,
        marginal_violation=final_marginal_violation,
        runtime_seconds=runtime_seconds,
        regularization_strength=regularization_strength,
        iterations=current_iteration,
        rho=rho,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        solver_message="ADMM hit the iteration limit before meeting the tolerance.",
        history=history,
    )


def solve_quadratic_ot_ipm(
    instance: OTInstance,
    regularization_strength: float,
    *,
    solver: str = "clarabel",
    solver_options: dict[str, float | int | bool] | None = None,
) -> OTQuadraticSolution:
    """Solve the quadratic OT problem with a CVXPY interior-point solver."""

    validate_ot_instance(instance)
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    solver_options = solver_options or {}
    normalized_solver = solver.strip().lower()
    solver_name = _cvxpy_ipm_solver(normalized_solver)

    plan_variable = cp.Variable(instance.cost_matrix.shape, nonneg=True)
    objective = cp.sum(cp.multiply(instance.cost_matrix, plan_variable)) + (
        regularization_strength / 2.0
    ) * cp.sum_squares(plan_variable)
    constraints = [
        cp.sum(plan_variable, axis=1) == instance.source_weights,
        cp.sum(plan_variable, axis=0) == instance.target_weights,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    start_time = perf_counter()
    problem.solve(solver=solver_name, verbose=False, **solver_options)
    runtime_seconds = perf_counter() - start_time

    success = problem.status in {"optimal", "optimal_inaccurate"}
    if not success or plan_variable.value is None:
        return OTQuadraticSolution(
            method="ipm",
            status=problem.status,
            success=False,
            plan=None,
            transport_cost=None,
            objective_value=None,
            marginal_violation=None,
            runtime_seconds=runtime_seconds,
            regularization_strength=regularization_strength,
            solver_message="The IPM solver did not return a valid plan.",
        )

    plan = np.asarray(plan_variable.value, dtype=float)
    return OTQuadraticSolution(
        method="ipm",
        status=problem.status,
        success=True,
        plan=plan,
        transport_cost=transport_cost(instance.cost_matrix, plan),
        objective_value=float(problem.value),
        marginal_violation=marginal_violation(
            plan,
            instance.source_weights,
            instance.target_weights,
        ),
        runtime_seconds=runtime_seconds,
        regularization_strength=regularization_strength,
        iterations=problem.solver_stats.num_iters,
        solver_message=f"Solved with CVXPY and {normalized_solver.upper()}.",
    )


def _initialize_admm_plan(instance: OTInstance, initial_plan: Array | None) -> Array:
    """Create the initial ADMM iterate."""

    if initial_plan is None:
        return np.outer(instance.source_weights, instance.target_weights).astype(float)

    initial_plan = np.asarray(initial_plan, dtype=float)
    if initial_plan.shape != instance.cost_matrix.shape:
        raise ValueError("initial_plan must match the OT cost matrix shape.")
    if np.any(initial_plan < 0):
        raise ValueError("initial_plan must be nonnegative.")
    return initial_plan.copy()


def _admm_thresholds(
    plan: Array,
    projected_plan: Array,
    scaled_dual: Array,
    rho: float,
    abs_tolerance: float,
    rel_tolerance: float,
) -> tuple[float, float]:
    """Compute the standard ADMM primal and dual stopping thresholds."""

    problem_dimension = float(np.sqrt(plan.size))
    primal_threshold = problem_dimension * abs_tolerance + rel_tolerance * max(
        np.linalg.norm(plan),
        np.linalg.norm(projected_plan),
    )
    dual_threshold = problem_dimension * abs_tolerance + rel_tolerance * rho * np.linalg.norm(
        scaled_dual
    )
    return float(primal_threshold), float(dual_threshold)


def _cvxpy_ipm_solver(name: str) -> str:
    """Map a simple solver name to the matching CVXPY constant."""

    if name == "clarabel":
        return cp.CLARABEL
    if name == "ecos":
        return cp.ECOS
    raise ValueError(f"Unsupported quadratic OT IPM solver: {name}")
