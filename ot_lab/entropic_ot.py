"""Entropically regularized optimal transport solvers for Part 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import ot
from scipy.special import logsumexp

from .lp import marginal_violation, transport_cost
from .problem_setup import OTInstance, validate_ot_instance

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OTSinkhornConfig:
    """Configuration for Sinkhorn solvers."""

    max_iterations: int = 50_000
    tolerance: float = 1e-6
    record_every: int = 25


@dataclass
class OTEntropicHistory:
    """Iteration traces for entropic OT solvers."""

    iterations: list[int] = field(default_factory=list)
    transport_costs: list[float] = field(default_factory=list)
    objective_values: list[float] = field(default_factory=list)
    marginal_violations: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class OTEntropicSolution:
    """Solver output for the entropic OT problem."""

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
    solver_message: str | None = None
    history: OTEntropicHistory | None = None


def entropic_transport_objective(
    cost_matrix: Array,
    plan: Array,
    regularization_strength: float,
) -> float:
    """Compute <C, P> + lambda * sum_ij P_ij log P_ij with 0 log 0 := 0."""

    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    plan = np.asarray(plan, dtype=float)
    positive_entries = plan > 0
    entropy_term = np.sum(plan[positive_entries] * np.log(plan[positive_entries]))
    return float(transport_cost(cost_matrix, plan) + regularization_strength * entropy_term)


def gibbs_kernel(cost_matrix: Array, regularization_strength: float) -> Array:
    """Return K_ij = exp(-C_ij / lambda)."""

    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")
    return np.exp(-np.asarray(cost_matrix, dtype=float) / regularization_strength)


def solve_entropic_ot(
    instance: OTInstance,
    regularization_strength: float,
    method: str,
    *,
    sinkhorn_config: OTSinkhornConfig | None = None,
    conic_solver: str = "clarabel",
    conic_options: dict[str, float | int | bool] | None = None,
) -> OTEntropicSolution:
    """Dispatch to one of the Part 3 entropic OT solvers."""

    normalized_method = method.lower()
    if normalized_method == "sinkhorn":
        return solve_entropic_ot_sinkhorn_pot(
            instance,
            regularization_strength,
            config=sinkhorn_config,
        )
    if normalized_method == "sinkhorn_numpy":
        return solve_entropic_ot_sinkhorn_numpy(
            instance,
            regularization_strength,
            config=sinkhorn_config,
        )
    if normalized_method == "ipm":
        return solve_entropic_ot_ipm(
            instance,
            regularization_strength,
            solver=conic_solver,
            solver_options=conic_options,
        )
    raise ValueError(f"Unsupported entropic OT method: {method}")


def solve_entropic_ot_sinkhorn_pot(
    instance: OTInstance,
    regularization_strength: float,
    *,
    config: OTSinkhornConfig | None = None,
) -> OTEntropicSolution:
    """Solve entropic OT with POT's log-domain Sinkhorn implementation."""

    validate_ot_instance(instance)
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    config = config or OTSinkhornConfig()

    start_time = perf_counter()
    plan, log = ot.bregman.sinkhorn_log(
        instance.source_weights,
        instance.target_weights,
        instance.cost_matrix,
        regularization_strength,
        numItermax=config.max_iterations,
        stopThr=config.tolerance,
        log=True,
        warn=False,
    )
    runtime_seconds = perf_counter() - start_time

    plan = np.asarray(plan, dtype=float)
    current_transport_cost = transport_cost(instance.cost_matrix, plan)
    current_objective_value = entropic_transport_objective(
        instance.cost_matrix,
        plan,
        regularization_strength,
    )
    current_marginal_violation = marginal_violation(
        plan,
        instance.source_weights,
        instance.target_weights,
    )

    history = OTEntropicHistory()
    if "err" in log:
        error_trace = list(log["err"])
        history.iterations = [10 * (index + 1) for index in range(len(error_trace))]
        history.marginal_violations = [float(error) for error in error_trace]

    success = current_marginal_violation <= 10.0 * config.tolerance
    status = "converged" if success else "max_iterations"
    return OTEntropicSolution(
        method="sinkhorn",
        status=status,
        success=success,
        plan=plan,
        transport_cost=current_transport_cost,
        objective_value=current_objective_value,
        marginal_violation=current_marginal_violation,
        runtime_seconds=runtime_seconds,
        regularization_strength=regularization_strength,
        iterations=int(log.get("niter", 0)),
        solver_message="Solved with POT sinkhorn_log.",
        history=history,
    )


def solve_entropic_ot_sinkhorn_numpy(
    instance: OTInstance,
    regularization_strength: float,
    *,
    config: OTSinkhornConfig | None = None,
) -> OTEntropicSolution:
    """Solve entropic OT with a custom log-domain NumPy Sinkhorn loop."""

    validate_ot_instance(instance)
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    config = config or OTSinkhornConfig()

    source_weights = np.asarray(instance.source_weights, dtype=float)
    target_weights = np.asarray(instance.target_weights, dtype=float)
    cost_matrix = np.asarray(instance.cost_matrix, dtype=float)

    log_a = np.log(source_weights)
    log_b = np.log(target_weights)
    log_kernel = -cost_matrix / regularization_strength
    log_u = np.zeros_like(source_weights)
    log_v = np.zeros_like(target_weights)

    history = OTEntropicHistory()
    current_marginal_violation = np.inf
    current_transport_cost = np.inf
    current_objective_value = np.inf
    current_iteration = 0

    start_time = perf_counter()
    for iteration in range(1, config.max_iterations + 1):
        current_iteration = iteration

        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)

        if iteration % config.record_every == 0 or iteration == 1:
            current_transport_cost, current_objective_value, current_marginal_violation = (
                _sinkhorn_log_state_metrics(
                    cost_matrix=cost_matrix,
                    source_weights=source_weights,
                    target_weights=target_weights,
                    regularization_strength=regularization_strength,
                    log_kernel=log_kernel,
                    log_u=log_u,
                    log_v=log_v,
                )
            )
            history.iterations.append(iteration)
            history.transport_costs.append(current_transport_cost)
            history.objective_values.append(current_objective_value)
            history.marginal_violations.append(current_marginal_violation)

            if current_marginal_violation <= config.tolerance:
                break

    runtime_seconds = perf_counter() - start_time
    plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])

    if not np.isfinite(current_marginal_violation):
        current_transport_cost = transport_cost(cost_matrix, plan)
        current_objective_value = entropic_transport_objective(
            cost_matrix,
            plan,
            regularization_strength,
        )
        current_marginal_violation = marginal_violation(
            plan,
            source_weights,
            target_weights,
        )

    success = current_marginal_violation <= config.tolerance
    status = "converged" if success else "max_iterations"
    message = (
        "Custom log-domain Sinkhorn reached the requested tolerance."
        if success
        else "Custom Sinkhorn hit the iteration limit before meeting the tolerance."
    )

    return OTEntropicSolution(
        method="sinkhorn_numpy",
        status=status,
        success=success,
        plan=plan,
        transport_cost=current_transport_cost,
        objective_value=current_objective_value,
        marginal_violation=current_marginal_violation,
        runtime_seconds=runtime_seconds,
        regularization_strength=regularization_strength,
        iterations=current_iteration,
        solver_message=message,
        history=history,
    )


def solve_entropic_ot_ipm(
    instance: OTInstance,
    regularization_strength: float,
    *,
    solver: str = "clarabel",
    solver_options: dict[str, float | int | bool] | None = None,
) -> OTEntropicSolution:
    """Solve entropic OT with a conic solver through CVXPY."""

    validate_ot_instance(instance)
    if regularization_strength <= 0:
        raise ValueError("regularization_strength must be positive.")

    solver_options = solver_options or {}
    normalized_solver = solver.strip().lower()
    solver_name = _cvxpy_entropic_solver(normalized_solver)

    plan_variable = cp.Variable(instance.cost_matrix.shape, nonneg=True)
    objective = cp.sum(cp.multiply(instance.cost_matrix, plan_variable)) - (
        regularization_strength * cp.sum(cp.entr(plan_variable))
    )
    constraints = [
        cp.sum(plan_variable, axis=1) == instance.source_weights,
        cp.sum(plan_variable, axis=0) == instance.target_weights,
    ]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    start_time = perf_counter()
    used_solver_name = solver_name
    solver_label = normalized_solver
    try:
        problem.solve(solver=solver_name, verbose=False, **solver_options)
    except cp.SolverError:
        if normalized_solver != "clarabel":
            raise
        used_solver_name = cp.SCS
        solver_label = "scs"
        fallback_options = {"eps": 1e-5, "max_iters": 20_000}
        fallback_options.update(solver_options)
        problem.solve(solver=used_solver_name, verbose=False, **fallback_options)
    runtime_seconds = perf_counter() - start_time

    success = problem.status in {"optimal", "optimal_inaccurate"}
    if not success or plan_variable.value is None:
        return OTEntropicSolution(
            method="ipm",
            status=problem.status,
            success=False,
            plan=None,
            transport_cost=None,
            objective_value=None,
            marginal_violation=None,
            runtime_seconds=runtime_seconds,
            regularization_strength=regularization_strength,
            solver_message="The conic solver did not return a valid entropic OT plan.",
        )

    plan = np.asarray(plan_variable.value, dtype=float)
    return OTEntropicSolution(
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
        solver_message=f"Solved with CVXPY and {solver_label.upper()}.",
    )


def _sinkhorn_log_state_metrics(
    *,
    cost_matrix: Array,
    source_weights: Array,
    target_weights: Array,
    regularization_strength: float,
    log_kernel: Array,
    log_u: Array,
    log_v: Array,
) -> tuple[float, float, float]:
    """Compute transport cost, full objective, and marginal violation in log-domain Sinkhorn."""

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(log_plan)

    current_transport_cost = transport_cost(cost_matrix, plan)
    current_objective_value = entropic_transport_objective(
        cost_matrix,
        plan,
        regularization_strength,
    )
    current_marginal_violation = marginal_violation(
        plan,
        source_weights,
        target_weights,
    )
    return current_transport_cost, current_objective_value, current_marginal_violation


def _cvxpy_entropic_solver(name: str) -> str:
    """Map a simple entropic solver name to the matching CVXPY constant."""

    if name == "clarabel":
        return cp.CLARABEL
    if name == "scs":
        return cp.SCS
    raise ValueError(f"Unsupported entropic OT solver: {name}")
