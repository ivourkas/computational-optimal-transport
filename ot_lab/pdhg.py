"""NumPy PDHG implementation for the exact OT linear program."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import numpy.typing as npt

from .lp import marginal_violation, transport_cost
from .problem_setup import OTInstance, validate_ot_instance

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OTPDHGConfig:
    """Configuration for the NumPy PDHG solver."""

    max_iterations: int = 20_000
    tolerance: float = 1e-7
    step_size: float | None = None
    primal_weight: float = 1.0
    stability_safety_factor: float = 0.99
    record_every: int = 50
    initial_plan: Array | None = None


def recommended_ot_pdhg_config(
    n_source: int,
    n_target: int,
    *,
    max_iterations: int = 20_000,
    tolerance: float = 1e-7,
) -> OTPDHGConfig:
    """Return an empirically tuned PDHG configuration for the OT LP family in this project."""

    return OTPDHGConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        step_size=1.13 / ot_operator_norm(n_source, n_target),
        primal_weight=2.0,
    )


@dataclass
class OTPDHGHistory:
    """Iteration traces for analyzing PDHG behavior."""

    iterations: list[int] = field(default_factory=list)
    objective_values: list[float] = field(default_factory=list)
    marginal_violations: list[float] = field(default_factory=list)
    fixed_point_residuals: list[float] = field(default_factory=list)
    step_sizes: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class OTPDHGSolution:
    """Result of the NumPy PDHG OT solver."""

    method: str
    status: str
    success: bool
    plan: Array
    objective_value: float
    marginal_violation: float
    runtime_seconds: float
    iterations: int
    tau: float
    sigma: float
    fixed_point_residual: float
    history: OTPDHGHistory
    solver_message: str | None = None


def ot_operator_norm(n_source: int, n_target: int) -> float:
    """Return ||K|| for K(P) = (P1, P^T1) under the Frobenius/Euclidean norms."""

    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be positive.")
    return float(np.sqrt(n_source + n_target))


def default_pdhg_step_sizes(
    n_source: int,
    n_target: int,
    primal_weight: float = 1.0,
    stability_safety_factor: float = 0.99,
    step_size: float | None = None,
) -> tuple[float, float]:
    """Compute stable PDHG step sizes using tau = eta / omega and sigma = eta * omega."""

    if primal_weight <= 0:
        raise ValueError("primal_weight must be positive.")
    if stability_safety_factor <= 0:
        raise ValueError("stability_safety_factor must be positive.")

    eta = step_size
    if eta is None:
        eta = stability_safety_factor / ot_operator_norm(n_source, n_target)
    tau = eta / primal_weight
    sigma = eta * primal_weight
    return float(tau), float(sigma)


def solve_ot_lp_pdhg_numpy(
    instance: OTInstance,
    config: OTPDHGConfig | None = None,
) -> OTPDHGSolution:
    """Solve the exact OT LP with a NumPy implementation of the PDHG updates from the assignment."""

    validate_ot_instance(instance)
    config = config or recommended_ot_pdhg_config(
        n_source=instance.n_source,
        n_target=instance.n_target,
    )

    source_weights = np.asarray(instance.source_weights, dtype=float)
    target_weights = np.asarray(instance.target_weights, dtype=float)
    cost_matrix = np.asarray(instance.cost_matrix, dtype=float)

    tau, sigma = default_pdhg_step_sizes(
        n_source=instance.n_source,
        n_target=instance.n_target,
        primal_weight=config.primal_weight,
        stability_safety_factor=config.stability_safety_factor,
        step_size=config.step_size,
    )

    plan = _initialize_plan(instance, config.initial_plan)
    source_dual = np.zeros(instance.n_source, dtype=float)
    target_dual = np.zeros(instance.n_target, dtype=float)

    history = OTPDHGHistory()

    current_iteration = 0
    fixed_point_residual = np.inf

    start_time = perf_counter()
    for iteration in range(1, config.max_iterations + 1):
        current_iteration = iteration
        step_size = np.sqrt(tau * sigma)

        plan_next = np.maximum(
            0.0,
            plan - tau * (cost_matrix - source_dual[:, None] - target_dual[None, :]),
        )
        extrapolated_plan = 2.0 * plan_next - plan

        source_dual_next = source_dual + sigma * (
            source_weights - extrapolated_plan.sum(axis=1)
        )
        target_dual_next = target_dual + sigma * (
            target_weights - extrapolated_plan.sum(axis=0)
        )

        plan_delta = plan_next - plan
        source_dual_delta = source_dual_next - source_dual
        target_dual_delta = target_dual_next - target_dual
        fixed_point_residual = _fixed_point_residual(
            plan_delta=plan_delta,
            source_dual_delta=source_dual_delta,
            target_dual_delta=target_dual_delta,
        )

        objective_value = transport_cost(cost_matrix, plan_next)
        current_marginal_violation = marginal_violation(
            plan_next,
            source_weights,
            target_weights,
        )

        if iteration % config.record_every == 0 or iteration == 1:
            history.iterations.append(iteration)
            history.objective_values.append(objective_value)
            history.marginal_violations.append(current_marginal_violation)
            history.fixed_point_residuals.append(fixed_point_residual)
            history.step_sizes.append(step_size)

        plan = plan_next
        source_dual = source_dual_next
        target_dual = target_dual_next

        if (
            current_marginal_violation <= config.tolerance
            and fixed_point_residual <= config.tolerance
        ):
            runtime_seconds = perf_counter() - start_time
            return OTPDHGSolution(
                method="pdhg_numpy",
                status="converged",
                success=True,
                plan=plan_next,
                objective_value=objective_value,
                marginal_violation=current_marginal_violation,
                runtime_seconds=runtime_seconds,
                iterations=iteration,
                tau=tau,
                sigma=sigma,
                fixed_point_residual=fixed_point_residual,
                history=history,
                solver_message="PDHG reached the requested residual tolerance.",
            )

    runtime_seconds = perf_counter() - start_time
    final_objective_value = transport_cost(cost_matrix, plan)
    final_marginal_violation = marginal_violation(
        plan,
        source_weights,
        target_weights,
    )
    return OTPDHGSolution(
        method="pdhg_numpy",
        status="max_iterations",
        success=False,
        plan=plan,
        objective_value=final_objective_value,
        marginal_violation=final_marginal_violation,
        runtime_seconds=runtime_seconds,
        iterations=current_iteration,
        tau=tau,
        sigma=sigma,
        fixed_point_residual=fixed_point_residual,
        history=history,
        solver_message="PDHG hit the iteration limit before meeting the tolerance.",
    )


def _initialize_plan(instance: OTInstance, initial_plan: Array | None) -> Array:
    """Create the initial primal iterate for PDHG."""

    if initial_plan is None:
        return np.zeros_like(instance.cost_matrix, dtype=float)

    initial_plan = np.asarray(initial_plan, dtype=float)
    if initial_plan.shape != instance.cost_matrix.shape:
        raise ValueError("initial_plan must have the same shape as the OT cost matrix.")
    if np.any(initial_plan < 0):
        raise ValueError("initial_plan must be nonnegative.")
    return initial_plan.copy()


def _fixed_point_residual(
    plan_delta: Array,
    source_dual_delta: Array,
    target_dual_delta: Array,
) -> float:
    """Measure how much the primal-dual iterate changed in one PDHG step."""

    plan_norm = np.linalg.norm(plan_delta)
    dual_norm = np.sqrt(
        np.linalg.norm(source_dual_delta) ** 2 + np.linalg.norm(target_dual_delta) ** 2
    )
    return float(np.sqrt(plan_norm**2 + dual_norm**2))
