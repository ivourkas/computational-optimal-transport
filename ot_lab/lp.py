"""Linear-programming tools for the exact discrete OT problem."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
from ortools.linear_solver import pywraplp
from scipy import sparse
from scipy.optimize import linprog

from .problem_setup import OTInstance, validate_ot_instance

Array = npt.NDArray[np.float64]
SparseMatrix = sparse.csr_matrix


@dataclass(frozen=True)
class OTLinearProgram:
    """Standard-form LP data for the exact OT problem."""

    cost_vector: Array
    equality_matrix: SparseMatrix
    equality_rhs: Array
    n_source: int
    n_target: int
    source_weights: Array
    target_weights: Array

    @property
    def n_variables(self) -> int:
        """Total number of transport variables."""

        return self.cost_vector.size


@dataclass(frozen=True)
class OTLPSolution:
    """Solver output for the exact OT linear program."""

    method: str
    status: str
    success: bool
    plan: Array | None
    objective_value: float | None
    marginal_violation: float | None
    runtime_seconds: float
    iterations: int | None = None
    solver_message: str | None = None


def flatten_plan(plan: Array) -> Array:
    """Vectorize a transport plan using row-major order."""

    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError("Transport plan must be a 2D array.")
    return plan.reshape(-1, order="C")


def unflatten_plan(plan_vector: Array, n_source: int, n_target: int) -> Array:
    """Recover the matrix form of a row-major vectorized transport plan."""

    plan_vector = np.asarray(plan_vector, dtype=float)
    if plan_vector.ndim != 1:
        raise ValueError("Plan vector must be a 1D array.")
    if plan_vector.size != n_source * n_target:
        raise ValueError("Plan vector size does not match the requested shape.")
    return plan_vector.reshape((n_source, n_target), order="C")


def build_transport_equality_constraints(n_source: int, n_target: int) -> SparseMatrix:
    """Build the sparse marginal constraint matrix in row-major vectorization."""

    if n_source <= 0 or n_target <= 0:
        raise ValueError("n_source and n_target must be positive.")

    row_block = sparse.kron(
        sparse.eye(n_source, format="csr"),
        sparse.csr_matrix(np.ones((1, n_target), dtype=float)),
        format="csr",
    )
    col_block = sparse.kron(
        sparse.csr_matrix(np.ones((1, n_source), dtype=float)),
        sparse.eye(n_target, format="csr"),
        format="csr",
    )
    return sparse.vstack([row_block, col_block], format="csr")


def build_ot_lp(instance: OTInstance) -> OTLinearProgram:
    """Convert an OT instance into the standard LP form used by solvers."""

    validate_ot_instance(instance)
    equality_matrix = build_transport_equality_constraints(
        n_source=instance.n_source,
        n_target=instance.n_target,
    )
    equality_rhs = np.concatenate([instance.source_weights, instance.target_weights]).astype(float)

    return OTLinearProgram(
        cost_vector=flatten_plan(instance.cost_matrix),
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        n_source=instance.n_source,
        n_target=instance.n_target,
        source_weights=np.asarray(instance.source_weights, dtype=float),
        target_weights=np.asarray(instance.target_weights, dtype=float),
    )


def transport_cost(cost_matrix: Array, plan: Array) -> float:
    """Compute the OT objective value <C, P>."""

    cost_matrix = np.asarray(cost_matrix, dtype=float)
    plan = np.asarray(plan, dtype=float)

    if cost_matrix.shape != plan.shape:
        raise ValueError("Cost matrix and plan must have the same shape.")
    return float(np.sum(cost_matrix * plan))


def marginal_violation(plan: Array, source_weights: Array, target_weights: Array) -> float:
    """Compute the L1 marginal residual used in the project write-up."""

    plan = np.asarray(plan, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    row_residual = np.abs(plan.sum(axis=1) - source_weights).sum()
    col_residual = np.abs(plan.sum(axis=0) - target_weights).sum()
    return float(row_residual + col_residual)


def solve_ot_lp(
    instance: OTInstance,
    method: str,
    scipy_options: dict[str, float | int | bool] | None = None,
    time_limit_milliseconds: int | None = None,
) -> OTLPSolution:
    """Dispatch to one of the exact OT LP solvers used in Part 1."""

    normalized_method = method.lower()
    if normalized_method == "simplex":
        return solve_ot_lp_simplex(instance, options=scipy_options)
    if normalized_method == "ipm":
        return solve_ot_lp_ipm(instance, options=scipy_options)
    if normalized_method == "pdlp":
        return solve_ot_lp_pdlp(instance, time_limit_milliseconds=time_limit_milliseconds)
    raise ValueError(f"Unsupported OT LP method: {method}")


def solve_ot_lp_simplex(
    instance: OTInstance,
    options: dict[str, float | int | bool] | None = None,
) -> OTLPSolution:
    """Solve the OT LP with HiGHS dual simplex."""

    return _solve_ot_lp_with_scipy(instance, method_name="simplex", highs_method="highs-ds", options=options)


def solve_ot_lp_ipm(
    instance: OTInstance,
    options: dict[str, float | int | bool] | None = None,
) -> OTLPSolution:
    """Solve the OT LP with HiGHS interior point."""

    return _solve_ot_lp_with_scipy(instance, method_name="ipm", highs_method="highs-ipm", options=options)


def solve_ot_lp_pdlp(
    instance: OTInstance,
    time_limit_milliseconds: int | None = None,
) -> OTLPSolution:
    """Solve the OT LP with OR-Tools PDLP."""

    validate_ot_instance(instance)
    solver = pywraplp.Solver.CreateSolver("PDLP")
    if solver is None:
        raise RuntimeError("OR-Tools PDLP solver is not available in this environment.")
    if time_limit_milliseconds is not None:
        solver.SetTimeLimit(time_limit_milliseconds)

    infinity = solver.infinity()
    flat_cost = flatten_plan(instance.cost_matrix)
    variables = [
        solver.NumVar(0.0, infinity, f"p_{index}")
        for index in range(flat_cost.size)
    ]

    objective = solver.Objective()
    for variable, coefficient in zip(variables, flat_cost, strict=True):
        objective.SetCoefficient(variable, float(coefficient))
    objective.SetMinimization()

    # Row sums correspond to contiguous blocks under row-major vectorization.
    for row_index, target_mass in enumerate(instance.source_weights):
        constraint = solver.Constraint(float(target_mass), float(target_mass), f"row_{row_index}")
        start = row_index * instance.n_target
        stop = start + instance.n_target
        for variable in variables[start:stop]:
            constraint.SetCoefficient(variable, 1.0)

    # Column sums pick every n_target-th variable starting at the column index.
    for col_index, target_mass in enumerate(instance.target_weights):
        constraint = solver.Constraint(float(target_mass), float(target_mass), f"col_{col_index}")
        for variable in variables[col_index::instance.n_target]:
            constraint.SetCoefficient(variable, 1.0)

    start_time = perf_counter()
    status_code = solver.Solve()
    runtime_seconds = perf_counter() - start_time

    success = status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    status = _ortools_status_name(status_code)
    if not success:
        return OTLPSolution(
            method="pdlp",
            status=status,
            success=False,
            plan=None,
            objective_value=None,
            marginal_violation=None,
            runtime_seconds=runtime_seconds,
            iterations=solver.iterations(),
            solver_message="PDLP did not return a feasible solution.",
        )

    plan = unflatten_plan(
        np.array([variable.solution_value() for variable in variables], dtype=float),
        n_source=instance.n_source,
        n_target=instance.n_target,
    )
    return OTLPSolution(
        method="pdlp",
        status=status,
        success=True,
        plan=plan,
        objective_value=transport_cost(instance.cost_matrix, plan),
        marginal_violation=marginal_violation(plan, instance.source_weights, instance.target_weights),
        runtime_seconds=runtime_seconds,
        iterations=solver.iterations(),
        solver_message="Solved with OR-Tools PDLP.",
    )


def _solve_ot_lp_with_scipy(
    instance: OTInstance,
    method_name: str,
    highs_method: str,
    options: dict[str, float | int | bool] | None = None,
) -> OTLPSolution:
    """Shared SciPy wrapper for the HiGHS LP solvers."""

    lp = build_ot_lp(instance)
    start_time = perf_counter()
    result = linprog(
        c=lp.cost_vector,
        A_eq=lp.equality_matrix,
        b_eq=lp.equality_rhs,
        bounds=(0.0, None),
        method=highs_method,
        options=options,
    )
    runtime_seconds = perf_counter() - start_time

    if not result.success:
        return OTLPSolution(
            method=method_name,
            status=str(result.status),
            success=False,
            plan=None,
            objective_value=None,
            marginal_violation=None,
            runtime_seconds=runtime_seconds,
            iterations=getattr(result, "nit", None),
            solver_message=result.message,
        )

    plan = unflatten_plan(result.x, n_source=lp.n_source, n_target=lp.n_target)
    return OTLPSolution(
        method=method_name,
        status="optimal",
        success=True,
        plan=plan,
        objective_value=transport_cost(instance.cost_matrix, plan),
        marginal_violation=marginal_violation(plan, lp.source_weights, lp.target_weights),
        runtime_seconds=runtime_seconds,
        iterations=getattr(result, "nit", None),
        solver_message=result.message,
    )


def _ortools_status_name(status_code: int) -> str:
    """Translate OR-Tools status codes into readable names."""

    status_names = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
    }
    return status_names.get(status_code, f"unknown_status_{status_code}")
