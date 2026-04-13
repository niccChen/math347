from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from objectives import (
    flat_objective_and_gradient,
    gradient_norm,
    is_dense_problem,
    objective,
    objective_and_gradient,
    pack_factors,
    problem_shape,
    unpack_factors,
)

Array = np.ndarray


def _init_history(method: str, problem: Any, u: Array, v: Array, lam: float) -> dict[str, Any]:
    obj, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
    return {
        "method": method,
        "obj": [obj],
        "grad_norm": [gradient_norm(grad_u, grad_v)],
        "time": [0.0],
        "step_size": [],
        "U_final": u.copy(),
        "V_final": v.copy(),
        "status": "initialized",
        "iterations": 0,
    }


def _append_history(
    history: dict[str, Any],
    start_time: float,
    problem: Any,
    u: Array,
    v: Array,
    lam: float,
    step_size: float | None = None,
) -> tuple[float, float]:
    obj, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
    history["obj"].append(obj)
    history["grad_norm"].append(gradient_norm(grad_u, grad_v))
    history["time"].append(time.perf_counter() - start_time)
    if step_size is not None:
        history["step_size"].append(step_size)
    history["U_final"] = u.copy()
    history["V_final"] = v.copy()
    history["iterations"] = len(history["obj"]) - 1
    return obj, history["grad_norm"][-1]


def gd_backtracking(
    problem: Any,
    u0: Array,
    v0: Array,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-6,
    initial_step: float = 1.0,
    armijo_c: float = 1e-4,
    rho: float = 0.5,
    min_step: float = 1e-12,
) -> dict[str, Any]:
    u = u0.copy()
    v = v0.copy()
    history = _init_history("GD", problem, u, v, lam)
    start_time = time.perf_counter()
    step_guess = initial_step

    for _ in range(max_iter):
        obj, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
        grad_sq = float(np.sum(grad_u * grad_u) + np.sum(grad_v * grad_v))
        grad_norm_value = float(np.sqrt(grad_sq))
        if grad_norm_value <= tol:
            history["status"] = "converged"
            break

        step = step_guess
        accepted = False
        while step >= min_step:
            candidate_u = u - step * grad_u
            candidate_v = v - step * grad_v
            candidate_obj = objective(problem, candidate_u, candidate_v, lam)
            if candidate_obj <= obj - armijo_c * step * grad_sq:
                accepted = True
                break
            step *= rho

        if not accepted:
            history["status"] = "line_search_failed"
            break

        u = candidate_u
        v = candidate_v
        _append_history(history, start_time, problem, u, v, lam, step_size=step)
        step_guess = min(2.0 * step, initial_step)
    else:
        history["status"] = "max_iter"

    return history


def _solve_linear_system(gram: Array, rhs: Array, allow_pinv: bool = True) -> Array:
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        if not allow_pinv:
            raise
        return np.linalg.pinv(gram) @ rhs


def _als_dense(
    problem: Any,
    u0: Array,
    v0: Array,
    lam: float,
    max_iter: int,
    tol: float,
    grad_tol: float,
    allow_pinv: bool,
) -> dict[str, Any]:
    dense_problem = problem if isinstance(problem, dict) else {"matrix": problem}
    matrix = dense_problem["matrix"] if "matrix" in dense_problem else problem
    u = u0.copy()
    v = v0.copy()
    rank = u.shape[1]
    history = _init_history("ALS", problem, u, v, lam)
    start_time = time.perf_counter()
    identity = np.eye(rank)

    previous_obj = history["obj"][-1]

    for _ in range(max_iter):
        gram_u = u.T @ u + lam * identity
        v = _solve_linear_system(gram_u, (matrix.T @ u).T, allow_pinv=allow_pinv).T

        gram_v = v.T @ v + lam * identity
        u = _solve_linear_system(gram_v, (matrix @ v).T, allow_pinv=allow_pinv).T

        current_obj, current_grad_norm = _append_history(
            history, start_time, problem, u, v, lam
        )
        if current_grad_norm <= grad_tol:
            history["status"] = "converged"
            break

        rel_change = abs(previous_obj - current_obj) / max(1.0, abs(previous_obj))
        if rel_change <= tol:
            history["status"] = "objective_stalled"
            break
        previous_obj = current_obj
    else:
        history["status"] = "max_iter"

    return history


def _als_sparse(
    problem: dict[str, Any],
    u0: Array,
    v0: Array,
    lam: float,
    max_iter: int,
    tol: float,
    grad_tol: float,
    allow_pinv: bool,
) -> dict[str, Any]:
    u = u0.copy()
    v = v0.copy()
    m, n = problem_shape(problem)
    rank = u.shape[1]
    history = _init_history("ALS", problem, u, v, lam)
    start_time = time.perf_counter()
    identity = np.eye(rank)
    previous_obj = history["obj"][-1]

    row_to_cols = problem["row_to_cols"]
    row_to_vals = problem["row_to_vals"]
    col_to_rows = problem["col_to_rows"]
    col_to_vals = problem["col_to_vals"]

    for _ in range(max_iter):
        for i in range(m):
            cols = row_to_cols[i]
            vals = row_to_vals[i]
            if cols.size == 0:
                u[i] = 0.0
                continue
            v_obs = v[cols]
            gram = v_obs.T @ v_obs + lam * identity
            rhs = v_obs.T @ vals
            u[i] = _solve_linear_system(gram, rhs, allow_pinv=allow_pinv)

        for j in range(n):
            rows = col_to_rows[j]
            vals = col_to_vals[j]
            if rows.size == 0:
                v[j] = 0.0
                continue
            u_obs = u[rows]
            gram = u_obs.T @ u_obs + lam * identity
            rhs = u_obs.T @ vals
            v[j] = _solve_linear_system(gram, rhs, allow_pinv=allow_pinv)

        current_obj, current_grad_norm = _append_history(
            history, start_time, problem, u, v, lam
        )
        if current_grad_norm <= grad_tol:
            history["status"] = "converged"
            break

        rel_change = abs(previous_obj - current_obj) / max(1.0, abs(previous_obj))
        if rel_change <= tol:
            history["status"] = "objective_stalled"
            break
        previous_obj = current_obj
    else:
        history["status"] = "max_iter"

    return history


def als(
    problem: Any,
    u0: Array,
    v0: Array,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-8,
    grad_tol: float = 1e-6,
    allow_pinv: bool = True,
) -> dict[str, Any]:
    if is_dense_problem(problem):
        return _als_dense(problem, u0, v0, lam, max_iter, tol, grad_tol, allow_pinv)
    return _als_sparse(problem, u0, v0, lam, max_iter, tol, grad_tol, allow_pinv)


def lbfgs_solve(
    problem: Any,
    u0: Array,
    v0: Array,
    lam: float,
    max_iter: int = 500,
    tol: float = 1e-6,
    memory: int = 10,
) -> dict[str, Any]:
    rank = u0.shape[1]
    m, n = problem_shape(problem)
    x0 = pack_factors(u0, v0)
    start_time = time.perf_counter()
    history = {
        "method": "L-BFGS",
        "obj": [],
        "grad_norm": [],
        "time": [],
        "step_size": [],
        "U_final": u0.copy(),
        "V_final": v0.copy(),
        "status": "initialized",
        "iterations": 0,
    }

    def record_state(x: Array) -> None:
        u, v = unpack_factors(x, m, n, rank)
        obj, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
        history["obj"].append(obj)
        history["grad_norm"].append(gradient_norm(grad_u, grad_v))
        history["time"].append(time.perf_counter() - start_time)
        history["U_final"] = u.copy()
        history["V_final"] = v.copy()
        history["iterations"] = len(history["obj"]) - 1

    def callback(xk: Array) -> None:
        record_state(xk)

    record_state(x0)
    result = minimize(
        fun=lambda x: flat_objective_and_gradient(x, problem, rank, lam),
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={
            "maxiter": max_iter,
            "gtol": tol,
            "maxcor": memory,
            "ftol": 1e-12,
        },
    )

    final_u, final_v = unpack_factors(result.x, m, n, rank)
    final_obj, grad_u, grad_v = objective_and_gradient(problem, final_u, final_v, lam)
    final_grad_norm = gradient_norm(grad_u, grad_v)
    if (
        not history["obj"]
        or abs(history["obj"][-1] - final_obj) > 1e-12
        or abs(history["grad_norm"][-1] - final_grad_norm) > 1e-12
    ):
        record_state(result.x)

    history["U_final"] = final_u
    history["V_final"] = final_v
    history["status"] = result.message
    history["iterations"] = int(result.nit)
    history["result"] = result
    return history
