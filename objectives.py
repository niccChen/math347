from __future__ import annotations

from typing import Any

import numpy as np

Array = np.ndarray
Problem = dict[str, Any]


def pack_factors(u: Array, v: Array) -> Array:
    return np.concatenate([u.ravel(), v.ravel()])


def unpack_factors(x: Array, m: int, n: int, rank: int) -> tuple[Array, Array]:
    split = m * rank
    u = x[:split].reshape(m, rank)
    v = x[split:].reshape(n, rank)
    return u, v


def make_dense_problem(matrix: Array, name: str = "dense") -> Problem:
    matrix = np.asarray(matrix, dtype=float)
    return {
        "kind": "dense",
        "name": name,
        "matrix": matrix,
        "shape": matrix.shape,
    }


def _build_sparse_lookup(
    rows: Array,
    cols: Array,
    values: Array,
    shape: tuple[int, int],
) -> tuple[list[Array], list[Array], list[Array], list[Array]]:
    m, n = shape
    row_to_cols: list[list[int]] = [[] for _ in range(m)]
    row_to_vals: list[list[float]] = [[] for _ in range(m)]
    col_to_rows: list[list[int]] = [[] for _ in range(n)]
    col_to_vals: list[list[float]] = [[] for _ in range(n)]

    for row, col, value in zip(rows, cols, values):
        row_to_cols[int(row)].append(int(col))
        row_to_vals[int(row)].append(float(value))
        col_to_rows[int(col)].append(int(row))
        col_to_vals[int(col)].append(float(value))

    row_cols_arrays = [np.asarray(items, dtype=int) for items in row_to_cols]
    row_vals_arrays = [np.asarray(vals, dtype=float) for vals in row_to_vals]
    col_rows_arrays = [np.asarray(items, dtype=int) for items in col_to_rows]
    col_vals_arrays = [np.asarray(vals, dtype=float) for vals in col_to_vals]
    return row_cols_arrays, row_vals_arrays, col_rows_arrays, col_vals_arrays


def make_sparse_problem(
    rows: Array,
    cols: Array,
    values: Array,
    shape: tuple[int, int],
    name: str = "sparse",
) -> Problem:
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    values = np.asarray(values, dtype=float)
    if rows.shape != cols.shape or rows.shape != values.shape:
        raise ValueError("rows, cols, and values must have the same shape.")
    if rows.ndim != 1:
        raise ValueError("rows, cols, and values must be one-dimensional arrays.")

    row_to_cols, row_to_vals, col_to_rows, col_to_vals = _build_sparse_lookup(
        rows, cols, values, shape
    )
    return {
        "kind": "sparse",
        "name": name,
        "rows": rows,
        "cols": cols,
        "values": values,
        "shape": shape,
        "nnz": int(values.size),
        "row_to_cols": row_to_cols,
        "row_to_vals": row_to_vals,
        "col_to_rows": col_to_rows,
        "col_to_vals": col_to_vals,
    }


def _normalize_problem(problem: Problem | Array) -> Problem:
    if isinstance(problem, dict):
        return problem
    return make_dense_problem(problem)


def is_dense_problem(problem: Problem | Array) -> bool:
    return _normalize_problem(problem)["kind"] == "dense"


def problem_shape(problem: Problem | Array) -> tuple[int, int]:
    return tuple(_normalize_problem(problem)["shape"])


def problem_dense_matrix(problem: Problem | Array, fill_value: float = 0.0) -> Array:
    normalized = _normalize_problem(problem)
    if normalized["kind"] == "dense":
        return np.asarray(normalized["matrix"], dtype=float)

    dense = np.full(normalized["shape"], fill_value, dtype=float)
    dense[normalized["rows"], normalized["cols"]] = normalized["values"]
    return dense


def _dense_residual(u: Array, v: Array, matrix: Array) -> Array:
    return u @ v.T - matrix


def _sparse_predictions(u: Array, v: Array, rows: Array, cols: Array) -> Array:
    return np.sum(u[rows] * v[cols], axis=1)


def objective(problem: Problem | Array, u: Array, v: Array, lam: float) -> float:
    normalized = _normalize_problem(problem)
    reg = np.sum(u * u) + np.sum(v * v)
    if normalized["kind"] == "dense":
        res = _dense_residual(u, v, normalized["matrix"])
        loss = np.sum(res * res)
    else:
        res = _sparse_predictions(u, v, normalized["rows"], normalized["cols"]) - normalized["values"]
        loss = np.dot(res, res)
    return 0.5 * float(loss + lam * reg)


def gradients(problem: Problem | Array, u: Array, v: Array, lam: float) -> tuple[Array, Array]:
    _, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
    return grad_u, grad_v


def objective_and_gradient(
    problem: Problem | Array,
    u: Array,
    v: Array,
    lam: float,
) -> tuple[float, Array, Array]:
    normalized = _normalize_problem(problem)
    reg = np.sum(u * u) + np.sum(v * v)

    if normalized["kind"] == "dense":
        res = _dense_residual(u, v, normalized["matrix"])
        obj = 0.5 * float(np.sum(res * res) + lam * reg)
        grad_u = res @ v + lam * u
        grad_v = res.T @ u + lam * v
        return obj, grad_u, grad_v

    rows = normalized["rows"]
    cols = normalized["cols"]
    values = normalized["values"]
    u_obs = u[rows]
    v_obs = v[cols]
    res = np.sum(u_obs * v_obs, axis=1) - values
    obj = 0.5 * float(np.dot(res, res) + lam * reg)

    grad_u = lam * u.copy()
    grad_v = lam * v.copy()
    np.add.at(grad_u, rows, res[:, None] * v_obs)
    np.add.at(grad_v, cols, res[:, None] * u_obs)
    return obj, grad_u, grad_v


def flat_objective_and_gradient(
    x: Array,
    problem: Problem | Array,
    rank: int,
    lam: float,
) -> tuple[float, Array]:
    m, n = problem_shape(problem)
    u, v = unpack_factors(x, m, n, rank)
    obj, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
    return obj, pack_factors(grad_u, grad_v)


def gradient_norm(grad_u: Array, grad_v: Array) -> float:
    return float(np.sqrt(np.sum(grad_u * grad_u) + np.sum(grad_v * grad_v)))


def relative_reconstruction_error(u: Array, v: Array, matrix: Array) -> float:
    numerator = np.linalg.norm(u @ v.T - matrix, ord="fro")
    denominator = np.linalg.norm(matrix, ord="fro")
    return float(numerator / max(denominator, 1e-12))


def rmse_on_problem(u: Array, v: Array, problem: Problem | Array) -> float:
    normalized = _normalize_problem(problem)
    if normalized["kind"] == "dense":
        residual = u @ v.T - normalized["matrix"]
        return float(np.sqrt(np.mean(residual * residual)))

    residual = _sparse_predictions(u, v, normalized["rows"], normalized["cols"]) - normalized["values"]
    return float(np.sqrt(np.mean(residual * residual)))


def check_gradient(
    problem: Problem | Array,
    u: Array,
    v: Array,
    lam: float,
    eps: float = 1e-6,
    num_checks: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    _, grad_u, grad_v = objective_and_gradient(problem, u, v, lam)
    records: list[dict[str, float | str | int]] = []

    for _ in range(num_checks):
        i = int(rng.integers(u.shape[0]))
        j = int(rng.integers(u.shape[1]))
        u_plus = u.copy()
        u_minus = u.copy()
        u_plus[i, j] += eps
        u_minus[i, j] -= eps
        numerical = (
            objective(problem, u_plus, v, lam) - objective(problem, u_minus, v, lam)
        ) / (2.0 * eps)
        analytic = float(grad_u[i, j])
        denom = max(1.0, abs(numerical), abs(analytic))
        records.append(
            {
                "block": "U",
                "i": i,
                "j": j,
                "analytic": analytic,
                "numerical": float(numerical),
                "relative_error": float(abs(numerical - analytic) / denom),
            }
        )

    for _ in range(num_checks):
        i = int(rng.integers(v.shape[0]))
        j = int(rng.integers(v.shape[1]))
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i, j] += eps
        v_minus[i, j] -= eps
        numerical = (
            objective(problem, u, v_plus, lam) - objective(problem, u, v_minus, lam)
        ) / (2.0 * eps)
        analytic = float(grad_v[i, j])
        denom = max(1.0, abs(numerical), abs(analytic))
        records.append(
            {
                "block": "V",
                "i": i,
                "j": j,
                "analytic": analytic,
                "numerical": float(numerical),
                "relative_error": float(abs(numerical - analytic) / denom),
            }
        )

    max_error = max(record["relative_error"] for record in records)
    return {"max_relative_error": float(max_error), "checks": records}
