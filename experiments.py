from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from methods import als, gd_backtracking, lbfgs_solve
from objectives import (
    is_dense_problem,
    objective,
    problem_dense_matrix,
    problem_shape,
    relative_reconstruction_error,
    rmse_on_problem,
)
from plotting import (
    ensure_dir,
    plot_initialization_overlays,
    plot_initialization_summary,
    plot_landscape_scan,
    plot_method_histories,
    plot_rank_sensitivity,
    plot_regularization_sensitivity,
    plot_singular_value_decay,
)

Array = np.ndarray


def random_initialization(
    m: int,
    n: int,
    rank: int,
    scale: float = 0.1,
    seed: int = 0,
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    return (
        scale * rng.standard_normal((m, rank)),
        scale * rng.standard_normal((n, rank)),
    )


def svd_initialization(problem: Any, rank: int) -> tuple[Array, Array]:
    matrix = problem_dense_matrix(problem, fill_value=0.0)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s = s[:rank]
    root_s = np.sqrt(np.maximum(s, 0.0))
    u0 = u[:, :rank] * root_s
    v0 = vt[:rank, :].T * root_s
    return u0, v0


def run_all_methods(
    problem: Any,
    rank: int,
    lam: float,
    u0: Array,
    v0: Array,
    max_iter: int = 300,
    gd_kwargs: dict[str, Any] | None = None,
    lbfgs_kwargs: dict[str, Any] | None = None,
    als_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    gd_kwargs = gd_kwargs or {}
    lbfgs_kwargs = lbfgs_kwargs or {}
    als_kwargs = als_kwargs or {}
    return {
        "GD": gd_backtracking(problem, u0, v0, lam, max_iter=max_iter, **gd_kwargs),
        "L-BFGS": lbfgs_solve(problem, u0, v0, lam, max_iter=max_iter, **lbfgs_kwargs),
        "ALS": als(problem, u0, v0, lam, max_iter=max_iter, **als_kwargs),
    }


def _train_problem(dataset: dict[str, Any]) -> Any:
    return dataset.get("train_problem", dataset.get("problem", dataset.get("M")))


def _test_problem(dataset: dict[str, Any]) -> Any:
    return dataset.get("test_problem")


def _history_summary(
    dataset_name: str,
    rank: int,
    lam: float,
    histories: dict[str, dict[str, Any]],
    train_problem: Any,
    test_problem: Any | None = None,
    matrix_clean: Array | None = None,
) -> list[dict[str, Any]]:
    rows = []
    dense_train = is_dense_problem(train_problem)
    train_matrix = problem_dense_matrix(train_problem) if dense_train else None

    for method, history in histories.items():
        row = {
            "dataset": dataset_name,
            "method": method,
            "rank": rank,
            "lam": lam,
            "final_obj": history["obj"][-1],
            "final_grad_norm": history["grad_norm"][-1],
            "iterations": history["iterations"],
            "time_seconds": history["time"][-1],
            "train_rmse": rmse_on_problem(history["U_final"], history["V_final"], train_problem),
            "status": str(history["status"]),
        }
        if dense_train and train_matrix is not None:
            row["reconstruction_error"] = relative_reconstruction_error(
                history["U_final"], history["V_final"], train_matrix
            )
        if test_problem is not None:
            row["test_rmse"] = rmse_on_problem(history["U_final"], history["V_final"], test_problem)
        if matrix_clean is not None:
            row["recovery_error"] = relative_reconstruction_error(
                history["U_final"], history["V_final"], matrix_clean
            )
        rows.append(row)
    return rows


def _rank_metric_for_dataset(dataset: dict[str, Any]) -> tuple[str, str, str]:
    if _test_problem(dataset) is not None:
        return "test_rmse", "Test RMSE", f"{dataset['name']}: test RMSE vs rank"
    if dataset.get("M_clean") is not None:
        return "recovery_error", "Recovery error", f"{dataset['name']}: recovery error vs rank"
    return (
        "reconstruction_error",
        "Relative reconstruction error",
        f"{dataset['name']}: reconstruction error vs rank",
    )


def run_experiment_1(
    datasets: list[dict[str, Any]],
    output_dir: str | Path,
    results_dir: str | Path | None,
    rank_map: dict[str, int],
    lam: float = 1e-2,
    seed: int = 0,
    max_iter: int = 300,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(results_dir or output_dir)
    all_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        problem = _train_problem(dataset)
        rank = rank_map[dataset["name"]]
        m, n = problem_shape(problem)
        u0, v0 = random_initialization(m, n, rank, scale=0.1, seed=seed)
        histories = run_all_methods(problem, rank, lam, u0, v0, max_iter=max_iter)
        plot_method_histories(
            histories,
            title_prefix=f"Experiment 1 - {dataset['name']}",
            out_dir=output_dir / dataset["name"],
        )
        all_rows.extend(
            _history_summary(
                dataset["name"],
                rank,
                lam,
                histories,
                problem,
                test_problem=_test_problem(dataset),
                matrix_clean=dataset.get("M_clean"),
            )
        )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(results_dir / "experiment_1_summary.csv", index=False)
    return frame


def run_experiment_2(
    problem: Any,
    rank: int,
    lam: float,
    output_dir: str | Path,
    results_dir: str | Path | None = None,
    seeds: range = range(10),
    max_iter: int = 300,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(results_dir or output_dir)
    records: list[dict[str, Any]] = []
    convergence: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "GD": [],
        "L-BFGS": [],
        "ALS": [],
    }
    m, n = problem_shape(problem)

    for init_type, scale in [("small", 0.01), ("large", 1.0)]:
        for seed in seeds:
            u0, v0 = random_initialization(m, n, rank, scale=scale, seed=seed)
            histories = run_all_methods(problem, rank, lam, u0, v0, max_iter=max_iter)
            for method, history in histories.items():
                records.append(
                    {
                        "method": method,
                        "init_type": init_type,
                        "seed": seed,
                        "final_obj": history["obj"][-1],
                        "final_grad_norm": history["grad_norm"][-1],
                        "iterations": history["iterations"],
                        "time_seconds": history["time"][-1],
                    }
                )
                convergence[method].append((f"{init_type}-{seed}", history))

    u0, v0 = svd_initialization(problem, rank)
    histories = run_all_methods(problem, rank, lam, u0, v0, max_iter=max_iter)
    for method, history in histories.items():
        records.append(
            {
                "method": method,
                "init_type": "svd",
                "seed": -1,
                "final_obj": history["obj"][-1],
                "final_grad_norm": history["grad_norm"][-1],
                "iterations": history["iterations"],
                "time_seconds": history["time"][-1],
            }
        )
        convergence[method].append(("svd", history))

    plot_initialization_summary(records, output_dir / "initialization_summary.png")
    plot_initialization_overlays(convergence, output_dir / "initialization_overlays.png")

    frame = pd.DataFrame(records)
    frame.to_csv(results_dir / "experiment_2_summary.csv", index=False)
    return frame


def run_experiment_3(
    datasets: list[dict[str, Any]],
    rank_grid: dict[str, list[int]],
    lam: float,
    output_dir: str | Path,
    results_dir: str | Path | None = None,
    seed: int = 0,
    max_iter: int = 300,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(results_dir or output_dir)
    records: list[dict[str, Any]] = []

    for dataset in datasets:
        problem = _train_problem(dataset)
        matrix_for_svd = problem_dense_matrix(problem, fill_value=0.0)
        singular_values = np.linalg.svd(matrix_for_svd, compute_uv=False)
        dataset_dir = ensure_dir(output_dir / dataset["name"])
        title = (
            f"{dataset['name']} singular value decay"
            if is_dense_problem(problem)
            else f"{dataset['name']} zero-filled singular value decay"
        )
        plot_singular_value_decay(
            singular_values,
            title=title,
            out_path=dataset_dir / "singular_value_decay.png",
        )

        for rank in rank_grid[dataset["name"]]:
            m, n = problem_shape(problem)
            u0, v0 = random_initialization(m, n, rank, scale=0.1, seed=seed)
            histories = run_all_methods(problem, rank, lam, u0, v0, max_iter=max_iter)
            records.extend(
                _history_summary(
                    dataset["name"],
                    rank,
                    lam,
                    histories,
                    problem,
                    test_problem=_test_problem(dataset),
                    matrix_clean=dataset.get("M_clean"),
                )
            )

        dataset_records = [record for record in records if record["dataset"] == dataset["name"]]
        metric, ylabel, title = _rank_metric_for_dataset(dataset)
        plot_rank_sensitivity(
            dataset_records,
            metric=metric,
            ylabel=ylabel,
            title=title,
            out_path=dataset_dir / "reconstruction_error_vs_rank.png",
        )
        plot_rank_sensitivity(
            dataset_records,
            metric="iterations",
            ylabel="Iterations",
            title=f"{dataset['name']}: iterations vs rank",
            out_path=dataset_dir / "iterations_vs_rank.png",
        )

    frame = pd.DataFrame(records)
    frame.to_csv(results_dir / "experiment_3_summary.csv", index=False)
    return frame


def run_experiment_4(
    matrix: Array,
    matrix_clean: Array,
    rank: int,
    lam_grid: list[float],
    output_dir: str | Path,
    results_dir: str | Path | None = None,
    seed: int = 0,
    max_iter: int = 300,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(results_dir or output_dir)
    records: list[dict[str, Any]] = []

    for lam in lam_grid:
        m, n = matrix.shape
        u0, v0 = random_initialization(m, n, rank, scale=0.1, seed=seed)
        histories = run_all_methods(matrix, rank, lam, u0, v0, max_iter=max_iter)
        records.extend(
            _history_summary(
                "synthetic_regularization",
                rank,
                lam,
                histories,
                matrix,
                matrix_clean=matrix_clean,
            )
        )

    plot_regularization_sensitivity(
        records,
        metric="final_obj",
        ylabel="Final training objective",
        title="Regularization sensitivity: training objective",
        out_path=output_dir / "training_objective_vs_lambda.png",
    )
    plot_regularization_sensitivity(
        records,
        metric="recovery_error",
        ylabel="Recovery error to clean matrix",
        title="Regularization sensitivity: recovery error",
        out_path=output_dir / "recovery_error_vs_lambda.png",
    )

    frame = pd.DataFrame(records)
    frame.to_csv(results_dir / "experiment_4_summary.csv", index=False)
    return frame


def run_experiment_5(
    problem: Any,
    rank: int,
    lam: float,
    output_dir: str | Path,
    results_dir: str | Path | None = None,
    method: str = "GD",
    max_iter: int = 300,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    results_dir = ensure_dir(results_dir or output_dir)
    method_map = {
        "GD": gd_backtracking,
        "L-BFGS": lbfgs_solve,
        "ALS": als,
    }
    solver = method_map[method]

    candidates: list[dict[str, Any]] = []
    m, n = problem_shape(problem)
    for seed in range(8):
        u0, v0 = random_initialization(m, n, rank, scale=1.0, seed=seed)
        history = solver(problem, u0, v0, lam, max_iter=max_iter)
        candidates.append(
            {
                "seed": seed,
                "final_obj": history["obj"][-1],
                "U_final": history["U_final"],
                "V_final": history["V_final"],
            }
        )

    candidates.sort(key=lambda item: item["final_obj"])
    low = candidates[0]
    high = candidates[-1]

    t_values = np.linspace(0.0, 1.0, 100)
    values = np.array(
        [
            objective(
                problem,
                (1.0 - t) * low["U_final"] + t * high["U_final"],
                (1.0 - t) * low["V_final"] + t * high["V_final"],
                lam,
            )
            for t in t_values
        ]
    )

    plot_landscape_scan(
        t_values,
        values,
        title=f"Experiment 5 - {method} landscape scan",
        out_path=output_dir / "line_segment_landscape.png",
    )

    frame = pd.DataFrame(
        {
            "t": t_values,
            "objective": values,
            "low_seed": low["seed"],
            "high_seed": high["seed"],
        }
    )
    frame.to_csv(results_dir / "experiment_5_landscape.csv", index=False)
    return frame
