from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data import load_movielens_100k, load_superconductivity, load_synthetic, load_synthetic_sparse
from experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
)
from objectives import check_gradient, problem_shape
from plotting import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-rank matrix factorization project runner")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--superconductivity-path", default=None)
    parser.add_argument("--movielens-path", default=None)
    parser.add_argument("--include-movielens", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)

    if args.mode == "quick":
        synthetic_dense = load_synthetic(m=120, n=80, r_true=5, sigma=0.1, seed=args.seed)
        rank = 5
        lam = 1e-2
        rng = np.random.default_rng(args.seed)
        u0 = 0.1 * rng.standard_normal((synthetic_dense["shape"][0], rank))
        v0 = 0.1 * rng.standard_normal((synthetic_dense["shape"][1], rank))
        grad_report = check_gradient(synthetic_dense["problem"], u0, v0, lam, num_checks=5, seed=args.seed)
        summary = run_experiment_1(
            datasets=[synthetic_dense],
            output_dir=Path(args.output_dir) / "quick",
            results_dir=Path(args.results_dir) / "quick",
            rank_map={"synthetic_dense": rank},
            lam=lam,
            seed=args.seed,
            max_iter=80,
        )
        print(f"Gradient check max relative error: {grad_report['max_relative_error']:.3e}")
        print(summary.to_string(index=False))
        return

    synthetic_dense = load_synthetic(m=500, n=200, r_true=5, sigma=0.1, seed=args.seed)
    datasets = [synthetic_dense]

    # Add synthetic sparse with 80% sparsity
    synthetic_sparse = load_synthetic_sparse(m=500, n=200, r_true=5, sigma=0.1, sparsity=0.8, seed=args.seed)
    datasets.append(synthetic_sparse)

    if not args.skip_real:
        try:
            superconductivity = load_superconductivity(args.superconductivity_path)
            datasets.append(superconductivity)
        except Exception as exc:
            print(f"Skipping superconductivity dataset: {exc}")

    if args.include_movielens or args.movielens_path:
        try:
            movielens = load_movielens_100k(args.movielens_path, split_seed=args.seed)
            datasets.append(movielens)
        except Exception as exc:
            print(f"Skipping MovieLens dataset: {exc}")

    rank = 5
    lam = 1e-2
    rng = np.random.default_rng(args.seed)
    u0 = 0.1 * rng.standard_normal((synthetic_dense["shape"][0], rank))
    v0 = 0.1 * rng.standard_normal((synthetic_dense["shape"][1], rank))
    grad_report = check_gradient(synthetic_dense["problem"], u0, v0, lam, num_checks=5, seed=args.seed)
    print(f"Gradient check max relative error: {grad_report['max_relative_error']:.3e}")

    rank_map = {"synthetic_dense": 5, "synthetic_sparse": 5, "superconductivity": 10, "movielens_100k": 20}
    exp1 = run_experiment_1(
        datasets=datasets,
        output_dir=Path(args.output_dir) / "experiment_1",
        results_dir=Path(args.results_dir) / "experiment_1",
        rank_map={name: rank_map[name] for name in [dataset["name"] for dataset in datasets]},
        lam=lam,
        seed=args.seed,
        max_iter=250,
    )
    print(f"Experiment 1 rows: {len(exp1)}")

    exp2 = run_experiment_2(
        problem=synthetic_dense["problem"],
        rank=5,
        lam=lam,
        output_dir=Path(args.output_dir) / "experiment_2",
        results_dir=Path(args.results_dir) / "experiment_2",
        seeds=range(10),
        max_iter=200,
    )
    print(f"Experiment 2 rows: {len(exp2)}")

    rank_grid = {"synthetic_dense": [2, 3, 5, 8, 10, 20], "synthetic_sparse": [2, 3, 5, 8, 10, 20]}
    dataset_names = {dataset["name"] for dataset in datasets}
    if "superconductivity" in dataset_names:
        rank_grid["superconductivity"] = [2, 5, 10, 20, 40, 60]
    if "movielens_100k" in dataset_names:
        rank_grid["movielens_100k"] = [5, 10, 20, 40, 60]

    exp3 = run_experiment_3(
        datasets=datasets,
        rank_grid=rank_grid,
        lam=lam,
        output_dir=Path(args.output_dir) / "experiment_3",
        results_dir=Path(args.results_dir) / "experiment_3",
        seed=args.seed,
        max_iter=180,
    )
    print(f"Experiment 3 rows: {len(exp3)}")

    noisy_synthetic = load_synthetic(m=500, n=200, r_true=5, sigma=0.5, seed=args.seed)
    exp4 = run_experiment_4(
        matrix=noisy_synthetic["M"],
        matrix_clean=noisy_synthetic["M_clean"],
        rank=5,
        lam_grid=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        output_dir=Path(args.output_dir) / "experiment_4",
        results_dir=Path(args.results_dir) / "experiment_4",
        seed=args.seed,
        max_iter=200,
    )
    print(f"Experiment 4 rows: {len(exp4)}")

    exp5 = run_experiment_5(
        problem=synthetic["problem"],
        rank=5,
        lam=lam,
        output_dir=Path(args.output_dir) / "experiment_5",
        results_dir=Path(args.results_dir) / "experiment_5",
        method="GD",
        max_iter=200,
    )
    print(f"Experiment 5 rows: {len(exp5)}")

    if "movielens_100k" in dataset_names:
        movie_dataset = next(dataset for dataset in datasets if dataset["name"] == "movielens_100k")
        m, n = problem_shape(movie_dataset["train_problem"])
        print(f"MovieLens loaded with shape {(m, n)} from {movie_dataset['source']}")


if __name__ == "__main__":
    main()
