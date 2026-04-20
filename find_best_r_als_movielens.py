from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data import load_movielens_100k
from methods import als
from objectives import problem_shape, rmse_on_problem


def random_init(m: int, n: int, r: int, seed: int, scale: float = 0.1):
    rng = np.random.default_rng(seed)
    u0 = scale * rng.standard_normal((m, r))
    v0 = scale * rng.standard_normal((n, r))
    return u0, v0


def main():
    # ===== 可调参数 =====
    movielens_path = None  # 例如 "data/ml-100k"
    rank_grid = [5, 10, 20, 30, 40, 60]
    lam_grid = [1e-3, 1e-2, 1e-1]
    seeds = [0, 1, 2, 3, 4]

    max_iter = 800
    tol = 1e-8
    grad_tol = 1e-6
    init_scale = 0.1

    out_dir = Path("results/movielens_als_robust_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    # ===================

    dataset = load_movielens_100k(path=movielens_path, split_seed=0)
    train_problem = dataset["train_problem"]
    test_problem = dataset["test_problem"]
    m, n = problem_shape(train_problem)

    print(f"Loaded MovieLens: m={m}, n={n}, source={dataset['source']}")
    print(f"rank_grid={rank_grid}")
    print(f"lam_grid={lam_grid}")
    print(f"seeds={seeds}\n")

    per_run_records = []
    best_single = {
        "test_rmse": float("inf"),
        "rank": None,
        "lam": None,
        "seed": None,
        "U": None,
        "V": None,
    }

    for lam in lam_grid:
        for r in rank_grid:
            for seed in seeds:
                u0, v0 = random_init(m, n, r, seed=seed, scale=init_scale)
                hist = als(
                    problem=train_problem,
                    u0=u0,
                    v0=v0,
                    lam=lam,
                    max_iter=max_iter,
                    tol=tol,
                    grad_tol=grad_tol,
                    allow_pinv=True,
                )

                u_hat = hist["U_final"]
                v_hat = hist["V_final"]

                train_rmse = rmse_on_problem(u_hat, v_hat, train_problem)
                test_rmse = rmse_on_problem(u_hat, v_hat, test_problem)

                row = {
                    "lam": lam,
                    "rank": r,
                    "seed": seed,
                    "train_rmse": float(train_rmse),
                    "test_rmse": float(test_rmse),
                    "final_obj": float(hist["obj"][-1]),
                    "final_grad_norm": float(hist["grad_norm"][-1]),
                    "iterations": int(hist["iterations"]),
                    "status": str(hist["status"]),
                    "time_seconds": float(hist["time"][-1]),
                }
                per_run_records.append(row)

                print(
                    f"lam={lam:<6g} r={r:<3d} seed={seed:<2d} "
                    f"train={train_rmse:.4f} test={test_rmse:.4f} "
                    f"iter={hist['iterations']} status={hist['status']}"
                )

                if test_rmse < best_single["test_rmse"]:
                    best_single.update(
                        {
                            "test_rmse": float(test_rmse),
                            "rank": r,
                            "lam": lam,
                            "seed": seed,
                            "U": u_hat.copy(),
                            "V": v_hat.copy(),
                        }
                    )

    # 保存每次运行结果
    runs_df = pd.DataFrame(per_run_records)
    runs_df.to_csv(out_dir / "all_runs.csv", index=False)

    # 按 (lam, rank) 聚合：均值/标准差
    summary_df = (
        runs_df.groupby(["lam", "rank"], as_index=False)
        .agg(
            mean_train_rmse=("train_rmse", "mean"),
            std_train_rmse=("train_rmse", "std"),
            mean_test_rmse=("test_rmse", "mean"),
            std_test_rmse=("test_rmse", "std"),
            mean_iterations=("iterations", "mean"),
        )
        .sort_values("mean_test_rmse", ascending=True)
    )
    summary_df.to_csv(out_dir / "summary_by_lam_rank.csv", index=False)

    best_combo = summary_df.iloc[0]
    best_lam = float(best_combo["lam"])
    best_rank = int(best_combo["rank"])
    best_mean_test = float(best_combo["mean_test_rmse"])

    # 保存“单次最佳”U/V（方便你后续直接用）
    np.save(out_dir / "U_best_single.npy", best_single["U"])
    np.save(out_dir / "V_best_single.npy", best_single["V"])

    report = (
        "===== BEST BY MEAN TEST RMSE (across seeds) =====\n"
        f"best_lam={best_lam}\n"
        f"best_rank={best_rank}\n"
        f"best_mean_test_rmse={best_mean_test:.8f}\n\n"
        "===== BEST SINGLE RUN =====\n"
        f"lam={best_single['lam']}\n"
        f"rank={best_single['rank']}\n"
        f"seed={best_single['seed']}\n"
        f"test_rmse={best_single['test_rmse']:.8f}\n"
        f"U_shape={best_single['U'].shape}\n"
        f"V_shape={best_single['V'].shape}\n"
    )
    (out_dir / "best_report.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()