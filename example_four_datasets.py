"""
Example script demonstrating the four dataset comparison for LRMF project.

The four datasets provide comprehensive evaluation across different data characteristics:
1. Synthetic Dense (500 × 200, rank 5, dense, known ground truth)
2. Synthetic Sparse (500 × 200, rank 5, 80% sparse, known ground truth)
3. Superconductivity (21263 × 81, dense real-world scientific data)
4. MovieLens 100K (943 × 1682, 93.7% sparse, user-item ratings)

Each provides unique insights into algorithm performance.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from data import (
    load_synthetic,
    load_synthetic_sparse,
    load_superconductivity,
    load_movielens_100k,
)
from experiments import run_experiment_1
from objectives import problem_shape
from plotting import ensure_dir


def main():
    """Load and compare all four datasets."""
    
    print("=" * 80)
    print("FOUR DATASETS COMPARISON FOR LOW-RANK MATRIX FACTORIZATION")
    print("=" * 80)
    
    # =========================================================================
    # 1. SYNTHETIC DENSE
    # =========================================================================
    print("\n[1/4] Loading Synthetic Dense...")
    synthetic_dense = load_synthetic(
        m=500, n=200, r_true=5, sigma=0.1, seed=0
    )
    print(f"  Name: {synthetic_dense['name']}")
    print(f"  Shape: {synthetic_dense['shape']}")
    print(f"  Type: Dense (full matrix stored)")
    print(f"  Known rank: {synthetic_dense['r_true']}")
    print(f"  Noise level (σ): {synthetic_dense['sigma']}")
    print(f"  ✓ Has M_clean: Can measure recovery error to ground truth")
    print(f"  ✓ Has U_true, V_true: For validation")
    
    # =========================================================================
    # 2. SYNTHETIC SPARSE
    # =========================================================================
    print("\n[2/4] Loading Synthetic Sparse...")
    synthetic_sparse = load_synthetic_sparse(
        m=500, n=200, r_true=5, sigma=0.1, sparsity=0.8, seed=0
    )
    print(f"  Name: {synthetic_sparse['name']}")
    print(f"  Shape: {synthetic_sparse['shape']}")
    print(f"  Type: Sparse (80% missing, 20% observed)")
    print(f"  Non-zeros (nnz): {synthetic_sparse['nnz']}")
    print(f"  Sparsity pattern: {100*synthetic_sparse['sparsity']:.1f}% masked")
    print(f"  Known rank: {synthetic_sparse['r_true']}")
    print(f"  ✓ Same low-rank structure as dense version")
    print(f"  ✓ Has M_clean: Can measure recovery error")
    
    # =========================================================================
    # 3. SUPERCONDUCTIVITY
    # =========================================================================
    print("\n[3/4] Loading Superconductivity (Real Scientific Data)...")
    try:
        superconductivity = load_superconductivity()
        print(f"  Name: {superconductivity['name']}")
        print(f"  Shape: {superconductivity['shape']}")
        print(f"  Type: Dense (real-world feature matrix)")
        print(f"  Source: {superconductivity['source']}")
        print(f"  ✓ Large scale: {superconductivity['shape'][0]:,} × {superconductivity['shape'][1]}")
        print(f"  ✓ Tests computational efficiency on large dense data")
    except Exception as e:
        print(f"  ✗ Could not load: {e}")
        superconductivity = None
    
    # =========================================================================
    # 4. MOVIELENS 100K
    # =========================================================================
    print("\n[4/4] Loading MovieLens 100K (Real Recommender System)...")
    try:
        movielens = load_movielens_100k()
        m, n = problem_shape(movielens["train_problem"])
        print(f"  Name: {movielens['name']}")
        print(f"  Shape: {movielens['shape']}")
        print(f"  Type: Sparse (user × item ratings)")
        print(f"  Source: {movielens['source']}")
        print(f"  Train entries: {movielens['train_problem']['nnz']:,}")
        print(f"  Test entries: {movielens['test_problem']['nnz']:,}")
        print(f"  Train/test split: {100*movielens['train_problem']['nnz']/(movielens['train_problem']['nnz']+movielens['test_problem']['nnz']):.1f}% / {100*movielens['test_problem']['nnz']/(movielens['train_problem']['nnz']+movielens['test_problem']['nnz']):.1f}%")
        print(f"  Sparsity: ~93.7% missing entries")
        print(f"  ✓ Tests realistic sparse scenarios")
        print(f"  ✓ Has train/test split: Can measure test RMSE")
    except Exception as e:
        print(f"  ✗ Could not load: {e}")
        movielens = None
    
    # =========================================================================
    # COMPARISON SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATASET COMPARISON TABLE")
    print("=" * 80)
    
    datasets_info = []
    
    # Synthetic Dense
    datasets_info.append({
        "Dataset": "Synthetic Dense",
        "Rows": synthetic_dense["shape"][0],
        "Cols": synthetic_dense["shape"][1],
        "Type": "Dense",
        "Known Rank": synthetic_dense["r_true"],
        "Entries": f"{synthetic_dense['shape'][0] * synthetic_dense['shape'][1]:,}",
        "Purpose": "Correctness validation",
    })
    
    # Synthetic Sparse
    datasets_info.append({
        "Dataset": "Synthetic Sparse",
        "Rows": synthetic_sparse["shape"][0],
        "Cols": synthetic_sparse["shape"][1],
        "Type": "80% Sparse",
        "Known Rank": synthetic_sparse["r_true"],
        "Entries": f"{synthetic_sparse['nnz']:,}",
        "Purpose": "Sparse handling",
    })
    
    # Superconductivity
    if superconductivity:
        datasets_info.append({
            "Dataset": "Superconductivity",
            "Rows": superconductivity["shape"][0],
            "Cols": superconductivity["shape"][1],
            "Type": "Dense",
            "Known Rank": "Unknown",
            "Entries": f"{superconductivity['shape'][0] * superconductivity['shape'][1]:,}",
            "Purpose": "Scalability test",
        })
    
    # MovieLens
    if movielens:
        m, n = problem_shape(movielens["train_problem"])
        datasets_info.append({
            "Dataset": "MovieLens 100K",
            "Rows": m,
            "Cols": n,
            "Type": "Sparse",
            "Known Rank": "Unknown",
            "Entries": f"{movielens['train_problem']['nnz']:,}",
            "Purpose": "Real recommender",
        })
    
    df_comparison = pd.DataFrame(datasets_info)
    print(df_comparison.to_string(index=False))
    
    # =========================================================================
    # WHY FOUR DATASETS?
    # =========================================================================
    print("\n" + "=" * 80)
    print("WHY USE FOUR DATASETS?")
    print("=" * 80)
    
    rationale = """
    1. CONTROL & VALIDATION (Synthetic Dense)
       → Know ground truth, measure recovery error
       → Validate algorithm correctness
       → Pure dense problem, no sparsity complications
    
    2. SPARSE HANDLING (Synthetic Sparse)
       → Same rank structure, but 80% missing
       → Tests sparse-specific algorithms (ALS)
       → Controlled comparison: same signal, different sparsity pattern
       → Demonstrates initialization importance
    
    3. SCALABILITY (Superconductivity - Dense)
       → Large scale: 21K × 81 matrix
       → Tests computational efficiency on real dense data
       → L-BFGS should excel here (fewer iterations)
       → Reveals per-iteration cost differences
    
    4. REALISM (MovieLens - Sparse)
       → Real application: user-item recommender system
       → Natural sparsity (millions of ratings, missing data)
       → Has train/test split: evaluate generalization
       → Proves algorithm works on production-like data
    
    KEY INSIGHT: Different algorithms shine on different datasets!
    → GD: Works everywhere (stable baseline)
    → L-BFGS: Best on dense (fewer iterations)
    → ALS: Best on sparse (native missing data handling)
    """
    print(rationale)
    
    # =========================================================================
    # RUNNING EXPERIMENTS
    # =========================================================================
    print("=" * 80)
    print("RUNNING EXPERIMENT 1 (Method Comparison)")
    print("=" * 80)
    
    # Assemble datasets for experiment
    datasets = [synthetic_dense, synthetic_sparse]
    if superconductivity:
        datasets.append(superconductivity)
    if movielens:
        datasets.append(movielens)
    
    # Set rank for each dataset
    rank_map = {
        "synthetic": 5,
        "synthetic_sparse": 5,
        "superconductivity": 10,
        "movielens_100k": 20,
    }
    
    print(f"\nRunning experiment on {len(datasets)} datasets...")
    print(f"Datasets: {[d['name'] for d in datasets]}")
    print(f"Rank map: {rank_map}")
    
    output_dir = Path("figures/experiment_1_four_datasets")
    results_dir = Path("results/experiment_1_four_datasets")
    
    summary = run_experiment_1(
        datasets=datasets,
        output_dir=output_dir,
        results_dir=results_dir,
        rank_map=rank_map,
        lam=1e-2,
        seed=0,
        max_iter=250,
    )
    
    print(f"\n✓ Experiment complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {output_dir}")
    
    print("\nExperiment 1 Summary (first 20 rows):")
    print(summary.head(20).to_string(index=False))
    
    # =========================================================================
    # INTERPRETATION GUIDE
    # =========================================================================
    print("\n" + "=" * 80)
    print("HOW TO INTERPRET RESULTS")
    print("=" * 80)
    
    interpretation = """
    SYNTHETIC DENSE:
    ├─ Columns: reconstruction_error (measure of recovery vs original matrix)
    ├─ Use to validate: Do algorithms recover the true low-rank structure?
    └─ Compare GD, L-BFGS, ALS on same known problem
    
    SYNTHETIC SPARSE:
    ├─ Columns: reconstruction_error (recovery error to clean full matrix)
    ├─ Use to test: How well do algorithms handle 80% missing data?
    ├─ Key question: Does ALS outperform GD/L-BFGS on sparse?
    └─ Insight: Initialization quality matters more with high sparsity
    
    SUPERCONDUCTIVITY:
    ├─ Columns: iterations, time_seconds (computational efficiency)
    ├─ Use to measure: Wall-clock time on large dense matrix
    ├─ Key question: Which method is fastest in practice?
    └─ Insight: L-BFGS fewer iterations, but GD simpler per-iteration
    
    MOVIELENS 100K:
    ├─ Columns: test_rmse (generalization error on held-out ratings)
    ├─ Use to validate: Does the learned factorization predict new ratings?
    ├─ Key question: Which method generalizes best?
    └─ Insight: ALS natural for MovieLens, but all methods should work well
    """
    print(interpretation)


if __name__ == "__main__":
    main()
