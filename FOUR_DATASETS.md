# Four Datasets Comparison for Low-Rank Matrix Factorization

## Overview

The LRMF project now includes a **four-dataset comparison framework** to comprehensively evaluate optimization algorithms across different data characteristics.

### The Four Datasets

| Dataset | Size | Type | Characteristics | Purpose |
|---------|------|------|-----------------|---------|
| **Synthetic Dense** | 500 × 200 | Dense | Rank 5, σ=0.1 noise, known ground truth | ✓ Validate algorithm correctness |
| **Synthetic Sparse** | 500 × 200 | 80% Sparse | Rank 5, σ=0.1 noise, 80% missing, known ground truth | ✓ Test sparse handling with controlled structure |
| **Superconductivity** | 21263 × 81 | Dense | Real scientific data, 1.7M entries | ✓ Evaluate scalability on large dense matrices |
| **MovieLens 100K** | 943 × 1682 | 93.7% Sparse | Real recommender system, 100K ratings | ✓ Validate on production-like application |

---

## Why Use Four Datasets?

### 1. **Control & Validation (Synthetic Dense)**
- Know the exact low-rank structure: $M = U_{\text{true}} V_{\text{true}}^T + \sigma \cdot \mathcal{N}(0,1)$
- Measure **recovery error** to ground truth
- Verify algorithms converge to correct solution
- Pure dense problem without sparsity complications

### 2. **Sparse Handling (Synthetic Sparse)** - **NEW**
- **Same rank structure** as synthetic dense
- Random masking creates **80% sparsity**
- Controlled comparison: signal identical, sparsity different
- Tests whether ALS outperforms GD/L-BFGS on sparse data
- Reveals: initialization quality matters more with high sparsity

### 3. **Scalability (Superconductivity - Dense)**
- Large scale: $21,263 \times 81$ matrix (1.7M entries)
- Tests computational efficiency on real-world dense data
- Where does **L-BFGS excel** (fewer iterations)?
- Where is **GD more efficient** (simpler per-iteration cost)?

### 4. **Realism (MovieLens - Sparse)**
- Real application: user-item recommender system
- Natural sparsity pattern (millions of ratings, many missing)
- Train/test split: evaluate **generalization** via test RMSE
- Proves algorithms work on production-like data

---

## Key Insight: Different Algorithms, Different Datasets

```
Algorithm Performance Profile:
┌─────────────────────────────────────────────────────────┐
│                                                           │
│  GD (Gradient Descent)          ↔ Stable everywhere     │
│  ├─ Synthetic Dense              👍 Works fine           │
│  ├─ Synthetic Sparse             👍 Works fine (slower)  │
│  ├─ Superconductivity            👍 Works fine (slow)    │
│  └─ MovieLens                     👍 Works fine           │
│                                                           │
│  L-BFGS (Quasi-Newton)          ↔ Best on dense         │
│  ├─ Synthetic Dense              🏆 Fewer iterations    │
│  ├─ Synthetic Sparse             ⚠️  Can struggle        │
│  ├─ Superconductivity            🏆 Fast wall-clock      │
│  └─ MovieLens                     ⚠️  Not ideal           │
│                                                           │
│  ALS (Alternating LS)           ↔ Best on sparse        │
│  ├─ Synthetic Dense              ⚠️  Less optimal         │
│  ├─ Synthetic Sparse             🏆 Natural fit         │
│  ├─ Superconductivity            ⚠️  Extra overhead      │
│  └─ MovieLens                     🏆 Designed for this   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## Using the Four Datasets

### Quick Test

```bash
# Test all four datasets (first 20 entries)
python example_four_datasets.py
```

This script:
1. Loads all four datasets
2. Displays dataset characteristics
3. Shows comparison table
4. Runs Experiment 1 (method comparison) on all four
5. Prints interpretation guide

### Running via Main Script

```bash
# Full mode with all four datasets
python main.py --mode full

# Quick mode with reduced data (testing only)
python main.py --mode quick
```

The main.py automatically:
- Loads synthetic_dense and synthetic_sparse
- Optionally loads superconductivity and movieLens
- Runs all five experiments on all available datasets

### Using Programmatically

```python
from data import (
    load_synthetic,
    load_synthetic_sparse,
    load_superconductivity,
    load_movielens_100k,
)

# 1. Dense synthetic (baseline)
dense = load_synthetic(m=500, n=200, r_true=5, sigma=0.1)

# 2. Sparse synthetic (new!)
sparse = load_synthetic_sparse(m=500, n=200, r_true=5, sigma=0.1, sparsity=0.8)

# 3. Large dense scientific data
supercond = load_superconductivity()

# 4. Real recommender system
movielens = load_movielens_100k()

# Use them in experiments
datasets = [dense, sparse, supercond, movielens]
```

---

## Dataset Details

### Synthetic Dense vs Synthetic Sparse

Both share identical **low-rank signal**:

$$M_{\text{full}} = U_{\text{true}} V_{\text{true}}^T + \sigma \mathcal{N}(0,1)$$

| Aspect | Synthetic Dense | Synthetic Sparse |
|--------|---|---|
| Full matrix | $500 \times 200 = 100,000$ entries | Same structure |
| Observed entries | 100% (all entries) | 20% (~20,000 entries) |
| Missing entries | 0% | 80% (80,000 entries) |
| Recovery evaluation | Direct on dense | Via $M_{\text{clean}}$ |
| Expected algorithm ranking | GD ≈ L-BFGS > ALS | ALS >> L-BFGS > GD |
| Sparsity factor | 1x | 5x fewer per-iteration ops |

### Superconductivity Details

- **Source**: UCI Machine Learning Repository (Dataset 464)
- **Description**: 21,263 superconductor compounds with 81 material features
- **Preprocessing**: Standardized by columns
- **Purpose**: Feature reduction / latent factor discovery
- **Metric**: Reconstruction error (no ground truth rank known)
- **Expected insight**: L-BFGS faster on dense, but GD acceptable

### MovieLens 100K Details

- **Source**: GroupLens MovieLens 100K dataset
- **Users**: 943
- **Movies**: 1,682
- **Ratings**: ~100,000 (5-star scale)
- **Sparsity**: 93.7% missing
- **Train/Test**: Standard u1.base/u1.test split (80/20)
- **Metric**: Test RMSE on held-out ratings
- **Expected insight**: ALS natural fit, but GD/L-BFGS also work

---

## Interpreting Results

### Metrics by Dataset Type

**Synthetic Dense & Sparse:**
```
reconstruction_error = ||U_hat * V_hat^T - M_clean||_F / ||M_clean||_F
recovery_error = ||U_hat * V_hat^T - U_true * V_true^T||_F
```
Lower is better. Compare to "oracle" error.

**Superconductivity:**
```
reconstruction_error = ||U_hat * V_hat^T - M||_F / ||M||_F
```
No test set; evaluate reconstruction quality.

**MovieLens:**
```
test_rmse = sqrt(mean((predictions - observed_ratings)^2))
```
Lower is better. Measures generalization.

### Key Comparisons

1. **Dense vs Sparse: Algorithm Suitability**
   - Is ALS slower on dense? (Expected: Yes, overhead)
   - Is L-BFGS slower on sparse? (Expected: Yes, less iterations work)
   - Does GD work everywhere? (Expected: Yes, but slowest)

2. **Synthetic vs Real: Robustness**
   - Do synthetic findings transfer to real data?
   - Is rank selection from synthetic applicable?
   - Does regularization transfer?

3. **Scalability: Computational Trends**
   - Time per iteration: GD < ALS < L-BFGS
   - Total time: depends on iterations and data size
   - Dense (21K × 81): L-BFGS likely fastest
   - Sparse (943 × 1682): ALS likely fastest

---

## Code Examples

### Example 1: Compare Algorithms on All Four Datasets

```python
from pathlib import Path
from data import load_synthetic, load_synthetic_sparse, load_superconductivity, load_movielens_100k
from experiments import run_experiment_1
from plotting import ensure_dir

# Load all four
datasets = [
    load_synthetic(m=500, n=200, r_true=5, sigma=0.1),
    load_synthetic_sparse(m=500, n=200, r_true=5, sigma=0.1, sparsity=0.8),
    load_superconductivity(),
    load_movielens_100k(),
]

# Run Experiment 1 on all
rank_map = {
    "synthetic": 5,
    "synthetic_sparse": 5,
    "superconductivity": 10,
    "movielens_100k": 20,
}

summary = run_experiment_1(
    datasets=datasets,
    output_dir=Path("figures/exp1_four_datasets"),
    results_dir=Path("results/exp1_four_datasets"),
    rank_map=rank_map,
    lam=1e-2,
)

print(summary)
```

### Example 2: Measure Sparsity Effect

```python
from data import load_synthetic, load_synthetic_sparse
from methods import gd_backtracking, als
from objectives import rmse_on_problem

# Create dense and sparse versions
dense = load_synthetic(m=500, n=200, r_true=5, sigma=0.1, seed=42)
sparse = load_synthetic_sparse(m=500, n=200, r_true=5, sigma=0.1, sparsity=0.8, seed=42)

# Initialize identically
u0, v0 = ..., ...

# Run both
hist_dense_gd = gd_backtracking(dense["problem"], u0, v0, lam=1e-2)
hist_sparse_gd = gd_backtracking(sparse["problem"], u0, v0, lam=1e-2)

# Compare recovery
recovery_dense = relative_reconstruction_error(
    hist_dense_gd["U_final"], hist_dense_gd["V_final"], dense["M_clean"]
)
recovery_sparse = relative_reconstruction_error(
    hist_sparse_gd["U_final"], hist_sparse_gd["V_final"], sparse["M_clean"]
)

print(f"Dense: recovery_error = {recovery_dense:.6f}")
print(f"Sparse: recovery_error = {recovery_sparse:.6f}")
print(f"Difference due to sparsity: {abs(recovery_dense - recovery_sparse):.6f}")
```

---

## Summary

| Aspect | Benefit |
|--------|---------|
| **Synthetic Dense** | Baseline correctness, known ground truth |
| **Synthetic Sparse** | Controlled sparsity test, same signal |
| **Superconductivity** | Scalability on realistic dense data |
| **MovieLens** | Real application, generalization test |
| **Combined** | Comprehensive algorithm evaluation |

The four-dataset framework enables research-grade validation of LRMF algorithms across diverse scenarios.
