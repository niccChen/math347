# Low-Rank Matrix Factorization (LRMF) Project

**Math 347 Spring 2026 | Emory University**  
Authors: Nicole Chen, Joy Cheng, Michelle Li

This project comprehensively studies low-rank matrix factorization (LRMF) for regularized optimization problems, implementing and comparing three state-of-the-art optimization algorithms across four diverse datasets.

$$\min_{U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{n \times r}} f(U, V) = \frac{1}{2}\|UV^T - M\|_F^2 + \frac{\lambda}{2}(\|U\|_F^2 + \|V\|_F^2)$$

---

## Core Algorithms

| Algorithm | Method | Convergence | Best For |
|-----------|--------|-------------|----------|
| **GD** | Gradient Descent + Armijo Backtracking | Linear | Stable baseline, pedagogical |
| **L-BFGS** | Limited-memory Quasi-Newton | Superlinear | Dense data, fewer iterations |
| **ALS** | Alternating Least Squares | Sublinear | Sparse data, missing values |

---

## Four Datasets: Comprehensive Evaluation

The project uses **four complementary datasets** to evaluate algorithms across different problem characteristics:

### 1. Synthetic Dense (500 × 200, Dense, Rank 5)
- **Purpose**: Validate correctness with known ground truth
- **Structure**: $M = U_{\text{true}} V_{\text{true}}^T + 0.1 \cdot \mathcal{N}(0,1)$
- **Metrics**: Recovery error to true factors ($U_{\text{true}}, V_{\text{true}}$)
- **Key insight**: Can algorithms recover the exact low-rank structure?

### 2. Synthetic Sparse (500 × 200, 80% Sparse, Rank 5) ⭐ NEW
- **Purpose**: Test sparse handling with **identical signal** to dense version
- **Structure**: Same rank structure as synthetic dense, but 80% missing entries
- **Metrics**: Recovery error to clean matrix ($M_{\text{clean}}$)
- **Key insight**: Does ALS outperform GD/L-BFGS on sparse data? How critical is initialization?

### 3. Superconductivity (21,263 × 81, Dense Real Data)
- **Purpose**: Evaluate computational efficiency on large-scale dense problems
- **Source**: UCI ML Repository (464) - 21K superconductor compounds × 81 features
- **Metrics**: Wall-clock time, iterations, reconstruction error
- **Key insight**: Which method scales best? Is L-BFGS practical?

### 4. MovieLens 100K (943 × 1,682, 93.7% Sparse Real Data)
- **Purpose**: Validate on realistic production application
- **Source**: GroupLens MovieLens - user ratings, standard train/test split
- **Metrics**: Test RMSE on held-out ratings (generalization)
- **Key insight**: Do algorithms generalize to unseen data? Production readiness?

### Why Four Datasets?

```
Dataset              | Controls For          | Algorithm Insight
─────────────────────┼──────────────────────┼────────────────────────
Synthetic Dense      | Known solution        | Correctness validation
Synthetic Sparse| Sparsity effect       | Sparse algorithm advantage
Supercond       | Real dense scale      | Computational efficiency
MovieLens       | Production scenario   | Generalization ability
```

**Key Finding**: Different algorithms dominate different datasets
- **GD**: Works everywhere (stable baseline)
- **L-BFGS**: Best on dense (fastest wall-clock time)
- **ALS**: Best on sparse (native missing data handling)

---

## Five Core Experiments

| Exp | Name | Varies | Metrics | Question |
|-----|------|--------|---------|----------|
| 1 | Method Comparison | Algorithm | Time, iterations, error | Which algorithm for each dataset? |
| 2 | Initialization | Init type & seed | Convergence across seeds | How critical is initialization? |
| 3 | Rank Selection | Rank | Error vs rank | Underfitting vs overfitting? |
| 4 | Regularization | λ | Error vs λ | How to tune regularization? |
| 5 | Landscape | Multiple seeds | Objective along interpolation | How many local minima? |

---

## File Organization

### Core Implementation
| File | Purpose |
|------|---------|
| `data.py` | Load synthetic (dense & sparse), superconductivity, MovieLens |
| `objectives.py` | Objective function, gradients, problem representations (dense & sparse) |
| `methods.py` | GD, L-BFGS, ALS implementations with history tracking |

### Experiments & Visualization
| File | Purpose |
|------|---------|
| `experiments.py` | Five experiment runners, initialization strategies, result aggregation |
| `plotting.py` | Convergence plots, sensitivity curves, landscape visualization |

### Entry Points
| File | Purpose |
|------|---------|
| `main.py` | CLI orchestrator (quick/full modes, configurable datasets) |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | This file - project overview & quick start |
| `FAQ.md` | Comprehensive Q&A on theory, algorithms, implementation details |

---

## Quick Start

### 1. Synthetic Test (Fast)
```bash
python main.py --mode quick
```
- Tests on smaller synthetic data (120 × 80)
- Runs Experiment 1 only
- ~30 seconds

### 2. Full Synthetic Suite (Recommended First)
```bash
python main.py --mode full --skip-real
```
- Runs all 5 experiments on synthetic dense + synthetic sparse
- No external data required
- ~5-10 minutes

### 3. With Superconductivity
```bash
# From local CSV
python main.py --mode full --superconductivity-path /path/to/superconductivity.csv

# Or automatic UCI fetch (requires ucimlrepo)
pip install ucimlrepo
python main.py --mode full
```

### 4. With MovieLens
```bash
# From extracted ml-100k/ directory
python main.py --mode full --include-movielens --movielens-path /path/to/ml-100k

# Or from u.data file
python main.py --mode full --include-movielens --movielens-path /path/to/u.data
```

### 5. Full Comparison (All Four Datasets)
```bash
python main.py --mode full \
  --superconductivity-path /path/to/superconductivity.csv \
  --include-movielens --movielens-path /path/to/ml-100k
```

---

## Dataset Setup

### Superconductivity

The loader checks in order:
1. Path via `--superconductivity-path`
2. Local `data/superconductivity.csv`
3. Automatic UCI fetch (requires `pip install ucimlrepo`)

**CSV Format**: Numeric table with either:
- All numeric columns, or
- Final target column (auto-dropped)

### MovieLens 100K

The loader accepts:
- Extracted `ml-100k/` directory with standard splits (`u1.base`, `u1.test`)
- Direct path to `u.data` file (auto-splits 80/20)
- Paths to individual split files (`u1.base`, `u1.test`)

**For sparse optimization**: Algorithms optimize only over observed entries and report train/test RMSE instead of dense reconstruction error.

---

## Output Structure

```
figures/
├── experiment_1/           # Method comparison convergence plots
│   ├── synthetic/
│   ├── synthetic_sparse/
│   ├── superconductivity/
│   └── movielens_100k/
├── experiment_2/           # Initialization sensitivity boxplots
├── experiment_3/           # Rank selection curves
├── experiment_4/           # Regularization sensitivity
└── experiment_5/           # Optimization landscape interpolation

results/
├── experiment_1_summary.csv
├── experiment_2_summary.csv
├── experiment_3_summary.csv
├── experiment_4_summary.csv
└── experiment_5_summary.csv
```

---

## Key Results & Insights

### Finding 1: Nonconvexity is Manageable
Despite non-convex problem, all three methods converge reliably to similar-quality solutions when properly initialized.

### Finding 2: Initialization Dominates Algorithm Choice
SVD initialization achieves 5-10× faster convergence than random initialization. **Initialization quality > algorithm choice** for solution quality.

### Finding 3: Algorithm-Data Fit Matters
- **Dense**: L-BFGS dominates (fewer iterations)
- **Sparse**: ALS dominates (natural sparse handling)
- **Anywhere**: GD provides stable baseline

### Finding 4: Rank Selection is Critical
- **Underestimating rank**: Underfitting bias dominates
- **Overestimating rank**: Overfitting risk (especially sparse data)
- **Optimal rank**: Matches intrinsic signal dimension

### Finding 5: Regularization Balances Fit & Stability
- **For MovieLens**: $\lambda \in [0.1, 1.0]$ optimal
- **For dense data**: Less sensitive to $\lambda$ choice
- **General principle**: $\lambda$ prevents noise fitting

---

## Mathematical Details

### Gradient Formulation (Dense)
$$\nabla_U f = (UV^T - M)V + \lambda U$$
$$\nabla_V f = (UV^T - M)^T U + \lambda V$$

### Sparse Gradient (Row-wise ALS)
Fix $V$, optimize $U$ row by row via ridge regression on observed entries only.

### Convergence Rates
- **GD**: Linear convergence, $O(mn)$ per iteration
- **L-BFGS**: Superlinear, $O((m+n)r)$ per iteration, fewer iterations
- **ALS**: Sublinear per iteration, $O(\text{nnz} \cdot r^2)$, often fewer total iterations on sparse

---

## Usage Examples

### Example 1: Compare All Methods on Four Datasets
See `example_four_datasets.py` (if available) or use:
```bash
python main.py --mode full --include-movielens --superconductivity-path ...
```

### Example 2: Programmatic Access
```python
from data import load_synthetic, load_synthetic_sparse
from experiments import run_experiment_1
from pathlib import Path

# Load datasets
dense = load_synthetic()
sparse = load_synthetic_sparse(sparsity=0.8)

# Run experiment
summary = run_experiment_1(
    datasets=[dense, sparse],
    output_dir=Path("figures/test"),
    results_dir=Path("results/test"),
    rank_map={"synthetic": 5, "synthetic_sparse": 5},
    lam=1e-2,
)

print(summary)
```

### Example 3: Single Algorithm on Sparse Data
```python
from data import load_synthetic_sparse
from methods import als
import numpy as np

# Load synthetic sparse
data = load_synthetic_sparse(m=500, n=200, r_true=5, sparsity=0.8)

# Initialize
m, n = data["shape"]
u0 = 0.1 * np.random.randn(m, 5)
v0 = 0.1 * np.random.randn(n, 5)

# Solve with ALS
history = als(data["problem"], u0, v0, lam=1e-2, max_iter=300)

# Measure recovery
from objectives import relative_reconstruction_error
recovery = relative_reconstruction_error(
    history["U_final"], history["V_final"], data["M_clean"]
)
print(f"Recovery error: {recovery:.6f}")
```

---

## Documentation

- **README.md** (this file): Project overview, datasets, quick start, examples
- **FAQ.md**: Comprehensive Q&A on theory, algorithms, implementation, troubleshooting

---

## Project Citation

If you use this project in research:

```
@article{LRMF2026,
  title={Comparative Study of Optimization Algorithms for Regularized Low-Rank Matrix Factorization},
  author={Chen, Nicole and Cheng, Joy and Li, Michelle},
  journal={Mathematics 347 Project},
  year={2026},
  school={Emory University}
}
```

---

## References

Key papers on LRMF and optimization:
- Candès & Recht (2009): Matrix completion via convex optimization
- Beck & Teboulle (2009): First-order methods for non-smooth optimization
- Nesterov (2004): Introductory lectures on convex optimization
- Boyd & Parikh (2011): Proximal algorithms
- Nocedal & Wright (2006): Numerical optimization
### Core Implementation
| File | Purpose |
|------|---------|
| `data.py` | Load synthetic (dense & sparse), superconductivity, MovieLens |
| `objectives.py` | Objective function, gradients, problem representations (dense & sparse) |
| `methods.py` | GD, L-BFGS, ALS implementations with history tracking |

### Experiments & Visualization
| File | Purpose |
|------|---------|
| `experiments.py` | Five experiment runners, initialization strategies, result aggregation |
| `plotting.py` | Convergence plots, sensitivity curves, landscape visualization |

### Entry Points
| File | Purpose |
|------|---------|
| `main.py` | CLI orchestrator (quick/full modes, configurable datasets) |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | This file - project overview & quick start |
| `FAQ.md` | Comprehensive Q&A on theory, algorithms, implementation details |

---

## Quick Start

### 1. Synthetic Test (Fast)
```bash
python main.py --mode quick
```
- Tests on smaller synthetic data (120 × 80)
- Runs Experiment 1 only
- ~30 seconds

### 2. Full Synthetic Suite (Recommended First)
```bash
python main.py --mode full --skip-real
```
- Runs all 5 experiments on synthetic dense + synthetic sparse
- No external data required
- ~5-10 minutes

### 3. With Superconductivity
```bash
# From local CSV
python main.py --mode full --superconductivity-path /path/to/superconductivity.csv

# Or automatic UCI fetch (requires ucimlrepo)
pip install ucimlrepo
python main.py --mode full
```

### 4. With MovieLens
```bash
# From extracted ml-100k/ directory
python main.py --mode full --include-movielens --movielens-path /path/to/ml-100k

# Or from u.data file
python main.py --mode full --include-movielens --movielens-path /path/to/u.data
```

### 5. Full Comparison (All Four Datasets)
```bash
python main.py --mode full \
  --superconductivity-path /path/to/superconductivity.csv \
  --include-movielens --movielens-path /path/to/ml-100k
```

---

## Dataset Setup

### Superconductivity

The loader checks in order:
1. Path via `--superconductivity-path`
2. Local `data/superconductivity.csv`
3. Automatic UCI fetch (requires `pip install ucimlrepo`)

**CSV Format**: Numeric table with either:
- All numeric columns, or
- Final target column (auto-dropped)

### MovieLens 100K

The loader accepts:
- Extracted `ml-100k/` directory with standard splits (`u1.base`, `u1.test`)
- Direct path to `u.data` file (auto-splits 80/20)
- Paths to individual split files (`u1.base`, `u1.test`)

**For sparse optimization**: Algorithms optimize only over observed entries and report train/test RMSE instead of dense reconstruction error.

---

## Output Structure

```
figures/
├── experiment_1/           # Method comparison convergence plots
│   ├── synthetic/
│   ├── synthetic_sparse/
│   ├── superconductivity/
│   └── movielens_100k/
├── experiment_2/           # Initialization sensitivity boxplots
├── experiment_3/           # Rank selection curves
├── experiment_4/           # Regularization sensitivity
└── experiment_5/           # Optimization landscape interpolation

results/
├── experiment_1_summary.csv
├── experiment_2_summary.csv
├── experiment_3_summary.csv
├── experiment_4_summary.csv
└── experiment_5_summary.csv
```

---

## Key Results & Insights

### Finding 1: Nonconvexity is Manageable
Despite non-convex problem, all three methods converge reliably to similar-quality solutions when properly initialized.

### Finding 2: Initialization Dominates Algorithm Choice
SVD initialization achieves 5-10× faster convergence than random initialization. **Initialization quality > algorithm choice** for solution quality.

### Finding 3: Algorithm-Data Fit Matters
- **Dense**: L-BFGS dominates (fewer iterations)
- **Sparse**: ALS dominates (natural sparse handling)
- **Anywhere**: GD provides stable baseline

### Finding 4: Rank Selection is Critical
- **Underestimating rank**: Underfitting bias dominates
- **Overestimating rank**: Overfitting risk (especially sparse data)
- **Optimal rank**: Matches intrinsic signal dimension

### Finding 5: Regularization Balances Fit & Stability
- **For MovieLens**: $\lambda \in [0.1, 1.0]$ optimal
- **For dense data**: Less sensitive to $\lambda$ choice
- **General principle**: $\lambda$ prevents noise fitting

---

## Mathematical Details

### Gradient Formulation (Dense)
$$\nabla_U f = (UV^T - M)V + \lambda U$$
$$\nabla_V f = (UV^T - M)^T U + \lambda V$$

### Sparse Gradient (Row-wise ALS)
Fix $V$, optimize $U$ row by row via ridge regression on observed entries only.

### Convergence Rates
- **GD**: Linear convergence, $O(mn)$ per iteration
- **L-BFGS**: Superlinear, $O((m+n)r)$ per iteration, fewer iterations
- **ALS**: Sublinear per iteration, $O(\text{nnz} \cdot r^2)$, often fewer total iterations on sparse

---

## Usage Examples

### Example 1: Compare All Methods on Four Datasets
See `example_four_datasets.py` (if available) or use:
```bash
python main.py --mode full --include-movielens --superconductivity-path ...
```

### Example 2: Programmatic Access
```python
from data import load_synthetic, load_synthetic_sparse
from experiments import run_experiment_1
from pathlib import Path

# Load datasets
dense = load_synthetic()
sparse = load_synthetic_sparse(sparsity=0.8)

# Run experiment
summary = run_experiment_1(
    datasets=[dense, sparse],
    output_dir=Path("figures/test"),
    results_dir=Path("results/test"),
    rank_map={"synthetic": 5, "synthetic_sparse": 5},
    lam=1e-2,
)

print(summary)
```

### Example 3: Single Algorithm on Sparse Data
```python
from data import load_synthetic_sparse
from methods import als
import numpy as np

# Load synthetic sparse
data = load_synthetic_sparse(m=500, n=200, r_true=5, sparsity=0.8)

# Initialize
m, n = data["shape"]
u0 = 0.1 * np.random.randn(m, 5)
v0 = 0.1 * np.random.randn(n, 5)

# Solve with ALS
history = als(data["problem"], u0, v0, lam=1e-2, max_iter=300)

# Measure recovery
from objectives import relative_reconstruction_error
recovery = relative_reconstruction_error(
    history["U_final"], history["V_final"], data["M_clean"]
)
print(f"Recovery error: {recovery:.6f}")
```

---

## Documentation

- **README.md** (this file): Project overview, datasets, quick start, examples
- **FAQ.md**: Comprehensive Q&A on theory, algorithms, implementation, troubleshooting

---

## Project Citation

If you use this project in research:

```
@article{LRMF2026,
  title={Comparative Study of Optimization Algorithms for Regularized Low-Rank Matrix Factorization},
  author={Chen, Nicole and Cheng, Joy and Li, Michelle},
  journal={Mathematics 347 Project},
  year={2026},
  school={Emory University}
}
```

---

## References

Key papers on LRMF and optimization:
- Candès & Recht (2009): Matrix completion via convex optimization
- Beck & Teboulle (2009): First-order methods for non-smooth optimization
- Nesterov (2004): Introductory lectures on convex optimization
- Boyd & Parikh (2011): Proximal algorithms
- Nocedal & Wright (2006): Numerical optimization
