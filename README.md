# Low-Rank Matrix Factorization Project

This project implements the final project plan for low-rank matrix factorization:

- Gradient descent with Armijo backtracking
- L-BFGS via `scipy.optimize.minimize`
- Alternating minimization / ALS

It supports:

- synthetic low-rank matrices with noise
- the superconductivity dataset through a local CSV or `ucimlrepo`
- MovieLens 100K through a local `u.data` file or extracted `ml-100k/` directory

## Files

- `data.py`: synthetic and superconductivity data loaders
- `objectives.py`: objective, gradients, packing/unpacking, gradient checks
- `methods.py`: GD, L-BFGS, ALS
- `experiments.py`: experiment runners for comparison, initialization, rank, regularization, and landscape
- `plotting.py`: figure generation
- `main.py`: command-line entry point

## Quick start

Run a synthetic smoke test:

```bash
python3 main.py --mode quick
```

Run the full synthetic suite:

```bash
python3 main.py --mode full --skip-real
```

Run the full suite including superconductivity from a local CSV:

```bash
python3 main.py --mode full --superconductivity-path /absolute/path/to/superconductivity.csv
```

Run with MovieLens 100K included:

```bash
python3 main.py --mode full --include-movielens --movielens-path /absolute/path/to/ml-100k
```

## Superconductivity dataset

The loader checks the following in order:

1. a path passed via `--superconductivity-path`
2. `data/superconductivity.csv`
3. `ucimlrepo` with dataset id `464`

If you want the automatic UCI fetch, install:

```bash
python3 -m pip install ucimlrepo
```

The expected CSV format is a numeric table of features with either:

- all numeric columns, or
- a final target column, which will be dropped automatically

## MovieLens 100K

The MovieLens loader accepts either:

- the extracted `ml-100k/` directory
- a direct path to `u.data`
- the fixed split files `u1.base` and `u1.test`

For sparse MovieLens runs, the project optimizes only over observed ratings and reports RMSE on train and test observations instead of dense reconstruction error.

## Outputs

Generated plots go to `figures/` and tabular experiment summaries go to `results/`.
