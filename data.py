from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from objectives import make_dense_problem, make_sparse_problem

Array = np.ndarray


def standardize_columns(matrix: Array, eps: float = 1e-12) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (matrix - mean) / std


def load_synthetic(
    m: int = 500,
    n: int = 200,
    r_true: int = 5,
    sigma: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    u_true = rng.standard_normal((m, r_true))
    v_true = rng.standard_normal((n, r_true))
    clean = u_true @ v_true.T
    noise = sigma * rng.standard_normal((m, n))
    observed = clean + noise
    return {
        "name": "synthetic",
        "M": observed,
        "M_clean": clean,
        "U_true": u_true,
        "V_true": v_true,
        "sigma": sigma,
        "r_true": r_true,
        "shape": observed.shape,
        "problem": make_dense_problem(observed, name="synthetic"),
    }


def load_synthetic_sparse(
    m: int = 500,
    n: int = 200,
    r_true: int = 5,
    sigma: float = 0.1,
    sparsity: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Create synthetic sparse data with same low-rank structure as dense version.
    
    Args:
        m: Number of rows
        n: Number of columns
        r_true: True rank of the underlying signal
        sigma: Noise standard deviation
        sparsity: Fraction of entries to mask (0.8 = 80% missing, 20% observed)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with sparse problem representation and full dense clean matrix
    """
    rng = np.random.default_rng(seed)
    
    # Generate true low-rank structure
    u_true = rng.standard_normal((m, r_true))
    v_true = rng.standard_normal((n, r_true))
    clean = u_true @ v_true.T
    noise = sigma * rng.standard_normal((m, n))
    observed = clean + noise
    
    # Create sparse version by random masking
    mask = rng.random((m, n)) > sparsity  # sparsity fraction are False (masked)
    
    # Extract observed entries in triplet format
    rows, cols = np.where(mask)
    values = observed[rows, cols]
    
    return {
        "name": "synthetic_sparse",
        "M_clean": clean,  # Keep full clean matrix for recovery evaluation
        "U_true": u_true,
        "V_true": v_true,
        "sigma": sigma,
        "r_true": r_true,
        "sparsity": sparsity,
        "shape": observed.shape,
        "nnz": len(values),
        "problem": make_sparse_problem(rows, cols, values, shape=observed.shape, name="synthetic_sparse"),
    }


def _read_superconductivity_csv(path: Path) -> Array:
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"No numeric columns found in {path}.")
    if numeric.shape[1] > 1:
        numeric = numeric.iloc[:, :-1]
    return numeric.to_numpy(dtype=float)


def _fetch_superconductivity_from_uci() -> Array:
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "ucimlrepo is not installed. Provide a local CSV path or install ucimlrepo."
        ) from exc

    dataset = fetch_ucirepo(id=464)
    features = dataset.data.features
    return np.asarray(features, dtype=float)


def load_superconductivity(
    csv_path: str | Path | None = None,
    standardize: bool = True,
) -> dict[str, Any]:
    candidate_paths = []
    if csv_path is not None:
        candidate_paths.append(Path(csv_path))
    candidate_paths.append(Path("data/superconductivity.csv"))

    matrix = None
    source = None
    for path in candidate_paths:
        if path.exists():
            matrix = _read_superconductivity_csv(path)
            source = str(path)
            break

    if matrix is None:
        matrix = _fetch_superconductivity_from_uci()
        source = "ucimlrepo:464"

    if standardize:
        matrix = standardize_columns(matrix)

    return {
        "name": "superconductivity",
        "M": matrix,
        "source": source,
        "shape": matrix.shape,
        "problem": make_dense_problem(matrix, name="superconductivity"),
    }


def _read_movielens_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"[\t:]+",
        engine="python",
        header=None,
        usecols=[0, 1, 2],
        names=["user_id", "item_id", "rating"],
    )


def _dataframe_to_sparse_problem(
    frame: pd.DataFrame,
    shape: tuple[int, int],
    name: str,
) -> dict[str, Any]:
    rows = frame["user_id"].to_numpy(dtype=int) - 1
    cols = frame["item_id"].to_numpy(dtype=int) - 1
    values = frame["rating"].to_numpy(dtype=float)
    return make_sparse_problem(rows, cols, values, shape=shape, name=name)


def load_movielens_100k(
    path: str | Path | None = None,
    split_seed: int = 0,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("data/ml-100k"),
            Path("data/movielens/ml-100k"),
            Path("data/u.data"),
            Path("data/ml-100k/u.data"),
        ]
    )

    source = None
    base_path = None
    test_path = None
    full_path = None

    for candidate in candidates:
        if candidate.is_dir():
            if (candidate / "u1.base").exists() and (candidate / "u1.test").exists():
                base_path = candidate / "u1.base"
                test_path = candidate / "u1.test"
                source = str(candidate)
                break
            if (candidate / "u.data").exists():
                full_path = candidate / "u.data"
                source = str(candidate / "u.data")
                break
        elif candidate.exists():
            if candidate.name == "u1.base":
                sibling = candidate.with_name("u1.test")
                base_path = candidate
                test_path = sibling if sibling.exists() else None
                source = str(candidate)
                break
            full_path = candidate
            source = str(candidate)
            break

    if base_path is None and full_path is None:
        raise FileNotFoundError(
            "MovieLens 100K not found. Provide a path to `u.data` or the extracted `ml-100k/` directory."
        )

    if base_path is not None and test_path is not None:
        train_frame = _read_movielens_split(base_path)
        test_frame = _read_movielens_split(test_path)
    else:
        frame = _read_movielens_split(full_path)
        rng = np.random.default_rng(split_seed)
        perm = rng.permutation(len(frame))
        cutoff = int((1.0 - test_fraction) * len(frame))
        train_frame = frame.iloc[perm[:cutoff]].copy()
        test_frame = frame.iloc[perm[cutoff:]].copy()

    combined = pd.concat([train_frame, test_frame], ignore_index=True)
    shape = (
        int(combined["user_id"].max()),
        int(combined["item_id"].max()),
    )

    train_problem = _dataframe_to_sparse_problem(train_frame, shape, "movielens_100k_train")
    test_problem = _dataframe_to_sparse_problem(test_frame, shape, "movielens_100k_test")

    return {
        "name": "movielens_100k",
        "shape": shape,
        "source": source,
        "train_problem": train_problem,
        "test_problem": test_problem,
        "rating_range": (
            float(combined["rating"].min()),
            float(combined["rating"].max()),
        ),
    }
