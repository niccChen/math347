from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save(fig: plt.Figure, path: str | Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_singular_value_decay(singular_values: np.ndarray, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.arange(1, len(singular_values) + 1), singular_values, "o-", linewidth=1.5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, out_path)


def plot_method_histories(
    histories: dict[str, dict[str, Any]],
    title_prefix: str,
    out_dir: str | Path,
) -> None:
    out_dir = ensure_dir(out_dir)
    metrics = [
        ("obj", "Objective", "iteration", False, out_dir / "objective_vs_iteration.png"),
        ("grad_norm", "Gradient norm", "iteration", True, out_dir / "gradnorm_vs_iteration.png"),
        ("obj", "Objective", "time", False, out_dir / "objective_vs_time.png"),
    ]

    for key, ylabel, x_key, semilogy, out_path in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method_name, history in histories.items():
            x_values = history[x_key] if x_key == "time" else np.arange(len(history[key]))
            y_values = np.asarray(history[key], dtype=float)
            if semilogy:
                ax.semilogy(x_values, y_values, label=method_name, linewidth=2)
            else:
                ax.plot(x_values, y_values, label=method_name, linewidth=2)
        ax.set_xlabel("Time (seconds)" if x_key == "time" else "Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}: {ylabel}")
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, out_path)


def plot_initialization_summary(records: list[dict[str, Any]], out_path: str | Path) -> None:
    methods = sorted({record["method"] for record in records})
    init_types = ["small", "large", "svd"]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        method_records = [record for record in records if record["method"] == method]
        positions = np.arange(len(init_types))
        box_data = []
        labels = []
        for init_type in init_types:
            values = [record["final_obj"] for record in method_records if record["init_type"] == init_type]
            if init_type == "svd":
                if values:
                    ax.scatter([positions[len(labels)]], values, color="tab:red", s=50, label="SVD init")
                labels.append("svd")
                continue
            box_data.append(values)
            labels.append(init_type)

        if box_data:
            ax.boxplot(box_data, positions=positions[: len(box_data)], widths=0.5)
        svd_values = [record["final_obj"] for record in method_records if record["init_type"] == "svd"]
        if svd_values:
            ax.scatter([2], svd_values, color="tab:red", s=50, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(init_types)
        ax.set_title(method)
        ax.set_ylabel("Final objective")
        ax.grid(alpha=0.3)

    fig.suptitle("Initialization sensitivity")
    _save(fig, out_path)


def plot_initialization_overlays(
    convergence: dict[str, list[tuple[str, dict[str, Any]]]],
    out_path: str | Path,
) -> None:
    methods = list(convergence)
    fig, axes = plt.subplots(len(methods), 1, figsize=(7, 4 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for label, history in convergence[method]:
            alpha = 0.9 if "svd" in label else 0.35
            linewidth = 2.2 if "svd" in label else 1.1
            ax.plot(np.arange(len(history["obj"])), history["obj"], alpha=alpha, linewidth=linewidth)
        ax.set_title(f"{method}: objective across initializations")
        ax.set_ylabel("Objective")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Iteration")
    _save(fig, out_path)


def plot_rank_sensitivity(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({record["method"] for record in records})
    for method in methods:
        subset = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["rank"],
        )
        ax.plot(
            [record["rank"] for record in subset],
            [record[metric] for record in subset],
            "o-",
            linewidth=2,
            label=method,
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_regularization_sensitivity(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({record["method"] for record in records})
    for method in methods:
        subset = sorted(
            [record for record in records if record["method"] == method],
            key=lambda record: record["lam"],
        )
        ax.semilogx(
            [record["lam"] for record in subset],
            [record[metric] for record in subset],
            "o-",
            linewidth=2,
            label=method,
        )
    ax.set_xlabel("lambda")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_landscape_scan(t_values: np.ndarray, values: np.ndarray, title: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_values, values, linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel("Objective along line segment")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, out_path)

