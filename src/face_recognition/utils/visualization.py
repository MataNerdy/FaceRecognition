"""Visualization helpers for notebooks and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show_faces(images, titles: list[str] | None = None, cols: int = 6, figsize: tuple[int, int] = (14, 8)):
    """Render a grid of face images and return the matplotlib figure."""
    if not images:
        raise ValueError("No images were provided")
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= len(images):
            continue
        axis.imshow(images[index])
        if titles:
            axis.set_title(titles[index], fontsize=8)
    fig.tight_layout()
    return fig


def plot_ir_curves(results: dict[str, dict[float, dict[str, float]]]):
    """Plot TPR@FPR curves for several face recognition models."""
    fig, axis = plt.subplots(figsize=(8, 5))
    for name, values in results.items():
        fprs = sorted(values.keys(), reverse=True)
        tprs = [values[fpr]["tpr"] for fpr in fprs]
        axis.plot(fprs, tprs, marker="o", label=name)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title("Identification Rate: TPR@FPR")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return fig

