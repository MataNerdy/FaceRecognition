"""Embedding and verification-style metrics for face recognition."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    torch = None
    F = None


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch.get("label")
    return batch[:2]


def compute_embeddings(model: torch.nn.Module, dataloader: Iterable, device: torch.device | str) -> tuple[np.ndarray, np.ndarray]:
    """Compute L2-normalized embeddings and labels for a dataloader."""
    if torch is None or F is None:
        raise ModuleNotFoundError("torch is required to compute model embeddings")
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images, target = _unpack_batch(batch)
            images = images.to(device)
            output = model(images)
            embedding = output[1] if isinstance(output, tuple) else output
            embeddings.append(F.normalize(embedding, dim=1).cpu().numpy())
            if target is not None:
                labels.append(torch.as_tensor(target).cpu().numpy())

    if not embeddings:
        raise ValueError("Dataloader produced no batches")

    label_array = np.concatenate(labels) if labels else np.empty((len(np.vstack(embeddings)),), dtype=np.int64)
    return np.vstack(embeddings), label_array


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    """Return pairwise cosine similarity matrix for two embedding arrays."""
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), eps, None)
    if b is None:
        b_norm = a_norm
    else:
        b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), eps, None)
    return a_norm @ b_norm.T


def compute_ir(
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    distractor_embeddings: np.ndarray,
    fprs: tuple[float, ...] = (0.5, 0.2, 0.1, 0.05),
) -> dict[float, dict[str, float]]:
    """Compute Identification Rate as TPR at selected false positive rates."""
    qsim = cosine_matrix(query_embeddings)
    dsim = cosine_matrix(query_embeddings, distractor_embeddings)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for i, j in itertools.combinations(range(len(query_labels)), 2):
        if query_labels[i] == query_labels[j]:
            positive_scores.append(float(qsim[i, j]))
        else:
            negative_scores.append(float(qsim[i, j]))

    if dsim.size:
        negative_scores.extend(dsim.reshape(-1).astype(float).tolist())
    if not negative_scores:
        raise ValueError("IR requires at least one negative or distractor pair")

    positives = np.asarray(positive_scores, dtype=np.float32)
    negatives = np.asarray(negative_scores, dtype=np.float32)

    results: dict[float, dict[str, float]] = {}
    for fpr in fprs:
        threshold = float(np.quantile(negatives, 1 - fpr))
        tpr = float((positives >= threshold).mean()) if positives.size else 0.0
        results[float(fpr)] = {"threshold": threshold, "tpr": tpr}
    return results


def triplet_accuracy(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.3) -> float:
    """Return the share of triplets where positive is closer than negative by margin."""
    if torch is None:
        raise ModuleNotFoundError("torch is required to compute triplet accuracy")
    pos_dist = torch.norm(anchor - positive, dim=1)
    neg_dist = torch.norm(anchor - negative, dim=1)
    return float(((pos_dist + margin) < neg_dist).float().mean().item())
