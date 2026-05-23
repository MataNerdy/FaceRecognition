"""Landmark post-processing and face alignment utilities."""

from __future__ import annotations

import cv2
import numpy as np
import torch

REFERENCE_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def heatmaps_to_landmarks(heatmaps: torch.Tensor | np.ndarray, image_size: int | tuple[int, int] = 256) -> np.ndarray:
    """Convert landmark heatmaps with shape `[K, H, W]` to image coordinates."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected heatmaps with shape [K, H, W], got {heatmaps.shape}")

    if isinstance(image_size, int):
        image_width = image_height = image_size
    else:
        image_width, image_height = image_size

    _, heatmap_height, heatmap_width = heatmaps.shape
    points = []
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([x * image_width / heatmap_width, y * image_height / heatmap_height])
    return np.asarray(points, dtype=np.float32)


def reference_landmarks(output_size: int = 112) -> np.ndarray:
    """Return the canonical five-point face template scaled to `output_size`."""
    reference = REFERENCE_5PTS.copy()
    reference *= output_size / 112
    return reference


def estimate_alignment_matrix(landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Estimate a partial affine transform from landmarks to the canonical template."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected five 2D landmarks, got shape {landmarks.shape}")
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, reference_landmarks(output_size), method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not estimate affine transform for landmarks")
    return matrix.astype(np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    """Align a face image by five landmarks using a similarity transform."""
    matrix = estimate_alignment_matrix(landmarks, output_size=output_size)
    return cv2.warpAffine(np.asarray(image), matrix, (output_size, output_size), borderValue=0.0)

