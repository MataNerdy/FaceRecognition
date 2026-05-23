"""Reusable building blocks for the face recognition pipeline."""

from face_recognition.alignment import align_face, heatmaps_to_landmarks
from face_recognition.metrics import compute_embeddings, compute_ir, cosine_matrix

__all__ = [
    "align_face",
    "heatmaps_to_landmarks",
    "compute_embeddings",
    "compute_ir",
    "cosine_matrix",
]

