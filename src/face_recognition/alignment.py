import cv2
import numpy as np
import torch


REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def heatmaps_to_landmarks(heatmaps, image_size=256):
    """Convert heatmaps [K,H,W] to landmark coordinates in original image scale."""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()
    k, h, w = heatmaps.shape
    points = []
    for i in range(k):
        y, x = np.unravel_index(np.argmax(heatmaps[i]), (h, w))
        points.append([x * image_size / w, y * image_size / h])
    return np.asarray(points, dtype=np.float32)


def align_face(image, landmarks, output_size=112):
    """Similarity-transform face by five predicted landmarks."""
    ref = REFERENCE_5PTS.copy()
    ref[:, 0] *= output_size / 112
    ref[:, 1] *= output_size / 112
    matrix, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, dtype=np.float32), ref, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Could not estimate affine transform for landmarks")
    return cv2.warpAffine(np.asarray(image), matrix, (output_size, output_size), borderValue=0.0)
