import numpy as np

from src.face_recognition.alignment import heatmaps_to_landmarks


def test_heatmaps_to_landmarks_scales_coordinates():
    heatmaps = np.zeros((2, 4, 4), dtype=np.float32)
    heatmaps[0, 1, 2] = 1.0
    heatmaps[1, 3, 0] = 1.0

    points = heatmaps_to_landmarks(heatmaps, image_size=8)

    np.testing.assert_allclose(points, np.array([[4.0, 2.0], [0.0, 6.0]], dtype=np.float32))
