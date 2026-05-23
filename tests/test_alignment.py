import numpy as np

from src.face_recognition.alignment import align_face, reference_landmarks


def test_align_face_returns_expected_shape():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[40:80, 40:80] = 255
    landmarks = reference_landmarks(112)

    aligned = align_face(image, landmarks, output_size=112)

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == image.dtype
