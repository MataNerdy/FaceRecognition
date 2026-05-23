import numpy as np

from face_recognition.metrics import compute_ir


def test_compute_ir_returns_expected_tpr_for_simple_embeddings():
    query_embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float32)
    query_labels = np.array([0, 0, 1])
    distractor_embeddings = np.array([[0.0, -1.0]], dtype=np.float32)

    results = compute_ir(query_embeddings, query_labels, distractor_embeddings, fprs=(0.5,))

    assert 0.5 in results
    assert results[0.5]["tpr"] == 1.0
    assert -1.0 <= results[0.5]["threshold"] <= 1.0
