import itertools
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    model.eval()
    embeddings, labels = [], []
    for batch in dataloader:
        x, y = batch[:2]
        x = x.to(device)
        out = model(x)
        emb = out[1] if isinstance(out, tuple) else out
        embeddings.append(F.normalize(emb, dim=1).cpu().numpy())
        labels.extend(y)
    return np.vstack(embeddings), np.asarray(labels)


def cosine_matrix(a, b=None):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = a if b is None else b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def compute_ir(query_embeddings, query_labels, distractor_embeddings, fprs=(0.5, 0.2, 0.1, 0.05)):
    """Identification Rate / TPR@FPR from query positives, query negatives and distractors."""
    qsim = cosine_matrix(query_embeddings)
    dsim = cosine_matrix(query_embeddings, distractor_embeddings)

    pos, neg = [], []
    n = len(query_labels)
    for i, j in itertools.combinations(range(n), 2):
        if query_labels[i] == query_labels[j]:
            pos.append(qsim[i, j])
        else:
            neg.append(qsim[i, j])
    neg = np.concatenate([np.asarray(neg), dsim.reshape(-1)])
    pos = np.asarray(pos)

    results = {}
    for fpr in fprs:
        thr = np.quantile(neg, 1 - fpr)
        tpr = float((pos >= thr).mean()) if len(pos) else 0.0
        results[fpr] = {"threshold": float(thr), "tpr": tpr}
    return results
