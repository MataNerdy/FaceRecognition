"""Evaluate face embeddings with Identification Rate / TPR@FPR.

The script can either print the saved experiment table from the notebooks or
compute IR for a checkpoint and two ImageFolder-style evaluation splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



EXPERIMENT_RESULTS = {
    "CE": {0.50: {"tpr": 0.8678, "threshold": 0.9655}, 0.20: {"tpr": 0.5805, "threshold": 0.9782}, 0.10: {"tpr": 0.3858, "threshold": 0.9835}, 0.05: {"tpr": 0.2431, "threshold": 0.9871}},
    "ArcFace": {0.50: {"tpr": 0.6393, "threshold": 0.6294}, 0.20: {"tpr": 0.2936, "threshold": 0.8009}, 0.10: {"tpr": 0.1677, "threshold": 0.8634}, 0.05: {"tpr": 0.0976, "threshold": 0.9003}},
    "Triplet": {0.50: {"tpr": 0.7215, "threshold": 0.8385}, 0.20: {"tpr": 0.3877, "threshold": 0.9443}, 0.10: {"tpr": 0.2285, "threshold": 0.9688}, 0.05: {"tpr": 0.1333, "threshold": 0.9802}},
    "CE+ArcFace": {0.50: {"tpr": 0.7172, "threshold": 0.4576}, 0.20: {"tpr": 0.3668, "threshold": 0.6828}, 0.10: {"tpr": 0.2127, "threshold": 0.7635}, 0.05: {"tpr": 0.1282, "threshold": 0.8135}},
    "ArcFace+Triplet": {0.50: {"tpr": 0.7192, "threshold": 0.3941}, 0.20: {"tpr": 0.3587, "threshold": 0.6227}, 0.10: {"tpr": 0.1991, "threshold": 0.7176}, 0.05: {"tpr": 0.1112, "threshold": 0.7804}},
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate embeddings with Identification Rate / TPR@FPR")
    parser.add_argument("--checkpoint", type=Path, help="EmbeddingNet checkpoint to evaluate")
    parser.add_argument("--query-dir", type=Path, help="ImageFolder with query identities")
    parser.add_argument("--distractor-dir", type=Path, help="ImageFolder with distractor identities")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--device", default="auto", help="cuda, cpu or auto")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def load_embedding_model(checkpoint: Path, embedding_dim: int, device: str):
    """Load an EmbeddingNet from a checkpoint."""
    import torch

    from face_recognition.models.embedding import EmbeddingNet

    model = EmbeddingNet(embedding_dim=embedding_dim, pretrained=False).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload.get("encoder_state_dict", payload))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main() -> None:
    """Run saved-table or checkpoint-based IR evaluation."""
    args = parse_args()
    if not (args.checkpoint and args.query_dir and args.distractor_dir):
        print(json.dumps(EXPERIMENT_RESULTS, indent=2, ensure_ascii=False))
        return

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from face_recognition.datasets import default_face_transform
    from face_recognition.metrics import compute_embeddings, compute_ir

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    transform = default_face_transform(args.image_size)
    query_dataset = ImageFolder(args.query_dir, transform=transform)
    distractor_dataset = ImageFolder(args.distractor_dir, transform=transform)
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    distractor_loader = DataLoader(distractor_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = load_embedding_model(args.checkpoint, args.embedding_dim, device)
    query_embeddings, query_labels = compute_embeddings(model, query_loader, device)
    distractor_embeddings, _ = compute_embeddings(model, distractor_loader, device)
    results = compute_ir(query_embeddings, query_labels, distractor_embeddings)

    serializable = {str(fpr): value for fpr, value in results.items()}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
