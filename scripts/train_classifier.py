"""Train a face classifier or embedding model on aligned face images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from tqdm import tqdm

    from face_recognition.datasets import TripletDatasetFromFolder, default_face_transform
    from face_recognition.losses import ArcFaceCELoss, ArcFaceLoss, ArcFaceTripletLoss, TripletLoss
    from face_recognition.metrics import triplet_accuracy
    from face_recognition.models.embedding import EmbeddingNet, FaceClassifier
except ModuleNotFoundError as exc:  # Allows importing the script without ML dependencies installed.
    torch = None
    nn = None
    DataLoader = None
    ImageFolder = None
    tqdm = None
    TripletDatasetFromFolder = None
    default_face_transform = None
    ArcFaceCELoss = None
    ArcFaceLoss = None
    ArcFaceTripletLoss = None
    TripletLoss = None
    triplet_accuracy = None
    EmbeddingNet = None
    FaceClassifier = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_dependencies() -> None:
    """Raise a clear error when training dependencies are not installed."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError("Training requires project dependencies. Run `pip install -r requirements.txt`.") from _IMPORT_ERROR


def default_device() -> str:
    """Return CUDA when PyTorch is available and CUDA is visible, otherwise CPU."""
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def no_grad():
    """Return torch.no_grad when available, otherwise a no-op decorator."""
    if torch is not None:
        return torch.no_grad()

    def decorator(func):
        return func

    return decorator


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train face recognition models on aligned ImageFolder data")
    parser.add_argument("--data-dir", type=Path, required=True, help="ImageFolder with aligned training faces")
    parser.add_argument("--val-dir", type=Path, help="Optional ImageFolder validation split")
    parser.add_argument("--loss", choices=["ce", "arcface", "hybrid", "triplet", "arcface_triplet"], default="ce")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--arcface-weight", type=float, default=1.0)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet initialization")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def train_epoch(model, loader, criterion, optimizer, device: str, mode: str, arcface=None) -> dict[str, float]:
    """Train one epoch and return aggregate metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        if mode == "triplet":
            anchor, positive, negative = [item.to(device) for item in batch]
            emb_anchor, emb_positive, emb_negative = model(anchor), model(positive), model(negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            batch_size = anchor.size(0)
            total_correct += triplet_accuracy(emb_anchor, emb_positive, emb_negative, margin=criterion.margin) * batch_size
            total_items += batch_size
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits, embeddings = model(images, return_embedding=True)
            if mode == "ce":
                loss = criterion(logits, labels)
            elif mode == "arcface":
                loss = arcface(embeddings, labels)
            elif mode == "hybrid":
                loss = criterion(logits, embeddings, labels)
            else:
                loss = criterion(embeddings, labels)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.numel()

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch_size if mode == "triplet" else float(loss.item()) * labels.numel()

    return {"loss": total_loss / max(total_items, 1), "accuracy": total_correct / max(total_items, 1)}


@no_grad()
def evaluate(model, loader, criterion, device: str, mode: str, arcface=None) -> dict[str, float]:
    """Evaluate one epoch and return aggregate metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for batch in tqdm(loader, desc="val", leave=False):
        if mode == "triplet":
            anchor, positive, negative = [item.to(device) for item in batch]
            emb_anchor, emb_positive, emb_negative = model(anchor), model(positive), model(negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            batch_size = anchor.size(0)
            total_correct += triplet_accuracy(emb_anchor, emb_positive, emb_negative, margin=criterion.margin) * batch_size
            total_items += batch_size
        else:
            images, labels = batch[0].to(device), batch[1].to(device)
            logits, embeddings = model(images, return_embedding=True)
            if mode == "ce":
                loss = criterion(logits, labels)
            elif mode == "arcface":
                loss = arcface(embeddings, labels)
            elif mode == "hybrid":
                loss = criterion(logits, embeddings, labels)
            else:
                loss = criterion(embeddings, labels)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.numel()
        total_loss += float(loss.item()) * batch_size if mode == "triplet" else float(loss.item()) * labels.numel()
    return {"loss": total_loss / max(total_items, 1), "accuracy": total_correct / max(total_items, 1)}


def main() -> None:
    """Train the selected model and save the best checkpoint."""
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transform = default_face_transform(args.image_size)

    if args.loss == "triplet":
        train_dataset = TripletDatasetFromFolder(args.data_dir, transform=transform)
        val_dataset = TripletDatasetFromFolder(args.val_dir, transform=transform) if args.val_dir else None
        model = EmbeddingNet(args.embedding_dim, pretrained=not args.no_pretrained, dropout=args.dropout).to(args.device)
        criterion = TripletLoss(margin=args.triplet_margin)
        arcface = None
    else:
        train_dataset = ImageFolder(args.data_dir, transform=transform)
        val_dataset = ImageFolder(args.val_dir, transform=transform) if args.val_dir else None
        model = FaceClassifier(len(train_dataset.classes), args.embedding_dim, pretrained=not args.no_pretrained, dropout=args.dropout).to(args.device)
        arcface = ArcFaceLoss(args.embedding_dim, len(train_dataset.classes)).to(args.device)
        if args.loss == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "arcface":
            criterion = arcface
        elif args.loss == "hybrid":
            criterion = ArcFaceCELoss(arcface, ce_weight=1.0, arcface_weight=args.arcface_weight)
        else:
            criterion = ArcFaceTripletLoss(
                arcface,
                TripletLoss(margin=args.triplet_margin),
                arcface_weight=args.arcface_weight,
                triplet_weight=args.triplet_weight,
            )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None
    optimizer = torch.optim.AdamW(list(model.parameters()) + ([] if arcface is None else list(arcface.parameters())), lr=args.lr)

    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, args.loss, arcface=arcface)
        val_metrics = evaluate(model, val_loader, criterion, args.device, args.loss, arcface=arcface) if val_loader else train_metrics
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["accuracy"] >= best_metric:
            best_metric = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "arcface_state_dict": arcface.state_dict() if arcface is not None else None,
                    "loss": args.loss,
                    "embedding_dim": args.embedding_dim,
                    "best_accuracy": best_metric,
                },
                args.output_dir / f"best_{args.loss}.pth",
            )


if __name__ == "__main__":
    main()

