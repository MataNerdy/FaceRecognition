"""Train a Stacked Hourglass landmark detector from a CSV annotation file.

The CSV must contain an image filename column plus ten landmark columns:
`x1,y1,x2,y2,x3,y3,x4,y4,x5,y5`. Dataset paths are provided by CLI and are not
hardcoded to Colab or Google Drive.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import transforms
    from tqdm import tqdm

    from face_recognition.models.hourglass import StackedHourglassNet
except ModuleNotFoundError as exc:  # Allows importing the script without ML dependencies installed.
    np = None
    torch = None
    F = None
    Image = None
    DataLoader = None
    Dataset = object
    random_split = None
    transforms = None
    tqdm = None
    StackedHourglassNet = None
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


class LandmarkHeatmapDataset(Dataset):
    """Load images and five-point landmark heatmaps from a CSV file."""

    def __init__(self, images_dir: Path, annotations: Path, image_size: int = 256, heatmap_size: int = 64, sigma: float = 2.0) -> None:
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        with Path(annotations).open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"No annotations found in {annotations}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        filename = row.get("filename") or row.get("image") or row.get("file")
        if filename is None:
            raise KeyError("CSV must contain one of: filename, image, file")
        image = Image.open(self.images_dir / filename).convert("RGB")
        width, height = image.size
        landmarks = np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 6)], dtype=np.float32)
        landmarks[:, 0] *= self.heatmap_size / width
        landmarks[:, 1] *= self.heatmap_size / height
        return self.transform(image), self._to_heatmaps(landmarks)

    def _to_heatmaps(self, landmarks: np.ndarray) -> torch.Tensor:
        yy, xx = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size), indexing="ij")
        maps = [np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma**2)) for x, y in landmarks]
        return torch.tensor(np.stack(maps), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = argparse.ArgumentParser(description="Train Stacked Hourglass facial landmark detector")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="CSV with filename,x1,y1,...,x5,y5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--heatmap-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint-out", type=Path, default=Path("checkpoints/hourglass_best.pth"))
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def run_epoch(model, loader, optimizer, device: str) -> float:
    """Train one epoch and return mean MSE loss."""
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = sum(F.mse_loss(output, targets) for output in outputs) / len(outputs)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1)


@no_grad()
def evaluate(model, loader, device: str) -> float:
    """Evaluate mean MSE loss."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    for images, targets in tqdm(loader, desc="val", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = sum(F.mse_loss(output, targets) for output in outputs) / len(outputs)
        total_loss += float(loss.item()) * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1)


def main() -> None:
    """Train the landmark detector and save the best checkpoint."""
    args = parse_args()
    require_dependencies()
    dataset = LandmarkHeatmapDataset(args.images_dir, args.annotations, args.image_size, args.heatmap_size)
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) if val_size else (dataset, None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_dataset else None

    model = StackedHourglassNet(num_stacks=2, num_blocks=4, channels=128, num_keypoints=5).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, args.device)
        val_loss = evaluate(model, val_loader, args.device) if val_loader else train_loss
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "val_loss": best_loss}, args.checkpoint_out)


if __name__ == "__main__":
    main()

