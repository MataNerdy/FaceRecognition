"""Dataset helpers for aligned face classification and metric learning."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TripletDatasetFromFolder(Dataset):
    """Sample triplets from an ImageFolder-like directory: `root/class_name/*.jpg`."""

    def __init__(self, root_dir: str | Path, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        self.class_to_images: dict[str, list[Path]] = defaultdict(list)
        for class_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
            if images:
                self.class_to_images[class_dir.name] = images

        self.valid_positive_labels = [label for label, images in self.class_to_images.items() if len(images) >= 2]
        if len(self.class_to_images) < 2 or not self.valid_positive_labels:
            raise ValueError("Triplet sampling requires at least two classes and one class with two images")

        self.samples = [(path, label) for label, images in self.class_to_images.items() for path in images]

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, index: int):
        anchor_path, label = self.samples[index]
        if len(self.class_to_images[label]) < 2:
            label = random.choice(self.valid_positive_labels)
            anchor_path = random.choice(self.class_to_images[label])

        positive_candidates = [path for path in self.class_to_images[label] if path != anchor_path]
        negative_label = random.choice([name for name in self.class_to_images if name != label])
        positive_path = random.choice(positive_candidates)
        negative_path = random.choice(self.class_to_images[negative_label])
        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)


def default_face_transform(image_size: int = 224):
    """Return the default ImageNet-style transform used by the ResNet backbone."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

