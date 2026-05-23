from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDatasetFromFolder(Dataset):
    """Triplet sampler over ImageFolder-like directory: root/class_name/*.jpg."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_images = {c.name: sorted(c.glob("*")) for c in self.classes}
        self.samples = [(img, c.name) for c in self.classes for img in self.class_to_images[c.name]]

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
        anchor_path, label = self.samples[idx]
        positive_path = random.choice([p for p in self.class_to_images[label] if p != anchor_path])
        negative_label = random.choice([c for c in self.class_to_images.keys() if c != label])
        negative_path = random.choice(self.class_to_images[negative_label])
        return self._load(anchor_path), self._load(positive_path), self._load(negative_path)


def default_face_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
