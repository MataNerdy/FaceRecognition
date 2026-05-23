import argparse
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from face_recognition.datasets import default_face_transform
from face_recognition.models.embedding import FaceClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="ImageFolder with aligned faces")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset = ImageFolder(args.data_dir, transform=default_face_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = FaceClassifier(num_classes=len(dataset.classes))
    print(f"Loaded {len(dataset)} images / {len(dataset.classes)} identities")
    print(model)
    print("Training loop is available in notebooks/2_AllModels_Clean (1).ipynb.")


if __name__ == "__main__":
    main()
