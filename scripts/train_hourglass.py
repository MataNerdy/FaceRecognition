"""Train Stacked Hourglass landmark detector.

This is a clean entrypoint. Dataset paths are intentionally configurable because
CelebA files are not stored in the repository.
"""
import argparse
import torch
from face_recognition.models.hourglass import StackedHourglassNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-keypoints", type=int, default=5)
    parser.add_argument("--checkpoint-out", default="hourglass_best.pth")
    args = parser.parse_args()

    model = StackedHourglassNet(num_stacks=2, num_blocks=4, channels=128, num_keypoints=args.num_keypoints)
    print(model)
    print("Add CelebA dataloaders from notebooks/1_StackedHourGlassNetwork_Clean.ipynb and train loop here.")
    torch.save({"model_state_dict": model.state_dict()}, args.checkpoint_out)


if __name__ == "__main__":
    main()
