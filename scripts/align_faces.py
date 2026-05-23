"""Batch-align face images with a trained Stacked Hourglass landmark model."""

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
    from PIL import Image

    from face_recognition.pipeline import align_with_landmarks, list_images, load_landmark_model, predict_landmarks, save_aligned_face
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_dependencies() -> None:
    """Raise a clear error when alignment dependencies are missing."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError("Batch alignment requires project dependencies. Run `pip install -r requirements.txt`.") from _IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Align a directory of face crops using a landmark checkpoint")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmark-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=112)
    parser.add_argument("--preserve-subdirs", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Align all images and write a CSV processing log."""
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_landmark_model(args.landmark_checkpoint, device=args.device)
    rows = []

    for image_path in list_images(args.input_dir):
        rel = image_path.relative_to(args.input_dir) if args.input_dir.is_dir() else image_path.name
        output_rel = rel if args.preserve_subdirs else Path(image_path.name)
        output_path = (args.output_dir / output_rel).with_suffix(".jpg")
        try:
            image = Image.open(image_path).convert("RGB").resize((args.image_size, args.image_size))
            landmarks = predict_landmarks(model, image, device=args.device, image_size=args.image_size)
            aligned = align_with_landmarks(image, landmarks, output_size=args.output_size)
            save_aligned_face(aligned, output_path)
            rows.append({"source": str(image_path), "output": str(output_path), "status": "ok", "error": ""})
        except Exception as exc:
            rows.append({"source": str(image_path), "output": str(output_path), "status": "error", "error": str(exc)})
            print(f"skip {image_path}: {exc}")

    log_path = args.output_dir / "alignment_log.csv"
    with log_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "output", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aligned {sum(row['status'] == 'ok' for row in rows)} / {len(rows)} images. Log: {log_path}")


if __name__ == "__main__":
    main()
