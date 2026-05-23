"""Run the full face recognition pipeline on an image or directory."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from src.face_recognition.pipeline import (
        FaceRecognitionPipeline,
        list_images,
        load_face_model,
        load_gallery,
        load_landmark_model,
        save_aligned_face,
    )
except ModuleNotFoundError as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def require_dependencies() -> None:
    """Raise a clear error when inference dependencies are missing."""
    if _IMPORT_ERROR is not None:
        raise RuntimeError("Pipeline inference requires project dependencies. Run `pip install -r requirements.txt`.") from _IMPORT_ERROR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run face alignment and optional recognition on images")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--landmark-checkpoint", type=Path, required=True)
    parser.add_argument("--face-checkpoint", type=Path, help="Optional embedding/classifier checkpoint")
    parser.add_argument("--face-model-type", choices=["embedding", "classifier"], default="embedding")
    parser.add_argument("--num-classes", type=int, help="Required for classifier checkpoints")
    parser.add_argument("--class-names", type=Path, help="Text file with one class name per line")
    parser.add_argument("--gallery", type=Path, help="NPZ with embeddings and labels for nearest identity search")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-detect", action="store_true", help="Treat each input image as a face crop")
    parser.add_argument("--landmark-image-size", type=int, default=256)
    parser.add_argument("--aligned-size", type=int, default=112)
    parser.add_argument("--embedding-image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def read_class_names(path: Path | None) -> list[str] | None:
    """Load class names from a text file."""
    if path is None:
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    """Run the pipeline and save aligned faces plus JSON/CSV metadata."""
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    landmark_model = load_landmark_model(args.landmark_checkpoint, device=args.device)
    face_model = None
    if args.face_checkpoint:
        face_model = load_face_model(
            args.face_checkpoint,
            model_type=args.face_model_type,
            num_classes=args.num_classes,
            embedding_dim=args.embedding_dim,
            device=args.device,
        )
    gallery = load_gallery(args.gallery) if args.gallery else None
    pipeline = FaceRecognitionPipeline(
        landmark_model=landmark_model,
        face_model=face_model,
        device=args.device,
        landmark_image_size=args.landmark_image_size,
        aligned_size=args.aligned_size,
        embedding_image_size=args.embedding_image_size,
        class_names=read_class_names(args.class_names),
        gallery=gallery,
    )

    rows = []
    for path in list_images(args.input):
        try:
            results = pipeline.process_image(path, detect=not args.no_detect, top_k=args.top_k)
        except Exception as exc:  # keep batch inference moving
            rows.append({"source": str(path), "status": "error", "error": str(exc)})
            continue
        for result in results:
            stem = f"{path.stem}_face{result.face_index:02d}.jpg"
            aligned_path = args.output_dir / "aligned" / stem
            save_aligned_face(result.aligned_face, aligned_path)
            rows.append(
                {
                    "source": result.source,
                    "face_index": result.face_index,
                    "status": "ok",
                    "aligned_path": str(aligned_path),
                    "landmarks": result.landmarks.tolist(),
                    "predictions": result.predictions or [],
                    "nearest": result.nearest or [],
                }
            )

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "face_index", "status", "aligned_path", "predictions", "nearest", "error"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    print(f"Saved {len(rows)} result rows to {json_path}")


if __name__ == "__main__":
    main()
