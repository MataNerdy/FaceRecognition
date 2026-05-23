"""Prepare query/distractor splits for Identification Rate evaluation."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create ImageFolder-style query/distractor splits from aligned faces")
    parser.add_argument("--source-dir", type=Path, required=True, help="ImageFolder-like aligned faces: source/identity/*.jpg")
    parser.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--query-identities", type=int, default=50)
    parser.add_argument("--min-query-images", type=int, default=3)
    parser.add_argument("--max-query-images-per-id", type=int, default=0, help="0 means keep all")
    parser.add_argument("--max-distractor-identities", type=int, default=0, help="0 means keep all remaining identities")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    return parser.parse_args()


def collect_identity_images(source_dir: Path) -> dict[str, list[Path]]:
    """Collect images grouped by identity folder name."""
    identities = {}
    for identity_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        images = sorted(p for p in identity_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            identities[identity_dir.name] = images
    if not identities:
        raise ValueError(f"No identity folders with images found in {source_dir}")
    return identities


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    """Copy or symlink one file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def write_split(images_by_identity: dict[str, list[Path]], identities: list[str], output_dir: Path, copy: bool, max_images: int = 0) -> int:
    """Write one split and return image count."""
    count = 0
    for identity in identities:
        images = images_by_identity[identity]
        if max_images > 0:
            images = images[:max_images]
        for image in images:
            link_or_copy(image, output_dir / identity / image.name, copy=copy)
            count += 1
    return count


def main() -> None:
    """Create query and distractor folders plus metadata CSV."""
    args = parse_args()
    random.seed(args.seed)
    identities = collect_identity_images(args.source_dir)
    query_candidates = [name for name, images in identities.items() if len(images) >= args.min_query_images]
    if len(query_candidates) < args.query_identities:
        raise ValueError(f"Only {len(query_candidates)} identities satisfy min-query-images={args.min_query_images}")

    query_ids = sorted(random.sample(query_candidates, args.query_identities))
    distractor_ids = [name for name in identities if name not in set(query_ids)]
    if args.max_distractor_identities > 0:
        distractor_ids = sorted(random.sample(distractor_ids, min(args.max_distractor_identities, len(distractor_ids))))

    query_dir = args.output_dir / "query"
    distractor_dir = args.output_dir / "distractors"
    if query_dir.exists():
        shutil.rmtree(query_dir)
    if distractor_dir.exists():
        shutil.rmtree(distractor_dir)

    query_count = write_split(identities, query_ids, query_dir, args.copy, max_images=args.max_query_images_per_id)
    distractor_count = write_split(identities, distractor_ids, distractor_dir, args.copy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "split_metadata.csv"
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "identity", "num_images"])
        writer.writeheader()
        for identity in query_ids:
            writer.writerow({"split": "query", "identity": identity, "num_images": len(list((query_dir / identity).iterdir()))})
        for identity in distractor_ids:
            writer.writerow({"split": "distractors", "identity": identity, "num_images": len(list((distractor_dir / identity).iterdir()))})

    print(f"Query: {query_count} images / {len(query_ids)} identities")
    print(f"Distractors: {distractor_count} images / {len(distractor_ids)} identities")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
