"""Build a movie-to-book label dataset with deterministic train/val/test splits.

This script runs the movie-to-book matcher over all MovieNet movies with
scene-fusion outputs, extracts the top-1 book as the teacher label, and
writes split files suitable for downstream supervised training.

Outputs:
    datasets/book_movie_dataset/
        movie_book_labels.jsonl
        train.jsonl
        val.jsonl
        test.jsonl
        manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.book.match_movie_books import match_movie_to_books, select_device


MOVIE_FUSION_DIR = ROOT / "datasets" / "movienet" / "scene_fusion"
OUTPUT_DIR = ROOT / "datasets" / "book_movie_dataset"


def list_movie_ids() -> list[str]:
    movie_ids = {
        path.name.removesuffix("_movie_vector.npy")
        for path in MOVIE_FUSION_DIR.glob("*_movie_vector.npy")
    }
    return sorted(movie_ids)


def split_for_movie(movie_id: str) -> str:
    """Deterministic 80/10/10 split based on movie_id."""
    digest = hashlib.sha256(movie_id.encode("utf-8")).digest()
    bucket = digest[0] % 10
    if bucket == 0:
        return "val"
    if bucket == 1:
        return "test"
    return "train"


def load_processed_movie_ids(output_file: Path) -> set[str]:
    if not output_file.is_file():
        return set()
    processed: set[str] = set()
    with output_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            movie_id = payload.get("movie_id")
            if movie_id:
                processed.add(str(movie_id))
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-id", action="append", help="Can be passed multiple times.")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--band-ratio", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movie_ids = list_movie_ids()
    if args.movie_id:
        requested = set(args.movie_id)
        movie_ids = [movie_id for movie_id in movie_ids if movie_id in requested]
    if args.limit is not None:
        movie_ids = movie_ids[: args.limit]

    if not movie_ids:
        print("No movies found.")
        return

    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_path = args.output_dir / "movie_book_labels.jsonl"
    processed = set() if args.overwrite else load_processed_movie_ids(labels_path)
    mode = "w" if args.overwrite else "a"

    split_paths = {
        "train": args.output_dir / "train.jsonl",
        "val": args.output_dir / "val.jsonl",
        "test": args.output_dir / "test.jsonl",
    }
    split_handles = {
        name: path.open(mode, encoding="utf-8")
        for name, path in split_paths.items()
    }

    manifest = {
        "movie_count": 0,
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "top1_count": 0,
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "device": device,
        "top_k": args.top_k,
        "band_ratio": args.band_ratio,
    }

    try:
        with labels_path.open(mode, encoding="utf-8") as labels_file:
            for movie_id in movie_ids:
                if movie_id in processed:
                    print(f"Skip current: {movie_id}")
                    continue

                result = match_movie_to_books(
                    movie_id=movie_id,
                    top_k=args.top_k,
                    model_name=args.model_name,
                    device=device,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    band_ratio=args.band_ratio,
                )

                results = result.get("results") or []
                top1 = results[0] if results else {}
                split = split_for_movie(movie_id)
                record = {
                    "movie_id": movie_id,
                    "split": split,
                    "label_book_id": top1.get("book_id"),
                    "label_title": top1.get("title"),
                    "label_coarse_score": top1.get("coarse_score"),
                    "label_dtw_score": top1.get("dtw_score"),
                    "label_dtw_normalized": top1.get("dtw_normalized"),
                    "label_movie_len": result.get("movie_len"),
                    "label_book_len": top1.get("book_len"),
                    "candidates": results,
                }

                line = json.dumps(record, ensure_ascii=False)
                labels_file.write(line + "\n")
                split_handles[split].write(line + "\n")
                labels_file.flush()
                split_handles[split].flush()
                processed.add(movie_id)

                manifest["movie_count"] += 1
                manifest["split_counts"][split] += 1
                manifest["top1_count"] += 1 if top1.get("book_id") else 0
                print(
                    f"Saved {movie_id}: split={split} "
                    f"top={top1.get('book_id', 'none')}"
                )
    finally:
        for handle in split_handles.values():
            handle.close()

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
