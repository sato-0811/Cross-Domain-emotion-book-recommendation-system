"""Batch movie-to-book label generation.

This script runs the matcher over all MovieNet movies that already have
scene-fusion outputs, and writes one JSONL line per movie with the best
candidate book labels.

Output:
    ``datasets/book_movie_matches/movie_book_labels.jsonl``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.book.match_movie_books import match_movie_to_books, select_device


MOVIE_FUSION_DIR = ROOT / "datasets" / "movienet" / "scene_fusion"
OUTPUT_DIR = ROOT / "datasets" / "book_movie_matches"


def list_movie_ids() -> list[str]:
    movie_ids = {
        path.name.removesuffix("_movie_vector.npy")
        for path in MOVIE_FUSION_DIR.glob("*_movie_vector.npy")
    }
    return sorted(movie_ids)


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
    parser.add_argument("--output-file", type=Path, default=OUTPUT_DIR / "movie_book_labels.jsonl")
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
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    processed = set() if args.overwrite else load_processed_movie_ids(args.output_file)
    mode = "w" if args.overwrite else "a"
    with args.output_file.open(mode, encoding="utf-8") as file:
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
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            processed.add(movie_id)
            print(
                f"Saved {movie_id}: candidates={result['candidate_count']} "
                f"top={result['results'][0]['book_id'] if result['results'] else 'none'}"
            )


if __name__ == "__main__":
    main()
