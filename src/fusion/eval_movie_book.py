"""Evaluate the movie-to-book Siamese model.

Main evaluation:
    - Uses the project-specific hold-out test split from
      datasets/book_movie_dataset/test.jsonl.
    - Reports Recall@1, Recall@5, and MRR.

Supplementary evaluation:
    - Optionally inspects NarrativeQA to report how much movie/gutenberg
      material is present in the requested split.
    - NarrativeQA is not a direct original-book recommendation benchmark,
      so it is treated as auxiliary analysis rather than a primary score.

Example:
    python3 src/fusion/eval_movie_book.py \
        --checkpoint datasets/siamese_runs/siamese_best_state.pt \
        --device cuda \
        --narrativeqa-split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.train_siamese_movie_book import (  # noqa: E402
    BOOK_DIR,
    MOVIE_DIR,
    PairExample,
    SiameseRanker,
    build_examples,
    collect_book_ids,
    load_all_examples,
    load_book_vector,
    load_movie_vector,
    mean_reciprocal_rank,
    recall_at_k,
    retrieval_metrics,
)


DATASET_DIR = ROOT / "datasets" / "book_movie_dataset"


@dataclass(frozen=True)
class NarrativeQASample:
    kind: str
    document_id: str
    document_title: str
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "datasets" / "siamese_runs" / "siamese_best_state.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--narrativeqa-split")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return state


def build_model(device: torch.device, checkpoint: Path) -> tuple[SiameseRanker, list[PairExample], list[PairExample], list[PairExample], list[str]]:
    splits = load_all_examples()
    train_examples = build_examples(splits["train"])
    val_examples = build_examples(splits["val"])
    test_examples = build_examples(splits["test"])
    all_book_ids = collect_book_ids(train_examples, val_examples, test_examples)

    if not train_examples:
        raise RuntimeError("No training examples available.")

    movie_dim = int(load_movie_vector(train_examples[0].movie_id).shape[0])  # type: ignore[union-attr]
    book_dim = int(load_book_vector(train_examples[0].book_id).shape[0])  # type: ignore[union-attr]

    model = SiameseRanker(movie_dim, book_dim, hidden_dim=256, shared_dim=128).to(device)
    state = load_state_dict(checkpoint, device)
    model.load_state_dict(state)
    model.eval()
    return model, train_examples, val_examples, test_examples, all_book_ids


def evaluate_holdout(model: SiameseRanker, test_examples: list[PairExample], all_book_ids: list[str], device: torch.device) -> dict[str, float]:
    return {
        "recall_at_1": recall_at_k(model, test_examples, all_book_ids, k=1, device=device),
        "recall_at_5": recall_at_k(model, test_examples, all_book_ids, k=5, device=device),
        "mrr": mean_reciprocal_rank(model, test_examples, all_book_ids, device=device),
    }


def load_narrativeqa_rows(split_name: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "NarrativeQA requires the `datasets` package. Install it with `pip install datasets`."
        ) from error

    dataset = load_dataset("deepmind/narrativeqa", split=split_name)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(dict(row))
    return rows


def summarize_narrativeqa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts = Counter()
    with_answers = 0
    with_documents = 0
    for row in rows:
        document = row.get("document") or {}
        kind = str(document.get("kind") or "unknown")
        kind_counts[kind] += 1
        if document.get("text"):
            with_documents += 1
        if row.get("answer"):
            with_answers += 1

    return {
        "rows": len(rows),
        "document_kind_counts": dict(sorted(kind_counts.items())),
        "rows_with_document_text": with_documents,
        "rows_with_answer_text": with_answers,
        "note": "NarrativeQA is supplementary only; it does not provide original-book recommendation labels directly.",
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model, train_examples, val_examples, test_examples, all_book_ids = build_model(device, args.checkpoint)

    holdout_metrics = evaluate_holdout(model, test_examples, all_book_ids, device)
    train_metrics = {
        "recall_at_1": recall_at_k(model, train_examples, all_book_ids, k=1, device=device),
        "recall_at_5": recall_at_k(model, train_examples, all_book_ids, k=5, device=device),
        "mrr": mean_reciprocal_rank(model, train_examples, all_book_ids, device=device),
    }
    val_metrics = {
        "recall_at_1": recall_at_k(model, val_examples, all_book_ids, k=1, device=device),
        "recall_at_5": recall_at_k(model, val_examples, all_book_ids, k=5, device=device),
        "mrr": mean_reciprocal_rank(model, val_examples, all_book_ids, device=device),
    }

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": holdout_metrics,
    }

    if args.narrativeqa_split:
        try:
            narrative_rows = load_narrativeqa_rows(args.narrativeqa_split)
            result["narrativeqa"] = summarize_narrativeqa(narrative_rows)
        except Exception as error:  # pragma: no cover - optional path
            result["narrativeqa"] = {
                "error": str(error),
                "note": "NarrativeQA could not be loaded in this environment.",
            }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
