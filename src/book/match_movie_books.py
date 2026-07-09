"""Match a movie against books with coarse retrieval + DTW refinement.

This script compares a movie's scene text sequence against the PG19 book
sequence database. It uses the same Transformer model for both sides so the
embeddings live in one comparable space.

Pipeline:

1. Build movie scene texts from ``datasets/movienet/scene_outputs``.
2. Embed movie scenes and book blocks with the same model.
3. Coarsely rank all books by mean-vector cosine similarity.
4. Run banded DTW on the top candidates.

The current implementation is text-only. PG19 has no face channel, so this
matcher uses the movie's screenplay-like text side as the common modality.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MOVIE_DIR = ROOT / "datasets" / "movienet" / "scene_outputs"
BOOK_DIR = ROOT / "datasets" / "pg19_embeddings"
OUTPUT_DIR = ROOT / "datasets" / "book_movie_matches"

DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K = 25
DEFAULT_BAND_RATIO = 0.25


@dataclass(frozen=True)
class SequenceItem:
    index: int
    text: str
    valid: bool


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_text(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned


def build_movie_items(movie_id: str) -> list[SequenceItem]:
    path = MOVIE_DIR / f"{movie_id}_integrated.json"
    scenes = _load_json(path)
    items: list[SequenceItem] = []
    for index, scene in enumerate(scenes):
        dialogs = _normalize_text(scene.get("dialogs"))
        descriptions = _normalize_text(scene.get("situation_descriptions"))
        parts: list[str] = []
        if descriptions:
            parts.append("Situation: " + " ".join(descriptions))
        if dialogs:
            parts.append("Dialog: " + " ".join(dialogs))
        text = "\n".join(parts).strip()
        items.append(SequenceItem(index=index, text=text, valid=bool(text)))
    return items


def build_book_items(book_id: str) -> list[SequenceItem]:
    path = BOOK_DIR / f"{book_id}_block_texts.json"
    texts = _load_json(path)
    items: list[SequenceItem] = []
    for index, text in enumerate(texts):
        clean = str(text).strip()
        items.append(SequenceItem(index=index, text=clean, valid=bool(clean)))
    return items


def load_encoder(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    try:
        model.to(device)
    except RuntimeError:
        if device != "cpu":
            device = "cpu"
            print("CUDA unavailable here; falling back to CPU.")
            model.to(device)
        else:
            raise
    model.eval()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return tokenizer, model


def select_device(device_arg: str) -> str:
    import torch

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def mean_embedding(items: list[SequenceItem], embeddings: np.ndarray) -> np.ndarray:
    valid = np.asarray([item.valid for item in items], dtype=bool)
    rows = embeddings[valid]
    if len(rows) == 0:
        return np.full((embeddings.shape[1],), np.nan, dtype=np.float32)
    return rows.mean(axis=0).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("-inf")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("-inf")
    return float(np.dot(a, b) / denom)


def banded_dtw(
    left: np.ndarray,
    right: np.ndarray,
    band_ratio: float,
) -> tuple[float, list[tuple[int, int]]]:
    """Compute cosine-distance DTW with a Sakoe-Chiba band."""
    if left.size == 0 or right.size == 0:
        return float("inf"), []

    left = l2_normalize_rows(left.astype(np.float32))
    right = l2_normalize_rows(right.astype(np.float32))

    n, m = left.shape[0], right.shape[0]
    band = max(abs(n - m), int(math.ceil(max(n, m) * band_ratio)))
    inf = float("inf")

    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    prev_i = np.full((n + 1, m + 1), -1, dtype=np.int32)
    prev_j = np.full((n + 1, m + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = 1.0 - float(np.dot(left[i - 1], right[j - 1]))
            choices = (
                (dp[i - 1, j], i - 1, j),
                (dp[i, j - 1], i, j - 1),
                (dp[i - 1, j - 1], i - 1, j - 1),
            )
            best_prev, best_i, best_j = min(choices, key=lambda item: item[0])
            dp[i, j] = cost + best_prev
            prev_i[i, j] = best_i
            prev_j[i, j] = best_j

    if not np.isfinite(dp[n, m]):
        return float("inf"), []

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i <= 0 or j <= 0:
            break
        path.append((i - 1, j - 1))
        ni, nj = int(prev_i[i, j]), int(prev_j[i, j])
        if ni < 0 or nj < 0:
            break
        i, j = ni, nj
    path.reverse()
    return float(dp[n, m]), path


def load_book_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for meta_path in sorted(BOOK_DIR.glob("*_book_metadata.json")):
        book_id = meta_path.name.removesuffix("_book_metadata.json")
        book_vector_path = BOOK_DIR / f"{book_id}_book_vector.npy"
        book_texts_path = BOOK_DIR / f"{book_id}_block_texts.json"
        if not book_vector_path.is_file():
            continue
        try:
            metadata = _load_json(meta_path)
            vector = np.load(book_vector_path)
            sequence_len = len(_load_json(book_texts_path)) if book_texts_path.is_file() else 0
        except Exception:
            continue
        catalog.append(
            {
                "book_id": book_id,
                "title": metadata.get("book_id", book_id),
                "source_file": metadata.get("source_file"),
                "vector": vector.astype(np.float32),
                "sequence_len": int(sequence_len),
            }
        )
    return catalog


def rank_books(
    movie_vector: np.ndarray,
    movie_len: int,
    catalog: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for entry in catalog:
        cosine = cosine_similarity(movie_vector, entry["vector"])
        book_len = max(1, int(entry.get("sequence_len") or 1))
        length_penalty = abs(movie_len - book_len) / max(movie_len, book_len)
        score = cosine - 0.25 * length_penalty
        if math.isfinite(score):
            ranked.append(
                {
                    **entry,
                    "coarse_score": score,
                    "coarse_cosine": cosine,
                    "length_penalty": length_penalty,
                }
            )
    ranked.sort(key=lambda item: item["coarse_score"], reverse=True)
    return ranked[:top_k]


def match_movie_to_books(
    movie_id: str,
    top_k: int,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    band_ratio: float,
    book_id: str | None = None,
) -> dict[str, Any]:
    tokenizer, model = load_encoder(model_name, device)

    movie_items = build_movie_items(movie_id)
    movie_texts = [item.text for item in movie_items if item.valid]
    movie_embeddings = embed_texts(
        movie_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    movie_vector = movie_embeddings.mean(axis=0).astype(np.float32)
    movie_len = int(movie_embeddings.shape[0])

    if book_id is not None:
        catalog = [
            {
                "book_id": book_id,
                "title": book_id,
                "source_file": None,
                "vector": np.load(BOOK_DIR / f"{book_id}_book_vector.npy").astype(np.float32),
            }
        ]
    else:
        catalog = load_book_catalog()

    candidates = rank_books(movie_vector, movie_len, catalog, top_k=top_k)

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        current_book_id = candidate["book_id"]
        book_items = build_book_items(current_book_id)
        book_texts = [item.text for item in book_items if item.valid]
        book_embeddings = embed_texts(
            book_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        dtw_cost, path = banded_dtw(movie_embeddings, book_embeddings, band_ratio=band_ratio)
        dtw_normalized = (
            dtw_cost / max(1, len(path))
            if math.isfinite(dtw_cost) and len(path) > 0
            else float("inf")
        )
        results.append(
            {
                "book_id": current_book_id,
                "title": candidate.get("title"),
                "source_file": candidate.get("source_file"),
                "coarse_score": candidate["coarse_score"],
                "coarse_cosine": candidate.get("coarse_cosine"),
                "length_penalty": candidate.get("length_penalty"),
                "dtw_cost": dtw_cost,
                "dtw_normalized": dtw_normalized,
                "dtw_score": math.exp(-dtw_normalized) if math.isfinite(dtw_normalized) else 0.0,
                "movie_len": movie_len,
                "book_len": int(book_embeddings.shape[0]),
                "path_length": len(path),
                "alignment_path": path[:200],
            }
        )

    results.sort(
        key=lambda item: (
            item["dtw_score"],
            item["coarse_score"],
            item["coarse_cosine"],
        ),
        reverse=True,
    )
    return {
        "movie_id": movie_id,
        "model_name": model_name,
        "device": device,
        "movie_len": int(movie_embeddings.shape[0]),
        "candidate_count": len(candidates),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-id", required=True)
    parser.add_argument("--book-id")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--band-ratio", type=float, default=DEFAULT_BAND_RATIO)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    result = match_movie_to_books(
        movie_id=args.movie_id,
        top_k=args.top_k,
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        band_ratio=args.band_ratio,
        book_id=args.book_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.movie_id}_matches.json"
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")
    for index, item in enumerate(result["results"][:10], start=1):
        print(
            f"{index}. {item['book_id']} coarse={item['coarse_score']:.4f} "
            f"dtw={item['dtw_score']:.4f} len={item['book_len']}"
        )


if __name__ == "__main__":
    main()
