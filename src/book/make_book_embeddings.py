"""Build chapter-like sliding windows and embed them for book data.

This script is the book-side counterpart to the movie pipeline. It:

1. Reads plain-text book files.
2. Splits them into 1000-character windows with 100-character overlap.
3. Encodes each window with a Transformer model.
4. Saves a time-series matrix per book.

GPU behavior:
    ``--device auto`` uses CUDA when available, otherwise CPU.
    ``--device cuda`` forces GPU execution.

Output files are written to ``datasets/books/embeddings``:

* ``<book_id>_block_embeddings.npy`` - one row per text window
* ``<book_id>_block_valid_mask.npy`` - valid-window mask
* ``<book_id>_block_spans.json`` - character span for each window
* ``<book_id>_block_texts.json`` - normalized window text
* ``<book_id>_book_vector.npy`` - mean pooled book-level vector
* ``<book_id>_book_metadata.json`` - provenance and schema metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "datasets" / "pg19"
OUTPUT_DIR = ROOT / "datasets" / "pg19_embeddings"

DEFAULT_MODEL_NAME = "roberta-base"
DEFAULT_BLOCK_SIZE = 1000
DEFAULT_OVERLAP = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_LENGTH = 512
OUTPUT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BookBlock:
    """One sliding-window chunk from a book."""

    block_id: int
    start_char: int
    end_char: int
    text: str
    valid: bool


@dataclass(frozen=True)
class BookSource:
    """One book input, either from a file or from a dataset row."""

    book_id: str
    title: str
    text: str
    source_label: str
    source_sha256: str


def _stable_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def slugify(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "book"


def load_text_file(path: Path) -> str:
    """Read a text file with forgiving decoding."""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_into_blocks(text: str, block_size: int, overlap: int) -> list[BookBlock]:
    """Split text into overlapping character windows.

    The implementation is intentionally simple and deterministic:
    the next window starts ``block_size - overlap`` characters after the
    previous one. The final block is kept even if it is shorter than
    ``block_size`` so the full text is covered.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if overlap < 0 or overlap >= block_size:
        raise ValueError("overlap must satisfy 0 <= overlap < block_size")

    if not text:
        return [BookBlock(1, 0, 0, "", False)]

    step = block_size - overlap
    blocks: list[BookBlock] = []
    start = 0
    block_id = 1
    text_length = len(text)

    while start < text_length:
        end = min(text_length, start + block_size)
        chunk = text[start:end].strip()
        blocks.append(
            BookBlock(
                block_id=block_id,
                start_char=start,
                end_char=end,
                text=chunk,
                valid=bool(chunk),
            )
        )
        if end >= text_length:
            break
        start += step
        block_id += 1

    return blocks


def select_device(device_arg: str) -> str:
    """Resolve the execution device lazily so import works without torch."""
    if device_arg != "auto":
        return device_arg
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_name: str, device: str):
    """Load tokenizer and model on the requested device."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return tokenizer, model


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def normalize_embeddings(embeddings):
    import torch

    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def embed_blocks(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encode a list of blocks into a dense matrix."""
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
            pooled = normalize_embeddings(pooled)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def load_book_files(input_dir: Path) -> list[Path]:
    """Collect PG19 book text files recursively.

    The default PG19 layout in this project is a directory of plain-text
    files, but the loader also accepts a few adjacent text formats so the
    script keeps working if the dataset is unpacked slightly differently.
    """
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".txt", ".text", ".md", ".html", ".htm"}
    )


def load_hf_sources(
    dataset_id: str,
    split: str,
    text_field: str,
    title_field: str,
    streaming: bool,
) -> list[BookSource]:
    """Load books from Hugging Face Datasets."""
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - runtime dependency only.
        raise RuntimeError(
            "Missing Hugging Face Datasets dependency. Install it with:\n"
            "  pip install -r src/book/requirements-book.txt"
        ) from error

    dataset = load_dataset(dataset_id, split=split, streaming=streaming)
    sources: list[BookSource] = []
    for index, row in enumerate(dataset):
        if not isinstance(row, dict):
            continue
        text = str(row.get(text_field) or "")
        if not text.strip():
            continue
        title = str(row.get(title_field) or f"{dataset_id}_{index}")
        book_id = f"{slugify(title)}__{index:05d}"
        source_label = f"{dataset_id}:{split}:{index}"
        sources.append(
            BookSource(
                book_id=book_id,
                title=title,
                text=text,
                source_label=source_label,
                source_sha256=_text_sha256(text),
            )
        )
    return sources


def load_local_sources(input_dir: Path) -> list[BookSource]:
    sources: list[BookSource] = []
    for path in load_book_files(input_dir):
        text = load_text_file(path)
        sources.append(
            BookSource(
                book_id=path.stem,
                title=path.stem,
                text=text,
                source_label=str(path),
                source_sha256=_stable_sha256(path),
            )
        )
    return sources


def output_is_current(book_id: str, source_sha256: str, output_dir: Path = OUTPUT_DIR) -> bool:
    vector_path = output_dir / f"{book_id}_block_embeddings.npy"
    metadata_path = output_dir / f"{book_id}_book_metadata.json"
    if not vector_path.is_file() or not metadata_path.is_file():
        return False

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    return (
        metadata.get("format_version") == OUTPUT_FORMAT_VERSION
        and metadata.get("source_sha256") == source_sha256
    )


def save_outputs(
    book_id: str,
    source_label: str,
    source_sha256: str,
    blocks: list[BookBlock],
    embeddings: np.ndarray,
    model_name: str,
    device: str,
    block_size: int,
    overlap: int,
    batch_size: int,
    max_length: int,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    block_spans = [
        {
            "block_id": block.block_id,
            "start_char": block.start_char,
            "end_char": block.end_char,
            "valid": block.valid,
        }
        for block in blocks
    ]
    block_texts = [block.text for block in blocks]
    valid_mask = np.asarray([block.valid for block in blocks], dtype=bool)

    np.save(output_dir / f"{book_id}_block_embeddings.npy", embeddings.astype(np.float32))
    np.save(output_dir / f"{book_id}_block_valid_mask.npy", valid_mask)
    np.save(output_dir / f"{book_id}_book_vector.npy", _book_vector(embeddings, valid_mask))

    with (output_dir / f"{book_id}_block_spans.json").open("w", encoding="utf-8") as file:
        json.dump(block_spans, file, ensure_ascii=False, indent=2)
    with (output_dir / f"{book_id}_block_texts.json").open("w", encoding="utf-8") as file:
        json.dump(block_texts, file, ensure_ascii=False, indent=2)

    metadata = {
        "format_version": OUTPUT_FORMAT_VERSION,
        "book_id": book_id,
        "source_file": source_label,
        "source_sha256": source_sha256,
        "model_name": model_name,
        "device": device,
        "block_size_chars": block_size,
        "overlap_chars": overlap,
        "batch_size": batch_size,
        "max_length": max_length,
        "block_count": len(blocks),
        "valid_block_count": int(valid_mask.sum()),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "output_files": {
            "block_embeddings": f"{book_id}_block_embeddings.npy",
            "block_valid_mask": f"{book_id}_block_valid_mask.npy",
            "block_spans": f"{book_id}_block_spans.json",
            "block_texts": f"{book_id}_block_texts.json",
            "book_vector": f"{book_id}_book_vector.npy",
        },
        "segmentation": "sliding window over raw characters with fixed overlap",
    }
    with (output_dir / f"{book_id}_book_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _book_vector(embeddings: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    valid_rows = embeddings[valid_mask]
    if len(valid_rows) == 0:
        return np.full((embeddings.shape[1],), np.nan, dtype=np.float32)
    return valid_rows.mean(axis=0).astype(np.float32)


def process_book(
    source: BookSource,
    tokenizer,
    model,
    device: str,
    block_size: int,
    overlap: int,
    batch_size: int,
    max_length: int,
    overwrite: bool,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    book_id = source.book_id
    if not overwrite and output_is_current(book_id, source.source_sha256, output_dir):
        print(f"Skip current: {book_id}")
        return

    text = source.text
    blocks = split_into_blocks(text, block_size=block_size, overlap=overlap)
    valid_texts = [block.text for block in blocks if block.valid]
    embeddings = np.full((len(blocks), model.config.hidden_size), np.nan, dtype=np.float32)

    if valid_texts:
        valid_embeddings = embed_blocks(
            valid_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        valid_indices = np.asarray([index for index, block in enumerate(blocks) if block.valid])
        embeddings[valid_indices] = valid_embeddings

    save_outputs(
        book_id=book_id,
        source_label=source.source_label,
        source_sha256=source.source_sha256,
        blocks=blocks,
        embeddings=embeddings,
        model_name=model.name_or_path,
        device=device,
        block_size=block_size,
        overlap=overlap,
        batch_size=batch_size,
        max_length=max_length,
        output_dir=output_dir,
    )
    print(
        f"Saved: {book_id}_block_embeddings.npy "
        f"shape={embeddings.shape} valid={int(np.isfinite(embeddings).all(axis=1).sum())}/{len(blocks)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--dataset-id", default="emozilla/pg19")
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--title-field", default="short_book_title")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--book-id", action="append", help="Can be passed multiple times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    if args.dataset_id:
        sources = load_hf_sources(
            dataset_id=args.dataset_id,
            split=args.split,
            text_field=args.text_field,
            title_field=args.title_field,
            streaming=args.streaming,
        )
    else:
        sources = load_local_sources(args.input_dir)

    if args.book_id:
        requested = set(args.book_id)
        sources = [source for source in sources if source.book_id in requested]
    if args.limit is not None:
        sources = sources[: args.limit]

    if not sources:
        if args.dataset_id:
            print(f"No rows found in {args.dataset_id} split={args.split}")
        else:
            print(f"No book files found under {args.input_dir}")
        return

    tokenizer, model = load_encoder(args.model_name, device=device)

    for source in sources:
        process_book(
            source=source,
            tokenizer=tokenizer,
            model=model,
            device=device,
            block_size=args.block_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            max_length=args.max_length,
            overwrite=args.overwrite,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
