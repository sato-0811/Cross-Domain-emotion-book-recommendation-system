"""Generate scene-level text embeddings from MovieNet script text.

The script consumes the aligned scene JSON files in
``datasets/movienet/scene_outputs`` and encodes each scene with a
Transformer model. It is GPU-ready: when CUDA is available, the model is
automatically moved to GPU and inference runs there.

Output files are written to ``datasets/movienet/script_embeddings``:

* ``<movie_id>_script_embeddings.npy`` - one vector per scene
* ``<movie_id>_script_valid_mask.npy`` - valid-text mask
* ``<movie_id>_scene_ids.json`` - scene id list
* ``<movie_id>_script_texts.json`` - normalized text used for embedding
* ``<movie_id>_script_metadata.json`` - provenance and schema metadata
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as error:  # pragma: no cover - exercised only at runtime.
    raise RuntimeError(
        "Missing text embedding dependencies. Install them with:\n"
        "  pip install -r src/movie/requirements-script.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = ROOT / "datasets" / "movienet" / "scene_outputs"
OUTPUT_DIR = ROOT / "datasets" / "movienet" / "script_embeddings"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
OUTPUT_FORMAT_VERSION = 1

PLACEHOLDER_TEXTS = {
    "No dialogue / Situation missing",
    "No situation description matched from script.",
}


@dataclass(frozen=True)
class TextExample:
    """One scene and the text we want to embed."""

    scene_id: int
    text: str
    valid: bool


def _stable_json_sha256(path: Path) -> str:
    """Hash the exact JSON bytes so metadata can detect stale outputs."""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _normalize_text(values: Any) -> list[str]:
    """Convert a JSON field into cleaned text lines."""
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = str(value).strip()
        if not text or text in PLACEHOLDER_TEXTS:
            continue
        result.append(text)
    return result


def build_scene_text(scene: dict[str, Any]) -> TextExample:
    """Build the final text string for one scene.

    We keep dialogs and situation descriptions separate inside the same
    embedding input so the model sees both the spoken content and the
    screenplay context.
    """
    try:
        scene_id = int(scene.get("scene_id"))
    except (TypeError, ValueError):
        scene_id = -1

    dialogs = _normalize_text(scene.get("dialogs"))
    descriptions = _normalize_text(scene.get("situation_descriptions"))

    parts: list[str] = []
    if descriptions:
        parts.append("Situation: " + " ".join(descriptions))
    if dialogs:
        parts.append("Dialog: " + " ".join(dialogs))

    text = "\n".join(parts).strip()
    return TextExample(scene_id=scene_id, text=text, valid=bool(text))


def load_scene_file(movie_id: str, json_dir: Path = JSON_DIR) -> list[dict[str, Any]]:
    path = json_dir / f"{movie_id}_integrated.json"
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_text_examples(scenes: list[dict[str, Any]]) -> list[TextExample]:
    return [build_scene_text(scene) for scene in scenes]


def select_device(device_arg: str) -> str:
    """Resolve the execution device.

    ``auto`` prefers CUDA when available, otherwise falls back to CPU.
    """
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_name: str, device: str):
    """Load tokenizer and model on the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average token embeddings while ignoring padding."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize embeddings so cosine similarity is stable."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def embed_texts(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Encode a batch of texts into a NumPy array."""
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


def output_is_current(movie_id: str, input_path: Path, output_dir: Path = OUTPUT_DIR) -> bool:
    """Check whether the cached output still matches the source JSON."""
    vector_path = output_dir / f"{movie_id}_script_embeddings.npy"
    metadata_path = output_dir / f"{movie_id}_script_metadata.json"
    if not vector_path.is_file() or not metadata_path.is_file():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError:
        return False

    return (
        metadata.get("format_version") == OUTPUT_FORMAT_VERSION
        and metadata.get("input_json_sha256") == _stable_json_sha256(input_path)
    )


def save_movie_outputs(
    movie_id: str,
    scenes: list[dict[str, Any]],
    examples: list[TextExample],
    embeddings: np.ndarray,
    input_path: Path,
    model_name: str,
    device: str,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Persist scene embeddings and provenance files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = [example.scene_id for example in examples]
    valid_mask = np.asarray([example.valid for example in examples], dtype=bool)
    texts = [example.text for example in examples]

    np.save(output_dir / f"{movie_id}_script_embeddings.npy", embeddings.astype(np.float32))
    np.save(output_dir / f"{movie_id}_script_valid_mask.npy", valid_mask)
    with (output_dir / f"{movie_id}_scene_ids.json").open("w", encoding="utf-8") as file:
        json.dump(scene_ids, file, ensure_ascii=False, indent=2)
    with (output_dir / f"{movie_id}_script_texts.json").open("w", encoding="utf-8") as file:
        json.dump(texts, file, ensure_ascii=False, indent=2)

    metadata = {
        "format_version": OUTPUT_FORMAT_VERSION,
        "movie_id": movie_id,
        "scene_count": len(scenes),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "model_name": model_name,
        "device": device,
        "input_json": str(input_path),
        "input_json_sha256": _stable_json_sha256(input_path),
        "output_files": {
            "embeddings": f"{movie_id}_script_embeddings.npy",
            "valid_mask": f"{movie_id}_script_valid_mask.npy",
            "scene_ids": f"{movie_id}_scene_ids.json",
            "texts": f"{movie_id}_script_texts.json",
        },
        "field_policy": "combined situation_descriptions + dialogs",
    }
    with (output_dir / f"{movie_id}_script_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def process_movie(
    movie_id: str,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    overwrite: bool,
    json_dir: Path = JSON_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Generate embeddings for one movie."""
    input_path = json_dir / f"{movie_id}_integrated.json"
    if not input_path.is_file():
        print(f"Skip {movie_id}: missing {input_path.name}")
        return
    if not overwrite and output_is_current(movie_id, input_path, output_dir):
        print(f"Skip current: {movie_id}")
        return

    scenes = load_scene_file(movie_id, json_dir=json_dir)
    examples = build_text_examples(scenes)
    texts = [example.text if example.valid else "" for example in examples]
    embeddings = np.full((len(examples), model.config.hidden_size), np.nan, dtype=np.float32)

    valid_indices = [index for index, example in enumerate(examples) if example.valid]
    valid_texts = [texts[index] for index in valid_indices]
    if valid_texts:
        valid_embeddings = embed_texts(
            valid_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        embeddings[np.asarray(valid_indices, dtype=np.int64)] = valid_embeddings

    save_movie_outputs(
        movie_id=movie_id,
        scenes=scenes,
        examples=examples,
        embeddings=embeddings,
        input_path=input_path,
        model_name=model.name_or_path,
        device=device,
        output_dir=output_dir,
    )
    print(
        f"Saved: {movie_id}_script_embeddings.npy "
        f"shape={embeddings.shape} valid={len(valid_indices)}/{len(examples)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-id", action="append", help="Can be passed multiple times.")
    parser.add_argument("--json-dir", type=Path, default=JSON_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer, model = load_encoder(args.model_name, device=device)

    json_files = sorted(args.json_dir.glob("*_integrated.json"))
    movie_ids = [path.stem.removesuffix("_integrated") for path in json_files]
    if args.movie_id:
        requested = set(args.movie_id)
        movie_ids = [movie_id for movie_id in movie_ids if movie_id in requested]
    if args.limit is not None:
        movie_ids = movie_ids[: args.limit]

    for movie_id in movie_ids:
        process_movie(
            movie_id=movie_id,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            overwrite=args.overwrite,
            json_dir=args.json_dir,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
