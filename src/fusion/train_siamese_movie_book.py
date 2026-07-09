"""Train a PyTorch Siamese network for movie-book matching.

This version learns from precomputed movie and book vectors, but the
projection layers and the ranking objective are fully implemented in
PyTorch.

Inputs:
    datasets/book_movie_dataset/{train,val,test}.jsonl
    datasets/movienet/scene_fusion/*_movie_vector.npy
    datasets/pg19_embeddings/*_book_vector.npy

Model:
    Two projection MLPs map movie and book vectors into a shared space.
    Training uses a pairwise ranking loss with sampled negatives.

Outputs:
    datasets/siamese_runs/siamese_metrics.json
    datasets/siamese_runs/siamese_state.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_DIR = ROOT / "datasets" / "book_movie_dataset"
MOVIE_DIR = ROOT / "datasets" / "movienet" / "scene_fusion"
BOOK_DIR = ROOT / "datasets" / "pg19_embeddings"
OUTPUT_DIR = ROOT / "datasets" / "siamese_runs"


@dataclass(frozen=True)
class PairExample:
    movie_id: str
    book_id: str
    split: str


def load_split(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    if not path.is_file():
        return examples
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            movie_id = str(payload.get("movie_id") or "")
            book_id = str(payload.get("label_book_id") or "")
            split = str(payload.get("split") or "train")
            if movie_id and book_id:
                examples.append(PairExample(movie_id=movie_id, book_id=book_id, split=split))
    return examples


def load_all_examples() -> dict[str, list[PairExample]]:
    return {
        "train": load_split(DATASET_DIR / "train.jsonl"),
        "val": load_split(DATASET_DIR / "val.jsonl"),
        "test": load_split(DATASET_DIR / "test.jsonl"),
    }


def load_vector(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    array = np.load(path).astype(np.float32)
    if not np.isfinite(array).all():
        return None
    return array


@lru_cache(maxsize=None)
def load_movie_vector(movie_id: str) -> np.ndarray | None:
    return load_vector(MOVIE_DIR / f"{movie_id}_movie_vector.npy")


@lru_cache(maxsize=None)
def load_book_vector(book_id: str) -> np.ndarray | None:
    return load_vector(BOOK_DIR / f"{book_id}_book_vector.npy")


def build_examples(examples: list[PairExample]) -> list[PairExample]:
    filtered: list[PairExample] = []
    for example in examples:
        if load_movie_vector(example.movie_id) is None:
            continue
        if load_book_vector(example.book_id) is None:
            continue
        filtered.append(example)
    return filtered


def collect_book_ids(*splits: list[PairExample]) -> list[str]:
    return sorted({example.book_id for split in splits for example in split})


def rank_of_positive(movie_score: float, candidate_scores: torch.Tensor) -> int:
    return int((candidate_scores > movie_score).sum().item()) + 1


class MovieBookDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        book_pool: list[str],
        seed: int = 13,
        negative_book_ids: list[str] | None = None,
    ):
        self.examples = examples
        self.book_pool = book_pool
        self.rng = random.Random(seed)
        self.negative_book_ids = negative_book_ids or []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        example = self.examples[index]
        movie_vec = load_movie_vector(example.movie_id)
        pos_book_vec = load_book_vector(example.book_id)
        if movie_vec is None or pos_book_vec is None:
            raise IndexError(index)

        neg_book_id = example.book_id
        if index < len(self.negative_book_ids):
            candidate = self.negative_book_ids[index]
            if candidate and candidate != example.book_id:
                neg_book_id = candidate
        if neg_book_id == example.book_id and self.book_pool:
            for _ in range(10):
                candidate = self.rng.choice(self.book_pool)
                if candidate != example.book_id:
                    neg_book_id = candidate
                    break
        neg_book_vec = load_book_vector(neg_book_id)
        if neg_book_vec is None:
            neg_book_vec = pos_book_vec

        return {
            "movie": movie_vec,
            "pos_book": pos_book_vec,
            "neg_book": neg_book_vec,
        }


def collate_batch(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    movie = torch.from_numpy(np.stack([item["movie"] for item in batch])).float()
    pos_book = torch.from_numpy(np.stack([item["pos_book"] for item in batch])).float()
    neg_book = torch.from_numpy(np.stack([item["neg_book"] for item in batch])).float()
    return {"movie": movie, "pos_book": pos_book, "neg_book": neg_book}


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shared_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class SiameseRanker(nn.Module):
    def __init__(self, movie_dim: int, book_dim: int, hidden_dim: int, shared_dim: int):
        super().__init__()
        self.movie_encoder = ProjectionMLP(movie_dim, hidden_dim, shared_dim)
        self.book_encoder = ProjectionMLP(book_dim, hidden_dim, shared_dim)

    def forward_movie(self, movie: torch.Tensor) -> torch.Tensor:
        return self.movie_encoder(movie)

    def forward_book(self, book: torch.Tensor) -> torch.Tensor:
        return self.book_encoder(book)

    def score(self, movie: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        movie_z = self.forward_movie(movie)
        book_z = self.forward_book(book)
        return (movie_z * book_z).sum(dim=1)


def pairwise_ranking_loss(movie: torch.Tensor, pos_book: torch.Tensor, neg_book: torch.Tensor, model: SiameseRanker, margin: float) -> torch.Tensor:
    pos_score = model.score(movie, pos_book)
    neg_score = model.score(movie, neg_book)
    return torch.relu(margin - pos_score + neg_score).mean()


def recall_at_k(
    model: SiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    k: int,
    device: torch.device,
) -> float:
    if not examples:
        return float("nan")

    book_vectors = []
    for book_id in book_ids:
        book_vec = load_book_vector(book_id)
        if book_vec is None:
            continue
        book_vectors.append(book_vec)
    if not book_vectors:
        return float("nan")

    book_tensor = torch.from_numpy(np.stack(book_vectors)).float().to(device)
    with torch.no_grad():
        book_z = model.forward_book(book_tensor)

    hits = 0
    total = 0
    for example in examples:
        movie_vec = load_movie_vector(example.movie_id)
        pos_vec = load_book_vector(example.book_id)
        if movie_vec is None or pos_vec is None:
            continue
        movie_tensor = torch.from_numpy(movie_vec[None, :]).float().to(device)
        pos_tensor = torch.from_numpy(pos_vec[None, :]).float().to(device)
        with torch.no_grad():
            movie_z = model.forward_movie(movie_tensor)
            pos_score = float((movie_z * model.forward_book(pos_tensor)).sum(dim=1).item())
            scores = (movie_z @ book_z.T).squeeze(0)
            rank = rank_of_positive(pos_score, scores)
        if rank <= k:
            hits += 1
        total += 1
    return hits / max(1, total)


def retrieval_metrics(
    model: SiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    device: torch.device,
) -> dict[str, float]:
    metrics = {
        "recall_at_1": recall_at_k(model, examples, book_ids, k=1, device=device),
        "recall_at_5": recall_at_k(model, examples, book_ids, k=5, device=device),
        "mrr": mean_reciprocal_rank(model, examples, book_ids, device=device),
    }
    return metrics


def encode_book_vectors(
    model: SiameseRanker,
    book_ids: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> tuple[list[str], torch.Tensor]:
    encoded_ids: list[str] = []
    encoded_vectors: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(book_ids), batch_size):
            chunk = book_ids[start : start + batch_size]
            vectors = []
            chunk_ids: list[str] = []
            for book_id in chunk:
                book_vec = load_book_vector(book_id)
                if book_vec is None:
                    continue
                vectors.append(book_vec)
                chunk_ids.append(book_id)
            if not vectors:
                continue
            book_tensor = torch.from_numpy(np.stack(vectors)).float().to(device)
            encoded_vectors.append(model.forward_book(book_tensor).detach().cpu())
            encoded_ids.extend(chunk_ids)
    if not encoded_vectors:
        return [], torch.empty(0, 0)
    return encoded_ids, torch.cat(encoded_vectors, dim=0)


def mine_hard_negative_ids(
    model: SiameseRanker,
    examples: list[PairExample],
    candidate_book_ids: list[str],
    device: torch.device,
    sample_size: int,
    seed: int,
    batch_size: int = 64,
) -> list[str]:
    if not examples:
        return []

    rng = random.Random(seed)
    pool = list(candidate_book_ids)
    if sample_size > 0 and sample_size < len(pool):
        pool = rng.sample(pool, sample_size)

    pool_ids, pool_z_cpu = encode_book_vectors(model, pool, device=device)
    if not pool_ids:
        return [example.book_id for example in examples]

    pool_z = pool_z_cpu.to(device)
    pool_index = {book_id: idx for idx, book_id in enumerate(pool_ids)}
    negatives: list[str] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            movie_vectors = []
            valid_examples: list[PairExample] = []
            for example in batch:
                movie_vec = load_movie_vector(example.movie_id)
                if movie_vec is None:
                    continue
                movie_vectors.append(movie_vec)
                valid_examples.append(example)
            if not movie_vectors:
                continue
            movie_tensor = torch.from_numpy(np.stack(movie_vectors)).float().to(device)
            movie_z = model.forward_movie(movie_tensor)
            scores = movie_z @ pool_z.T

            for row, example in zip(scores, valid_examples):
                masked = row.clone()
                pos_index = pool_index.get(example.book_id)
                if pos_index is not None:
                    masked[pos_index] = -torch.inf
                best_index = int(torch.argmax(masked).item())
                negatives.append(pool_ids[best_index])

    if len(negatives) < len(examples):
        negatives.extend(example.book_id for example in examples[len(negatives) :])
    return negatives


def mean_reciprocal_rank(
    model: SiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    device: torch.device,
) -> float:
    if not examples:
        return float("nan")

    book_vectors = []
    for book_id in book_ids:
        book_vec = load_book_vector(book_id)
        if book_vec is not None:
            book_vectors.append(book_vec)
    if not book_vectors:
        return float("nan")

    book_tensor = torch.from_numpy(np.stack(book_vectors)).float().to(device)
    with torch.no_grad():
        book_z = model.forward_book(book_tensor)

    reciprocal_ranks: list[float] = []
    for example in examples:
        movie_vec = load_movie_vector(example.movie_id)
        pos_vec = load_book_vector(example.book_id)
        if movie_vec is None or pos_vec is None:
            continue
        movie_tensor = torch.from_numpy(movie_vec[None, :]).float().to(device)
        pos_tensor = torch.from_numpy(pos_vec[None, :]).float().to(device)
        with torch.no_grad():
            movie_z = model.forward_movie(movie_tensor)
            pos_score = float((movie_z * model.forward_book(pos_tensor)).sum(dim=1).item())
            scores = (movie_z @ book_z.T).squeeze(0)
            rank = rank_of_positive(pos_score, scores)
        reciprocal_ranks.append(1.0 / rank)
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--shared-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hard-negative-sample-size", type=int, default=4096)
    parser.add_argument("--hard-negative-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_all_examples()
    train_examples = build_examples(splits["train"])
    val_examples = build_examples(splits["val"])
    test_examples = build_examples(splits["test"])
    all_book_ids = collect_book_ids(train_examples, val_examples, test_examples)

    if args.limit is not None:
        train_examples = train_examples[: args.limit]

    if not train_examples:
        print("No training pairs found.")
        return

    movie_dim = int(load_movie_vector(train_examples[0].movie_id).shape[0])  # type: ignore[union-attr]
    book_dim = int(load_book_vector(train_examples[0].book_id).shape[0])  # type: ignore[union-attr]
    device = select_device(args.device)

    model = SiameseRanker(movie_dim, book_dim, args.hidden_dim, args.shared_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val_mrr = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        if args.hard_negative_sample_size > 0:
            negative_book_ids = mine_hard_negative_ids(
                model=model,
                examples=train_examples,
                candidate_book_ids=all_book_ids,
                device=device,
                sample_size=args.hard_negative_sample_size,
                seed=args.seed + epoch,
                batch_size=args.hard_negative_batch_size,
            )
        else:
            negative_book_ids = []

        train_dataset = MovieBookDataset(
            train_examples,
            all_book_ids,
            seed=args.seed + epoch,
            negative_book_ids=negative_book_ids,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            drop_last=False,
        )

        for batch in train_loader:
            movie = batch["movie"].to(device)
            pos_book = batch["pos_book"].to(device)
            neg_book = batch["neg_book"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(movie, pos_book, neg_book, model, args.margin)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        model.eval()
        train_metrics = retrieval_metrics(model, train_examples, all_book_ids, device=device)
        val_metrics = retrieval_metrics(model, val_examples, all_book_ids, device=device)
        test_metrics = retrieval_metrics(model, test_examples, all_book_ids, device=device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "train_recall_at_1": float(train_metrics["recall_at_1"]),
                "train_recall_at_5": float(train_metrics["recall_at_5"]),
                "train_mrr": float(train_metrics["mrr"]),
                "val_recall_at_1": float(val_metrics["recall_at_1"]),
                "val_recall_at_5": float(val_metrics["recall_at_5"]),
                "val_mrr": float(val_metrics["mrr"]),
                "test_recall_at_1": float(test_metrics["recall_at_1"]),
                "test_recall_at_5": float(test_metrics["recall_at_5"]),
                "test_mrr": float(test_metrics["mrr"]),
            }
        )
        print(
            f"epoch={epoch} loss={avg_loss:.6f} "
            f"train_r1={train_metrics['recall_at_1']:.4f} train_r5={train_metrics['recall_at_5']:.4f} "
            f"val_r1={val_metrics['recall_at_1']:.4f} val_r5={val_metrics['recall_at_5']:.4f} "
            f"test_r1={test_metrics['recall_at_1']:.4f} test_r5={test_metrics['recall_at_5']:.4f}"
        )

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = float(val_metrics["mrr"])
            best_epoch = epoch
            best_metrics = {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "train_recall_at_1": float(train_metrics["recall_at_1"]),
                "train_recall_at_5": float(train_metrics["recall_at_5"]),
                "train_mrr": float(train_metrics["mrr"]),
                "val_recall_at_1": float(val_metrics["recall_at_1"]),
                "val_recall_at_5": float(val_metrics["recall_at_5"]),
                "val_mrr": float(val_metrics["mrr"]),
                "test_recall_at_1": float(test_metrics["recall_at_1"]),
                "test_recall_at_5": float(test_metrics["recall_at_5"]),
                "test_mrr": float(test_metrics["mrr"]),
            }
            torch.save(model.state_dict(), args.output_dir / "siamese_best_state.pt")

    model.eval()
    final_train_metrics = retrieval_metrics(model, train_examples, all_book_ids, device=device)
    final_val_metrics = retrieval_metrics(model, val_examples, all_book_ids, device=device)
    final_test_metrics = retrieval_metrics(model, test_examples, all_book_ids, device=device)
    final_metrics = {
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "movie_dim": movie_dim,
        "book_dim": book_dim,
        "train_recall_at_1": float(final_train_metrics["recall_at_1"]),
        "train_recall_at_5": float(final_train_metrics["recall_at_5"]),
        "train_mrr": float(final_train_metrics["mrr"]),
        "val_recall_at_1": float(final_val_metrics["recall_at_1"]),
        "val_recall_at_5": float(final_val_metrics["recall_at_5"]),
        "val_mrr": float(final_val_metrics["mrr"]),
        "test_recall_at_1": float(final_test_metrics["recall_at_1"]),
        "test_recall_at_5": float(final_test_metrics["recall_at_5"]),
        "test_mrr": float(final_test_metrics["mrr"]),
        "best_epoch": int(best_epoch),
        "best_val_mrr": float(best_val_mrr),
    }

    torch.save(model.state_dict(), args.output_dir / "siamese_state.pt")
    with (args.output_dir / "siamese_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "history": history,
                "metrics": final_metrics,
                "best_epoch_metrics": best_metrics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
