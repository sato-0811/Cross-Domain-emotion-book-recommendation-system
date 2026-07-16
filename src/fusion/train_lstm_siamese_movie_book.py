"""Train an LSTM-based Siamese network for movie-book matching.

This experiment uses time-series features directly:

    movie scene fusion sequence -> LSTM -> shared embedding
    book block embedding sequence -> LSTM -> shared embedding

It is intended as a comparison against train_siamese_movie_book.py, which
uses precomputed mean-pooled movie/book vectors.

Inputs:
    datasets/book_movie_dataset/{train,val,test}.jsonl
    datasets/movienet/scene_fusion/*_scene_fused_vectors.npy
    datasets/pg19_embeddings/*_block_embeddings.npy

Outputs:
    datasets/lstm_siamese_runs/lstm_siamese_metrics.json
    datasets/lstm_siamese_runs/lstm_siamese_state.pt
    datasets/lstm_siamese_runs/lstm_siamese_best_state.pt
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.train_siamese_movie_book import (  # noqa: E402
    BOOK_DIR,
    MOVIE_DIR,
    PairExample,
    collect_book_ids,
    load_all_examples,
)


OUTPUT_DIR = ROOT / "datasets" / "lstm_siamese_runs"


@dataclass(frozen=True)
class SequenceBatch:
    values: torch.Tensor
    lengths: torch.Tensor


def load_mask(path: Path, expected_len: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    mask = np.load(path).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != expected_len:
        return None
    return mask


def resample_sequence(array: np.ndarray, max_len: int | None) -> np.ndarray:
    if max_len is None or array.shape[0] <= max_len:
        return array
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    indices = np.linspace(0, array.shape[0] - 1, max_len).round().astype(np.int64)
    return array[indices]


def load_sequence(array_path: Path, mask_path: Path, max_len: int | None) -> np.ndarray | None:
    if not array_path.is_file():
        return None
    array = np.load(array_path).astype(np.float32)
    if array.ndim != 2:
        return None

    mask = load_mask(mask_path, expected_len=array.shape[0])
    finite_mask = np.isfinite(array).all(axis=1)
    valid_mask = finite_mask if mask is None else (mask & finite_mask)
    array = array[valid_mask]
    if array.shape[0] == 0:
        return None

    array = resample_sequence(array, max_len=max_len)
    if not np.isfinite(array).all():
        return None
    return np.ascontiguousarray(array, dtype=np.float32)


@lru_cache(maxsize=1024)
def load_movie_sequence(movie_id: str, max_len: int | None) -> np.ndarray | None:
    return load_sequence(
        MOVIE_DIR / f"{movie_id}_scene_fused_vectors.npy",
        MOVIE_DIR / f"{movie_id}_scene_valid_mask.npy",
        max_len=max_len,
    )


@lru_cache(maxsize=1024)
def load_book_sequence(book_id: str, max_len: int | None) -> np.ndarray | None:
    return load_sequence(
        BOOK_DIR / f"{book_id}_block_embeddings.npy",
        BOOK_DIR / f"{book_id}_block_valid_mask.npy",
        max_len=max_len,
    )


def list_all_book_sequence_ids(max_book_len: int | None) -> list[str]:
    book_ids: list[str] = []
    for path in sorted(BOOK_DIR.glob("*_block_embeddings.npy")):
        book_id = path.name.removesuffix("_block_embeddings.npy")
        if load_book_sequence(book_id, max_book_len) is not None:
            book_ids.append(book_id)
    return book_ids


def build_sequence_examples(
    examples: list[PairExample],
    max_movie_len: int | None,
    max_book_len: int | None,
) -> list[PairExample]:
    filtered: list[PairExample] = []
    for example in examples:
        if load_movie_sequence(example.movie_id, max_movie_len) is None:
            continue
        if load_book_sequence(example.book_id, max_book_len) is None:
            continue
        filtered.append(example)
    return filtered


class MovieBookSequenceDataset(Dataset):
    def __init__(
        self,
        examples: list[PairExample],
        book_pool: list[str],
        max_movie_len: int | None,
        max_book_len: int | None,
        seed: int = 13,
        negative_book_ids: list[str] | None = None,
    ):
        self.examples = examples
        self.book_pool = book_pool
        self.max_movie_len = max_movie_len
        self.max_book_len = max_book_len
        self.rng = random.Random(seed)
        self.negative_book_ids = negative_book_ids or []

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        example = self.examples[index]
        movie_seq = load_movie_sequence(example.movie_id, self.max_movie_len)
        pos_book_seq = load_book_sequence(example.book_id, self.max_book_len)
        if movie_seq is None or pos_book_seq is None:
            raise IndexError(index)

        neg_book_id = example.book_id
        if index < len(self.negative_book_ids):
            candidate = self.negative_book_ids[index]
            if candidate and candidate != example.book_id:
                neg_book_id = candidate
        if neg_book_id == example.book_id and self.book_pool:
            for _ in range(20):
                candidate = self.rng.choice(self.book_pool)
                if candidate != example.book_id:
                    neg_book_id = candidate
                    break

        neg_book_seq = load_book_sequence(neg_book_id, self.max_book_len)
        if neg_book_seq is None:
            neg_book_seq = pos_book_seq

        return {
            "movie": movie_seq,
            "movie_len": int(movie_seq.shape[0]),
            "pos_book": pos_book_seq,
            "pos_book_len": int(pos_book_seq.shape[0]),
            "neg_book": neg_book_seq,
            "neg_book_len": int(neg_book_seq.shape[0]),
        }


def pad_sequences(sequences: list[np.ndarray]) -> SequenceBatch:
    if not sequences:
        raise ValueError("Cannot pad an empty sequence list")
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    dim = int(sequences[0].shape[1])
    values = torch.zeros((len(sequences), max_len, dim), dtype=torch.float32)
    for row, sequence in enumerate(sequences):
        length = sequence.shape[0]
        values[row, :length] = torch.from_numpy(sequence)
    return SequenceBatch(values=values, lengths=lengths)


def collate_train_batch(batch: list[dict[str, np.ndarray | int]]) -> dict[str, SequenceBatch]:
    return {
        "movie": pad_sequences([item["movie"] for item in batch if isinstance(item["movie"], np.ndarray)]),
        "pos_book": pad_sequences([item["pos_book"] for item in batch if isinstance(item["pos_book"], np.ndarray)]),
        "neg_book": pad_sequences([item["neg_book"] for item in batch if isinstance(item["neg_book"], np.ndarray)]),
    }


def move_batch(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(values=batch.values.to(device), lengths=batch.lengths.to(device))


class LSTMSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        proj_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        shared_dim: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, shared_dim),
        )

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.detach().cpu()
        projected = self.input_projection(sequence)
        packed = pack_padded_sequence(
            projected,
            lengths=lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        if self.bidirectional:
            final = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final = hidden[-1]
        output = self.output_projection(final)
        return torch.nn.functional.normalize(output, p=2, dim=1)


class LSTMSiameseRanker(nn.Module):
    def __init__(
        self,
        movie_dim: int,
        book_dim: int,
        proj_dim: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        shared_dim: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        self.movie_encoder = LSTMSequenceEncoder(
            input_dim=movie_dim,
            proj_dim=proj_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            shared_dim=shared_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.book_encoder = LSTMSequenceEncoder(
            input_dim=book_dim,
            proj_dim=proj_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            shared_dim=shared_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward_movie(self, batch: SequenceBatch) -> torch.Tensor:
        return self.movie_encoder(batch.values, batch.lengths)

    def forward_book(self, batch: SequenceBatch) -> torch.Tensor:
        return self.book_encoder(batch.values, batch.lengths)

    def score(self, movie: SequenceBatch, book: SequenceBatch) -> torch.Tensor:
        movie_z = self.forward_movie(movie)
        book_z = self.forward_book(book)
        return (movie_z * book_z).sum(dim=1)


def pairwise_ranking_loss(
    movie: SequenceBatch,
    pos_book: SequenceBatch,
    neg_book: SequenceBatch,
    model: LSTMSiameseRanker,
    margin: float,
) -> torch.Tensor:
    pos_score = model.score(movie, pos_book)
    neg_score = model.score(movie, neg_book)
    return torch.relu(margin - pos_score + neg_score).mean()


def encode_book_sequences(
    model: LSTMSiameseRanker,
    book_ids: list[str],
    device: torch.device,
    max_book_len: int | None,
    batch_size: int,
) -> tuple[list[str], torch.Tensor]:
    encoded_ids: list[str] = []
    encoded_vectors: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(book_ids), batch_size):
            chunk = book_ids[start : start + batch_size]
            sequences: list[np.ndarray] = []
            chunk_ids: list[str] = []
            for book_id in chunk:
                sequence = load_book_sequence(book_id, max_book_len)
                if sequence is None:
                    continue
                sequences.append(sequence)
                chunk_ids.append(book_id)
            if not sequences:
                continue
            batch = move_batch(pad_sequences(sequences), device)
            encoded_vectors.append(model.forward_book(batch).detach().cpu())
            encoded_ids.extend(chunk_ids)
    if not encoded_vectors:
        return [], torch.empty(0, 0)
    return encoded_ids, torch.cat(encoded_vectors, dim=0)


def evaluate_split(
    model: LSTMSiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    device: torch.device,
    max_movie_len: int | None,
    max_book_len: int | None,
    batch_size: int,
) -> dict[str, float]:
    if not examples:
        return {"recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan")}

    encoded_book_ids, book_z_cpu = encode_book_sequences(
        model=model,
        book_ids=book_ids,
        device=device,
        max_book_len=max_book_len,
        batch_size=batch_size,
    )
    if not encoded_book_ids:
        return {"recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan")}

    book_index = {book_id: index for index, book_id in enumerate(encoded_book_ids)}
    book_z = book_z_cpu.to(device)
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            movie_sequences: list[np.ndarray] = []
            valid_examples: list[PairExample] = []
            for example in chunk:
                if example.book_id not in book_index:
                    continue
                sequence = load_movie_sequence(example.movie_id, max_movie_len)
                if sequence is None:
                    continue
                movie_sequences.append(sequence)
                valid_examples.append(example)
            if not movie_sequences:
                continue

            movie_batch = move_batch(pad_sequences(movie_sequences), device)
            movie_z = model.forward_movie(movie_batch)
            scores = movie_z @ book_z.T
            for row, example in zip(scores, valid_examples):
                pos_index = book_index[example.book_id]
                pos_score = row[pos_index]
                rank = int((row > pos_score).sum().item()) + 1
                hits_at_1 += 1 if rank <= 1 else 0
                hits_at_5 += 1 if rank <= 5 else 0
                reciprocal_ranks.append(1.0 / rank)

    total = max(1, len(reciprocal_ranks))
    return {
        "recall_at_1": hits_at_1 / total,
        "recall_at_5": hits_at_5 / total,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
    }


def mine_hard_negative_ids(
    model: LSTMSiameseRanker,
    examples: list[PairExample],
    candidate_book_ids: list[str],
    device: torch.device,
    max_movie_len: int | None,
    max_book_len: int | None,
    sample_size: int,
    seed: int,
    batch_size: int,
) -> list[str]:
    if not examples:
        return []

    rng = random.Random(seed)
    pool = list(candidate_book_ids)
    if sample_size > 0 and sample_size < len(pool):
        pool = rng.sample(pool, sample_size)

    pool_ids, pool_z_cpu = encode_book_sequences(
        model=model,
        book_ids=pool,
        device=device,
        max_book_len=max_book_len,
        batch_size=batch_size,
    )
    if not pool_ids:
        return [example.book_id for example in examples]

    pool_index = {book_id: index for index, book_id in enumerate(pool_ids)}
    pool_z = pool_z_cpu.to(device)
    negatives: list[str] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            movie_sequences: list[np.ndarray] = []
            valid_examples: list[PairExample] = []
            for example in chunk:
                sequence = load_movie_sequence(example.movie_id, max_movie_len)
                if sequence is None:
                    continue
                movie_sequences.append(sequence)
                valid_examples.append(example)
            if not movie_sequences:
                continue

            movie_batch = move_batch(pad_sequences(movie_sequences), device)
            movie_z = model.forward_movie(movie_batch)
            scores = movie_z @ pool_z.T
            for row, example in zip(scores, valid_examples):
                masked = row.clone()
                pos_index = pool_index.get(example.book_id)
                if pos_index is not None:
                    masked[pos_index] = -torch.inf
                negatives.append(pool_ids[int(torch.argmax(masked).item())])

    if len(negatives) < len(examples):
        negatives.extend(example.book_id for example in examples[len(negatives) :])
    return negatives


def parse_optional_len(value: int) -> int | None:
    return None if value <= 0 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--shared-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--max-movie-len", type=int, default=128, help="Use <=0 to disable truncation.")
    parser.add_argument("--max-book-len", type=int, default=256, help="Use <=0 to disable truncation.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-pool",
        choices=("labels", "all-books"),
        default="all-books",
        help="all-books evaluates against every available PG19 book; labels matches the earlier 12-candidate paper metric.",
    )
    parser.add_argument(
        "--hard-negative-sample-size",
        type=int,
        default=0,
        help="0 disables hard negative mining. Use a positive value to sample that many candidate books.",
    )
    parser.add_argument("--hard-negative-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
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

    max_movie_len = parse_optional_len(args.max_movie_len)
    max_book_len = parse_optional_len(args.max_book_len)

    splits = load_all_examples()
    train_examples = build_sequence_examples(splits["train"], max_movie_len, max_book_len)
    val_examples = build_sequence_examples(splits["val"], max_movie_len, max_book_len)
    test_examples = build_sequence_examples(splits["test"], max_movie_len, max_book_len)
    if args.candidate_pool == "all-books":
        all_book_ids = list_all_book_sequence_ids(max_book_len)
    else:
        all_book_ids = collect_book_ids(train_examples, val_examples, test_examples)

    if args.limit is not None:
        train_examples = train_examples[: args.limit]

    if not train_examples:
        print("No training sequence pairs found.")
        return

    first_movie = load_movie_sequence(train_examples[0].movie_id, max_movie_len)
    first_book = load_book_sequence(train_examples[0].book_id, max_book_len)
    if first_movie is None or first_book is None:
        raise RuntimeError("First training example could not be loaded.")

    movie_dim = int(first_movie.shape[1])
    book_dim = int(first_book.shape[1])
    device = select_device(args.device)

    model = LSTMSiameseRanker(
        movie_dim=movie_dim,
        book_dim=book_dim,
        proj_dim=args.proj_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config = {
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "candidate_pool": args.candidate_pool,
        "candidate_book_count": len(all_book_ids),
        "movie_dim": movie_dim,
        "book_dim": book_dim,
        "max_movie_len": max_movie_len,
        "max_book_len": max_book_len,
        "proj_dim": args.proj_dim,
        "lstm_hidden_dim": args.lstm_hidden_dim,
        "lstm_layers": args.lstm_layers,
        "shared_dim": args.shared_dim,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
        "margin": args.margin,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "hard_negative_sample_size": args.hard_negative_sample_size,
        "device": str(device),
    }

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
                max_movie_len=max_movie_len,
                max_book_len=max_book_len,
                sample_size=args.hard_negative_sample_size,
                seed=args.seed + epoch,
                batch_size=args.hard_negative_batch_size,
            )
        else:
            negative_book_ids = []

        train_dataset = MovieBookSequenceDataset(
            train_examples,
            all_book_ids,
            max_movie_len=max_movie_len,
            max_book_len=max_book_len,
            seed=args.seed + epoch,
            negative_book_ids=negative_book_ids,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_train_batch,
            drop_last=False,
            num_workers=args.num_workers,
        )

        for batch in train_loader:
            movie = move_batch(batch["movie"], device)
            pos_book = move_batch(batch["pos_book"], device)
            neg_book = move_batch(batch["neg_book"], device)

            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_ranking_loss(movie, pos_book, neg_book, model, args.margin)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        train_metrics = evaluate_split(
            model,
            train_examples,
            all_book_ids,
            device=device,
            max_movie_len=max_movie_len,
            max_book_len=max_book_len,
            batch_size=args.eval_batch_size,
        )
        val_metrics = evaluate_split(
            model,
            val_examples,
            all_book_ids,
            device=device,
            max_movie_len=max_movie_len,
            max_book_len=max_book_len,
            batch_size=args.eval_batch_size,
        )
        test_metrics = evaluate_split(
            model,
            test_examples,
            all_book_ids,
            device=device,
            max_movie_len=max_movie_len,
            max_book_len=max_book_len,
            batch_size=args.eval_batch_size,
        )

        epoch_metrics = {
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
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} loss={avg_loss:.6f} "
            f"train_r1={train_metrics['recall_at_1']:.4f} train_r5={train_metrics['recall_at_5']:.4f} "
            f"val_r1={val_metrics['recall_at_1']:.4f} val_r5={val_metrics['recall_at_5']:.4f} "
            f"test_r1={test_metrics['recall_at_1']:.4f} test_r5={test_metrics['recall_at_5']:.4f}"
        )

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = float(val_metrics["mrr"])
            best_epoch = epoch
            best_metrics = epoch_metrics
            torch.save(model.state_dict(), args.output_dir / "lstm_siamese_best_state.pt")

    final_train_metrics = evaluate_split(
        model,
        train_examples,
        all_book_ids,
        device=device,
        max_movie_len=max_movie_len,
        max_book_len=max_book_len,
        batch_size=args.eval_batch_size,
    )
    final_val_metrics = evaluate_split(
        model,
        val_examples,
        all_book_ids,
        device=device,
        max_movie_len=max_movie_len,
        max_book_len=max_book_len,
        batch_size=args.eval_batch_size,
    )
    final_test_metrics = evaluate_split(
        model,
        test_examples,
        all_book_ids,
        device=device,
        max_movie_len=max_movie_len,
        max_book_len=max_book_len,
        batch_size=args.eval_batch_size,
    )

    final_metrics = {
        "train_size": len(train_examples),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "candidate_pool": args.candidate_pool,
        "candidate_book_count": len(all_book_ids),
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

    torch.save(model.state_dict(), args.output_dir / "lstm_siamese_state.pt")
    with (args.output_dir / "lstm_siamese_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": config,
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
