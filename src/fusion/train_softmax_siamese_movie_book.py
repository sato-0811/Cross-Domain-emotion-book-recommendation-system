"""Train a full-candidate softmax Siamese model for movie-book matching.

Unlike the pairwise ranking experiment, this trains each movie against all
candidate books with cross-entropy. It is useful when evaluating against the
full PG19 book pool because every update directly penalizes the model if the
positive book is not ranked above the other candidates.

Inputs:
    datasets/book_movie_dataset/{train,val,test}.jsonl
    datasets/movienet/scene_fusion/*_movie_vector.npy
    datasets/pg19_embeddings/*_book_vector.npy

Outputs:
    datasets/softmax_siamese_runs/softmax_siamese_metrics.json
    datasets/softmax_siamese_runs/softmax_siamese_state.pt
    datasets/softmax_siamese_runs/softmax_siamese_best_state.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion.train_siamese_movie_book import (  # noqa: E402
    BOOK_DIR,
    OUTPUT_DIR as PAIRWISE_OUTPUT_DIR,
    PairExample,
    ProjectionMLP,
    build_examples,
    collect_book_ids,
    list_all_book_vector_ids,
    load_all_examples,
    load_book_vector,
    load_movie_vector,
)


OUTPUT_DIR = ROOT / "datasets" / "softmax_siamese_runs"


class SoftmaxMovieBookDataset(Dataset):
    def __init__(self, examples: list[PairExample], book_to_index: dict[str, int]):
        self.examples = [example for example in examples if example.book_id in book_to_index]
        self.book_to_index = book_to_index

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        example = self.examples[index]
        movie_vec = load_movie_vector(example.movie_id)
        if movie_vec is None:
            raise IndexError(index)
        return {
            "movie": movie_vec,
            "target": self.book_to_index[example.book_id],
        }


def collate_batch(batch: list[dict[str, np.ndarray | int]]) -> dict[str, torch.Tensor]:
    movies = torch.from_numpy(np.stack([item["movie"] for item in batch if isinstance(item["movie"], np.ndarray)])).float()
    targets = torch.tensor([int(item["target"]) for item in batch], dtype=torch.long)
    return {"movie": movies, "target": targets}


class SoftmaxSiameseRanker(nn.Module):
    def __init__(self, movie_dim: int, book_dim: int, hidden_dim: int, shared_dim: int, dropout: float):
        super().__init__()
        self.movie_encoder = ProjectionMLP(movie_dim, hidden_dim, shared_dim)
        self.book_encoder = ProjectionMLP(book_dim, hidden_dim, shared_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)

    def forward_movie(self, movie: torch.Tensor) -> torch.Tensor:
        return self.movie_encoder(movie)

    def forward_book(self, book: torch.Tensor) -> torch.Tensor:
        return self.book_encoder(book)

    def logits(self, movie: torch.Tensor, book_matrix: torch.Tensor) -> torch.Tensor:
        movie_z = self.forward_movie(self.dropout(movie))
        book_z = self.forward_book(book_matrix)
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        return scale * (movie_z @ book_z.T)


def load_book_matrix(book_ids: list[str]) -> np.ndarray:
    vectors: list[np.ndarray] = []
    kept_ids: list[str] = []
    for book_id in book_ids:
        vector = load_book_vector(book_id)
        if vector is None:
            continue
        vectors.append(vector)
        kept_ids.append(book_id)
    if len(kept_ids) != len(book_ids):
        missing = set(book_ids) - set(kept_ids)
        raise RuntimeError(f"Missing book vectors for {len(missing)} candidates")
    return np.stack(vectors).astype(np.float32)


def rank_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    target_scores = logits.gather(1, targets[:, None]).squeeze(1)
    ranks = (logits > target_scores[:, None]).sum(dim=1) + 1
    ranks_float = ranks.float()
    return {
        "recall_at_1": float((ranks <= 1).float().mean().item()),
        "recall_at_5": float((ranks <= 5).float().mean().item()),
        "mrr": float((1.0 / ranks_float).mean().item()),
    }


def evaluate(
    model: SoftmaxSiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    book_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    book_to_index = {book_id: index for index, book_id in enumerate(book_ids)}
    dataset = SoftmaxMovieBookDataset(examples, book_to_index)
    if len(dataset) == 0:
        return {"recall_at_1": float("nan"), "recall_at_5": float("nan"), "mrr": float("nan")}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            movie = batch["movie"].to(device)
            target = batch["target"].to(device)
            all_logits.append(model.logits(movie, book_matrix).detach().cpu())
            all_targets.append(target.detach().cpu())

    return rank_metrics_from_logits(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0))


def build_class_weights(
    examples: list[PairExample],
    book_to_index: dict[str, int],
    candidate_count: int,
    mode: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None

    counts = Counter(example.book_id for example in examples if example.book_id in book_to_index)
    weights = torch.ones(candidate_count, dtype=torch.float32)
    if not counts:
        return weights

    present_weights: list[float] = []
    for book_id, count in counts.items():
        if mode == "inverse":
            value = 1.0 / count
        elif mode == "inverse-sqrt":
            value = 1.0 / (count ** 0.5)
        else:
            raise ValueError(f"Unknown class weighting mode: {mode}")
        weights[book_to_index[book_id]] = float(value)
        present_weights.append(float(value))

    mean_present = float(np.mean(present_weights))
    if mean_present > 0:
        for book_id in counts:
            weights[book_to_index[book_id]] /= mean_present
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--shared-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse", "inverse-sqrt"),
        default="none",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-pool",
        choices=("labels", "all-books"),
        default="all-books",
        help="all-books evaluates against every available PG19 book; labels matches the earlier 12-candidate paper metric.",
    )
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

    if args.candidate_pool == "all-books":
        book_ids = list_all_book_vector_ids()
    else:
        book_ids = collect_book_ids(train_examples, val_examples, test_examples)

    if args.limit is not None:
        train_examples = train_examples[: args.limit]

    if not train_examples:
        print("No training examples found.")
        return

    book_to_index = {book_id: index for index, book_id in enumerate(book_ids)}
    train_dataset = SoftmaxMovieBookDataset(train_examples, book_to_index)
    if len(train_dataset) == 0:
        print("No training examples remain after candidate filtering.")
        return

    first_movie = load_movie_vector(train_examples[0].movie_id)
    first_book = load_book_vector(train_examples[0].book_id)
    if first_movie is None or first_book is None:
        raise RuntimeError("First training example could not be loaded.")

    movie_dim = int(first_movie.shape[0])
    book_dim = int(first_book.shape[0])
    device = select_device(args.device)
    model = SoftmaxSiameseRanker(
        movie_dim=movie_dim,
        book_dim=book_dim,
        hidden_dim=args.hidden_dim,
        shared_dim=args.shared_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = build_class_weights(
        train_examples,
        book_to_index,
        candidate_count=len(book_ids),
        mode=args.class_weighting,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    book_matrix_np = load_book_matrix(book_ids)
    book_matrix = torch.from_numpy(book_matrix_np).float().to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    config = {
        "train_size": len(train_dataset),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "candidate_pool": args.candidate_pool,
        "candidate_book_count": len(book_ids),
        "movie_dim": movie_dim,
        "book_dim": book_dim,
        "hidden_dim": args.hidden_dim,
        "shared_dim": args.shared_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weighting": args.class_weighting,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
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
        for batch in train_loader:
            movie = batch["movie"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(movie, book_matrix)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        train_metrics = evaluate(model, train_examples, book_ids, book_matrix, device, args.eval_batch_size)
        val_metrics = evaluate(model, val_examples, book_ids, book_matrix, device, args.eval_batch_size)
        test_metrics = evaluate(model, test_examples, book_ids, book_matrix, device, args.eval_batch_size)
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
            torch.save(model.state_dict(), args.output_dir / "softmax_siamese_best_state.pt")

    final_train_metrics = evaluate(model, train_examples, book_ids, book_matrix, device, args.eval_batch_size)
    final_val_metrics = evaluate(model, val_examples, book_ids, book_matrix, device, args.eval_batch_size)
    final_test_metrics = evaluate(model, test_examples, book_ids, book_matrix, device, args.eval_batch_size)
    final_metrics = {
        "train_size": len(train_dataset),
        "val_size": len(val_examples),
        "test_size": len(test_examples),
        "candidate_pool": args.candidate_pool,
        "candidate_book_count": len(book_ids),
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

    torch.save(model.state_dict(), args.output_dir / "softmax_siamese_state.pt")
    with (args.output_dir / "softmax_siamese_metrics.json").open("w", encoding="utf-8") as file:
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
