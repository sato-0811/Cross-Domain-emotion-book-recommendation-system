"""Compare saved Siamese experiment metric files.

Example:
    python3 src/fusion/compare_siamese_runs.py \
      datasets/siamese_runs/siamese_metrics.json \
      datasets/lstm_siamese_runs/lstm_siamese_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected metrics file: {path}")
    return payload


def run_name(path: Path) -> str:
    parent = path.parent.name
    if parent:
        return parent
    return path.stem


def metric(payload: dict[str, Any], key: str, default: str = "-") -> str:
    value = payload.get(key)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return default
    return str(value)


def summarize(path: Path) -> dict[str, str]:
    payload = load_metrics(path)
    config = payload.get("config") or {}
    final = payload.get("metrics") or {}
    best = payload.get("best_epoch_metrics") or {}
    if not best:
        best_epoch = final.get("best_epoch")
        for row in payload.get("history") or []:
            if row.get("epoch") == float(best_epoch):
                best = row
                break

    return {
        "run": run_name(path),
        "pool": str(config.get("candidate_pool") or final.get("candidate_pool") or "labels"),
        "candidates": metric(config or final, "candidate_book_count"),
        "best_epoch": metric(best, "epoch"),
        "best_val_mrr": metric(best, "val_mrr"),
        "best_test_r1": metric(best, "test_recall_at_1"),
        "best_test_r5": metric(best, "test_recall_at_5"),
        "best_test_mrr": metric(best, "test_mrr"),
        "final_test_r1": metric(final, "test_recall_at_1"),
        "final_test_r5": metric(final, "test_recall_at_5"),
        "final_test_mrr": metric(final, "test_mrr"),
    }


def print_table(rows: list[dict[str, str]]) -> None:
    headers = [
        "run",
        "pool",
        "candidates",
        "best_epoch",
        "best_val_mrr",
        "best_test_r1",
        "best_test_r5",
        "best_test_mrr",
        "final_test_r1",
        "final_test_r5",
        "final_test_mrr",
    ]
    widths = {
        header: max(len(header), *(len(row.get(header, "")) for row in rows))
        for header in headers
    }
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row.get(header, "").ljust(widths[header]) for header in headers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_table([summarize(path) for path in args.metrics])


if __name__ == "__main__":
    main()
