"""Visualize the rank distribution of the positive book for Softmax Siamese.

For each test movie, this script reloads the trained Softmax Siamese model,
scores every candidate book, records the rank of the pseudo-label book, and
writes a rank-distribution chart plus per-example rank details.
"""

from __future__ import annotations

import argparse
import csv
import html
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
    PairExample,
    build_examples,
    collect_book_ids,
    list_all_book_vector_ids,
    load_all_examples,
    load_book_vector,
    load_movie_vector,
)
from src.fusion.train_softmax_siamese_movie_book import (  # noqa: E402
    OUTPUT_DIR,
    SoftmaxSiameseRanker,
    load_book_matrix,
)


DEFAULT_CHECKPOINT = OUTPUT_DIR / "softmax_siamese_best_state.pt"
DEFAULT_METRICS = OUTPUT_DIR / "softmax_siamese_metrics.json"
DEFAULT_CSV = OUTPUT_DIR / "softmax_test_rank_details.csv"
DEFAULT_JSON = OUTPUT_DIR / "softmax_test_rank_distribution.json"
DEFAULT_SVG = ROOT / "figures" / "softmax_test_rank_distribution.svg"
DEFAULT_PNG = ROOT / "figures" / "softmax_test_rank_distribution.png"


@dataclass(frozen=True)
class RankRecord:
    movie_id: str
    label_book_id: str
    rank: int
    target_score: float
    top_book_id: str
    top_score: float


def read_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint payload in {path}")
    return state


def build_model(metrics_path: Path, checkpoint: Path, device: torch.device) -> SoftmaxSiameseRanker:
    payload = read_metrics(metrics_path)
    config = payload.get("config") or {}
    metrics = payload.get("metrics") or {}

    movie_dim = int(config.get("movie_dim") or metrics.get("movie_dim"))
    book_dim = int(config.get("book_dim") or metrics.get("book_dim"))
    hidden_dim = int(config.get("hidden_dim", 256))
    shared_dim = int(config.get("shared_dim", 128))
    dropout = float(config.get("dropout", 0.1))

    model = SoftmaxSiameseRanker(
        movie_dim=movie_dim,
        book_dim=book_dim,
        hidden_dim=hidden_dim,
        shared_dim=shared_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(load_state_dict(checkpoint, device))
    model.eval()
    return model


def build_candidate_book_ids(candidate_pool: str) -> list[str]:
    splits = load_all_examples()
    train_examples = build_examples(splits["train"])
    val_examples = build_examples(splits["val"])
    test_examples = build_examples(splits["test"])
    if candidate_pool == "all-books":
        return list_all_book_vector_ids()
    if candidate_pool == "labels":
        return collect_book_ids(train_examples, val_examples, test_examples)
    raise ValueError(f"Unknown candidate pool: {candidate_pool}")


def collect_rank_records(
    model: SoftmaxSiameseRanker,
    examples: list[PairExample],
    book_ids: list[str],
    device: torch.device,
) -> list[RankRecord]:
    book_to_index = {book_id: index for index, book_id in enumerate(book_ids)}
    book_matrix = torch.from_numpy(load_book_matrix(book_ids)).float().to(device)
    records: list[RankRecord] = []

    with torch.no_grad():
        for example in examples:
            target_index = book_to_index.get(example.book_id)
            movie_vector = load_movie_vector(example.movie_id)
            if target_index is None or movie_vector is None or load_book_vector(example.book_id) is None:
                continue

            movie = torch.from_numpy(movie_vector[None, :].astype(np.float32)).to(device)
            logits = model.logits(movie, book_matrix).squeeze(0).detach().cpu()
            target_score = float(logits[target_index].item())
            rank = int((logits > target_score).sum().item()) + 1
            top_index = int(torch.argmax(logits).item())
            records.append(
                RankRecord(
                    movie_id=example.movie_id,
                    label_book_id=example.book_id,
                    rank=rank,
                    target_score=target_score,
                    top_book_id=book_ids[top_index],
                    top_score=float(logits[top_index].item()),
                )
            )
    return records


def summarize(records: list[RankRecord]) -> dict[str, Any]:
    ranks = [record.rank for record in records]
    counts = Counter(ranks)
    total = len(ranks)
    return {
        "test_size": total,
        "rank_counts": {str(rank): counts[rank] for rank in sorted(counts)},
        "recall_at_1": sum(rank <= 1 for rank in ranks) / total if total else float("nan"),
        "recall_at_5": sum(rank <= 5 for rank in ranks) / total if total else float("nan"),
        "mrr": float(np.mean([1.0 / rank for rank in ranks])) if ranks else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
        "max_rank": max(ranks) if ranks else None,
    }


def write_csv(records: list[RankRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "movie_id",
                "label_book_id",
                "rank",
                "target_score",
                "top_book_id",
                "top_score",
                "is_top1",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "movie_id": record.movie_id,
                    "label_book_id": record.label_book_id,
                    "rank": record.rank,
                    "target_score": f"{record.target_score:.8f}",
                    "top_book_id": record.top_book_id,
                    "top_score": f"{record.top_score:.8f}",
                    "is_top1": int(record.rank == 1),
                }
            )


def write_json(summary: dict[str, Any], records: list[RankRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **summary,
        "records": [
            {
                "movie_id": record.movie_id,
                "label_book_id": record.label_book_id,
                "rank": record.rank,
                "target_score": record.target_score,
                "top_book_id": record.top_book_id,
                "top_score": record.top_score,
            }
            for record in records
        ],
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 12,
    weight: int | str = 400,
    anchor: str = "middle",
    fill: str = "#202124",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"/>'
    )


def svg_rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}"/>'


def render_svg(summary: dict[str, Any], output: Path) -> None:
    rank_counts = {int(rank): int(count) for rank, count in summary["rank_counts"].items()}
    ranks = sorted(rank_counts)
    total = int(summary["test_size"])
    width = 980
    height = 560
    left = 88
    right = 42
    top = 122
    chart_height = 310
    chart_width = width - left - right
    bottom = top + chart_height
    max_count = max(rank_counts.values(), default=1)
    y_max = max_count if max_count % 5 == 0 else max_count + (5 - max_count % 5)
    slot_width = chart_width / max(1, len(ranks))
    bar_width = min(72, slot_width * 0.62)

    def y_for(value: float) -> float:
        return bottom - (value / y_max) * chart_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#FFFFFF"),
        svg_text(width / 2, 38, "Softmax Siamese: rank distribution of pseudo-label books", size=26, weight=700),
        svg_text(
            width / 2,
            68,
            f"Test split n={total}, candidate books=406, R@1={summary['recall_at_1']:.3f}, R@5={summary['recall_at_5']:.3f}, MRR={summary['mrr']:.3f}",
            size=16,
            fill="#5F6368",
        ),
        svg_text(
            width / 2,
            94,
            f"Median rank={summary['median_rank']:.1f}, mean rank={summary['mean_rank']:.2f}, worst rank={summary['max_rank']}",
            size=16,
            fill="#5F6368",
        ),
    ]

    tick_step = 5 if y_max > 20 else 2 if y_max > 10 else 1
    for value in range(0, y_max + 1, tick_step):
        y = y_for(value)
        parts.append(svg_line(left, y, left + chart_width, y, "#E0E0E0"))
        parts.append(svg_text(left - 14, y + 5, value, size=14, anchor="end", fill="#5F6368"))
    parts.append(svg_line(left, top, left, bottom, "#202124", 1.2))
    parts.append(svg_line(left, bottom, left + chart_width, bottom, "#202124", 1.2))
    parts.append(svg_text(width / 2, height - 32, "Rank of pseudo-label book in recommendation list", size=19, fill="#3C4043"))
    parts.append(svg_text(left, top - 17, "samples", size=13, weight=700, fill="#3C4043"))
    parts.append(svg_text(left + chart_width, bottom + 52, "rank", size=13, weight=700, anchor="end", fill="#3C4043"))

    for index, rank in enumerate(ranks):
        count = rank_counts[rank]
        x = left + index * slot_width + (slot_width - bar_width) / 2
        y = y_for(count)
        fill = "#4E79A7" if rank == 1 else "#59A14F" if rank <= 5 else "#F28E2B"
        parts.append(svg_rect(x, y, bar_width, bottom - y, fill))
        parts.append(svg_text(x + bar_width / 2, y - 9, count, size=16, weight=700))
        parts.append(svg_text(x + bar_width / 2, bottom + 27, rank, size=15, weight=700))

    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def load_font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def render_png(summary: dict[str, Any], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("PNG output requires Pillow. Install pillow or omit --png-output.") from exc

    rank_counts = {int(rank): int(count) for rank, count in summary["rank_counts"].items()}
    ranks = sorted(rank_counts)
    total = int(summary["test_size"])
    width = 1960
    height = 1120
    left = 164
    right = 84
    top = 244
    chart_height = 620
    chart_width = width - left - right
    bottom = top + chart_height
    max_count = max(rank_counts.values(), default=1)
    y_max = max_count if max_count % 5 == 0 else max_count + (5 - max_count % 5)
    slot_width = chart_width / max(1, len(ranks))
    bar_width = min(144, slot_width * 0.62)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(52, bold=True)
    font_subtitle = load_font(32)
    font_axis = load_font(30)
    font_x_axis = load_font(36)
    font_unit = load_font(28, bold=True)
    font_value = load_font(34, bold=True)
    font_rank = load_font(30, bold=True)

    def y_for(value: float) -> float:
        return bottom - (value / y_max) * chart_height

    draw.text((width / 2, 76), "Softmax Siamese: rank distribution of pseudo-label books", fill=rgb("#202124"), font=font_title, anchor="mm")
    draw.text(
        (width / 2, 136),
        f"Test split n={total}, candidate books=406, R@1={summary['recall_at_1']:.3f}, R@5={summary['recall_at_5']:.3f}, MRR={summary['mrr']:.3f}",
        fill=rgb("#5F6368"),
        font=font_subtitle,
        anchor="mm",
    )
    draw.text(
        (width / 2, 186),
        f"Median rank={summary['median_rank']:.1f}, mean rank={summary['mean_rank']:.2f}, worst rank={summary['max_rank']}",
        fill=rgb("#5F6368"),
        font=font_subtitle,
        anchor="mm",
    )

    tick_step = 5 if y_max > 20 else 2 if y_max > 10 else 1
    for value in range(0, y_max + 1, tick_step):
        y = y_for(value)
        draw.line((left, y, left + chart_width, y), fill=rgb("#E0E0E0"), width=2)
        draw.text((left - 28, y), str(value), fill=rgb("#5F6368"), font=font_axis, anchor="rm")
    draw.line((left, top, left, bottom), fill=rgb("#202124"), width=3)
    draw.line((left, bottom, left + chart_width, bottom), fill=rgb("#202124"), width=3)
    draw.text((width / 2, height - 68), "Rank of pseudo-label book in recommendation list", fill=rgb("#3C4043"), font=font_x_axis, anchor="mm")
    draw.text((left, top - 34), "samples", fill=rgb("#3C4043"), font=font_unit, anchor="mm")
    draw.text((left + chart_width, bottom + 104), "rank", fill=rgb("#3C4043"), font=font_unit, anchor="rm")

    for index, rank in enumerate(ranks):
        count = rank_counts[rank]
        x = left + index * slot_width + (slot_width - bar_width) / 2
        y = y_for(count)
        fill = "#4E79A7" if rank == 1 else "#59A14F" if rank <= 5 else "#F28E2B"
        draw.rectangle((x, y, x + bar_width, bottom), fill=rgb(fill))
        draw.text((x + bar_width / 2, y - 22), str(count), fill=rgb("#202124"), font=font_value, anchor="mm")
        draw.text((x + bar_width / 2, bottom + 48), str(rank), fill=rgb("#202124"), font=font_rank, anchor="mm")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidate-pool", choices=("all-books", "labels"), default="all-books")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--svg-output", type=Path, default=DEFAULT_SVG)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = build_model(args.metrics, args.checkpoint, device)
    splits = load_all_examples()
    test_examples = build_examples(splits["test"])
    book_ids = build_candidate_book_ids(args.candidate_pool)
    records = collect_rank_records(model, test_examples, book_ids, device)
    summary = summarize(records)

    write_csv(records, args.csv_output)
    write_json(summary, records, args.json_output)
    render_svg(summary, args.svg_output)
    render_png(summary, args.png_output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.csv_output}")
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.svg_output}")
    print(f"Wrote {args.png_output}")


if __name__ == "__main__":
    main()
