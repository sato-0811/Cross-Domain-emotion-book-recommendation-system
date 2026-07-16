"""Generate all bar charts used for the Siamese model evaluation.

This script creates:
1. Model-performance comparison over all 406 candidate books.
2. Rank distribution of pseudo-label books on the Softmax Siamese test split.

Examples:
    python3 src/fusion/plot_all_bar_charts.py
    python3 src/fusion/plot_all_bar_charts.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fusion import plot_siamese_model_comparison as model_plot
from src.fusion import plot_softmax_rank_distribution as rank_plot
from src.fusion.train_siamese_movie_book import build_examples, load_all_examples


def generate_model_comparison(svg_output: Path, png_output: Path) -> None:
    rows = model_plot.collect_metrics(model_plot.DEFAULT_METRICS)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.write_text(model_plot.render_svg(rows), encoding="utf-8")
    model_plot.render_png(rows, png_output)


def generate_rank_distribution(
    *,
    device_arg: str,
    checkpoint: Path,
    metrics: Path,
    candidate_pool: str,
    csv_output: Path,
    json_output: Path,
    svg_output: Path,
    png_output: Path,
) -> dict[str, object]:
    device = rank_plot.select_device(device_arg)
    model = rank_plot.build_model(metrics, checkpoint, device)
    splits = load_all_examples()
    test_examples = build_examples(splits["test"])
    book_ids = rank_plot.build_candidate_book_ids(candidate_pool)
    records = rank_plot.collect_rank_records(model, test_examples, book_ids, device)
    summary = rank_plot.summarize(records)

    rank_plot.write_csv(records, csv_output)
    rank_plot.write_json(summary, records, json_output)
    rank_plot.render_svg(summary, svg_output)
    rank_plot.render_png(summary, png_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidate-pool", choices=("all-books", "labels"), default="all-books")
    parser.add_argument("--softmax-checkpoint", type=Path, default=rank_plot.DEFAULT_CHECKPOINT)
    parser.add_argument("--softmax-metrics", type=Path, default=rank_plot.DEFAULT_METRICS)
    parser.add_argument("--model-svg", type=Path, default=model_plot.DEFAULT_OUTPUT)
    parser.add_argument("--model-png", type=Path, default=model_plot.DEFAULT_PNG_OUTPUT)
    parser.add_argument("--rank-svg", type=Path, default=rank_plot.DEFAULT_SVG)
    parser.add_argument("--rank-png", type=Path, default=rank_plot.DEFAULT_PNG)
    parser.add_argument("--rank-csv", type=Path, default=rank_plot.DEFAULT_CSV)
    parser.add_argument("--rank-json", type=Path, default=rank_plot.DEFAULT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generate_model_comparison(args.model_svg, args.model_png)
    print(f"Wrote {args.model_svg}")
    print(f"Wrote {args.model_png}")

    summary = generate_rank_distribution(
        device_arg=args.device,
        checkpoint=args.softmax_checkpoint,
        metrics=args.softmax_metrics,
        candidate_pool=args.candidate_pool,
        csv_output=args.rank_csv,
        json_output=args.rank_json,
        svg_output=args.rank_svg,
        png_output=args.rank_png,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.rank_svg}")
    print(f"Wrote {args.rank_png}")
    print(f"Wrote {args.rank_csv}")
    print(f"Wrote {args.rank_json}")


if __name__ == "__main__":
    main()
