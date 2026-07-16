"""Create a grouped bar chart for Siamese model comparison metrics.

SVG output uses only the Python standard library. PNG output is optional and
requires Pillow.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = [
    ROOT / "datasets" / "siamese_runs" / "siamese_metrics.json",
    ROOT / "datasets" / "lstm_siamese_runs" / "lstm_siamese_metrics.json",
    ROOT / "datasets" / "softmax_siamese_runs" / "softmax_siamese_metrics.json",
]
DEFAULT_OUTPUT = ROOT / "figures" / "model_comparison_all_books_best_epoch.svg"
DEFAULT_PNG_OUTPUT = ROOT / "figures" / "model_comparison_all_books_best_epoch.png"

RUN_LABELS = {
    "siamese_runs": "Pairwise Siamese",
    "lstm_siamese_runs": "LSTM Siamese",
    "softmax_siamese_runs": "Softmax Siamese",
}

METRIC_SPECS = [
    ("Recall@1", "test_recall_at_1", "#4E79A7"),
    ("Recall@5", "test_recall_at_5", "#59A14F"),
    ("MRR", "test_mrr", "#F28E2B"),
]


@dataclass(frozen=True)
class ModelMetrics:
    label: str
    best_epoch: float
    val_mrr: float
    candidate_count: int
    values: dict[str, float]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def find_best_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_epoch_metrics")
    if isinstance(best, dict) and best:
        return best

    final = payload.get("metrics") or {}
    best_epoch = final.get("best_epoch")
    for row in payload.get("history") or []:
        if row.get("epoch") == float(best_epoch):
            return row
    raise ValueError("Could not find best epoch metrics")


def get_candidate_count(payload: dict[str, Any]) -> int:
    for section_name in ("config", "metrics"):
        section = payload.get(section_name) or {}
        value = section.get("candidate_book_count")
        if value is not None:
            return int(value)
    raise ValueError("Could not find candidate_book_count")


def collect_metrics(paths: list[Path]) -> list[ModelMetrics]:
    rows: list[ModelMetrics] = []
    for path in paths:
        payload = load_json(path)
        best = find_best_metrics(payload)
        values = {key: float(best[key]) for _, key, _ in METRIC_SPECS}
        rows.append(
            ModelMetrics(
                label=RUN_LABELS.get(path.parent.name, path.parent.name or path.stem),
                best_epoch=float(best["epoch"]),
                val_mrr=float(best["val_mrr"]),
                candidate_count=get_candidate_count(payload),
                values=values,
            )
        )
    return rows


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 12,
    weight: int | str = 400,
    anchor: str = "middle",
    fill: str = "#202124",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" {extra}>{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"/>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str, extra: str = "") -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" {extra}/>'
    )


def render_svg(rows: list[ModelMetrics]) -> str:
    if not rows:
        raise ValueError("No metrics to plot")

    width = 920
    height = 540
    left = 82
    right = 36
    top = 94
    chart_height = 332
    chart_width = width - left - right
    bottom = top + chart_height
    group_width = chart_width / len(rows)
    bar_width = 52
    bar_gap = 14
    group_bar_width = len(METRIC_SPECS) * bar_width + (len(METRIC_SPECS) - 1) * bar_gap
    max_score = 1.0
    candidate_counts = sorted({row.candidate_count for row in rows})
    candidate_label = str(candidate_counts[0]) if len(candidate_counts) == 1 else "mixed"

    def y_for(value: float) -> float:
        return bottom - (value / max_score) * chart_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "#FFFFFF"),
        text(width / 2, 36, "Siamese model performance over 406 candidate books", size=26, weight=700),
        text(
            width / 2,
            66,
            f"Best validation epoch, test split, candidate_pool=all-books, candidates={candidate_label}",
            size=16,
            fill="#5F6368",
        ),
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = y_for(value)
        parts.append(line(left, y, left + chart_width, y, "#E0E0E0", 1.0))
        parts.append(text(left - 14, y + 5, f"{value:.1f}", size=14, anchor="end", fill="#5F6368"))

    parts.extend(
        [
            line(left, top, left, bottom, "#202124", 1.2),
            line(left, bottom, left + chart_width, bottom, "#202124", 1.2),
            text(26, top + chart_height / 2, "Score", size=17, fill="#3C4043", extra=f'transform="rotate(-90 26 {top + chart_height / 2:.1f})"'),
        ]
    )

    for index, row in enumerate(rows):
        group_left = left + index * group_width + (group_width - group_bar_width) / 2
        center = left + index * group_width + group_width / 2
        for metric_index, (metric_label, metric_key, color) in enumerate(METRIC_SPECS):
            value = row.values[metric_key]
            bar_x = group_left + metric_index * (bar_width + bar_gap)
            bar_y = y_for(value)
            parts.append(rect(bar_x, bar_y, bar_width, bottom - bar_y, color))
            parts.append(text(bar_x + bar_width / 2, bar_y + 20, f"{value:.3f}", size=15, weight=700, fill="#FFFFFF"))
        parts.append(text(center, bottom + 33, row.label, size=16, weight=700))
        parts.append(text(center, bottom + 57, f"best epoch {row.best_epoch:g}, val MRR {row.val_mrr:.3f}", size=14, fill="#5F6368"))

    legend_y = height - 36
    legend_start = width / 2 - 150
    for index, (metric_label, _, color) in enumerate(METRIC_SPECS):
        x = legend_start + index * 122
        parts.append(rect(x, legend_y - 11, 15, 15, color))
        parts.append(text(x + 23, legend_y + 2, metric_label, size=15, anchor="start"))

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


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


def render_png(rows: list[ModelMetrics], output: Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("PNG output requires Pillow. Install pillow or omit --png-output.") from exc

    if not rows:
        raise ValueError("No metrics to plot")

    width = 1840
    height = 1080
    scale = 2
    left = 82 * scale
    top = 94 * scale
    chart_height = 332 * scale
    chart_width = (920 - 82 - 36) * scale
    bottom = top + chart_height
    group_width = chart_width / len(rows)
    bar_width = 52 * scale
    bar_gap = 14 * scale
    group_bar_width = len(METRIC_SPECS) * bar_width + (len(METRIC_SPECS) - 1) * bar_gap
    candidate_counts = sorted({row.candidate_count for row in rows})
    candidate_label = str(candidate_counts[0]) if len(candidate_counts) == 1 else "mixed"

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = load_font(52, bold=True)
    font_subtitle = load_font(32)
    font_axis = load_font(30)
    font_value = load_font(30, bold=True)
    font_model = load_font(32, bold=True)
    font_note = load_font(28)
    font_legend = load_font(30)

    def y_for(value: float) -> float:
        return bottom - value * chart_height

    draw.text((width / 2, 72), "Siamese model performance over 406 candidate books", fill=rgb("#202124"), font=font_title, anchor="mm")
    draw.text(
        (width / 2, 132),
        f"Best validation epoch, test split, candidate_pool=all-books, candidates={candidate_label}",
        fill=rgb("#5F6368"),
        font=font_subtitle,
        anchor="mm",
    )

    for tick in range(0, 6):
        value = tick / 5
        y = y_for(value)
        draw.line((left, y, left + chart_width, y), fill=rgb("#E0E0E0"), width=2)
        draw.text((left - 28, y), f"{value:.1f}", fill=rgb("#5F6368"), font=font_axis, anchor="rm")

    draw.line((left, top, left, bottom), fill=rgb("#202124"), width=3)
    draw.line((left, bottom, left + chart_width, bottom), fill=rgb("#202124"), width=3)

    label_bbox = draw.textbbox((0, 0), "Score", font=font_axis)
    label_width = label_bbox[2] - label_bbox[0]
    label_height = label_bbox[3] - label_bbox[1]
    label_image = Image.new("RGBA", (label_width + 8, label_height + 8), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((4 - label_bbox[0], 4 - label_bbox[1]), "Score", fill=rgb("#3C4043"), font=font_axis)
    rotated_label = label_image.rotate(90, expand=True)
    image.paste(
        rotated_label,
        (48 - rotated_label.width // 2, int(top + chart_height / 2 - rotated_label.height / 2)),
        rotated_label,
    )

    for index, row in enumerate(rows):
        group_left = left + index * group_width + (group_width - group_bar_width) / 2
        center = left + index * group_width + group_width / 2
        for metric_index, (_, metric_key, color) in enumerate(METRIC_SPECS):
            value = row.values[metric_key]
            bar_x = group_left + metric_index * (bar_width + bar_gap)
            bar_y = y_for(value)
            draw.rectangle((bar_x, bar_y, bar_x + bar_width, bottom), fill=rgb(color))
            draw.text((bar_x + bar_width / 2, bar_y + 34), f"{value:.3f}", fill="white", font=font_value, anchor="mm")
        draw.text((center, bottom + 64), row.label, fill=rgb("#202124"), font=font_model, anchor="mm")
        draw.text((center, bottom + 110), f"best epoch {row.best_epoch:g}, val MRR {row.val_mrr:.3f}", fill=rgb("#5F6368"), font=font_note, anchor="mm")

    legend_y = height - 70
    legend_start = width / 2 - 300
    for index, (metric_label, _, color) in enumerate(METRIC_SPECS):
        x = legend_start + index * 244
        draw.rectangle((x, legend_y - 22, x + 30, legend_y + 8), fill=rgb(color))
        draw.text((x + 46, legend_y - 6), metric_label, fill=rgb("#202124"), font=font_legend, anchor="lm")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        nargs="*",
        type=Path,
        default=DEFAULT_METRICS,
        help="Metrics JSON files to plot. Defaults to the three Siamese runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"SVG output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=None,
        help=f"Optional PNG output path. Example: {DEFAULT_PNG_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_metrics(args.metrics)
    svg = render_svg(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")
    if args.png_output:
        render_png(rows, args.png_output)
        print(f"Wrote {args.png_output}")


if __name__ == "__main__":
    main()
