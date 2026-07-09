"""既存の integrated JSON を、正確なキーフレーム時刻で180秒窓へ並べ直す。

なぜ必要か
----------
以前の ``generate_complete_scene.py`` は ``shot_idx // 30`` を scene_id にしていた。
しかしショットの長さは一定ではないため、30ショットは180秒とは限らない。

MovieNet の shot ファイルには、各ショットについて次が保存されている::

    start_frame end_frame keyframe_0 keyframe_1 keyframe_2

``keyframe_<img_idx> / fps`` で画像の実時間（秒）が分かる。この実時間を使い、
字幕と同じ「180秒窓・30秒オーバーラップ」へ visual_characters を入れ直す。

既存JSON内の字幕・Script由来テキストは、すでにSRT時刻で窓分けされているので
そのまま保持する。処理前のJSONは backup ディレクトリへ一度だけ保存する。

例::

    python src/movie/align_scenes_to_time.py \
      --shot-dir datasets/movienet/datas/shot \
      --video-info datasets/movienet/datas/movie1K.video_info.v1.json

すでに補正済みJSONを、バックアップから作り直す場合::

    python src/movie/align_scenes_to_time.py \
      --source-json-dir datasets/movienet/scene_outputs_30shot_backup \
      --json-dir datasets/movienet/scene_outputs \
      --shot-dir ... --video-info ...
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "datasets" / "movienet" / "datas"
ANNOTATION_DIR = DATA_DIR / "annotation"
DEFAULT_JSON_DIR = ROOT / "datasets" / "movienet" / "scene_outputs"
DEFAULT_BACKUP_DIR = ROOT / "datasets" / "movienet" / "scene_outputs_30shot_backup"
DEFAULT_TIMING_META_DIR = ROOT / "datasets" / "movienet" / "scene_timing_metadata"

WINDOW_SECONDS = 180.0
OVERLAP_SECONDS = 30.0
STEP_SECONDS = WINDOW_SECONDS - OVERLAP_SECONDS
TIMING_FORMAT_VERSION = 1


def window_ids_for_time(second: float) -> list[int]:
    """1つの時刻が含まれる全ウィンドウのscene_idを返す。

    30秒オーバーラップがあるので、150〜180秒などは2シーンに入る。
    scene_id は既存JSONに合わせて1始まり。
    """
    if not math.isfinite(second) or second < 0:
        return []
    last_index = int(math.floor(second / STEP_SECONDS))
    first_index = max(0, int(math.floor((second - WINDOW_SECONDS) / STEP_SECONDS)) + 1)
    return [
        index + 1
        for index in range(first_index, last_index + 1)
        if index * STEP_SECONDS <= second < index * STEP_SECONDS + WINDOW_SECONDS
    ]


def parse_shot_file(path: Path) -> list[list[int]]:
    """MovieNet shot txtを読み、各行を整数配列へする。"""
    shots: list[list[int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values = [int(value) for value in stripped.split()]
            except ValueError as error:
                raise ValueError(f"Invalid shot row: {path}:{line_number}") from error
            if len(values) < 2:
                raise ValueError(f"Shot row needs start/end frames: {path}:{line_number}")
            shots.append(values)
    if not shots:
        raise ValueError(f"No shots found: {path}")
    return shots


def keyframe_second(shots: list[list[int]], shot_idx: Any, img_idx: Any, fps: float) -> float:
    """shot_idx/img_idxが示すキーフレームの実時間を秒で返す。"""
    shot_index = int(shot_idx)
    image_index = int(img_idx)
    if shot_index < 0 or shot_index >= len(shots):
        raise IndexError(f"shot_idx out of range: {shot_index}/{len(shots)}")
    row = shots[shot_index]

    # row[2:5] が img_0, img_1, img_2 のフレーム番号。
    keyframe_column = 2 + image_index
    if 0 <= image_index <= 2 and keyframe_column < len(row):
        frame = row[keyframe_column]
    else:
        # データ欠損時だけ、ショット中央時刻へフォールバックする。
        frame = round((row[0] + row[1]) / 2)
    return frame / fps


def _extract_fps(entry: Any) -> float | None:
    """版によるキー名の差を吸収してfpsを取り出す。"""
    if isinstance(entry, (int, float)):
        value = float(entry)
        return value if value > 0 else None
    if not isinstance(entry, dict):
        return None
    for key in ("fps", "frame_rate", "framerate"):
        value = entry.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    for key in ("video", "video_info", "info"):
        nested = _extract_fps(entry.get(key))
        if nested is not None:
            return nested
    return None


def load_fps_map(path: Path) -> dict[str, float]:
    """MovieNet video_info JSONを ``movie_id -> fps`` に正規化する。"""
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    entries: list[tuple[str, Any]] = []
    if isinstance(raw, dict):
        # 通常版: {"tt...": {"fps": ...}, ...}
        entries.extend((str(key), value) for key, value in raw.items())
        for container_key in ("movies", "data", "video_info"):
            container = raw.get(container_key)
            if isinstance(container, list):
                for item in container:
                    if isinstance(item, dict):
                        movie_id = item.get("imdb_id") or item.get("movie_id") or item.get("id")
                        if movie_id:
                            entries.append((str(movie_id), item))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                movie_id = item.get("imdb_id") or item.get("movie_id") or item.get("id")
                if movie_id:
                    entries.append((str(movie_id), item))

    result: dict[str, float] = {}
    for movie_id, entry in entries:
        fps = _extract_fps(entry)
        if movie_id.startswith("tt") and fps is not None:
            result[movie_id] = fps
    return result


def build_shot_index(shot_dir: Path) -> dict[str, Path]:
    """展開方法が違っても見つけられるよう、shot txtを再帰的に索引化する。"""
    index: dict[str, Path] = {}
    for path in shot_dir.rglob("*.txt"):
        if path.stem.startswith("tt"):
            index.setdefault(path.stem, path)
    return index


def _new_scene(scene_id: int) -> dict[str, Any]:
    start = (scene_id - 1) * STEP_SECONDS
    end = start + WINDOW_SECONDS
    return {
        "scene_id": scene_id,
        "time_range": f"{int(start)}s - {int(end)}s",
        "start_second": start,
        "end_second": end,
        "situation_descriptions": [],
        "dialogs": [],
        "visual_characters": [],
    }


def rebuild_movie_scenes(
    old_scenes: list[dict[str, Any]],
    casts: list[dict[str, Any]],
    shots: list[list[int]],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """字幕テキストを保持し、人物だけを実時間で再配置する。"""
    movie_duration = shots[-1][1] / fps
    video_scene_count = max(
        1, int(math.floor(max(0.0, movie_duration - 1e-9) / STEP_SECONDS)) + 1
    )

    # MovieNetの動画版とSRT版は尺が少しずれる作品がある。映像の最終時刻だけで
    # 打ち切ると末尾字幕が消えるため、実際に字幕がある窓までは残す。
    # 旧「30 shots」処理だけが作った空の窓は、ここでは延長理由にしない。
    text_scene_ids: list[int] = []
    for old_scene in old_scenes:
        has_text = bool(old_scene.get("dialogs")) or any(
            value
            and value
            not in {
                "No dialogue / Situation missing",
                "No situation description matched from script.",
            }
            for value in (old_scene.get("situation_descriptions") or [])
        )
        try:
            old_scene_id = int(old_scene.get("scene_id"))
        except (TypeError, ValueError):
            continue
        if has_text and old_scene_id > 0:
            text_scene_ids.append(old_scene_id)

    total_scenes = max(video_scene_count, max(text_scene_ids, default=1))
    scenes = {scene_id: _new_scene(scene_id) for scene_id in range(1, total_scenes + 1)}

    # 既存のdialogs/説明は時刻窓へ正しく配置済み。人物だけを捨てて再構築する。
    for old_scene in old_scenes:
        try:
            scene_id = int(old_scene.get("scene_id"))
        except (TypeError, ValueError):
            continue
        if scene_id not in scenes:
            continue
        # 変更点: 台詞が0件でも説明文は存在し得る。以前の条件では、その説明文まで
        # 消えていたため、dialogsと説明文をそれぞれ独立して引き継ぐ。
        scenes[scene_id]["dialogs"] = list(old_scene.get("dialogs") or [])
        descriptions = [
            value
            for value in (old_scene.get("situation_descriptions") or [])
            if value and value != "No dialogue / Situation missing"
        ]
        scenes[scene_id]["situation_descriptions"] = descriptions

    errors: list[dict[str, Any]] = []
    for cast in casts:
        shot_idx = cast.get("shot_idx")
        img_idx = cast.get("img_idx")
        body = cast.get("body") or {}
        try:
            second = keyframe_second(shots, shot_idx, img_idx, fps)
        except (TypeError, ValueError, IndexError) as error:
            errors.append(
                {
                    "shot_idx": shot_idx,
                    "img_idx": img_idx,
                    "reason": str(error),
                }
            )
            continue

        character = {
            "shot_idx": int(shot_idx),
            "img_idx": int(img_idx),
            "pid": cast.get("pid"),
            "bbox": body.get("bbox"),
            "bbox_type": "body",
            "resolution": cast.get("resolution"),
            "keyframe_second": second,
        }
        for scene_id in window_ids_for_time(second):
            if scene_id in scenes:
                scenes[scene_id]["visual_characters"].append(character.copy())

    for scene in scenes.values():
        if not scene["situation_descriptions"]:
            scene["situation_descriptions"] = ["No situation description matched from script."]
    return list(scenes.values()), errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-dir", type=Path, required=True)
    parser.add_argument("--video-info", type=Path, required=True)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument(
        "--source-json-dir",
        type=Path,
        help="読み込み元JSON。省略時は--json-dirと同じ。バックアップからの再構築用。",
    )
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--timing-meta-dir", type=Path, default=DEFAULT_TIMING_META_DIR)
    parser.add_argument("--movie-id", action="append", help="複数指定可。省略時は全作品。")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_json_dir = args.source_json_dir or args.json_dir
    json_files = sorted(source_json_dir.glob("*_integrated.json"))
    if args.movie_id:
        requested = set(args.movie_id)
        json_files = [p for p in json_files if p.stem.removesuffix("_integrated") in requested]

    fps_map = load_fps_map(args.video_info)
    shot_index = build_shot_index(args.shot_dir)
    movie_ids = [path.stem.removesuffix("_integrated") for path in json_files]

    # 途中まで上書きしてから不足に気付く事故を防ぐため、最初に全作品を検証する。
    missing_shots = [movie_id for movie_id in movie_ids if movie_id not in shot_index]
    missing_fps = [movie_id for movie_id in movie_ids if movie_id not in fps_map]
    if missing_shots or missing_fps:
        raise RuntimeError(
            f"Timing data incomplete: missing shot files={len(missing_shots)} "
            f"{missing_shots[:10]}, missing fps={len(missing_fps)} {missing_fps[:10]}"
        )

    print("Movies:", len(json_files))
    print("Shot files:", len(shot_index))
    print("FPS entries:", len(fps_map))
    if args.dry_run:
        print("Dry run validation passed; no files changed.")
        return

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    args.timing_meta_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)
    for index, source_json_path in enumerate(json_files, start=1):
        movie_id = source_json_path.stem.removesuffix("_integrated")
        json_path = args.json_dir / source_json_path.name
        with source_json_path.open("r", encoding="utf-8") as file:
            old_scenes = json.load(file)
        annotation_path = ANNOTATION_DIR / f"{movie_id}.json"
        with annotation_path.open("r", encoding="utf-8") as file:
            casts = json.load(file).get("cast") or []
        shots = parse_shot_file(shot_index[movie_id])

        new_scenes, errors = rebuild_movie_scenes(old_scenes, casts, shots, fps_map[movie_id])

        backup_path = args.backup_dir / json_path.name
        if not backup_path.exists() and json_path.exists():
            shutil.copy2(json_path, backup_path)

        temporary_path = json_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(new_scenes, file, ensure_ascii=False, indent=2)
        temporary_path.replace(json_path)

        metadata = {
            "timing_format_version": TIMING_FORMAT_VERSION,
            "movie_id": movie_id,
            "source_json": str(source_json_path),
            "fps": fps_map[movie_id],
            "shot_file": str(shot_index[movie_id]),
            "window_seconds": WINDOW_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS,
            "scene_count": len(new_scenes),
            "cast_count": len(casts),
            "timing_errors": errors,
        }
        with (args.timing_meta_dir / f"{movie_id}.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"[{index}/{len(json_files)}] {movie_id}: scenes={len(new_scenes)}, errors={len(errors)}")


if __name__ == "__main__":
    main()
