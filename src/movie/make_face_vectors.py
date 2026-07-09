"""MovieNet のキーフレームから、シーンごとの表情ベクトルを作る。

以前の実装との主な違い
----------------------
1. CLIP は使わない。
   CLIP は画像全体の意味を表すモデルであり、表情専用モデルではないため。
2. annotation の bbox は「顔」ではなく「全身」なので、その中から顔を再検出する。
3. annotation と 240P keyframe の解像度差を補正してから bbox を使う。
4. 未展開の ``<movie_id>.tar`` からも画像を直接読めるようにする。
5. 顔が見つからないシーンをゼロベクトルにしない。
   ``NaN`` と valid mask を保存し、「無表情」と「データなし」を区別する。

出力ベクトルは次の 10 次元で、映画と本の共通の感情軸として使える。

    [anger, contempt, disgust, fear, happiness,
     neutral, sadness, surprise, valence, arousal]

実行前に表情モデルをインストールすること::

    pip install -r src/movie/requirements-face.txt

例::

    python src/movie/make_face_vectors.py --movie-id tt0056869 --overwrite

注意:
    このファイルは既存の integrated JSON の scene_id を使用する。
    ``generate_complete_scene.py`` に残る「30 shot = 1 scene」は暫定ルールであり、
    正確な 3 分窓にするには MovieNet の shot boundary と fps が別途必要。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

JSON_DIR = ROOT / "datasets" / "movienet" / "scene_outputs"
ANNOTATION_DIR = ROOT / "datasets" / "movienet" / "datas" / "annotation"
KEYFRAMES_DIR = ROOT / "datasets" / "movienet" / "datas" / "keyframes_data"
OUTPUT_DIR = ROOT / "datasets" / "movienet" / "face_vectors"

# 変更点: CLIP の 512 次元ではなく、解釈可能な 8 感情 + Valence/Arousal を使う。
EMOTION_NAMES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]
FEATURE_NAMES = EMOTION_NAMES + ["valence", "arousal"]
FEATURE_DIM = len(FEATURE_NAMES)

# 8 感情と Valence/Arousal を同時に出す EmotiEffLib の軽量モデル。
DEFAULT_MODEL_NAME = "enet_b0_8_va_mtl"
OUTPUT_FORMAT_VERSION = 2
DEFAULT_BATCH_SIZE = 64
# MovieNet keyframe は高さ240pxなので、16px未満だけを除外する。
DEFAULT_MIN_FACE_SIZE = 16
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx?raw=true"
)
YUNET_MODEL_PATH = (
    ROOT
    / "datasets"
    / "movienet"
    / ".cache"
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


def _as_int(value: Any) -> int | None:
    """JSON の数値を安全に int へ変換する。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def keyframe_filename(shot_idx: Any, img_idx: Any, suffix: str = ".jpg") -> str:
    """MovieNet のキーフレーム名を作る。"""
    shot = _as_int(shot_idx)
    image = _as_int(img_idx)
    if shot is None or image is None:
        raise ValueError(f"Invalid shot/img index: shot={shot_idx}, img={img_idx}")
    return f"shot_{shot:04d}_img_{image}{suffix}"


class KeyframeStore:
    """展開済みディレクトリと tar の両方からキーフレームを読む。

    変更点:
        以前は ``keyframes_data/<movie_id>/...jpg`` だけを探していた。
        実データの多くは ``keyframes_data/<movie_id>.tar`` の中にあるため、
        tar を作品ごとに一度だけ開いて直接読み込む。
    """

    SUFFIXES = (".jpg", ".jpeg", ".png")

    def __init__(self, movie_id: str, base_dir: Path = KEYFRAMES_DIR) -> None:
        self.movie_id = movie_id
        self.base_dir = Path(base_dir)
        self.movie_dir = self.base_dir / movie_id
        self.tar_path = self.base_dir / f"{movie_id}.tar"
        self._tar: tarfile.TarFile | None = None
        self._tar_members: dict[str, tarfile.TarInfo] | None = None

    def __enter__(self) -> "KeyframeStore":
        if self.tar_path.is_file():
            self._tar = tarfile.open(self.tar_path, mode="r")
            # tar.getmember を何万回も線形探索させないため、最初に索引を作る。
            self._tar_members = {
                member.name.lstrip("./"): member
                for member in self._tar.getmembers()
                if member.isfile() and not Path(member.name).name.startswith("._")
            }
        return self

    def __exit__(self, *_: object) -> None:
        if self._tar is not None:
            self._tar.close()

    def load(self, shot_idx: Any, img_idx: Any) -> Image.Image | None:
        """該当画像を RGB の PIL Image として返す。なければ None。"""
        for suffix in self.SUFFIXES:
            filename = keyframe_filename(shot_idx, img_idx, suffix)

            # まず展開済み画像を使う。
            direct_path = self.movie_dir / filename
            if direct_path.is_file():
                try:
                    with Image.open(direct_path) as image:
                        return image.convert("RGB")
                except (OSError, ValueError):
                    continue

            # 展開されていなければ tar の中を探す。
            if self._tar is None or self._tar_members is None:
                continue
            for member_name in (f"{self.movie_id}/{filename}", filename):
                member = self._tar_members.get(member_name)
                if member is None:
                    continue
                extracted = self._tar.extractfile(member)
                if extracted is None:
                    continue
                try:
                    with Image.open(io.BytesIO(extracted.read())) as image:
                        return image.convert("RGB")
                except (OSError, ValueError):
                    continue

        return None


def _bbox_key(bbox: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return tuple(int(v) for v in bbox)
    except (TypeError, ValueError):
        return None


def build_resolution_index(movie_id: str) -> dict[tuple[Any, ...], tuple[int, int]]:
    """raw annotation から bbox の基準解像度を引ける索引を作る。

    integrated JSON の旧版には resolution が保存されていないため、raw annotation
    から補う。キーは厳密一致用とフォールバック用の2種類を登録する。
    """
    annotation_path = ANNOTATION_DIR / f"{movie_id}.json"
    if not annotation_path.is_file():
        return {}

    with annotation_path.open("r", encoding="utf-8") as file:
        annotation = json.load(file)

    index: dict[tuple[Any, ...], tuple[int, int]] = {}
    for cast in annotation.get("cast") or []:
        resolution = cast.get("resolution")
        bbox = _bbox_key((cast.get("body") or {}).get("bbox"))
        shot = _as_int(cast.get("shot_idx"))
        image = _as_int(cast.get("img_idx"))
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or shot is None
            or image is None
        ):
            continue
        try:
            width, height = int(resolution[0]), int(resolution[1])
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue

        normalized = (width, height)
        index[(shot, image)] = normalized
        if bbox is not None:
            index[(shot, image, cast.get("pid"), bbox)] = normalized
    return index


def find_source_resolution(
    item: dict[str, Any],
    resolution_index: dict[tuple[Any, ...], tuple[int, int]],
) -> tuple[int, int] | None:
    """visual_character が使っている bbox の基準解像度を取得する。"""
    resolution = item.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        try:
            width, height = int(resolution[0]), int(resolution[1])
            if width > 0 and height > 0:
                return width, height
        except (TypeError, ValueError):
            pass

    shot = _as_int(item.get("shot_idx"))
    image = _as_int(item.get("img_idx"))
    bbox = _bbox_key(item.get("bbox") or item.get("body_bbox"))
    exact_key = (shot, image, item.get("pid"), bbox)
    return resolution_index.get(exact_key) or resolution_index.get((shot, image))


def scale_bbox(
    bbox: Any,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """annotation 解像度の body bbox を、実際の 240P 画像座標へ変換する。

    変更点:
        以前は 1280x720 等の座標を 240P 画像にそのまま適用していたため、
        多くの bbox が画像外になっていた。
    """
    normalized_bbox = _bbox_key(bbox)
    if normalized_bbox is None:
        return None

    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        return None

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    x1, y1, x2, y2 = normalized_bbox
    x1 = max(0, min(round(x1 * scale_x), target_width - 1))
    y1 = max(0, min(round(y1 * scale_y), target_height - 1))
    x2 = max(0, min(round(x2 * scale_x), target_width))
    y2 = max(0, min(round(y2 * scale_y), target_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class BodyGuidedFaceDetector:
    """body bbox 内から OpenCV YuNet で顔を探す。

    MovieNet には顔 bbox がないため、全身 bbox を検索範囲として利用する。
    顔検出を画像全体へ何度もかけるより軽く、別人物の顔を拾いにくい。

    YuNet は OpenCV から直接使える軽量な顔検出器で、モデルは初回だけ
    ``datasets/movienet/.cache/models`` へダウンロードする。
    """

    def __init__(self, min_face_size: int = DEFAULT_MIN_FACE_SIZE) -> None:
        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                "This OpenCV build does not provide YuNet (FaceDetectorYN_create). "
                "Install the version in src/movie/requirements-face.txt."
            )
        if not YUNET_MODEL_PATH.is_file():
            YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = YUNET_MODEL_PATH.with_suffix(".download")
            print("Downloading YuNet face detector:", YUNET_MODEL_URL)
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, temporary_path)
                temporary_path.replace(YUNET_MODEL_PATH)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

        self.detector = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL_PATH),
            "",
            (320, 320),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=100,
        )
        self.min_face_size = min_face_size

    def detect(
        self,
        image: Image.Image,
        body_bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        image_array = np.asarray(image)
        image_height, image_width = image_array.shape[:2]
        x1, y1, x2, y2 = body_bbox

        # body bbox の端で顔が切れないように少しだけ外側へ広げる。
        pad_x = max(2, round((x2 - x1) * 0.08))
        pad_y = max(2, round((y2 - y1) * 0.05))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)

        body = image_array[y1:y2, x1:x2]
        if body.shape[0] < self.min_face_size or body.shape[1] < self.min_face_size:
            return None

        # YuNet は BGR 画像を受け取る。body crop のサイズを毎回明示する。
        body_bgr = cv2.cvtColor(body, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((body.shape[1], body.shape[0]))
        _, detections = self.detector.detect(body_bgr)
        if detections is None:
            return None

        # 1行は [x, y, w, h, 5 landmarks..., confidence]。
        faces = [
            detection
            for detection in detections
            if detection[2] >= self.min_face_size and detection[3] >= self.min_face_size
        ]
        if not faces:
            return None

        # 人間の顔は通常 body bbox の上側にあるため、面積と上側らしさで選ぶ。
        def face_score(face: Iterable[int]) -> float:
            _, face_y, face_width, face_height = (float(v) for v in list(face)[:4])
            vertical_bonus = 1.0 + max(0.0, 0.5 - face_y / max(1, body.shape[0]))
            return face_width * face_height * vertical_bonus

        face_x, face_y, face_width, face_height = (
            int(round(v)) for v in list(max(faces, key=face_score))[:4]
        )

        # 表情モデルへ渡す前に、額や顎が欠けないよう顔の周囲を15%広げる。
        margin_x = round(face_width * 0.15)
        margin_y = round(face_height * 0.15)
        local_x1 = max(0, face_x - margin_x)
        local_y1 = max(0, face_y - margin_y)
        local_x2 = min(body.shape[1], face_x + face_width + margin_x)
        local_y2 = min(body.shape[0], face_y + face_height + margin_y)
        face_image = body[local_y1:local_y2, local_x1:local_x2].copy()
        if face_image.size == 0:
            return None

        face_bbox = (
            x1 + local_x1,
            y1 + local_y1,
            x1 + local_x2,
            y1 + local_y2,
        )
        return face_image, face_bbox


def load_emotion_recognizer(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """EmotiEffLib の ONNX 表情モデルを読み込む。

    モデルは初回だけ ``datasets/movienet/.cache/.emotiefflib`` に保存する。
    ライブラリ既定の ``~/.emotiefflib`` はこのプロジェクト外になるため、モデル
    初期化中だけ HOME をプロジェクト内キャッシュへ切り替える。
    """
    try:
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer
    except ImportError as error:
        raise RuntimeError(
            "EmotiEffLib is not installed. Run: "
            "pip install -r src/movie/requirements-face.txt"
        ) from error

    cache_home = ROOT / "datasets" / "movienet" / ".cache"
    cache_home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(cache_home)
    try:
        # ONNX は CUDA がない環境でも扱いやすく、今回の軽量バッチ処理に向く。
        return EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device="cpu")
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home


def predict_expression_vectors(
    face_images: list[np.ndarray],
    recognizer: Any,
    batch_size: int,
) -> np.ndarray:
    """顔画像を 8 感情確率 + Valence/Arousal の10次元へ変換する。"""
    if not face_images:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    vectors: list[np.ndarray] = []
    for start in range(0, len(face_images), batch_size):
        batch = face_images[start : start + batch_size]
        # logits=False により、先頭8値を比較可能な確率へ変換する。
        _, scores = recognizer.predict_emotions(batch, logits=False)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                f"Unexpected EmotiEffLib output shape: {scores.shape}; "
                f"expected (*, {FEATURE_DIM})"
            )

        # 学習ラベルの範囲は -1..1。外れ値がDTW距離を壊さないよう制限する。
        scores[:, -2:] = np.clip(scores[:, -2:], -1.0, 1.0)
        vectors.append(scores)
    return np.concatenate(vectors, axis=0)


def _visual_character_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """同じキーフレーム上の同じ人物を識別するキーを返す。"""
    return (
        _as_int(item.get("shot_idx")),
        _as_int(item.get("img_idx")),
        item.get("pid"),
        _bbox_key(item.get("bbox") or item.get("body_bbox")),
    )


def _unique_visual_characters(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同じ人物bboxが重複して表情平均へ二重加算されるのを防ぐ。"""
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        key = _visual_character_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def make_scene_face_vectors(
    json_path: Path,
    recognizer: Any,
    face_detector: BodyGuidedFaceDetector,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """1作品を処理し、シーン単位の表情ベクトルと品質情報を保存する。"""
    movie_id = json_path.stem.removesuffix("_integrated")
    scene_source_sha256 = hashlib.sha256(json_path.read_bytes()).hexdigest()

    with json_path.open("r", encoding="utf-8") as file:
        scenes = json.load(file)

    resolution_index = build_resolution_index(movie_id)
    scene_vectors: list[np.ndarray] = []
    scene_std_vectors: list[np.ndarray] = []
    scene_ids: list[Any] = []
    valid_mask: list[bool] = []
    quality: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    # 変更点: 30秒の重複区間にある人物は隣接する2シーンへ入る。同じ顔を
    # シーンごとに再検出・再推論せず、映画内で一度だけ処理して共有する。
    scene_items: list[list[dict[str, Any]]] = []
    unique_items: dict[tuple[Any, ...], dict[str, Any]] = {}
    for scene in scenes:
        items = _unique_visual_characters(scene.get("visual_characters") or [])
        scene_items.append(items)
        for item in items:
            unique_items.setdefault(_visual_character_key(item), item)

    items_by_image: dict[tuple[int, int], list[tuple[tuple[Any, ...], dict[str, Any]]]] = {}
    failure_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, item in unique_items.items():
        shot_idx = _as_int(item.get("shot_idx"))
        img_idx = _as_int(item.get("img_idx"))
        bbox = item.get("bbox") or item.get("body_bbox")
        if shot_idx is None or img_idx is None or _bbox_key(bbox) is None:
            failure_by_key[key] = {"reason": "missing_shot_img_or_body_bbox"}
            continue
        items_by_image.setdefault((shot_idx, img_idx), []).append((key, item))

    vector_by_key: dict[tuple[Any, ...], np.ndarray] = {}
    pending_keys: list[tuple[Any, ...]] = []
    pending_faces: list[np.ndarray] = []

    def flush_face_batch() -> None:
        """検出済みの顔をまとめて推論し、人物キーへ対応付ける。"""
        if not pending_faces:
            return
        batch_vectors = predict_expression_vectors(pending_faces, recognizer, batch_size)
        vector_by_key.update(zip(pending_keys, batch_vectors))
        pending_keys.clear()
        pending_faces.clear()

    with KeyframeStore(movie_id) as keyframes:
        for (shot_idx, img_idx), grouped_items in tqdm(
            items_by_image.items(),
            desc=f"Processing {movie_id}",
            disable=os.environ.get("TQDM_DISABLE") == "1",
        ):
            # 同じキーフレームに複数人物がいても、画像の展開は1回だけ。
            image = keyframes.load(shot_idx, img_idx)
            for key, item in grouped_items:
                bbox = item.get("bbox") or item.get("body_bbox")
                if image is None:
                    failure_by_key[key] = {
                        "reason": "image_not_found_in_directory_or_tar"
                    }
                    continue

                source_size = find_source_resolution(item, resolution_index)
                if source_size is None:
                    failure_by_key[key] = {"reason": "annotation_resolution_not_found"}
                    continue

                scaled_body_bbox = scale_bbox(bbox, source_size, image.size)
                if scaled_body_bbox is None:
                    failure_by_key[key] = {"reason": "invalid_scaled_body_bbox"}
                    continue

                detected = face_detector.detect(image, scaled_body_bbox)
                if detected is None:
                    failure_by_key[key] = {"reason": "face_not_detected_in_body_bbox"}
                    continue

                face_image, _ = detected
                pending_keys.append(key)
                pending_faces.append(face_image)
                if len(pending_faces) >= batch_size:
                    flush_face_batch()

        flush_face_batch()

    # 共有済みの人物ベクトルを、各180秒窓へ集約する。
    for scene, items in zip(scenes, scene_items):
        scene_id = scene.get("scene_id")
        face_vectors = np.asarray(
            [
                vector_by_key[_visual_character_key(item)]
                for item in items
                if _visual_character_key(item) in vector_by_key
            ],
            dtype=np.float32,
        )

        for item in items:
            key = _visual_character_key(item)
            failure = failure_by_key.get(key)
            if failure is None:
                continue
            log = {
                "scene_id": scene_id,
                "shot_idx": _as_int(item.get("shot_idx")),
                "img_idx": _as_int(item.get("img_idx")),
                "pid": item.get("pid"),
                **failure,
            }
            logs.append({key: value for key, value in log.items() if value is not None})

        if len(face_vectors):
            scene_vector = face_vectors.mean(axis=0).astype(np.float32)
            scene_std = face_vectors.std(axis=0).astype(np.float32)
            is_valid = True
        else:
            # 変更点: 欠損を neutral やゼロと混同しないよう NaN にする。
            scene_vector = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
            scene_std = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
            is_valid = False

        scene_ids.append(scene_id)
        scene_vectors.append(scene_vector)
        scene_std_vectors.append(scene_std)
        valid_mask.append(is_valid)
        quality.append(
            {
                "scene_id": scene_id,
                "attempted_characters": len(items),
                "detected_faces": len(face_vectors),
                "detection_rate": len(face_vectors) / len(items) if items else 0.0,
                "valid": is_valid,
            }
        )

    vectors_array = np.stack(scene_vectors, axis=0)
    std_array = np.stack(scene_std_vectors, axis=0)
    valid_array = np.asarray(valid_mask, dtype=bool)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{movie_id}_face_vectors.npy", vectors_array)
    np.save(OUTPUT_DIR / f"{movie_id}_face_std_vectors.npy", std_array)
    np.save(OUTPUT_DIR / f"{movie_id}_face_valid_mask.npy", valid_array)

    with (OUTPUT_DIR / f"{movie_id}_scene_ids.json").open("w", encoding="utf-8") as file:
        json.dump(scene_ids, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / f"{movie_id}_face_log.json").open("w", encoding="utf-8") as file:
        json.dump(logs, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / f"{movie_id}_face_quality.json").open("w", encoding="utf-8") as file:
        json.dump(quality, file, ensure_ascii=False, indent=2)
    with (OUTPUT_DIR / f"{movie_id}_face_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "format_version": OUTPUT_FORMAT_VERSION,
                "model": model_name,
                "face_detector": "OpenCV YuNet 2023mar, body-bbox guided",
                "feature_names": FEATURE_NAMES,
                "shape": list(vectors_array.shape),
                "valid_scenes": int(valid_array.sum()),
                "total_scenes": len(valid_array),
                "missing_value": "NaN; use *_face_valid_mask.npy",
                "aggregation": "mean per scene; std saved separately",
                # 時刻補正後に integrated JSON が変わったら、再処理対象にするため保存。
                "scene_source_sha256": scene_source_sha256,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {movie_id}_face_vectors.npy")
    print("shape:", vectors_array.shape)
    print("valid scenes:", f"{valid_array.sum()}/{len(valid_array)}")
    print("log entries:", len(logs))


def output_is_current(json_path: Path) -> bool:
    """旧CLIP出力を誤って再利用せず、現形式だけをスキップ対象にする。"""
    movie_id = json_path.stem.removesuffix("_integrated")
    vector_path = OUTPUT_DIR / f"{movie_id}_face_vectors.npy"
    metadata_path = OUTPUT_DIR / f"{movie_id}_face_metadata.json"
    if not vector_path.is_file() or not metadata_path.is_file():
        return False
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        return (
            metadata.get("format_version") == OUTPUT_FORMAT_VERSION
            and metadata.get("feature_names") == FEATURE_NAMES
            and metadata.get("scene_source_sha256")
            == hashlib.sha256(json_path.read_bytes()).hexdigest()
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--movie-id",
        action="append",
        help="処理するIMDb ID。複数指定可。省略時は全 integrated JSON。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="現形式の出力が存在しても再生成する。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="テスト用に処理作品数を制限する。",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-face-size", type=int, default=DEFAULT_MIN_FACE_SIZE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(JSON_DIR.glob("*_integrated.json"))
    if args.movie_id:
        requested = set(args.movie_id)
        json_files = [
            path
            for path in json_files
            if path.stem.removesuffix("_integrated") in requested
        ]
    if args.limit is not None:
        json_files = json_files[: max(0, args.limit)]

    pending_files = []
    for json_path in json_files:
        movie_id = json_path.stem.removesuffix("_integrated")
        if not args.overwrite and output_is_current(json_path):
            print(f"Skip current output: {movie_id}")
            continue
        pending_files.append(json_path)

    print("JSON_DIR:", JSON_DIR)
    print("KEYFRAMES_DIR:", KEYFRAMES_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    print("Movies to process:", len(pending_files))
    if not pending_files:
        return

    recognizer = load_emotion_recognizer(args.model_name)
    face_detector = BodyGuidedFaceDetector(args.min_face_size)
    for json_path in pending_files:
        make_scene_face_vectors(
            json_path=json_path,
            recognizer=recognizer,
            face_detector=face_detector,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )


if __name__ == "__main__":
    main()
