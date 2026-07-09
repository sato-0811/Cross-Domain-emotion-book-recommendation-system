import tempfile
import unittest
from pathlib import Path

from src.movie.align_scenes_to_time import (
    keyframe_second,
    load_fps_map,
    parse_shot_file,
    rebuild_movie_scenes,
    window_ids_for_time,
)


class WindowAssignmentTest(unittest.TestCase):
    def test_overlap_assigns_to_both_windows(self):
        self.assertEqual(window_ids_for_time(149.9), [1])
        self.assertEqual(window_ids_for_time(150.0), [1, 2])
        self.assertEqual(window_ids_for_time(179.9), [1, 2])
        self.assertEqual(window_ids_for_time(180.0), [2])
        self.assertEqual(window_ids_for_time(300.0), [2, 3])


class ShotTimingTest(unittest.TestCase):
    def test_uses_img_index_keyframe(self):
        shots = [[0, 99, 10, 50, 90], [100, 199, 110, 150, 190]]
        self.assertEqual(keyframe_second(shots, 1, 2, 25.0), 7.6)

    def test_parses_official_five_column_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tt0000001.txt"
            path.write_text("0 99 10 50 90\n100 199 110 150 190\n")
            self.assertEqual(parse_shot_file(path)[1], [100, 199, 110, 150, 190])


class RebuildTest(unittest.TestCase):
    def test_preserves_dialog_and_places_cast_by_real_time(self):
        # 25fpsで180秒を超える2ショットを用意。
        shots = [[0, 4499, 10, 2250, 4490], [4500, 8999, 4510, 6750, 8990]]
        old = [
            {
                "scene_id": 1,
                "dialogs": ["hello"],
                "situation_descriptions": ["A person speaks."],
                "visual_characters": [{"wrong": "old 30-shot assignment"}],
            }
        ]
        casts = [
            {
                "shot_idx": 1,
                "img_idx": 1,
                "pid": "nm1",
                "resolution": [1280, 720],
                "body": {"bbox": [0, 0, 100, 200]},
            }
        ]
        scenes, errors = rebuild_movie_scenes(old, casts, shots, fps=25.0)
        self.assertEqual(errors, [])
        self.assertEqual(scenes[0]["dialogs"], ["hello"])
        self.assertEqual(scenes[0]["visual_characters"], [])
        self.assertEqual(scenes[1]["visual_characters"][0]["keyframe_second"], 270.0)

    def test_preserves_description_when_scene_has_no_dialog(self):
        """台詞がない窓でも、Script由来の説明文だけは消してはいけない。"""
        shots = [[0, 4499, 10, 2250, 4490]]
        old = [
            {
                "scene_id": 1,
                "dialogs": [],
                "situation_descriptions": ["A silent chase continues."],
                "visual_characters": [],
            }
        ]

        scenes, errors = rebuild_movie_scenes(old, [], shots, fps=25.0)

        self.assertEqual(errors, [])
        self.assertEqual(
            scenes[0]["situation_descriptions"], ["A silent chase continues."]
        )

    def test_keeps_subtitle_window_past_video_duration(self):
        """動画版とSRT版の尺がずれても、末尾字幕を捨てない。"""
        shots = [[0, 4499, 10, 2250, 4490]]  # 25fps = 約180秒
        old = [
            {
                "scene_id": 3,
                "dialogs": ["A subtitle from a slightly longer release."],
                "situation_descriptions": [],
                "visual_characters": [],
            }
        ]

        scenes, errors = rebuild_movie_scenes(old, [], shots, fps=25.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[2]["dialogs"], old[0]["dialogs"])


class VideoInfoTest(unittest.TestCase):
    def test_loads_mapping_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video_info.json"
            path.write_text('{"tt0000001": {"fps": 23.976}}')
            self.assertEqual(load_fps_map(path), {"tt0000001": 23.976})


if __name__ == "__main__":
    unittest.main()
