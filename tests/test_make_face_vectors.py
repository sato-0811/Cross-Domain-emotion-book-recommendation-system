import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.movie.make_face_vectors import KeyframeStore, scale_bbox  # noqa: E402


class ScaleBboxTest(unittest.TestCase):
    def test_scales_annotation_bbox_to_240p_image(self):
        # 1280x690 -> 445x240 の実データと同じ縮尺条件。
        result = scale_bbox([935, 381, 1056, 680], (1280, 690), (445, 240))
        self.assertEqual(result, (325, 133, 367, 237))

    def test_rejects_invalid_bbox(self):
        self.assertIsNone(scale_bbox([10, 10, 5, 5], (100, 100), (50, 50)))


class KeyframeStoreTest(unittest.TestCase):
    @staticmethod
    def _jpeg_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), color=(255, 0, 0)).save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_loads_unextracted_image_from_tar(self):
        movie_id = "tt0000001"
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            image_bytes = self._jpeg_bytes()
            member_name = f"{movie_id}/shot_0001_img_2.jpg"
            with tarfile.open(base_dir / f"{movie_id}.tar", mode="w") as archive:
                info = tarfile.TarInfo(member_name)
                info.size = len(image_bytes)
                archive.addfile(info, io.BytesIO(image_bytes))

            with KeyframeStore(movie_id, base_dir=base_dir) as store:
                image = store.load(1, 2)

            self.assertIsNotNone(image)
            self.assertEqual(image.size, (32, 24))


if __name__ == "__main__":
    unittest.main()
