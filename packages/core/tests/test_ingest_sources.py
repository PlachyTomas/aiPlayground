import io

import imageio.v3 as iio
import numpy as np
from PIL import Image as PILImage

from visionsuite_core.ingest import (
    extract_video_frames,
    hf_images_from_iterable,
    iter_folder_images,
)


def _write_png(p, color=(0, 128, 0)):
    PILImage.new("RGB", (32, 24), color).save(p, format="PNG")


def test_iter_folder_finds_images(tmp_path):
    _write_png(tmp_path / "a.png")
    (tmp_path / "sub").mkdir()
    _write_png(tmp_path / "sub" / "b.JPG")  # note case + jpg
    (tmp_path / "notes.txt").write_text("ignore me")
    found = sorted(p.name.lower() for p in iter_folder_images(tmp_path))
    assert found == ["a.png", "b.jpg"]


def test_extract_video_frames(tmp_path):
    vid = tmp_path / "clip.mp4"
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in (10, 20, 30, 40, 50, 60)]
    iio.imwrite(vid, np.stack(frames), fps=6, codec="libx264")
    out = list(extract_video_frames(vid, every_n=2))
    assert 1 <= len(out) <= 6
    PILImage.open(io.BytesIO(out[0])).load()  # decodes


def test_hf_images_autodetect():
    ds = [{"image": PILImage.new("RGB", (8, 8)), "label": 1},
          {"image": PILImage.new("RGB", (8, 8)), "label": 0}]
    out = list(hf_images_from_iterable(ds))
    assert len(out) == 2
    PILImage.open(io.BytesIO(out[0])).load()
