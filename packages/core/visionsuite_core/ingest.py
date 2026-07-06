from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

THUMB_MAX = 256
_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "bmp": "bmp"}


@dataclass
class IngestedImage:
    image_id: str
    filename: str
    width: int
    height: int
    source: str


def save_image_bytes(data: bytes, images_dir: Path, thumbs_dir: Path, source: str) -> IngestedImage:
    image_id = hashlib.sha1(data).hexdigest()[:16]
    img = PILImage.open(io.BytesIO(data))
    img.load()
    ext = _EXT.get((img.format or "").lower(), "png")
    filename = f"{image_id}.{ext}"
    full = images_dir / filename
    thumb = thumbs_dir / f"{image_id}.webp"
    width, height = img.size
    if not full.exists():
        full.write_bytes(data)
    if not thumb.exists():
        t = img.convert("RGB")
        t.thumbnail((THUMB_MAX, THUMB_MAX))
        t.save(thumb, format="WEBP")
    return IngestedImage(image_id=image_id, filename=filename, width=width, height=height, source=source)


from collections.abc import Iterable, Iterator

import imageio.v3 as _iio

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_folder_images(folder: Path) -> Iterator[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _encode_png(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def extract_video_frames(video_path: Path, every_n: int = 30) -> Iterator[bytes]:
    if every_n < 1:
        every_n = 1
    for i, frame in enumerate(_iio.imiter(video_path)):
        if i % every_n == 0:
            yield _encode_png(PILImage.fromarray(frame))


def hf_images_from_iterable(examples: Iterable[dict], image_column: str | None = None) -> Iterator[bytes]:
    col = image_column
    for ex in examples:
        if col is None:
            col = next((k for k, v in ex.items() if isinstance(v, PILImage.Image)), None)
            if col is None:
                raise ValueError("no PIL image column found in HF example")
        yield _encode_png(ex[col])


def iter_hf_images(dataset_id: str, split: str = "train", config: str | None = None,
                   image_column: str | None = None) -> Iterator[bytes]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
    yield from hf_images_from_iterable(ds, image_column=image_column)
