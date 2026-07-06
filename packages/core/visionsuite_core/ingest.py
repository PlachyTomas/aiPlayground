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
