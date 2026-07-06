import io

from PIL import Image as PILImage

from visionsuite_core.ingest import IngestedImage, save_image_bytes


def _png_bytes(w=80, h=40, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_save_writes_image_and_thumb(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    rec = save_image_bytes(_png_bytes(), images, thumbs, source="folder")
    assert isinstance(rec, IngestedImage)
    assert (images / rec.filename).exists()
    assert (thumbs / f"{rec.image_id}.webp").exists()
    assert rec.width == 80 and rec.height == 40 and rec.source == "folder"


def test_dedup_same_bytes(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    b = _png_bytes()
    a1 = save_image_bytes(b, images, thumbs, source="folder")
    a2 = save_image_bytes(b, images, thumbs, source="folder")
    assert a1.image_id == a2.image_id
    assert len(list(images.iterdir())) == 1


def test_rejects_non_image(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    try:
        save_image_bytes(b"not an image", images, thumbs, source="folder")
        assert False, "expected an error"
    except Exception:
        pass
