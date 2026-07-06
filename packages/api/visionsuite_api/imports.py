import os
import tempfile
from pathlib import Path
from pathlib import Path as _Path

from sqlmodel import Session, select

from visionsuite_core import workspace
from visionsuite_core.backends import RunEvent, RunStatus
from visionsuite_core.ingest import (
    extract_video_frames,
    iter_folder_images,
    iter_hf_images,
    save_image_bytes,
)

from .db import Image


def _dirs(ds_id: int):
    d = workspace.dataset_dir(str(ds_id))
    images, thumbs = d / "images", d / "thumbs"
    images.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    return images, thumbs


def save_and_record(engine, ds_id: int, data: bytes, source: str) -> dict:
    images, thumbs = _dirs(ds_id)
    rec = save_image_bytes(data, images, thumbs, source=source)
    root = workspace.workspace_root()
    with Session(engine) as s:
        exists = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == rec.image_id)).first()
        if exists is None:
            s.add(Image(
                dataset_id=ds_id, image_id=rec.image_id, filename=rec.filename,
                width=rec.width, height=rec.height, source=source,
                path=str((images / rec.filename).relative_to(root)),
                thumb_path=str((thumbs / f"{rec.image_id}.webp").relative_to(root)),
            ))
            s.commit()
    return {"image_id": rec.image_id, "filename": rec.filename, "width": rec.width,
            "height": rec.height, "source": source}


async def folder_producer(engine, ds_id: int, folder: str):
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    paths = list(iter_folder_images(Path(folder)))
    total = len(paths)
    yield RunEvent(type="log", message=f"importing {total} images from {folder}")
    for i, p in enumerate(paths, 1):
        save_and_record(engine, ds_id, p.read_bytes(), source="folder")
        yield RunEvent(type="progress", progress=i / total if total else 1.0)
    yield RunEvent(type="status", status=RunStatus.DONE)


async def hf_producer(engine, ds_id, dataset_id, split, config, image_column, limit):
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    yield RunEvent(type="log", message=f"streaming {dataset_id} [{split}]")
    n = 0
    for data in iter_hf_images(dataset_id, split=split, config=config, image_column=image_column):
        save_and_record(engine, ds_id, data, source="hf")
        n += 1
        yield RunEvent(type="progress", progress=(n / limit) if limit else None, message=f"{n} images")
        if limit and n >= limit:
            break
    yield RunEvent(type="status", status=RunStatus.DONE)


async def video_producer(engine, ds_id, video_path, every_n):
    try:
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        n = 0
        for data in extract_video_frames(_Path(video_path), every_n=every_n):
            save_and_record(engine, ds_id, data, source="video")
            n += 1
            yield RunEvent(type="progress", message=f"{n} frames")
        yield RunEvent(type="log", message=f"extracted {n} frames")
        yield RunEvent(type="status", status=RunStatus.DONE)
    finally:
        _Path(video_path).unlink(missing_ok=True)


def save_upload_tempfile(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    _Path(path).write_bytes(data)
    return path
