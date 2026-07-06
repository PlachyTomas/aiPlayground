import shutil
from uuid import uuid4

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, func, select

from visionsuite_core import workspace

from ..db import Dataset, Image
from ..imports import folder_producer, hf_producer, save_and_record, save_upload_tempfile, video_producer

router = APIRouter()


class CreateDataset(BaseModel):
    name: str
    task: str


def _count(session, ds_id: int) -> int:
    return session.exec(select(func.count()).select_from(Image).where(Image.dataset_id == ds_id)).one()


@router.post("/api/datasets")
def create_dataset(request: Request, body: CreateDataset) -> dict:
    with Session(request.app.state.engine) as s:
        ds = Dataset(name=body.name, task=body.task)
        s.add(ds); s.commit(); s.refresh(ds)
        return {"id": ds.id, "name": ds.name, "task": ds.task, "image_count": 0}


@router.get("/api/datasets")
def list_datasets(request: Request) -> list:
    with Session(request.app.state.engine) as s:
        out = []
        for ds in s.exec(select(Dataset)).all():
            out.append({"id": ds.id, "name": ds.name, "task": ds.task, "image_count": _count(s, ds.id)})
        return out


@router.delete("/api/datasets/{ds_id}")
def delete_dataset(request: Request, ds_id: int) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        for img in s.exec(select(Image).where(Image.dataset_id == ds_id)).all():
            s.delete(img)
        s.delete(ds); s.commit()
    shutil.rmtree(workspace.workspace_root() / "datasets" / str(ds_id), ignore_errors=True)
    return {"deleted": True}


@router.get("/api/datasets/{ds_id}/images")
def list_images(request: Request, ds_id: int, offset: int = 0, limit: int = 60) -> dict:
    with Session(request.app.state.engine) as s:
        total = _count(s, ds_id)
        rows = s.exec(select(Image).where(Image.dataset_id == ds_id).offset(offset).limit(limit)).all()
        images = [{
            "image_id": r.image_id, "filename": r.filename, "width": r.width, "height": r.height,
            "source": r.source,
            "thumb_url": f"/api/datasets/{ds_id}/images/{r.image_id}/thumb",
            "file_url": f"/api/datasets/{ds_id}/images/{r.image_id}/file",
        } for r in rows]
    return {"total": total, "images": images}


def _one_image(request: Request, ds_id: int, image_id: str) -> Image:
    with Session(request.app.state.engine) as s:
        row = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == image_id)).first()
    if row is None:
        raise HTTPException(404)
    return row


@router.get("/api/datasets/{ds_id}/images/{image_id}/thumb")
def image_thumb(request: Request, ds_id: int, image_id: str):
    return FileResponse(workspace.workspace_root() / _one_image(request, ds_id, image_id).thumb_path)


@router.get("/api/datasets/{ds_id}/images/{image_id}/file")
def image_file(request: Request, ds_id: int, image_id: str):
    return FileResponse(workspace.workspace_root() / _one_image(request, ds_id, image_id).path)


@router.delete("/api/datasets/{ds_id}/images/{image_id}")
def delete_image(request: Request, ds_id: int, image_id: str) -> dict:
    row = _one_image(request, ds_id, image_id)
    root = workspace.workspace_root()
    (root / row.path).unlink(missing_ok=True)
    (root / row.thumb_path).unlink(missing_ok=True)
    with Session(request.app.state.engine) as s:
        db_row = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == image_id)).first()
        if db_row:
            s.delete(db_row); s.commit()
    return {"deleted": True}


class FolderImport(BaseModel):
    path: str


@router.post("/api/datasets/{ds_id}/import/folder")
async def import_folder(request: Request, ds_id: int, body: FolderImport) -> dict:
    from pathlib import Path as _P
    if not _P(body.path).is_dir():
        raise HTTPException(400, f"not a folder: {body.path}")
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import", lambda: folder_producer(engine, ds_id, body.path))
    return {"job_id": job_id}


@router.post("/api/datasets/{ds_id}/import/webcam")
async def import_webcam(request: Request, ds_id: int, file: UploadFile) -> dict:
    data = await file.read()
    return save_and_record(request.app.state.engine, ds_id, data, source="webcam")


class HFImport(BaseModel):
    dataset_id: str
    split: str = "train"
    config: str | None = None
    image_column: str | None = None
    limit: int | None = 200


@router.post("/api/datasets/{ds_id}/import/hf")
async def import_hf(request: Request, ds_id: int, body: HFImport) -> dict:
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import",
        lambda: hf_producer(engine, ds_id, body.dataset_id, body.split, body.config,
                            body.image_column, body.limit))
    return {"job_id": job_id}


@router.post("/api/datasets/{ds_id}/import/video")
async def import_video(request: Request, ds_id: int, file: UploadFile, every_n: int = Form(30)) -> dict:
    data = await file.read()
    path = save_upload_tempfile(data, suffix=".mp4")
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import", lambda: video_producer(engine, ds_id, path, every_n))
    return {"job_id": job_id}
