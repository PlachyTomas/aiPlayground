from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from ..labelstudio import get_gateway

router = APIRouter()


@router.get("/api/labelstudio/status")
async def labelstudio_status(request: Request) -> dict:
    return await run_in_threadpool(get_gateway(request).status)


import json

from fastapi import HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from visionsuite_core import workspace
from visionsuite_core.labelstudio_convert import ls_config_for

from ..db import Dataset


class CreateLabelingProject(BaseModel):
    class_names: list[str]


@router.post("/api/datasets/{ds_id}/labeling/project")
async def create_labeling_project(request: Request, ds_id: int, body: CreateLabelingProject) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        task = ds.task
    gateway = get_gateway(request)
    config = ls_config_for(task, body.class_names)
    project_id = await run_in_threadpool(gateway.create_project, f"visionsuite-{ds_id}", config)
    images_dir = workspace.dataset_dir(str(ds_id)) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    await run_in_threadpool(
        gateway.connect_local_storage, project_id, str(images_dir), r".*\.(jpg|jpeg|png|webp|bmp)$")
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        ds.ls_project_id = project_id
        ds.class_names = json.dumps(body.class_names)
        s.add(ds); s.commit()
    return {"ls_project_id": project_id, "ls_url": f"{gateway.url}/projects/{project_id}/data"}


from uuid import uuid4

from visionsuite_core.backends import RunEvent, RunStatus
from visionsuite_core.labelstudio_convert import ls_json_to_coco

from ..db import Annotation


@router.get("/api/datasets/{ds_id}/labeling/status")
async def labeling_status(request: Request, ds_id: int) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        pid = ds.ls_project_id
    if pid is None:
        return {"configured": False}
    gateway = get_gateway(request)
    stats = await run_in_threadpool(gateway.project_stats, pid)
    return {"configured": True, "ls_project_id": pid, "total": stats["total"],
            "annotated": stats["annotated"], "ls_url": f"{gateway.url}/projects/{pid}/data"}


def _pull_producer(engine, ds_id, gateway):
    async def producer():
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        with Session(engine) as s:
            ds = s.get(Dataset, ds_id)
            pid = ds.ls_project_id
            class_names = json.loads(ds.class_names)
        yield RunEvent(type="log", message=f"exporting project {pid}")
        tasks = await run_in_threadpool(gateway.export_json, pid)
        coco = ls_json_to_coco(tasks, class_names)
        with Session(engine) as s:
            for old in s.exec(select(Annotation).where(Annotation.dataset_id == ds_id)).all():
                s.delete(old)
            s.commit()
            for i, img in enumerate(coco["images"], 1):
                if img["classification"] is not None:
                    payload = [{"category_id": img["classification"]}]
                else:
                    payload = img["annotations"]
                s.add(Annotation(dataset_id=ds_id, image_id=img["image_id"],
                                 coco_json=json.dumps(payload), n_objects=len(payload)))
                yield RunEvent(type="progress", progress=i / len(coco["images"]))
            s.commit()
        yield RunEvent(type="log", message=f"stored {len(coco['images'])} annotated images")
        yield RunEvent(type="status", status=RunStatus.DONE)
    return producer


@router.post("/api/datasets/{ds_id}/labeling/pull")
async def labeling_pull(request: Request, ds_id: int) -> dict:
    gateway = get_gateway(request)
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(job_id, "pull", _pull_producer(engine, ds_id, gateway))
    return {"job_id": job_id}
