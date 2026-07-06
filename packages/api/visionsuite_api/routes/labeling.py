from fastapi import APIRouter, Request

from ..labelstudio import get_gateway

router = APIRouter()


@router.get("/api/labelstudio/status")
async def labelstudio_status(request: Request) -> dict:
    return get_gateway(request).status()


import json

from fastapi import HTTPException
from pydantic import BaseModel
from sqlmodel import Session

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
    project_id = gateway.create_project(f"visionsuite-{ds_id}", config)
    images_dir = workspace.dataset_dir(str(ds_id)) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    gateway.connect_local_storage(project_id, str(images_dir), r".*\.(jpg|jpeg|png|webp|bmp)$")
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        ds.ls_project_id = project_id
        ds.class_names = json.dumps(body.class_names)
        s.add(ds); s.commit()
    return {"ls_project_id": project_id, "ls_url": f"{gateway.url}/projects/{project_id}/data"}
