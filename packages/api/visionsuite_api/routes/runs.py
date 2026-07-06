from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session

from visionsuite_core.backends import RunSpec

from ..db import Run

router = APIRouter()


class CreateRunRequest(BaseModel):
    model_id: str = "dummy/echo"
    dataset_id: str = "dummy"


@router.post("/api/runs")
async def create_run(request: Request, body: CreateRunRequest) -> dict:
    run_id = uuid4().hex
    with Session(request.app.state.engine) as session:
        session.add(Run(id=run_id, model_id=body.model_id, dataset_id=body.dataset_id))
        session.commit()
    await request.app.state.manager.submit(
        RunSpec(run_id=run_id, model_id=body.model_id, dataset_id=body.dataset_id)
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("/api/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict:
    job = request.app.state.manager.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "status": job.status.value}
