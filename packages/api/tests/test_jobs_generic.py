from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.jobs import JobManager
from visionsuite_api.main import create_app
from visionsuite_core.backends import RunEvent, RunStatus


async def _producer():
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    yield RunEvent(type="progress", progress=1.0)
    yield RunEvent(type="status", status=RunStatus.DONE)


async def test_submit_stream_runs_any_producer():
    jm = JobManager()
    job = await jm.submit_stream("j1", "import", _producer)
    import asyncio
    for _ in range(200):
        if job.status == RunStatus.DONE:
            break
        await asyncio.sleep(0.01)
    assert job.status == RunStatus.DONE and job.kind == "import"


def test_ws_jobs_endpoint_streams():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    run_id = client.post("/api/runs", json={}).json()["run_id"]
    received = []
    with client.websocket_connect(f"/api/jobs/{run_id}/events") as ws:
        while True:
            try:
                received.append(ws.receive_json())
            except Exception:
                break
    assert any(e.get("status") == "done" for e in received)
