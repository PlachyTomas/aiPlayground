import time

from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app(engine=make_engine("sqlite://")))


def test_create_run_returns_id():
    client = _client()
    r = client.post("/api/runs", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending" and len(body["run_id"]) > 0


def test_get_run_reaches_done():
    client = _client()
    run_id = client.post("/api/runs", json={}).json()["run_id"]
    for _ in range(200):
        status = client.get(f"/api/runs/{run_id}").json()["status"]
        if status == "done":
            break
        time.sleep(0.01)
    assert status == "done"


def test_get_unknown_run_404():
    assert _client().get("/api/runs/nope").status_code == 404
