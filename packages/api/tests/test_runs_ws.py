from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def test_run_events_stream_to_done():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    run_id = client.post("/api/runs", json={}).json()["run_id"]

    received = []
    with client.websocket_connect(f"/api/runs/{run_id}/events") as ws:
        while True:
            try:
                received.append(ws.receive_json())
            except Exception:
                break

    assert any(e.get("status") == "done" for e in received)
    assert any(e["type"] == "progress" and e["progress"] == 1.0 for e in received)
