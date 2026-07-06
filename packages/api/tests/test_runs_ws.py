import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def test_run_events_unknown_run_closes_4404():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/runs/does-not-exist/events") as ws:
            ws.receive_json()
    assert exc_info.value.code == 4404


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
