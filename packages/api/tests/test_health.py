from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def test_health_ok():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
