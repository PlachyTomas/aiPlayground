from fastapi.testclient import TestClient

import visionsuite_api.routes.labeling as labeling
from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


class FakeGateway:
    def __init__(self, url="http://ls:8080"):
        self.url = url
    def status(self):
        return {"connected": True, "url": self.url}


def test_status_uses_gateway(monkeypatch):
    monkeypatch.setattr(labeling, "get_gateway", lambda request: FakeGateway())
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    r = client.get("/api/labelstudio/status")
    assert r.status_code == 200 and r.json()["connected"] is True
