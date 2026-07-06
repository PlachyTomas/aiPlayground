import json

from fastapi.testclient import TestClient
from sqlmodel import Session, select

import visionsuite_api.routes.labeling as labeling
from visionsuite_api.db import Dataset, make_engine
from visionsuite_api.main import create_app


class FakeGateway:
    url = "http://ls:8080"
    def __init__(self):
        self.created = None
        self.storage = None
    def create_project(self, title, label_config):
        self.created = (title, label_config)
        return 42
    def connect_local_storage(self, project_id, abs_path, regex):
        self.storage = (project_id, abs_path, regex)


def test_create_project(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    fake = FakeGateway()
    monkeypatch.setattr(labeling, "get_gateway", lambda request: fake)
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    ds_id = client.post("/api/datasets", json={"name": "d", "task": "detection"}).json()["id"]
    r = client.post(f"/api/datasets/{ds_id}/labeling/project", json={"class_names": ["car", "bus"]})
    assert r.status_code == 200 and r.json()["ls_project_id"] == 42
    assert "RectangleLabels" in fake.created[1] and fake.storage[0] == 42
    with Session(client.app.state.engine) as s:
        ds = s.exec(select(Dataset).where(Dataset.id == ds_id)).one()
    assert ds.ls_project_id == 42 and json.loads(ds.class_names) == ["car", "bus"]
