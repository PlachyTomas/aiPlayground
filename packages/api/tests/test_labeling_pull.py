import json
import time

from fastapi.testclient import TestClient
from sqlmodel import Session, select

import visionsuite_api.routes.labeling as labeling
from visionsuite_api.db import Annotation, make_engine
from visionsuite_api.main import create_app


class FakeGateway:
    url = "http://ls:8080"
    def project_stats(self, pid):
        return {"total": 2, "annotated": 1}
    def export_json(self, pid):
        return [{
            "data": {"image": "/data/local-files/?d=datasets/1/images/img1.png"},
            "annotations": [{"result": [{
                "type": "rectanglelabels", "original_width": 100, "original_height": 100,
                "image_rotation": 0,
                "value": {"x": 0, "y": 0, "width": 50, "height": 50, "rotation": 0,
                          "rectanglelabels": ["car"]}}]}],
        }]


def _seeded_client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setattr(labeling, "get_gateway", lambda request: FakeGateway())
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    ds_id = client.post("/api/datasets", json={"name": "d", "task": "detection"}).json()["id"]
    # mark it configured with a class list
    from visionsuite_api.db import Dataset
    with Session(client.app.state.engine) as s:
        ds = s.get(Dataset, ds_id); ds.ls_project_id = 1; ds.class_names = json.dumps(["car"]); s.add(ds); s.commit()
    return client, ds_id


def test_status_reports_stats(tmp_path, monkeypatch):
    client, ds_id = _seeded_client(tmp_path, monkeypatch)
    st = client.get(f"/api/datasets/{ds_id}/labeling/status").json()
    assert st["configured"] is True and st["total"] == 2 and st["annotated"] == 1


def test_pull_stores_annotations(tmp_path, monkeypatch):
    client, ds_id = _seeded_client(tmp_path, monkeypatch)
    # Keep the client's event loop alive so the background pull's threadpool export can resolve.
    with client:
        job_id = client.post(f"/api/datasets/{ds_id}/labeling/pull").json()["job_id"]
        for _ in range(300):
            job = client.app.state.manager.get(job_id)
            if job and job.status.value in ("done", "failed"):
                break
            time.sleep(0.01)
        assert job.status.value == "done"
        with Session(client.app.state.engine) as s:
            rows = s.exec(select(Annotation)).all()
    assert len(rows) == 1 and rows[0].image_id == "img1" and rows[0].n_objects == 1
    assert json.loads(rows[0].coco_json)[0]["category_id"] == 0
