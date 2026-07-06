import io
import time

from fastapi.testclient import TestClient
from PIL import Image as PILImage

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    return TestClient(create_app(engine=make_engine("sqlite://")))


def _png(color):
    b = io.BytesIO(); PILImage.new("RGB", (24, 24), color).save(b, format="PNG"); return b.getvalue()


def test_folder_import(tmp_path, monkeypatch):
    src = tmp_path / "src"; src.mkdir()
    for i in range(3):
        (src / f"{i}.png").write_bytes(_png((i * 10, 0, 0)))
    c = _client(tmp_path, monkeypatch)
    ds_id = c.post("/api/datasets", json={"name": "d", "task": "classification"}).json()["id"]
    job_id = c.post(f"/api/datasets/{ds_id}/import/folder", json={"path": str(src)}).json()["job_id"]
    for _ in range(300):
        job = c.app.state.manager.get(job_id)
        if job and job.status.value in ("done", "failed"):
            break
        time.sleep(0.01)
    assert job.status.value == "done"
    assert c.get(f"/api/datasets/{ds_id}/images").json()["total"] == 3


def test_webcam_import(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    ds_id = c.post("/api/datasets", json={"name": "w", "task": "detection"}).json()["id"]
    r = c.post(f"/api/datasets/{ds_id}/import/webcam",
               files={"file": ("frame.png", _png((0, 0, 200)), "image/png")})
    assert r.status_code == 200 and r.json()["source"] == "webcam"
    assert c.get(f"/api/datasets/{ds_id}/images").json()["total"] == 1
