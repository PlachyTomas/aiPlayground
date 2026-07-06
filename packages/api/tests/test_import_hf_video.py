import io
import time

import imageio.v3 as iio
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage

import visionsuite_api.imports as imports
from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    return TestClient(create_app(engine=make_engine("sqlite://")))


def _await_done(c, job_id):
    for _ in range(300):
        job = c.app.state.manager.get(job_id)
        if job and job.status.value in ("done", "failed"):
            return job
        time.sleep(0.01)
    return c.app.state.manager.get(job_id)


def test_hf_import_monkeypatched(tmp_path, monkeypatch):
    def fake_iter(dataset_id, split="train", config=None, image_column=None):
        for i in range(4):
            b = io.BytesIO(); PILImage.new("RGB", (12, 12), (i * 40, 0, 0)).save(b, format="PNG"); yield b.getvalue()
    monkeypatch.setattr(imports, "iter_hf_images", fake_iter)
    c = _client(tmp_path, monkeypatch)
    ds_id = c.post("/api/datasets", json={"name": "h", "task": "classification"}).json()["id"]
    job_id = c.post(f"/api/datasets/{ds_id}/import/hf", json={"dataset_id": "fake/ds", "limit": 3}).json()["job_id"]
    job = _await_done(c, job_id)
    assert job.status.value == "done"
    assert c.get(f"/api/datasets/{ds_id}/images").json()["total"] == 3  # limit respected


def test_video_import(tmp_path, monkeypatch):
    vid = tmp_path / "clip.mp4"
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in (10, 40, 70, 100)]
    iio.imwrite(vid, np.stack(frames), fps=4, codec="libx264")
    c = _client(tmp_path, monkeypatch)
    ds_id = c.post("/api/datasets", json={"name": "v", "task": "detection"}).json()["id"]
    r = c.post(f"/api/datasets/{ds_id}/import/video",
               files={"file": ("clip.mp4", vid.read_bytes(), "video/mp4")},
               data={"every_n": "1"})
    job = _await_done(c, r.json()["job_id"])
    assert job.status.value == "done"
    assert c.get(f"/api/datasets/{ds_id}/images").json()["total"] >= 1
