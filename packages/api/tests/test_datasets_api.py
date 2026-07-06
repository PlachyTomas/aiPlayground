from fastapi.testclient import TestClient
from sqlmodel import Session

from visionsuite_api.db import Image, make_engine
from visionsuite_api.main import create_app
from visionsuite_core import workspace


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    return TestClient(create_app(engine=make_engine("sqlite://")))


def test_create_and_list_dataset(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/datasets", json={"name": "cars", "task": "detection"})
    assert r.status_code == 200 and r.json()["image_count"] == 0
    ds_id = r.json()["id"]
    listing = c.get("/api/datasets").json()
    assert any(d["id"] == ds_id and d["name"] == "cars" for d in listing)


def test_image_list_serve_delete(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    ds_id = c.post("/api/datasets", json={"name": "d", "task": "classification"}).json()["id"]
    # seed one image on disk + a row, the way an import would
    d = workspace.dataset_dir(str(ds_id))
    (d / "images").mkdir(exist_ok=True); (d / "thumbs").mkdir(exist_ok=True)
    from PIL import Image as PILImage
    PILImage.new("RGB", (10, 10)).save(d / "images" / "x.png")
    PILImage.new("RGB", (10, 10)).save(d / "thumbs" / "x.webp", format="WEBP")
    root = workspace.workspace_root()
    with Session(c.app.state.engine) as s:
        s.add(Image(dataset_id=ds_id, image_id="x", filename="x.png", width=10, height=10,
                    source="folder",
                    path=str((d / "images" / "x.png").relative_to(root)),
                    thumb_path=str((d / "thumbs" / "x.webp").relative_to(root))))
        s.commit()
    imgs = c.get(f"/api/datasets/{ds_id}/images").json()
    assert imgs["total"] == 1 and imgs["images"][0]["image_id"] == "x"
    assert c.get(f"/api/datasets/{ds_id}/images/x/thumb").status_code == 200
    assert c.delete(f"/api/datasets/{ds_id}/images/x").json()["deleted"] is True
    assert c.get(f"/api/datasets/{ds_id}/images").json()["total"] == 0
