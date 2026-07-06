# VisionSuite Sub-Project 2: Annotation (Label Studio) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Label SP1's images in Label Studio (boxes for detection, classes for classification), track progress in VisionSuite, and pull completed annotations back into a COCO-style DB store — via a pure, fully-tested LS-JSON→COCO converter with all SDK calls isolated behind a gateway.

**Architecture:** Pure `visionsuite_core.labelstudio_convert` generates the LS label config and converts LS export JSON → COCO. `visionsuite_api.labelstudio.LabelStudioGateway` is the only place the SDK is touched (tests use a fake). `routes/labeling.py` creates projects, syncs images via LS Local Storage, and pulls annotations as a background job into an `Annotation` table. The React **Labeling** page orchestrates and links out to LS's own UI.

**Tech Stack:** Python 3.11+, `label-studio-sdk>=2,<3`, FastAPI, SQLModel; React 19 + Vite + Vitest. Label Studio server is user-installed.

## Global Constraints

- `visionsuite_core` imports NO web framework and NO `label_studio_sdk` (the SDK lives only in the api package). The converter is pure stdlib.
- Deferred ML stack stays UNINSTALLED. SP2 adds only `label-studio-sdk>=2,<3` (api package).
- LS SDK usage: `from label_studio_sdk import LabelStudio`; resource-namespaced (`ls.projects.*`, `ls.import_storage.local.*`, `ls.projects.exports.*`). The SDK is synchronous httpx → call it from FastAPI via `fastapi.concurrency.run_in_threadpool`. ALL SDK calls live in `LabelStudioGateway`; endpoints/tests never import the SDK directly.
- Config from env: `LABEL_STUDIO_URL` (default `http://localhost:8080`), `LABEL_STUDIO_API_KEY` (default empty).
- LS bbox coords are PERCENT (0–100); `original_width`/`original_height` are at the RESULT level (siblings of `value`). COCO bbox is `[x_px, y_px, w_px, h_px]` xywh top-left. Guard `value.rotation != 0` (and top-level `image_rotation`) by emitting the axis-aligned enclosing box.
- `image_id` for an LS task is the stem of its `data.image` filename (our images are stored as `<image_id>.<ext>`).
- Single-user; pull runs as a background job through the existing `JobManager.submit_stream`.
- Commits: conventional-commit; EVERY commit message ends with the trailer line exactly:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Branch: all work on `feat/subproject-2-annotation`.
- Schema note: this adds columns/table; the dev DB (`workspace/db.sqlite`) is gitignored and recreated by `init_db` — delete `workspace/` if a stale DB errors.

---

### Task 1: DB — dataset LS fields, `Annotation` table, SDK dep

**Files:**
- Modify: `packages/api/pyproject.toml` (add `label-studio-sdk>=2,<3`)
- Modify: `packages/api/visionsuite_api/db.py` (`Dataset` fields + `Annotation`)
- Test: `packages/api/tests/test_db_annotation.py`

**Interfaces:**
- Produces: `Dataset` gains `ls_project_id: int | None = None`, `class_names: str = "[]"` (JSON list). New `Annotation(id: int|None pk, dataset_id: int fk, image_id: str, coco_json: str, n_objects: int)`.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_db_annotation.py`:
```python
from sqlmodel import Session, select

from visionsuite_api.db import Annotation, Dataset, init_db, make_engine


def test_dataset_ls_fields_and_annotation():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        ds = Dataset(name="d", task="detection", ls_project_id=7, class_names='["car"]')
        s.add(ds); s.commit(); s.refresh(ds)
        s.add(Annotation(dataset_id=ds.id, image_id="abc", coco_json="[]", n_objects=0))
        s.commit()
    with Session(engine) as s:
        ds = s.exec(select(Dataset)).one()
        ann = s.exec(select(Annotation)).one()
    assert ds.ls_project_id == 7 and ds.class_names == '["car"]'
    assert ann.image_id == "abc" and ann.n_objects == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_db_annotation.py -v`
Expected: FAIL (unexpected kwargs / no `Annotation`).

- [ ] **Step 3: Extend the models + add the dep**

Add `"label-studio-sdk>=2,<3"` to `dependencies` in `packages/api/pyproject.toml`.

In `db.py`, replace the `Dataset` class and add `Annotation`:
```python
class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    task: str
    project_id: int | None = Field(default=None, foreign_key="project.id")
    ls_project_id: int | None = Field(default=None)
    class_names: str = "[]"


class Annotation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    image_id: str = Field(index=True)
    coco_json: str
    n_objects: int
```

- [ ] **Step 4: Sync + run**

Run: `uv sync --all-packages && uv run pytest packages/api/tests/test_db_annotation.py -v`
Expected: `label-studio-sdk` resolves (no ML stack); PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(annotate): dataset LS fields + Annotation table + label-studio-sdk dep"
```

---

### Task 2: Core — `ls_config_for` (LS label configs)

**Files:**
- Create: `packages/core/visionsuite_core/labelstudio_convert.py`
- Test: `packages/core/tests/test_ls_config.py`

**Interfaces:**
- Produces: `ls_config_for(task, class_names: list[str]) -> str` where `task` is a `VisionTask` or the strings `"detection"`/`"classification"`. Detection → `RectangleLabels name="label" toName="image"` + a `<Label>` per class. Classification → `Choices name="choice" toName="image" choice="single-radio"` + a `<Choice>` per class.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_ls_config.py`:
```python
from visionsuite_core.labelstudio_convert import ls_config_for
from visionsuite_core.types import VisionTask


def test_detection_config():
    xml = ls_config_for(VisionTask.DETECTION, ["car", "person"])
    assert 'RectangleLabels name="label" toName="image"' in xml
    assert '<Label value="car"/>' in xml and '<Label value="person"/>' in xml


def test_classification_config_accepts_string_task():
    xml = ls_config_for("classification", ["cat", "dog"])
    assert 'Choices name="choice" toName="image"' in xml
    assert '<Choice value="cat"/>' in xml and 'choice="single-radio"' in xml
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_ls_config.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `ls_config_for`**

`packages/core/visionsuite_core/labelstudio_convert.py`:
```python
from __future__ import annotations


def _task_str(task) -> str:
    return task.value if hasattr(task, "value") else str(task)


def ls_config_for(task, class_names: list[str]) -> str:
    if _task_str(task) == "detection":
        labels = "\n".join(f'    <Label value="{c}"/>' for c in class_names)
        return (
            '<View>\n  <Image name="image" value="$image"/>\n'
            '  <RectangleLabels name="label" toName="image">\n'
            f"{labels}\n  </RectangleLabels>\n</View>"
        )
    choices = "\n".join(f'    <Choice value="{c}"/>' for c in class_names)
    return (
        '<View>\n  <Image name="image" value="$image"/>\n'
        '  <Choices name="choice" toName="image" choice="single-radio">\n'
        f"{choices}\n  </Choices>\n</View>"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_ls_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): Label Studio label-config generator"
```

---

### Task 3: Core — `ls_json_to_coco` (export → COCO)

**Files:**
- Modify: `packages/core/visionsuite_core/labelstudio_convert.py`
- Test: `packages/core/tests/test_ls_to_coco.py`

**Interfaces:**
- Produces:
  - `image_id_from_task(task: dict) -> str` — stem of the `data.image` path (handles a `/data/local-files/?d=<rel>` URL and query strings).
  - `ls_json_to_coco(tasks: list[dict], class_names: list[str]) -> dict` returning
    `{"categories": [{"id": i, "name": n}], "images": [{"image_id": str, "annotations": [{"bbox": [x,y,w,h], "category_id": int}], "classification": int | None}]}`.
    Detection results (`type=="rectanglelabels"`) → boxes (percent→pixel, top-left xywh, rotation→axis-aligned). Classification results (`type=="choices"`) → `classification` category id. Tasks whose `annotations` are empty are skipped. Unknown label names are skipped (not crashed).

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_ls_to_coco.py`:
```python
import math

from visionsuite_core.labelstudio_convert import image_id_from_task, ls_json_to_coco


def _det_task(image_id, results):
    return {"data": {"image": f"/data/local-files/?d=datasets/1/images/{image_id}.png"},
            "annotations": [{"result": results}]}


def test_image_id_from_task():
    t = _det_task("abcd1234", [])
    assert image_id_from_task(t) == "abcd1234"


def test_detection_percent_to_pixels():
    res = [{"type": "rectanglelabels", "original_width": 200, "original_height": 100,
            "image_rotation": 0,
            "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rotation": 0,
                      "rectanglelabels": ["car"]}}]
    coco = ls_json_to_coco([_det_task("img1", res)], ["car", "person"])
    assert coco["categories"] == [{"id": 0, "name": "car"}, {"id": 1, "name": "person"}]
    box = coco["images"][0]["annotations"][0]
    assert box["category_id"] == 0
    assert box["bbox"] == [20.0, 20.0, 60.0, 40.0]  # x=10%*200, y=20%*100, w=30%*200, h=40%*100


def test_rotation_makes_enclosing_box_larger():
    res = [{"type": "rectanglelabels", "original_width": 100, "original_height": 100,
            "image_rotation": 0,
            "value": {"x": 40, "y": 40, "width": 20, "height": 20, "rotation": 45,
                      "rectanglelabels": ["car"]}}]
    coco = ls_json_to_coco([_det_task("i", res)], ["car"])
    x, y, w, h = coco["images"][0]["annotations"][0]["bbox"]
    assert w > 20 and h > 20  # rotated 20x20 encloses larger than axis-aligned
    assert math.isclose(w, h, rel_tol=0.05)


def test_classification():
    t = {"data": {"image": "/data/local-files/?d=datasets/1/images/z.png"},
         "annotations": [{"result": [{"type": "choices", "value": {"choices": ["dog"]}}]}]}
    coco = ls_json_to_coco([t], ["cat", "dog"])
    img = coco["images"][0]
    assert img["classification"] == 1 and img["annotations"] == []


def test_unannotated_skipped():
    t = {"data": {"image": "/data/local-files/?d=x/y.png"}, "annotations": []}
    assert ls_json_to_coco([t], ["a"])["images"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_ls_to_coco.py -v`
Expected: FAIL (functions missing).

- [ ] **Step 3: Implement the converter**

Append to `labelstudio_convert.py`:
```python
import math
from pathlib import Path
from urllib.parse import unquote


def image_id_from_task(task: dict) -> str:
    raw = str(task.get("data", {}).get("image", ""))
    rel = raw.split("d=")[-1] if "d=" in raw else raw
    return Path(unquote(rel)).stem


def _enclosing_bbox(x, y, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    pts = [(x + cx * ca - cy * sa, y + cx * sa + cy * ca) for cx, cy in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def ls_json_to_coco(tasks: list[dict], class_names: list[str]) -> dict:
    cat_id = {name: i for i, name in enumerate(class_names)}
    categories = [{"id": i, "name": n} for i, n in enumerate(class_names)]
    images = []
    for task in tasks:
        anns = task.get("annotations") or []
        results = anns[0].get("result", []) if anns else []
        if not results:
            continue
        boxes: list[dict] = []
        classification = None
        for r in results:
            rtype = r.get("type")
            val = r.get("value", {})
            if rtype == "rectanglelabels":
                name = (val.get("rectanglelabels") or [None])[0]
                if name not in cat_id:
                    continue
                ow, oh = r.get("original_width", 0), r.get("original_height", 0)
                x = val.get("x", 0) / 100 * ow
                y = val.get("y", 0) / 100 * oh
                w = val.get("width", 0) / 100 * ow
                h = val.get("height", 0) / 100 * oh
                rot = val.get("rotation", 0) or r.get("image_rotation", 0)
                bbox = _enclosing_bbox(x, y, w, h, rot) if rot else [x, y, w, h]
                boxes.append({"bbox": bbox, "category_id": cat_id[name]})
            elif rtype == "choices":
                name = (val.get("choices") or [None])[0]
                if name in cat_id:
                    classification = cat_id[name]
        images.append({"image_id": image_id_from_task(task), "annotations": boxes,
                       "classification": classification})
    return {"categories": categories, "images": images}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_ls_to_coco.py -v`
Expected: PASS (5 passed). Also `uv run pytest packages/core -q` (purity intact).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): Label Studio export JSON -> COCO converter"
```

---

### Task 4: API — `LabelStudioGateway` + status endpoint

**Files:**
- Create: `packages/api/visionsuite_api/labelstudio.py`
- Create: `packages/api/visionsuite_api/routes/labeling.py`
- Modify: `packages/api/visionsuite_api/main.py` (include the labeling router)
- Test: `packages/api/tests/test_labeling_status.py`

**Interfaces:**
- Produces:
  - `labelstudio.LabelStudioGateway` with methods `status() -> dict`, `create_project(title, label_config) -> int`, `connect_local_storage(project_id, abs_path, regex) -> None`, `project_stats(project_id) -> dict`, `export_json(project_id) -> list[dict]`. The real impl wraps the SDK; **construction of the real client is lazy** (inside `__init__` guarded so tests never trigger a network call).
  - `labelstudio.get_gateway(request) -> LabelStudioGateway` — factory reading `LABEL_STUDIO_URL`/`LABEL_STUDIO_API_KEY` from env; stored on `request.app.state` for override. Tests monkeypatch `visionsuite_api.routes.labeling.get_gateway`.
  - `GET /api/labelstudio/status` → `{connected: bool, url: str, detail?: str}`.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_labeling_status.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_labeling_status.py -v`
Expected: FAIL (module/route missing).

- [ ] **Step 3: Implement the gateway + status route**

`packages/api/visionsuite_api/labelstudio.py`:
```python
import os


class LabelStudioGateway:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self._client = None

    def _sdk(self):
        if self._client is None:
            from label_studio_sdk import LabelStudio
            self._client = LabelStudio(base_url=self.url, api_key=self.api_key)
        return self._client

    def status(self) -> dict:
        try:
            self._sdk().projects.list()
            return {"connected": True, "url": self.url}
        except Exception as exc:  # noqa: BLE001
            return {"connected": False, "url": self.url, "detail": str(exc)}

    def create_project(self, title: str, label_config: str) -> int:
        return self._sdk().projects.create(title=title, label_config=label_config).id

    def connect_local_storage(self, project_id: int, abs_path: str, regex: str) -> None:
        sdk = self._sdk()
        storage = sdk.import_storage.local.create(
            project=project_id, path=abs_path, use_blob_urls=True, regex_filter=regex)
        sdk.import_storage.local.sync(id=storage.id)

    def project_stats(self, project_id: int) -> dict:
        p = self._sdk().projects.get(id=project_id)
        return {"total": getattr(p, "task_number", 0) or 0,
                "annotated": getattr(p, "num_tasks_with_annotations", 0) or 0}

    def export_json(self, project_id: int) -> list:
        sdk = self._sdk()
        ex = sdk.projects.exports.create(id=project_id)
        while True:
            got = sdk.projects.exports.get(id=project_id, export_pk=ex.id)
            if getattr(got, "status", "") == "completed":
                break
        chunks = sdk.projects.exports.download(id=project_id, export_pk=ex.id, export_type="JSON")
        import json
        return json.loads(b"".join(chunks))


def get_gateway(request) -> LabelStudioGateway:
    url = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
    key = os.environ.get("LABEL_STUDIO_API_KEY", "")
    return LabelStudioGateway(url, key)
```

`packages/api/visionsuite_api/routes/labeling.py`:
```python
from fastapi import APIRouter, Request

from ..labelstudio import get_gateway

router = APIRouter()


@router.get("/api/labelstudio/status")
async def labelstudio_status(request: Request) -> dict:
    return get_gateway(request).status()
```

In `main.py`: add `labeling` to the routes import and `app.include_router(labeling.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api/tests/test_labeling_status.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): LabelStudioGateway + /api/labelstudio/status"
```

---

### Task 5: API — create labeling project

**Files:**
- Modify: `packages/api/visionsuite_api/routes/labeling.py`
- Test: `packages/api/tests/test_labeling_project.py`

**Interfaces:**
- Consumes: `get_gateway`; `visionsuite_core.labelstudio_convert.ls_config_for`; `visionsuite_core.workspace`; `db.Dataset`.
- Produces: `POST /api/datasets/{ds_id}/labeling/project` `{class_names: list[str]}` → generates the LS config from the dataset's `task` + classes, `create_project`, `connect_local_storage` over the dataset's images dir (abs path), persists `ls_project_id` + `class_names` (JSON) on the `Dataset`, returns `{ls_project_id, ls_url}` where `ls_url = f"{gateway.url}/projects/{ls_project_id}/data"`.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_labeling_project.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_labeling_project.py -v`
Expected: FAIL (route missing).

- [ ] **Step 3: Implement the endpoint**

Append to `routes/labeling.py`:
```python
import json

from fastapi import HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from visionsuite_core import workspace
from visionsuite_core.labelstudio_convert import ls_config_for

from ..db import Dataset


class CreateLabelingProject(BaseModel):
    class_names: list[str]


@router.post("/api/datasets/{ds_id}/labeling/project")
async def create_labeling_project(request: Request, ds_id: int, body: CreateLabelingProject) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        task = ds.task
    gateway = get_gateway(request)
    config = ls_config_for(task, body.class_names)
    project_id = gateway.create_project(f"visionsuite-{ds_id}", config)
    images_dir = workspace.dataset_dir(str(ds_id)) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    gateway.connect_local_storage(project_id, str(images_dir), r".*\.(jpg|jpeg|png|webp|bmp)$")
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        ds.ls_project_id = project_id
        ds.class_names = json.dumps(body.class_names)
        s.add(ds); s.commit()
    return {"ls_project_id": project_id, "ls_url": f"{gateway.url}/projects/{project_id}/data"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api/tests/test_labeling_project.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): create Label Studio project + local-storage sync"
```

---

### Task 6: API — labeling status + pull annotations (job)

**Files:**
- Modify: `packages/api/visionsuite_api/routes/labeling.py`
- Test: `packages/api/tests/test_labeling_pull.py`

**Interfaces:**
- Consumes: `get_gateway`; `ls_json_to_coco`; `db.Dataset`, `db.Annotation`; `app.state.manager.submit_stream`.
- Produces:
  - `GET /api/datasets/{ds_id}/labeling/status` → `{configured: bool, ls_project_id, total, annotated, ls_url}` (`configured=false` if the dataset has no `ls_project_id`).
  - `POST /api/datasets/{ds_id}/labeling/pull` → `{job_id}`; the job calls `export_json` → `ls_json_to_coco(tasks, class_names)` → upserts one `Annotation` per image (`coco_json` = the image's `annotations` list, or `[{"category_id": classification}]` for classification; `n_objects` accordingly), replacing prior annotations for that dataset; yields progress; terminal DONE. Returns per-class counts in a final log event.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_labeling_pull.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_labeling_pull.py -v`
Expected: FAIL (routes missing).

- [ ] **Step 3: Implement status + pull**

Append to `routes/labeling.py`:
```python
from uuid import uuid4

from visionsuite_core.backends import RunEvent, RunStatus
from visionsuite_core.labelstudio_convert import ls_json_to_coco

from ..db import Annotation


@router.get("/api/datasets/{ds_id}/labeling/status")
async def labeling_status(request: Request, ds_id: int) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        pid = ds.ls_project_id
    if pid is None:
        return {"configured": False}
    gateway = get_gateway(request)
    stats = gateway.project_stats(pid)
    return {"configured": True, "ls_project_id": pid, "total": stats["total"],
            "annotated": stats["annotated"], "ls_url": f"{gateway.url}/projects/{pid}/data"}


def _pull_producer(engine, ds_id, gateway):
    async def producer():
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        with Session(engine) as s:
            ds = s.get(Dataset, ds_id)
            pid = ds.ls_project_id
            class_names = json.loads(ds.class_names)
        yield RunEvent(type="log", message=f"exporting project {pid}")
        tasks = gateway.export_json(pid)
        coco = ls_json_to_coco(tasks, class_names)
        with Session(engine) as s:
            for old in s.exec(select(Annotation).where(Annotation.dataset_id == ds_id)).all():
                s.delete(old)
            s.commit()
            for i, img in enumerate(coco["images"], 1):
                if img["classification"] is not None:
                    payload = [{"category_id": img["classification"]}]
                else:
                    payload = img["annotations"]
                s.add(Annotation(dataset_id=ds_id, image_id=img["image_id"],
                                 coco_json=json.dumps(payload), n_objects=len(payload)))
                yield RunEvent(type="progress", progress=i / len(coco["images"]))
            s.commit()
        yield RunEvent(type="log", message=f"stored {len(coco['images'])} annotated images")
        yield RunEvent(type="status", status=RunStatus.DONE)
    return producer


@router.post("/api/datasets/{ds_id}/labeling/pull")
async def labeling_pull(request: Request, ds_id: int) -> dict:
    from sqlmodel import select  # local import to avoid top-level churn
    gateway = get_gateway(request)
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(job_id, "pull", _pull_producer(engine, ds_id, gateway))
    return {"job_id": job_id}
```
Note: `select` is used inside `_pull_producer`; ensure `from sqlmodel import Session, select` is imported at the top of `labeling.py` (it already imports `Session` from Task 5 — add `select`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS (whole api suite).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): labeling status + pull-annotations job"
```

---

### Task 7: Frontend — Labeling page

**Files:**
- Modify: `web/src/lib/api.ts` (labeling client functions)
- Create: `web/src/routes/Labeling.tsx` (replaces the stub)
- Test: `web/src/routes/Labeling.test.tsx`

**Interfaces:**
- Consumes: the Task 4–6 endpoints; the existing `useJobStream`, `listDatasets`, `apiUrl`.
- Produces in `api.ts`: `labelStudioStatus()`, `createLabelingProject(dsId, classNames: string[])`, `labelingStatus(dsId)`, `pullAnnotations(dsId) -> {job_id}`.
- `Labeling.tsx`: connection banner; pick a dataset; class-names input → create project; once configured, an "Open in Label Studio" link (`ls_url`, new tab), an `annotated/total` line, and a "Pull annotations" button that launches the pull job and shows `useJobStream` progress.

- [ ] **Step 1: Write the failing test**

`web/src/routes/Labeling.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Labeling from "./Labeling";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("Labeling", () => {
  it("shows LS connection and creates a project", async () => {
    vi.spyOn(api, "labelStudioStatus").mockResolvedValue({ connected: true, url: "http://ls:8080" });
    vi.spyOn(api, "listDatasets").mockResolvedValue([{ id: 1, name: "cars", task: "detection", image_count: 3 }]);
    vi.spyOn(api, "labelingStatus").mockResolvedValue({ configured: false });
    const create = vi.spyOn(api, "createLabelingProject").mockResolvedValue({ ls_project_id: 5, ls_url: "http://ls:8080/projects/5/data" });
    render(<Labeling />);
    await waitFor(() => expect(screen.getByText(/connected/i)).toBeTruthy());
    fireEvent.click(await screen.findByText(/cars/));
    fireEvent.change(screen.getByPlaceholderText(/classes/i), { target: { value: "car, bus" } });
    fireEvent.click(screen.getByRole("button", { name: /create labeling project/i }));
    await waitFor(() => expect(create).toHaveBeenCalledWith(1, ["car", "bus"]));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/routes/Labeling.test.tsx`
Expected: FAIL (stub only).

- [ ] **Step 3: Implement the client + page**

Append to `web/src/lib/api.ts`:
```ts
export async function labelStudioStatus(): Promise<{ connected: boolean; url: string; detail?: string }> {
  return (await fetch(apiUrl("/api/labelstudio/status"))).json();
}
export async function createLabelingProject(dsId: number, classNames: string[]):
  Promise<{ ls_project_id: number; ls_url: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/labeling/project`), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_names: classNames }),
  })).json();
}
export async function labelingStatus(dsId: number): Promise<{
  configured: boolean; ls_project_id?: number; total?: number; annotated?: number; ls_url?: string;
}> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/labeling/status`))).json();
}
export async function pullAnnotations(dsId: number): Promise<{ job_id: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/labeling/pull`), { method: "POST" })).json();
}
```

`web/src/routes/Labeling.tsx`:
```tsx
import { useEffect, useState } from "react";
import { createLabelingProject, labelingStatus, labelStudioStatus, listDatasets,
  pullAnnotations, type DatasetInfo } from "../lib/api";
import { useJobStream } from "../lib/useJobStream";

export default function Labeling() {
  const [conn, setConn] = useState<{ connected: boolean } | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [classes, setClasses] = useState("");
  const [status, setStatus] = useState<Awaited<ReturnType<typeof labelingStatus>> | null>(null);
  const job = useJobStream();

  useEffect(() => { labelStudioStatus().then(setConn); listDatasets().then(setDatasets); }, []);
  useEffect(() => { if (selected != null) labelingStatus(selected).then(setStatus); }, [selected]);
  useEffect(() => { if (job.status === "done" && selected != null) labelingStatus(selected).then(setStatus); }, [job.status]);

  async function onCreate() {
    if (selected == null) return;
    const names = classes.split(",").map((c) => c.trim()).filter(Boolean);
    await createLabelingProject(selected, names);
    labelingStatus(selected).then(setStatus);
  }

  return (
    <div>
      <h1>Labeling</h1>
      <p>Label Studio: {conn?.connected ? "connected" : "not connected — start it and set LABEL_STUDIO_URL/API_KEY"}</p>
      <ul>{datasets.map((d) => (
        <li key={d.id}><button onClick={() => setSelected(d.id)}>{d.name}</button> ({d.task})</li>
      ))}</ul>
      {selected != null && (
        <div>
          {!status?.configured && (
            <div>
              <input placeholder="classes (comma-separated)" value={classes}
                     onChange={(e) => setClasses(e.target.value)} />
              <button onClick={onCreate}>Create labeling project</button>
            </div>
          )}
          {status?.configured && (
            <div>
              <a href={status.ls_url} target="_blank" rel="noreferrer">Open in Label Studio</a>
              <p>{status.annotated ?? 0} / {status.total ?? 0} annotated</p>
              <button onClick={() => pullAnnotations(selected).then((r) => job.watch(r.job_id))}>
                Pull annotations
              </button>
              <p>{job.status} <progress value={job.progress} max={1} /></p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes + build**

Run: `cd web && npx vitest run && npm run build`
Expected: PASS + build succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web): Labeling page (LS connect, create project, pull annotations)"
```

---

### Task 8: Ops — Label Studio launcher + README

**Files:**
- Create: `scripts/labelstudio.sh`
- Modify: `README.md` (Label Studio setup section)
- Test: (no unit test — a shell launcher + docs; verified by review + the full existing suite staying green)

**Interfaces:**
- Produces: `scripts/labelstudio.sh` launching LS with local-file serving pointed at the workspace root; README documenting the one-time setup.

- [ ] **Step 1: Create the launcher**

`scripts/labelstudio.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

WS="${VISIONSUITE_WORKSPACE:-$(pwd)/workspace}"
mkdir -p "$WS"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(cd "$WS" && pwd -P)"

echo "Launching Label Studio on http://localhost:8080"
echo "Document root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
echo "First run: create an account, then Account & Settings -> copy the (legacy) token,"
echo "and export it for VisionSuite:  export LABEL_STUDIO_API_KEY=<token>"
exec label-studio start
```
Then `chmod +x scripts/labelstudio.sh`.

- [ ] **Step 2: Document setup in README**

Add a `## Labeling (Label Studio)` section to `README.md`:
```markdown
## Labeling (Label Studio)

VisionSuite uses a local Label Studio for annotation.

1. Install it once: `pip install label-studio` (or `uv tool install label-studio`).
2. Start it (serves local files from your workspace): `./scripts/labelstudio.sh`
3. On first run, create an account, open **Account & Settings**, copy the **legacy** access token, and:
   `export LABEL_STUDIO_API_KEY=<token>` (and `LABEL_STUDIO_URL` if not on :8080), then start the app.
4. In VisionSuite → **Labeling**: pick a dataset, enter classes, **Create labeling project** (this
   creates the LS project and syncs the dataset's images via Local Storage), click **Open in Label
   Studio** to label, then **Pull annotations** to import them back as COCO.
```

- [ ] **Step 3: Verify nothing regressed**

Run: `uv run pytest -q` (repo root) and `cd web && npm run build`
Expected: backend suite PASS; build succeeds. Confirm `scripts/labelstudio.sh` is executable (`ls -l scripts/labelstudio.sh` shows `x`).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(ops): Label Studio launcher script + README setup"
```

---

## Self-Review

**1. Spec coverage (spec §2–§5):** connect/status → Task 4; create project + config + local-storage sync → Tasks 2,5; progress/status → Task 6; pull → COCO → DB (job) → Tasks 3,6; pure converter (config + json→coco, %→px, rotation, image_id) → Tasks 2,3; DB fields/Annotation → Task 1; Labeling page → Task 7; ops/launcher → Task 8. ✅ SDK isolated in gateway; core stays SDK-free. ✅
**2. Placeholder scan:** all steps carry complete code; the gateway's `export_json`/`project_stats` are best-effort SDK shapes flagged as the open risk (spec §5) — real code present, not a placeholder. ✅
**3. Type consistency:** `ls_config_for`/`ls_json_to_coco`/`image_id_from_task`, `LabelStudioGateway` methods, `get_gateway`, `Annotation` fields, `RunEvent`/`RunStatus`, and the api client names are consistent across defining and consuming tasks. Pull stores `coco_json` as the per-image annotations list (detection) or `[{category_id}]` (classification), matching Task 6's test. ✅

## Notes for the executor
- Delete `workspace/` before running (schema changed).
- Tests NEVER touch a live Label Studio — every endpoint test monkeypatches `visionsuite_api.routes.labeling.get_gateway` with a fake. Do not add a test that constructs the real gateway/SDK.
- `LabelStudioGateway.export_json`/`project_stats` mirror the research brief's SDK shapes but are unverified against a live server; they're the one place to adjust when first run on the M5 — keep all SDK calls inside the gateway so that adjustment is localized.
- Run backend tests from repo root (`uv run pytest`), frontend from `web/`.
