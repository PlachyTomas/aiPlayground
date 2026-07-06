# VisionSuite Sub-Project 1: Data Pipeline & Ingestion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import images into VisionSuite from four sources (local folder, Hugging Face dataset, uploaded video, webcam), store them under `workspace/datasets/<id>/` with thumbnails + a SQLite manifest, and browse them as a paginated grid — imports running as live-progress background jobs.

**Architecture:** Pure I/O lives in `visionsuite_core.ingest` (hash → save → thumbnail, plus folder/video/HF iterators — no web, no DB). SP0's `JobManager`/WebSocket is generalized to run any event-producing async job. `visionsuite_api.routes.datasets` exposes dataset CRUD, image serving, and the four import endpoints (imports create jobs). The React **Datasets** page replaces its stub with create/list/grid/delete + import UIs sharing a `useJobStream` hook.

**Tech Stack:** Python 3.11+, Pillow, `datasets` (HF), `imageio[ffmpeg]`, FastAPI + `python-multipart`, SQLModel; React 19 + Vite + Vitest.

## Global Constraints

- Python `>=3.11`; `visionsuite_core` imports NO web framework (fastapi/starlette/uvicorn) — `test_core_purity.py` enforces it. Pillow/datasets/imageio ARE allowed in core.
- The deferred ML stack (`torch>=2.11`, `transformers>=4.54`, `label-studio-sdk>=2,<3`, `trackio==0.29.0`, `optimum[onnxruntime]`, `onnxruntime`) stays UNINSTALLED — SP1 adds only data-pipeline deps.
- Workspace root from `VISIONSUITE_WORKSPACE`, default `./workspace` (via `visionsuite_core.workspace`); store image paths RELATIVE to the workspace root — never absolute — so the workspace stays relocatable.
- Single-user: no auth; one background job at a time (the existing `JobManager` lock).
- Image identity: `image_id = sha1(file_bytes).hexdigest()[:16]`; dedup within a dataset by `image_id`.
- Supported image extensions: `.jpg .jpeg .png .webp .bmp`. Thumbnails: max 256px longest side, WEBP.
- Commits: conventional-commit; EVERY commit message ends with the trailer line exactly:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Branch: all work on `feat/subproject-1-data` (created before Task 1).
- Schema note: this plan changes the `Image` table. There is no migration system; the dev DB (`workspace/db.sqlite`) is gitignored and recreated by `init_db`. Delete `workspace/` when the schema changes.

---

### Task 1: SP1 dependencies + extend the `Image` table

**Files:**
- Modify: `packages/core/pyproject.toml` (add `pillow`, `datasets`, `imageio`, `imageio-ffmpeg`)
- Modify: `packages/api/pyproject.toml` (add `python-multipart`)
- Modify: `packages/api/visionsuite_api/db.py` (extend `Image`)
- Test: `packages/api/tests/test_db_image.py`

**Interfaces:**
- Produces: `Image(id: int|None pk, dataset_id: int fk, image_id: str, filename: str, width: int, height: int, source: str, path: str, thumb_path: str)` where `path`/`thumb_path` are relative to the workspace root.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_db_image.py`:
```python
from sqlmodel import Session

from visionsuite_api.db import Image, init_db, make_engine


def test_image_has_ingestion_fields():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        s.add(Image(dataset_id=1, image_id="abc123", filename="abc123.png",
                    width=64, height=48, source="folder",
                    path="datasets/1/images/abc123.png",
                    thumb_path="datasets/1/thumbs/abc123.webp"))
        s.commit()
    with Session(engine) as s:
        got = s.exec(__import__("sqlmodel").select(Image)).one()
    assert got.image_id == "abc123" and got.source == "folder" and got.width == 64
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_db_image.py -v`
Expected: FAIL (TypeError / unexpected keyword — old `Image` lacks these fields).

- [ ] **Step 3: Add deps and extend the model**

In `packages/core/pyproject.toml`, set `dependencies` to:
```toml
dependencies = ["pillow>=10", "datasets>=3", "imageio>=2.34", "imageio-ffmpeg>=0.5"]
```
In `packages/api/pyproject.toml`, add `"python-multipart>=0.0.9"` to `dependencies`.

Replace the `Image` class in `packages/api/visionsuite_api/db.py`:
```python
class Image(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id", index=True)
    image_id: str = Field(index=True)
    filename: str
    width: int
    height: int
    source: str
    path: str
    thumb_path: str
```

- [ ] **Step 4: Sync + run the test**

Run: `uv sync --all-packages && uv run pytest packages/api/tests/test_db_image.py -v`
Expected: deps resolve (no torch/transformers); PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(data): add SP1 deps and extend Image table with ingestion fields"
```

---

### Task 2: Core `ingest.save_image_bytes` + thumbnails + dedup

**Files:**
- Create: `packages/core/visionsuite_core/ingest.py`
- Test: `packages/core/tests/test_ingest_save.py`

**Interfaces:**
- Produces:
  - `IngestedImage` (dataclass): `image_id: str`, `filename: str`, `width: int`, `height: int`, `source: str`.
  - `save_image_bytes(data: bytes, images_dir: Path, thumbs_dir: Path, source: str) -> IngestedImage` — hashes bytes → `image_id`; opens with Pillow (raises on non-image); writes the full image to `images_dir/<image_id>.<ext>` (ext from the decoded format, lowercased, `jpeg`→`jpg`); writes a WEBP thumbnail (≤256px) to `thumbs_dir/<image_id>.webp`; returns the record. If the target file already exists, skips rewriting (dedup) but still returns the record.
  - `THUMB_MAX = 256`.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_ingest_save.py`:
```python
import io

from PIL import Image as PILImage

from visionsuite_core.ingest import IngestedImage, save_image_bytes


def _png_bytes(w=80, h=40, color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def test_save_writes_image_and_thumb(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    rec = save_image_bytes(_png_bytes(), images, thumbs, source="folder")
    assert isinstance(rec, IngestedImage)
    assert (images / rec.filename).exists()
    assert (thumbs / f"{rec.image_id}.webp").exists()
    assert rec.width == 80 and rec.height == 40 and rec.source == "folder"


def test_dedup_same_bytes(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    b = _png_bytes()
    a1 = save_image_bytes(b, images, thumbs, source="folder")
    a2 = save_image_bytes(b, images, thumbs, source="folder")
    assert a1.image_id == a2.image_id
    assert len(list(images.iterdir())) == 1


def test_rejects_non_image(tmp_path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(); thumbs.mkdir()
    try:
        save_image_bytes(b"not an image", images, thumbs, source="folder")
        assert False, "expected an error"
    except Exception:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_ingest_save.py -v`
Expected: FAIL (no module `visionsuite_core.ingest`).

- [ ] **Step 3: Implement `ingest.py`**

```python
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

THUMB_MAX = 256
_EXT = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "bmp": "bmp"}


@dataclass
class IngestedImage:
    image_id: str
    filename: str
    width: int
    height: int
    source: str


def save_image_bytes(data: bytes, images_dir: Path, thumbs_dir: Path, source: str) -> IngestedImage:
    image_id = hashlib.sha1(data).hexdigest()[:16]
    img = PILImage.open(io.BytesIO(data))
    img.load()
    ext = _EXT.get((img.format or "").lower(), "png")
    filename = f"{image_id}.{ext}"
    full = images_dir / filename
    thumb = thumbs_dir / f"{image_id}.webp"
    width, height = img.size
    if not full.exists():
        full.write_bytes(data)
    if not thumb.exists():
        t = img.convert("RGB")
        t.thumbnail((THUMB_MAX, THUMB_MAX))
        t.save(thumb, format="WEBP")
    return IngestedImage(image_id=image_id, filename=filename, width=width, height=height, source=source)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_ingest_save.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): ingest.save_image_bytes with thumbnails and dedup"
```

---

### Task 3: Core ingest source iterators — folder, video, HF

**Files:**
- Modify: `packages/core/visionsuite_core/ingest.py`
- Test: `packages/core/tests/test_ingest_sources.py`

**Interfaces:**
- Produces (all in `ingest.py`):
  - `SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}`.
  - `iter_folder_images(folder: Path) -> Iterator[Path]` — recursive; only `SUPPORTED_EXTS` (case-insensitive); raises `FileNotFoundError` if `folder` is missing.
  - `extract_video_frames(video_path: Path, every_n: int = 30) -> Iterator[bytes]` — yields PNG bytes for every `every_n`-th frame (imageio).
  - `hf_images_from_iterable(examples: Iterable[dict], image_column: str | None = None) -> Iterator[bytes]` — yields PNG bytes from an iterable of HF examples; auto-detects the column holding a PIL image if `image_column` is None; raises `ValueError` if none found.
  - `iter_hf_images(dataset_id: str, split: str = "train", config: str | None = None, image_column: str | None = None) -> Iterator[bytes]` — loads the HF dataset (streaming) and delegates to `hf_images_from_iterable`.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_ingest_sources.py`:
```python
import io

import imageio.v3 as iio
import numpy as np
from PIL import Image as PILImage

from visionsuite_core.ingest import (
    extract_video_frames,
    hf_images_from_iterable,
    iter_folder_images,
)


def _write_png(p, color=(0, 128, 0)):
    PILImage.new("RGB", (32, 24), color).save(p, format="PNG")


def test_iter_folder_finds_images(tmp_path):
    _write_png(tmp_path / "a.png")
    (tmp_path / "sub").mkdir()
    _write_png(tmp_path / "sub" / "b.JPG")  # note case + jpg
    (tmp_path / "notes.txt").write_text("ignore me")
    found = sorted(p.name.lower() for p in iter_folder_images(tmp_path))
    assert found == ["a.png", "b.jpg"]


def test_extract_video_frames(tmp_path):
    vid = tmp_path / "clip.mp4"
    frames = [np.full((16, 16, 3), i, dtype=np.uint8) for i in (10, 20, 30, 40, 50, 60)]
    iio.imwrite(vid, np.stack(frames), fps=6, codec="libx264")
    out = list(extract_video_frames(vid, every_n=2))
    assert 1 <= len(out) <= 6
    PILImage.open(io.BytesIO(out[0])).load()  # decodes


def test_hf_images_autodetect():
    ds = [{"image": PILImage.new("RGB", (8, 8)), "label": 1},
          {"image": PILImage.new("RGB", (8, 8)), "label": 0}]
    out = list(hf_images_from_iterable(ds))
    assert len(out) == 2
    PILImage.open(io.BytesIO(out[0])).load()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_ingest_sources.py -v`
Expected: FAIL (functions not defined).

- [ ] **Step 3: Extend `ingest.py`**

Append:
```python
from collections.abc import Iterable, Iterator

import imageio.v3 as _iio

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_folder_images(folder: Path) -> Iterator[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"not a folder: {folder}")
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def _encode_png(img: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def extract_video_frames(video_path: Path, every_n: int = 30) -> Iterator[bytes]:
    if every_n < 1:
        every_n = 1
    for i, frame in enumerate(_iio.imiter(video_path)):
        if i % every_n == 0:
            yield _encode_png(PILImage.fromarray(frame))


def hf_images_from_iterable(examples: Iterable[dict], image_column: str | None = None) -> Iterator[bytes]:
    col = image_column
    for ex in examples:
        if col is None:
            col = next((k for k, v in ex.items() if isinstance(v, PILImage.Image)), None)
            if col is None:
                raise ValueError("no PIL image column found in HF example")
        yield _encode_png(ex[col])


def iter_hf_images(dataset_id: str, split: str = "train", config: str | None = None,
                   image_column: str | None = None) -> Iterator[bytes]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, name=config, split=split, streaming=True)
    yield from hf_images_from_iterable(ds, image_column=image_column)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_ingest_sources.py -v`
Expected: PASS. (If `libx264` is unavailable in the imageio-ffmpeg build, the test writer may fall back to `codec="mpeg4"`; keep the extract assertion range-based.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): folder/video/HF ingest source iterators"
```

---

### Task 4: Generalize JobManager to run any producer + `/api/jobs/{id}/events`

**Files:**
- Modify: `packages/api/visionsuite_api/jobs.py`
- Modify: `packages/api/visionsuite_api/routes/runs.py` (WS endpoint moves to a shared helper; keep `/api/runs/{id}/events`)
- Create: `packages/api/visionsuite_api/routes/jobs.py`
- Modify: `packages/api/visionsuite_api/main.py` (include the jobs router)
- Test: `packages/api/tests/test_jobs_generic.py`

**Interfaces:**
- Consumes: `visionsuite_core.backends` (`RunEvent`, `RunStatus`, `LocalBackend`, `RunSpec`); existing `TERMINAL`.
- Produces:
  - `Job` gains `kind: str` (default `"train"`).
  - `JobManager.submit_stream(job_id: str, kind: str, producer: Callable[[], AsyncIterator[RunEvent]]) -> Job` — runs one at a time, records events + terminal status, retains the task (the SP0 failure-path try/except stays).
  - `JobManager.submit(spec)` becomes a wrapper: `submit_stream(spec.run_id, "train", lambda: self.backend.stream(spec))`.
  - `routes/jobs.py`: `WS /api/jobs/{job_id}/events` streaming any job (same JSON shape as the runs stream); unknown id → close `4404`. A shared `stream_job_events(websocket, job)` helper is reused by the `/api/runs/{run_id}/events` endpoint.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_jobs_generic.py`:
```python
from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.jobs import JobManager
from visionsuite_api.main import create_app
from visionsuite_core.backends import RunEvent, RunStatus


async def _producer():
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    yield RunEvent(type="progress", progress=1.0)
    yield RunEvent(type="status", status=RunStatus.DONE)


async def test_submit_stream_runs_any_producer():
    jm = JobManager()
    job = await jm.submit_stream("j1", "import", _producer)
    import asyncio
    for _ in range(200):
        if job.status == RunStatus.DONE:
            break
        await asyncio.sleep(0.01)
    assert job.status == RunStatus.DONE and job.kind == "import"


def test_ws_jobs_endpoint_streams():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    run_id = client.post("/api/runs", json={}).json()["run_id"]
    received = []
    with client.websocket_connect(f"/api/jobs/{run_id}/events") as ws:
        while True:
            try:
                received.append(ws.receive_json())
            except Exception:
                break
    assert any(e.get("status") == "done" for e in received)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_jobs_generic.py -v`
Expected: FAIL (`submit_stream` / `/api/jobs` not defined).

- [ ] **Step 3: Generalize the manager and add the router**

In `jobs.py`, change `Job.__init__` to accept `kind` and refactor the manager:
```python
class Job:
    def __init__(self, spec: RunSpec, kind: str = "train") -> None:
        self.spec = spec
        self.kind = kind
        self.events: list[RunEvent] = []
        self.status: RunStatus = RunStatus.PENDING


class JobManager:
    def __init__(self, backend: TrainingBackend | None = None) -> None:
        self.backend = backend or LocalBackend()
        self._jobs: dict[str, Job] = {}
        self._tasks: set = set()
        self._lock = asyncio.Lock()

    async def submit(self, spec: RunSpec) -> Job:
        return await self.submit_stream(spec.run_id, "train", lambda: self.backend.stream(spec), spec=spec)

    async def submit_stream(self, job_id, kind, producer, spec=None) -> Job:
        job = Job(spec or RunSpec(run_id=job_id, model_id="", dataset_id=""), kind=kind)
        self._jobs[job_id] = job
        task = asyncio.create_task(self._run(job, producer))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job, producer) -> None:
        async with self._lock:
            job.status = RunStatus.RUNNING
            try:
                async for event in producer():
                    job.events.append(event)
                    if event.status is not None:
                        job.status = event.status
            except Exception as exc:
                job.events.append(RunEvent(type="log", message=f"job failed: {exc}"))
                job.status = RunStatus.FAILED

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)
```
(Keep the `TERMINAL` set and existing imports.)

Add `packages/api/visionsuite_api/routes/jobs.py`:
```python
import asyncio

from fastapi import APIRouter, WebSocket

from ..jobs import TERMINAL

router = APIRouter()


async def stream_job_events(websocket: WebSocket, job) -> None:
    await websocket.accept()
    if job is None:
        await websocket.close(code=4404)
        return
    sent = 0
    while True:
        while sent < len(job.events):
            e = job.events[sent]
            sent += 1
            await websocket.send_json({
                "type": e.type, "message": e.message,
                "progress": e.progress, "status": e.status.value if e.status else None,
            })
        if job.status in TERMINAL:
            break
        await asyncio.sleep(0.05)
    await websocket.close()


@router.websocket("/api/jobs/{job_id}/events")
async def job_events(websocket: WebSocket, job_id: str) -> None:
    await stream_job_events(websocket, websocket.app.state.manager.get(job_id))
```

In `routes/runs.py`, replace the body of the existing `run_events` WS endpoint to delegate:
```python
from .jobs import stream_job_events

@router.websocket("/api/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await stream_job_events(websocket, websocket.app.state.manager.get(run_id))
```
(Remove the now-unused inline poll loop + its `TERMINAL`/`asyncio` imports from `runs.py` if they are no longer referenced.)

In `main.py`, add `from .routes import health, runs, jobs` and `app.include_router(jobs.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS — including the pre-existing `test_runs_ws.py` (behavior preserved via the shared helper).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): generalize JobManager.submit_stream + /api/jobs/{id}/events"
```

---

### Task 5: Datasets CRUD API + image serving

**Files:**
- Create: `packages/api/visionsuite_api/routes/datasets.py`
- Modify: `packages/api/visionsuite_api/main.py` (include the datasets router)
- Test: `packages/api/tests/test_datasets_api.py`

**Interfaces:**
- Consumes: `app.state.engine`; `visionsuite_api.db` (`Dataset`, `Image`); `visionsuite_core.workspace` (`dataset_dir`, `workspace_root`).
- Produces (`datasets.router`):
  - `POST /api/datasets` `{name, task}` → `{id, name, task, image_count: 0}`.
  - `GET /api/datasets` → `[{id, name, task, image_count}]`.
  - `DELETE /api/datasets/{ds_id}` → `{deleted: true}`; removes the on-disk `datasets/<ds_id>/` dir.
  - `GET /api/datasets/{ds_id}/images?offset=0&limit=60` → `{total, images: [{image_id, filename, width, height, source, thumb_url, file_url}]}` where the URLs point at the two serve endpoints.
  - `GET /api/datasets/{ds_id}/images/{image_id}/thumb` and `.../file` → the bytes (FileResponse), 404 if missing.
  - `DELETE /api/datasets/{ds_id}/images/{image_id}` → `{deleted: true}`; removes the row + on-disk full image + thumb.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_datasets_api.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_datasets_api.py -v`
Expected: FAIL (routes not registered).

- [ ] **Step 3: Implement `datasets.py` and register it**

`packages/api/visionsuite_api/routes/datasets.py`:
```python
import shutil

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, func, select

from visionsuite_core import workspace

from ..db import Dataset, Image

router = APIRouter()


class CreateDataset(BaseModel):
    name: str
    task: str


def _count(session, ds_id: int) -> int:
    return session.exec(select(func.count()).select_from(Image).where(Image.dataset_id == ds_id)).one()


@router.post("/api/datasets")
def create_dataset(request: Request, body: CreateDataset) -> dict:
    with Session(request.app.state.engine) as s:
        ds = Dataset(name=body.name, task=body.task)
        s.add(ds); s.commit(); s.refresh(ds)
        return {"id": ds.id, "name": ds.name, "task": ds.task, "image_count": 0}


@router.get("/api/datasets")
def list_datasets(request: Request) -> list:
    with Session(request.app.state.engine) as s:
        out = []
        for ds in s.exec(select(Dataset)).all():
            out.append({"id": ds.id, "name": ds.name, "task": ds.task, "image_count": _count(s, ds.id)})
        return out


@router.delete("/api/datasets/{ds_id}")
def delete_dataset(request: Request, ds_id: int) -> dict:
    with Session(request.app.state.engine) as s:
        ds = s.get(Dataset, ds_id)
        if ds is None:
            raise HTTPException(404)
        for img in s.exec(select(Image).where(Image.dataset_id == ds_id)).all():
            s.delete(img)
        s.delete(ds); s.commit()
    shutil.rmtree(workspace.workspace_root() / "datasets" / str(ds_id), ignore_errors=True)
    return {"deleted": True}


@router.get("/api/datasets/{ds_id}/images")
def list_images(request: Request, ds_id: int, offset: int = 0, limit: int = 60) -> dict:
    with Session(request.app.state.engine) as s:
        total = _count(s, ds_id)
        rows = s.exec(select(Image).where(Image.dataset_id == ds_id).offset(offset).limit(limit)).all()
        images = [{
            "image_id": r.image_id, "filename": r.filename, "width": r.width, "height": r.height,
            "source": r.source,
            "thumb_url": f"/api/datasets/{ds_id}/images/{r.image_id}/thumb",
            "file_url": f"/api/datasets/{ds_id}/images/{r.image_id}/file",
        } for r in rows]
    return {"total": total, "images": images}


def _one_image(request: Request, ds_id: int, image_id: str) -> Image:
    with Session(request.app.state.engine) as s:
        row = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == image_id)).first()
    if row is None:
        raise HTTPException(404)
    return row


@router.get("/api/datasets/{ds_id}/images/{image_id}/thumb")
def image_thumb(request: Request, ds_id: int, image_id: str):
    return FileResponse(workspace.workspace_root() / _one_image(request, ds_id, image_id).thumb_path)


@router.get("/api/datasets/{ds_id}/images/{image_id}/file")
def image_file(request: Request, ds_id: int, image_id: str):
    return FileResponse(workspace.workspace_root() / _one_image(request, ds_id, image_id).path)


@router.delete("/api/datasets/{ds_id}/images/{image_id}")
def delete_image(request: Request, ds_id: int, image_id: str) -> dict:
    row = _one_image(request, ds_id, image_id)
    root = workspace.workspace_root()
    (root / row.path).unlink(missing_ok=True)
    (root / row.thumb_path).unlink(missing_ok=True)
    with Session(request.app.state.engine) as s:
        db_row = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == image_id)).first()
        if db_row:
            s.delete(db_row); s.commit()
    return {"deleted": True}
```
In `main.py`: add `datasets` to the routes import and `app.include_router(datasets.router)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): dataset CRUD + image list/serve/delete"
```

---

### Task 6: Folder + webcam import endpoints

**Files:**
- Create: `packages/api/visionsuite_api/imports.py` (shared import helpers)
- Modify: `packages/api/visionsuite_api/routes/datasets.py` (add import endpoints)
- Test: `packages/api/tests/test_import_folder_webcam.py`

**Interfaces:**
- Consumes: `visionsuite_core.ingest` (`iter_folder_images`, `save_image_bytes`); `visionsuite_core.workspace` (`dataset_dir`); the `JobManager` (`submit_stream`); `db.Image`.
- Produces:
  - `imports.save_and_record(engine, ds_id, data, source) -> dict` — calls `save_image_bytes` into the dataset's `images/`+`thumbs/`, inserts an `Image` row (skipping if that `image_id` already exists in the dataset), returns the image record dict.
  - `imports.folder_producer(engine, ds_id, folder)` → async generator of `RunEvent` (progress per file).
  - `POST /api/datasets/{ds_id}/import/folder` `{path}` → `{job_id}` (unknown/relative-missing path → 400).
  - `POST /api/datasets/{ds_id}/import/webcam` (multipart `file`) → the saved image record dict directly (no job).

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_import_folder_webcam.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_import_folder_webcam.py -v`
Expected: FAIL (import endpoints not defined).

- [ ] **Step 3: Implement `imports.py` and the two endpoints**

`packages/api/visionsuite_api/imports.py`:
```python
from pathlib import Path

from sqlmodel import Session, select

from visionsuite_core import workspace
from visionsuite_core.backends import RunEvent, RunStatus
from visionsuite_core.ingest import iter_folder_images, save_image_bytes

from .db import Image


def _dirs(ds_id: int):
    d = workspace.dataset_dir(str(ds_id))
    images, thumbs = d / "images", d / "thumbs"
    images.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    return images, thumbs


def save_and_record(engine, ds_id: int, data: bytes, source: str) -> dict:
    images, thumbs = _dirs(ds_id)
    rec = save_image_bytes(data, images, thumbs, source=source)
    root = workspace.workspace_root()
    with Session(engine) as s:
        exists = s.exec(select(Image).where(Image.dataset_id == ds_id, Image.image_id == rec.image_id)).first()
        if exists is None:
            s.add(Image(
                dataset_id=ds_id, image_id=rec.image_id, filename=rec.filename,
                width=rec.width, height=rec.height, source=source,
                path=str((images / rec.filename).relative_to(root)),
                thumb_path=str((thumbs / f"{rec.image_id}.webp").relative_to(root)),
            ))
            s.commit()
    return {"image_id": rec.image_id, "filename": rec.filename, "width": rec.width,
            "height": rec.height, "source": source}


async def folder_producer(engine, ds_id: int, folder: str):
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    paths = list(iter_folder_images(Path(folder)))
    total = len(paths)
    yield RunEvent(type="log", message=f"importing {total} images from {folder}")
    for i, p in enumerate(paths, 1):
        save_and_record(engine, ds_id, p.read_bytes(), source="folder")
        yield RunEvent(type="progress", progress=i / total if total else 1.0)
    yield RunEvent(type="status", status=RunStatus.DONE)
```

In `routes/datasets.py`, add:
```python
from uuid import uuid4

from fastapi import UploadFile

from ..imports import folder_producer, save_and_record


class FolderImport(BaseModel):
    path: str


@router.post("/api/datasets/{ds_id}/import/folder")
async def import_folder(request: Request, ds_id: int, body: FolderImport) -> dict:
    from pathlib import Path as _P
    if not _P(body.path).is_dir():
        raise HTTPException(400, f"not a folder: {body.path}")
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import", lambda: folder_producer(engine, ds_id, body.path))
    return {"job_id": job_id}


@router.post("/api/datasets/{ds_id}/import/webcam")
async def import_webcam(request: Request, ds_id: int, file: UploadFile) -> dict:
    data = await file.read()
    return save_and_record(request.app.state.engine, ds_id, data, source="webcam")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): folder (job) + webcam import endpoints"
```

---

### Task 7: HF + video import endpoints

**Files:**
- Modify: `packages/api/visionsuite_api/imports.py` (add `hf_producer`, `video_producer`)
- Modify: `packages/api/visionsuite_api/routes/datasets.py` (add HF + video endpoints)
- Test: `packages/api/tests/test_import_hf_video.py`

**Interfaces:**
- Consumes: `visionsuite_core.ingest` (`extract_video_frames`, `iter_hf_images`); `imports.save_and_record`.
- Produces:
  - `imports.hf_producer(engine, ds_id, dataset_id, split, config, image_column, limit)` async generator (progress by count; `limit` caps how many to pull).
  - `imports.video_producer(engine, ds_id, video_path, every_n)` async generator.
  - `POST /api/datasets/{ds_id}/import/hf` `{dataset_id, split?, config?, image_column?, limit?}` → `{job_id}`.
  - `POST /api/datasets/{ds_id}/import/video` (multipart `file` + form `every_n`) → `{job_id}`; the upload is written to a temp file the producer reads.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_import_hf_video.py`:
```python
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
        for _ in range(4):
            b = io.BytesIO(); PILImage.new("RGB", (12, 12)).save(b, format="PNG"); yield b.getvalue()
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_import_hf_video.py -v`
Expected: FAIL (HF/video endpoints not defined).

- [ ] **Step 3: Implement the producers and endpoints**

Append to `imports.py`:
```python
import tempfile
from pathlib import Path as _Path

from visionsuite_core.ingest import extract_video_frames, iter_hf_images


async def hf_producer(engine, ds_id, dataset_id, split, config, image_column, limit):
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    yield RunEvent(type="log", message=f"streaming {dataset_id} [{split}]")
    n = 0
    for data in iter_hf_images(dataset_id, split=split, config=config, image_column=image_column):
        save_and_record(engine, ds_id, data, source="hf")
        n += 1
        yield RunEvent(type="progress", progress=(n / limit) if limit else None, message=f"{n} images")
        if limit and n >= limit:
            break
    yield RunEvent(type="status", status=RunStatus.DONE)


async def video_producer(engine, ds_id, video_path, every_n):
    yield RunEvent(type="status", status=RunStatus.RUNNING)
    n = 0
    for data in extract_video_frames(_Path(video_path), every_n=every_n):
        save_and_record(engine, ds_id, data, source="video")
        n += 1
        yield RunEvent(type="progress", message=f"{n} frames")
    yield RunEvent(type="log", message=f"extracted {n} frames")
    yield RunEvent(type="status", status=RunStatus.DONE)


def save_upload_tempfile(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    _Path(path).write_bytes(data)
    return path
```

Append to `routes/datasets.py`:
```python
from fastapi import Form
from ..imports import hf_producer, video_producer, save_upload_tempfile


class HFImport(BaseModel):
    dataset_id: str
    split: str = "train"
    config: str | None = None
    image_column: str | None = None
    limit: int | None = 200


@router.post("/api/datasets/{ds_id}/import/hf")
async def import_hf(request: Request, ds_id: int, body: HFImport) -> dict:
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import",
        lambda: hf_producer(engine, ds_id, body.dataset_id, body.split, body.config,
                            body.image_column, body.limit))
    return {"job_id": job_id}


@router.post("/api/datasets/{ds_id}/import/video")
async def import_video(request: Request, ds_id: int, file: UploadFile, every_n: int = Form(30)) -> dict:
    data = await file.read()
    path = save_upload_tempfile(data, suffix=".mp4")
    engine = request.app.state.engine
    job_id = uuid4().hex
    await request.app.state.manager.submit_stream(
        job_id, "import", lambda: video_producer(engine, ds_id, path, every_n))
    return {"job_id": job_id}
```
Note: the endpoint references `imports.iter_hf_images` indirectly — the HF test monkeypatches `imports.iter_hf_images`, so `hf_producer` must call the module-level `iter_hf_images` name imported into `imports.py` (it does). Keep that import at module scope.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): HF + video import endpoints"
```

---

### Task 8: Frontend Datasets page — list, create, grid, delete

**Files:**
- Modify: `web/src/lib/api.ts` (dataset + image client functions)
- Create: `web/src/routes/Datasets.tsx` (replaces the stub)
- Test: `web/src/routes/Datasets.test.tsx`

**Interfaces:**
- Consumes: the Task 5 endpoints.
- Produces in `api.ts`: `listDatasets()`, `createDataset(name, task)`, `deleteDataset(id)`, `listImages(dsId, offset?, limit?)` returning `{total, images}`; image `thumb_url`/`file_url` are absolute-from-root paths usable directly as `<img src>` (prefix with `VITE_API_BASE` if set — add a `apiUrl(path)` helper).

- [ ] **Step 1: Write the failing test**

`web/src/routes/Datasets.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Datasets from "./Datasets";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("Datasets", () => {
  it("lists datasets and creates one", async () => {
    vi.spyOn(api, "listDatasets").mockResolvedValue([
      { id: 1, name: "cars", task: "detection", image_count: 5 },
    ]);
    const create = vi.spyOn(api, "createDataset").mockResolvedValue({
      id: 2, name: "pets", task: "classification", image_count: 0,
    });
    render(<Datasets />);
    await waitFor(() => expect(screen.getByText(/cars/)).toBeTruthy());
    fireEvent.change(screen.getByPlaceholderText(/dataset name/i), { target: { value: "pets" } });
    fireEvent.click(screen.getByRole("button", { name: /create/i }));
    await waitFor(() => expect(create).toHaveBeenCalledWith("pets", expect.any(String)));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/routes/Datasets.test.tsx`
Expected: FAIL (Datasets exports only the stub).

- [ ] **Step 3: Implement api client additions + the page**

Append to `web/src/lib/api.ts`:
```ts
export function apiUrl(path: string): string {
  return `${BASE}${path}`;
}

export interface DatasetInfo { id: number; name: string; task: string; image_count: number; }
export interface ImageInfo {
  image_id: string; filename: string; width: number; height: number; source: string;
  thumb_url: string; file_url: string;
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return (await fetch(apiUrl("/api/datasets"))).json();
}
export async function createDataset(name: string, task: string): Promise<DatasetInfo> {
  return (await fetch(apiUrl("/api/datasets"), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, task }),
  })).json();
}
export async function deleteDataset(id: number): Promise<void> {
  await fetch(apiUrl(`/api/datasets/${id}`), { method: "DELETE" });
}
export async function listImages(dsId: number, offset = 0, limit = 60):
  Promise<{ total: number; images: ImageInfo[] }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/images?offset=${offset}&limit=${limit}`))).json();
}
```

`web/src/routes/Datasets.tsx`:
```tsx
import { useEffect, useState } from "react";
import { apiUrl, createDataset, deleteDataset, listDatasets, listImages,
  type DatasetInfo, type ImageInfo } from "../lib/api";

export default function Datasets() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [name, setName] = useState("");
  const [task, setTask] = useState("detection");
  const [selected, setSelected] = useState<number | null>(null);
  const [images, setImages] = useState<ImageInfo[]>([]);

  const refresh = () => listDatasets().then(setDatasets);
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (selected != null) listImages(selected).then((r) => setImages(r.images));
  }, [selected]);

  async function onCreate() {
    if (!name) return;
    await createDataset(name, task);
    setName(""); refresh();
  }

  return (
    <div>
      <h1>Datasets</h1>
      <div>
        <input placeholder="dataset name" value={name} onChange={(e) => setName(e.target.value)} />
        <select value={task} onChange={(e) => setTask(e.target.value)}>
          <option value="detection">detection</option>
          <option value="classification">classification</option>
        </select>
        <button onClick={onCreate}>Create</button>
      </div>
      <ul>
        {datasets.map((d) => (
          <li key={d.id}>
            <button onClick={() => setSelected(d.id)}>{d.name}</button>
            {" "}({d.task}, {d.image_count} images){" "}
            <button onClick={() => deleteDataset(d.id).then(refresh)}>delete</button>
          </li>
        ))}
      </ul>
      {selected != null && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {images.map((im) => (
            <img key={im.image_id} src={apiUrl(im.thumb_url)} width={96} height={96}
                 alt={im.filename} style={{ objectFit: "cover" }} />
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes + build**

Run: `cd web && npx vitest run src/routes/Datasets.test.tsx && npm run build`
Expected: PASS + build succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web): Datasets page with create/list/grid/delete"
```

---

### Task 9: Frontend import UIs (folder / HF / video) + shared `useJobStream`

**Files:**
- Create: `web/src/lib/useJobStream.ts`
- Modify: `web/src/lib/api.ts` (import launchers)
- Modify: `web/src/routes/Datasets.tsx` (import panel on the selected dataset)
- Test: `web/src/lib/useJobStream.test.tsx`

**Interfaces:**
- Produces:
  - `useJobStream()` hook → `{ logs: string[], progress: number, status: string, watch(jobId: string) }`; opens `WS apiUrl('/api/jobs/{jobId}/events')` (ws-scheme) and updates on messages.
  - `api.importFolder(dsId, path)`, `api.importHf(dsId, {dataset_id, split?, limit?})` → `{job_id}`; `api.importVideo(dsId, file, every_n)` (multipart).

- [ ] **Step 1: Write the failing test**

`web/src/lib/useJobStream.test.tsx`:
```tsx
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useJobStream } from "./useJobStream";

class FakeWS {
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(_url: string) {
    setTimeout(() => {
      this.onmessage?.({ data: JSON.stringify({ type: "progress", progress: 1, status: null, message: "" }) });
      this.onmessage?.({ data: JSON.stringify({ type: "status", progress: null, status: "done", message: "" }) });
      this.onclose?.();
    }, 0);
  }
  close() {}
}

afterEach(() => vi.restoreAllMocks());

describe("useJobStream", () => {
  it("tracks progress to done", async () => {
    vi.stubGlobal("WebSocket", FakeWS as unknown as typeof WebSocket);
    const { result } = renderHook(() => useJobStream());
    act(() => result.current.watch("j1"));
    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.progress).toBe(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/lib/useJobStream.test.tsx`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the hook, launchers, and panel**

`web/src/lib/useJobStream.ts`:
```ts
import { useRef, useState } from "react";
import { apiUrl } from "./api";

export function useJobStream() {
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef<WebSocket | null>(null);

  function watch(jobId: string) {
    setLogs([]); setProgress(0); setStatus("running");
    const base = apiUrl(`/api/jobs/${jobId}/events`);
    const url = (base.startsWith("http") ? base : window.location.origin + base).replace(/^http/, "ws");
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "log" && ev.message) setLogs((p) => [...p, ev.message]);
      if (ev.type === "progress" && ev.progress != null) setProgress(ev.progress);
      if (ev.type === "status" && ev.status) setStatus(ev.status);
    };
  }
  return { logs, progress, status, watch };
}
```

Append to `api.ts`:
```ts
export async function importFolder(dsId: number, path: string): Promise<{ job_id: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/folder`), {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }),
  })).json();
}
export async function importHf(dsId: number, dataset_id: string, limit = 200): Promise<{ job_id: string }> {
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/hf`), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id, limit }),
  })).json();
}
export async function importVideo(dsId: number, file: File, everyN = 30): Promise<{ job_id: string }> {
  const fd = new FormData(); fd.append("file", file); fd.append("every_n", String(everyN));
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/video`), { method: "POST", body: fd })).json();
}
```

In `Datasets.tsx`, add an import panel shown when a dataset is selected (wire to `useJobStream` and refresh the grid when `status === "done"`):
```tsx
// inside the component, after the images grid:
//   const job = useJobStream();
//   useEffect(() => { if (job.status === "done" && selected != null)
//       listImages(selected).then((r) => setImages(r.images)); }, [job.status]);
// Controls: a text input for folder path -> importFolder(selected, path).then(r => job.watch(r.job_id));
//           a text input for HF id -> importHf(...).then(watch);
//           a file input for video -> importVideo(...).then(watch);
//           render job.status + <progress value={job.progress} max={1} />
```
Implement those controls concretely with the handlers above; keep the existing create/list/grid intact.

- [ ] **Step 4: Run test to verify it passes + build**

Run: `cd web && npx vitest run && npm run build`
Expected: PASS + build succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web): useJobStream hook + folder/HF/video import UIs"
```

---

### Task 10: Frontend webcam capture

**Files:**
- Create: `web/src/routes/WebcamCapture.tsx`
- Modify: `web/src/routes/Datasets.tsx` (mount the capture panel for the selected dataset)
- Modify: `web/src/lib/api.ts` (`importWebcam`)
- Test: `web/src/routes/WebcamCapture.test.tsx`

**Interfaces:**
- Produces: `api.importWebcam(dsId, blob)` (multipart) → the saved image record; `WebcamCapture({ dsId, onCaptured })` — requests `getUserMedia`, shows a `<video>` preview, and a Capture button that grabs a frame from a `<canvas>`, POSTs it, and calls `onCaptured`.

- [ ] **Step 1: Write the failing test**

`web/src/routes/WebcamCapture.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import WebcamCapture from "./WebcamCapture";
import * as api from "../lib/api";

afterEach(() => vi.restoreAllMocks());

describe("WebcamCapture", () => {
  it("captures a frame and posts it", async () => {
    vi.stubGlobal("navigator", {
      mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }) },
    });
    // canvas.toBlob → provide a blob synchronously
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({ drawImage: vi.fn() });
    HTMLCanvasElement.prototype.toBlob = function (cb: BlobCallback) { cb(new Blob(["x"])); };
    const post = vi.spyOn(api, "importWebcam").mockResolvedValue({
      image_id: "z", filename: "z.png", width: 1, height: 1, source: "webcam",
    });
    const onCaptured = vi.fn();
    render(<WebcamCapture dsId={1} onCaptured={onCaptured} />);
    fireEvent.click(await screen.findByRole("button", { name: /capture/i }));
    await waitFor(() => expect(post).toHaveBeenCalled());
    await waitFor(() => expect(onCaptured).toHaveBeenCalled());
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/routes/WebcamCapture.test.tsx`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `importWebcam`, the component, and mount it**

Append to `api.ts`:
```ts
export async function importWebcam(dsId: number, blob: Blob): Promise<ImageInfo> {
  const fd = new FormData(); fd.append("file", blob, "frame.png");
  return (await fetch(apiUrl(`/api/datasets/${dsId}/import/webcam`), { method: "POST", body: fd })).json();
}
```

`web/src/routes/WebcamCapture.tsx`:
```tsx
import { useEffect, useRef } from "react";
import { importWebcam } from "../lib/api";

export default function WebcamCapture({ dsId, onCaptured }: { dsId: number; onCaptured: () => void }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    navigator.mediaDevices.getUserMedia({ video: true }).then((s) => {
      stream = s;
      if (videoRef.current) { videoRef.current.srcObject = s; videoRef.current.play?.(); }
    }).catch(() => {});
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, []);

  async function capture() {
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video?.videoWidth || 320;
    canvas.height = video?.videoHeight || 240;
    canvas.getContext("2d")?.drawImage(video as CanvasImageSource, 0, 0, canvas.width, canvas.height);
    await new Promise<void>((resolve) =>
      canvas.toBlob(async (blob) => {
        if (blob) { await importWebcam(dsId, blob); onCaptured(); }
        resolve();
      }, "image/png"));
  }

  return (
    <div>
      <video ref={videoRef} width={320} height={240} muted playsInline />
      <div><button onClick={capture}>Capture</button></div>
    </div>
  );
}
```

In `Datasets.tsx`, render `<WebcamCapture dsId={selected} onCaptured={() => listImages(selected).then(r => setImages(r.images))} />` when a dataset is selected.

- [ ] **Step 4: Run test to verify it passes + build + full backend suite**

Run: `cd web && npx vitest run && npm run build`
Then: `cd .. && uv run pytest -q`
Expected: all frontend + backend tests PASS; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web): webcam capture panel"
```

---

## Self-Review

**1. Spec coverage (spec §2–§6):**
- Dataset create/list/delete + image delete → Task 5. ✅
- 4 sources: folder + webcam → Task 6; HF + video → Task 7; each a background job (webcam is synchronous per spec) → ✅
- Thumbnails on import → Task 2 (`save_image_bytes`). ✅
- Generalized job infra + WS → Task 4. ✅
- Browse (paginated grid) → Task 5 (API) + Task 8 (UI). ✅
- Import UIs + webcam UI → Tasks 9, 10. ✅
- Relative paths / workspace root → Tasks 5, 6 (store `relative_to(root)`). ✅
- New deps installed, ML stack untouched → Task 1. ✅
- Images-only (no HF labels) → respected; nothing imports labels. ✅

**2. Placeholder scan:** Task 9 Step 3's import-panel controls are described as handlers to implement concretely (with the exact api calls given) rather than full JSX — the implementer has every function signature and the wiring pattern; this is the one spot to watch. All other steps carry complete code. `iter_hf_images` monkeypatch note is explicit.

**3. Type consistency:** `RunEvent`/`RunStatus` reused across producers; `save_and_record(engine, ds_id, data, source)`, `submit_stream(job_id, kind, producer)`, `IngestedImage`, `Image` fields, and the api client names (`listDatasets`/`createDataset`/`listImages`/`import*`/`useJobStream`/`apiUrl`) are consistent between the task that defines them and the tasks that use them.

## Notes for the executor
- Delete `workspace/` before running (schema changed): the dev DB is recreated by `init_db`.
- The HF import test monkeypatches `visionsuite_api.imports.iter_hf_images` — so `imports.py` must import that name at module scope and `hf_producer` must call it unqualified. Do not import it inside the function.
- Video tests depend on `imageio-ffmpeg`. If `libx264` isn't in the bundled ffmpeg, switch the TEST's `codec` to `"mpeg4"`; keep production `extract_video_frames` codec-agnostic (it only reads).
- Run backend tests from repo root (`uv run pytest`), frontend from `web/` (`npx vitest run`).
