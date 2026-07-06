# VisionSuite Sub-Project 0: Foundation (Walking Skeleton) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the VisionSuite monorepo as an end-to-end walking skeleton — React dashboard → FastAPI → job manager → core `TrainingBackend` → SQLite/WebSocket — that runs a *dummy* (zero-ML) training run and streams its logs/progress live to the browser.

**Architecture:** A `uv` workspace with two Python packages — `visionsuite_core` (pure Python, no web deps: model/dataset/training-backend contracts) and `visionsuite_api` (thin FastAPI over the core: DB, job manager, REST + WebSocket) — plus a React+Vite frontend. Everything reads/writes one on-disk `workspace/`. Later sub-projects swap each stub for a real implementation without touching the wiring.

**Tech Stack:** Python 3.11+, `uv`, FastAPI, SQLModel (SQLite), pytest + pytest-asyncio + httpx; React 18 + Vite + TypeScript, Vitest + Testing Library.

## Global Constraints

_Project-wide requirements. Every task implicitly includes these. Values copied verbatim from the [design spec](../specs/2026-07-06-visionsuite-design.md) §4._

- **Python:** `requires-python = ">=3.11"`.
- **ML deps are NOT installed in Sub-project 0.** They are added in Sub-project 3. Record (do not install) these locked pins now: `torch>=2.11`, `transformers>=4.54`, `label-studio-sdk>=2,<3`, `trackio==0.29.0`, `optimum[onnxruntime]`, `onnxruntime`.
- **Training env (future, documented now):** `PYTORCH_ENABLE_MPS_FALLBACK=1` set in the training subprocess; bf16 (never fp16); `torch.compile` OFF. Not exercised in #0.
- **Core purity:** `visionsuite_core` must import no web framework (no fastapi/starlette/uvicorn). Enforced by a test in Task 3.
- **Single-user:** no auth, one training run at a time.
- **Workspace root:** resolved from `VISIONSUITE_WORKSPACE` env, default `./workspace`. Never hard-code absolute paths.
- **Commits:** conventional-commit messages; every commit message ends with the trailer line `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Branch:** do all work on `feat/subproject-0-foundation` (the executing skill creates it / a worktree before Task 1).

---

### Task 1: Monorepo + tooling scaffold

**Files:**
- Create: `pyproject.toml` (virtual `uv` workspace root)
- Create: `packages/core/pyproject.toml`, `packages/core/visionsuite_core/__init__.py`
- Create: `packages/api/pyproject.toml`, `packages/api/visionsuite_api/__init__.py`, `packages/api/visionsuite_api/routes/__init__.py`
- Test: `packages/core/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installed `uv` env where `visionsuite_core` and `visionsuite_api` import cleanly and `uv run pytest` works from the repo root.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_smoke.py`:
```python
def test_core_imports():
    import visionsuite_core

    assert visionsuite_core.__version__ == "0.0.0"
```

- [ ] **Step 2: Create the workspace + package files**

`pyproject.toml` (repo root — virtual workspace, not a package):
```toml
[tool.uv.workspace]
members = ["packages/core", "packages/api"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["packages"]
```

`packages/core/pyproject.toml`:
```toml
[project]
name = "visionsuite-core"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["visionsuite_core"]
```

`packages/core/visionsuite_core/__init__.py`:
```python
__version__ = "0.0.0"
```

`packages/api/pyproject.toml`:
```toml
[project]
name = "visionsuite-api"
version = "0.0.0"
requires-python = ">=3.11"
dependencies = [
    "visionsuite-core",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlmodel>=0.0.22",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["visionsuite_api"]

[tool.uv.sources]
visionsuite-core = { workspace = true }

[dependency-groups]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "httpx>=0.27"]
```

`packages/api/visionsuite_api/__init__.py` and `packages/api/visionsuite_api/routes/__init__.py`: empty files.

- [ ] **Step 3: Sync the environment**

Run: `uv sync --all-packages`
Expected: creates `.venv`, resolves `visionsuite-core` + `visionsuite-api` (workspace) + dev group. No torch/transformers pulled.

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/core/tests/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold uv workspace with core + api packages"
```

---

### Task 2: Workspace path resolution (`workspace.py`)

**Files:**
- Create: `packages/core/visionsuite_core/workspace.py`
- Test: `packages/core/tests/test_workspace.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `workspace_root() -> pathlib.Path` — `VISIONSUITE_WORKSPACE` or `./workspace`.
  - `ensure_workspace() -> pathlib.Path` — creates root + `datasets/ runs/ models/ exports/`, returns root.
  - `dataset_dir(dataset_id: str) -> Path`, `run_dir(run_id: str) -> Path`, `model_dir(model_id: str) -> Path`, `export_dir(model_id: str) -> Path` — each creates and returns its subdir.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_workspace.py`:
```python
from visionsuite_core import workspace


def test_root_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    assert workspace.workspace_root() == tmp_path / "ws"


def test_ensure_creates_subdirs(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    root = workspace.ensure_workspace()
    for sub in ("datasets", "runs", "models", "exports"):
        assert (root / sub).is_dir()


def test_run_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("VISIONSUITE_WORKSPACE", str(tmp_path / "ws"))
    d = workspace.run_dir("abc123")
    assert d.is_dir() and d.name == "abc123" and d.parent.name == "runs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_workspace.py -v`
Expected: FAIL (ModuleNotFoundError: visionsuite_core.workspace).

- [ ] **Step 3: Implement `workspace.py`**

```python
import os
from pathlib import Path

_SUBDIRS = ("datasets", "runs", "models", "exports")


def workspace_root() -> Path:
    return Path(os.environ.get("VISIONSUITE_WORKSPACE", "./workspace")).resolve()


def ensure_workspace() -> Path:
    root = workspace_root()
    for sub in _SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def _child(sub: str, name: str) -> Path:
    d = ensure_workspace() / sub / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_dir(dataset_id: str) -> Path:
    return _child("datasets", dataset_id)


def run_dir(run_id: str) -> Path:
    return _child("runs", run_id)


def model_dir(model_id: str) -> Path:
    return _child("models", model_id)


def export_dir(model_id: str) -> Path:
    return _child("exports", model_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_workspace.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): workspace path resolution"
```

---

### Task 3: Core contracts — types, dataset, registry

**Files:**
- Create: `packages/core/visionsuite_core/types.py`
- Create: `packages/core/visionsuite_core/dataset.py`
- Create: `packages/core/visionsuite_core/registry.py`
- Test: `packages/core/tests/test_registry.py`, `packages/core/tests/test_dataset.py`, `packages/core/tests/test_core_purity.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `types.VisionTask` (str Enum): `DETECTION="detection"`, `CLASSIFICATION="classification"`.
  - `types.CompatVerdict` (str Enum): `KNOWN_GOOD_MPS="known_good_mps"`, `UNTESTED="untested"`, `UNSUPPORTED="unsupported"`.
  - `dataset.ImageRecord` (dataclass): `image_id: str`, `path: str`, `width: int`, `height: int`.
  - `dataset.Dataset` (dataclass): `dataset_id: str`, `task: VisionTask`, `class_names: list[str]`, `images: list[ImageRecord]`. Methods `to_coco() -> dict` and classmethods `from_coco(dict)`, `from_label_studio(list)` — the last two raise `NotImplementedError("lands in Sub-project 1/2")` in #0.
  - `registry.ModelAdapter` (runtime-checkable Protocol): attrs `hf_id: str`, `task: VisionTask`; method `mps_compat() -> CompatVerdict`. (Full adapter surface — `load/build_processor/train_config/evaluate/export` — is added in Sub-project 3.)
  - `registry.DummyAdapter` implementing that Protocol (`hf_id="dummy/echo"`, `task=CLASSIFICATION`, `mps_compat()->KNOWN_GOOD_MPS`).
  - `registry.ModelRegistry` with `register(adapter)`, `get(hf_id)->ModelAdapter`, `list()->list[ModelAdapter]`.
  - `registry.default_registry` — a `ModelRegistry` pre-registered with `DummyAdapter()`.

- [ ] **Step 1: Write the failing tests**

`packages/core/tests/test_registry.py`:
```python
from visionsuite_core.registry import ModelAdapter, DummyAdapter, ModelRegistry, default_registry
from visionsuite_core.types import CompatVerdict, VisionTask


def test_dummy_is_adapter():
    assert isinstance(DummyAdapter(), ModelAdapter)


def test_dummy_verdict():
    assert DummyAdapter().mps_compat() == CompatVerdict.KNOWN_GOOD_MPS
    assert DummyAdapter().task == VisionTask.CLASSIFICATION


def test_registry_register_and_get():
    reg = ModelRegistry()
    reg.register(DummyAdapter())
    assert reg.get("dummy/echo").hf_id == "dummy/echo"
    assert [a.hf_id for a in reg.list()] == ["dummy/echo"]


def test_default_registry_has_dummy():
    assert default_registry.get("dummy/echo") is not None
```

`packages/core/tests/test_dataset.py`:
```python
import pytest
from visionsuite_core.dataset import Dataset, ImageRecord
from visionsuite_core.types import VisionTask


def _sample() -> Dataset:
    return Dataset(
        dataset_id="d1",
        task=VisionTask.CLASSIFICATION,
        class_names=["cat", "dog"],
        images=[ImageRecord(image_id="i1", path="i1.jpg", width=64, height=48)],
    )


def test_to_coco_shape():
    coco = _sample().to_coco()
    assert {"images", "categories", "annotations"} <= coco.keys()
    assert coco["images"][0]["file_name"] == "i1.jpg"
    assert [c["name"] for c in coco["categories"]] == ["cat", "dog"]


def test_from_coco_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        Dataset.from_coco({})


def test_from_label_studio_not_yet_implemented():
    with pytest.raises(NotImplementedError):
        Dataset.from_label_studio([])
```

`packages/core/tests/test_core_purity.py`:
```python
import importlib
import pkgutil
import sys

import visionsuite_core


def test_core_imports_no_web_framework():
    for mod in pkgutil.walk_packages(visionsuite_core.__path__, "visionsuite_core."):
        importlib.import_module(mod.name)
    forbidden = {"fastapi", "starlette", "uvicorn"}
    assert forbidden.isdisjoint(sys.modules.keys())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/core/tests/test_registry.py packages/core/tests/test_dataset.py -v`
Expected: FAIL (ModuleNotFoundError for `types`/`dataset`/`registry`).

- [ ] **Step 3: Implement the three modules**

`packages/core/visionsuite_core/types.py`:
```python
from enum import Enum


class VisionTask(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"


class CompatVerdict(str, Enum):
    KNOWN_GOOD_MPS = "known_good_mps"
    UNTESTED = "untested"
    UNSUPPORTED = "unsupported"
```

`packages/core/visionsuite_core/dataset.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field

from .types import VisionTask


@dataclass
class ImageRecord:
    image_id: str
    path: str
    width: int
    height: int


@dataclass
class Dataset:
    dataset_id: str
    task: VisionTask
    class_names: list[str] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)

    def to_coco(self) -> dict:
        return {
            "images": [
                {"id": i, "file_name": im.path, "width": im.width, "height": im.height}
                for i, im in enumerate(self.images)
            ],
            "categories": [
                {"id": i, "name": name} for i, name in enumerate(self.class_names)
            ],
            "annotations": [],
        }

    @classmethod
    def from_coco(cls, coco: dict) -> "Dataset":
        raise NotImplementedError("lands in Sub-project 1")

    @classmethod
    def from_label_studio(cls, tasks: list) -> "Dataset":
        raise NotImplementedError("lands in Sub-project 2")
```

`packages/core/visionsuite_core/registry.py`:
```python
from typing import Protocol, runtime_checkable

from .types import CompatVerdict, VisionTask


@runtime_checkable
class ModelAdapter(Protocol):
    hf_id: str
    task: VisionTask

    def mps_compat(self) -> CompatVerdict: ...


class DummyAdapter:
    hf_id = "dummy/echo"
    task = VisionTask.CLASSIFICATION

    def mps_compat(self) -> CompatVerdict:
        return CompatVerdict.KNOWN_GOOD_MPS


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.hf_id] = adapter

    def get(self, hf_id: str) -> ModelAdapter:
        return self._adapters[hf_id]

    def list(self) -> list[ModelAdapter]:
        return list(self._adapters.values())


default_registry = ModelRegistry()
default_registry.register(DummyAdapter())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/core -v`
Expected: PASS (all core tests, including purity).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): task/verdict enums, Dataset contract, model registry"
```

---

### Task 4: Training backend contract + `LocalBackend` stub

**Files:**
- Create: `packages/core/visionsuite_core/backends.py`
- Test: `packages/core/tests/test_backends.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RunStatus` (str Enum): `PENDING`, `RUNNING`, `DONE`, `FAILED`, `CANCELLED`.
  - `RunSpec` (dataclass): `run_id: str`, `model_id: str`, `dataset_id: str`, `hyperparams: dict = {}`.
  - `RunEvent` (dataclass): `type: str` (`"log"|"progress"|"status"`), `message: str = ""`, `progress: float | None = None`, `status: RunStatus | None = None`.
  - `TrainingBackend` (Protocol): `stream(spec: RunSpec) -> AsyncIterator[RunEvent]`.
  - `LocalBackend` implementing it — emits a RUNNING status, 5 log+progress steps, then DONE.

- [ ] **Step 1: Write the failing test**

`packages/core/tests/test_backends.py`:
```python
from visionsuite_core.backends import LocalBackend, RunSpec, RunStatus


async def test_local_backend_streams_to_done():
    spec = RunSpec(run_id="r1", model_id="dummy/echo", dataset_id="d1")
    events = [e async for e in LocalBackend().stream(spec)]
    assert events[0].type == "status" and events[0].status == RunStatus.RUNNING
    assert events[-1].status == RunStatus.DONE
    progresses = [e.progress for e in events if e.type == "progress"]
    assert progresses[-1] == 1.0
    assert any(e.type == "log" and "r1" in e.message for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/core/tests/test_backends.py -v`
Expected: FAIL (ModuleNotFoundError: visionsuite_core.backends).

- [ ] **Step 3: Implement `backends.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Protocol


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RunSpec:
    run_id: str
    model_id: str
    dataset_id: str
    hyperparams: dict = field(default_factory=dict)


@dataclass
class RunEvent:
    type: str
    message: str = ""
    progress: float | None = None
    status: RunStatus | None = None


class TrainingBackend(Protocol):
    def stream(self, spec: RunSpec) -> AsyncIterator[RunEvent]: ...


class LocalBackend:
    steps = 5

    async def stream(self, spec: RunSpec) -> AsyncIterator[RunEvent]:
        yield RunEvent(type="status", status=RunStatus.RUNNING)
        for i in range(1, self.steps + 1):
            yield RunEvent(type="log", message=f"[{spec.run_id}] step {i}/{self.steps}")
            yield RunEvent(type="progress", progress=i / self.steps)
        yield RunEvent(type="status", status=RunStatus.DONE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/core/tests/test_backends.py -v`
Expected: PASS (asyncio_mode=auto runs the async test).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(core): TrainingBackend contract + LocalBackend stub"
```

---

### Task 5: API database layer (`db.py`)

**Files:**
- Create: `packages/api/visionsuite_api/db.py`
- Test: `packages/api/tests/test_db.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - SQLModel tables: `Project(id, name)`, `Dataset(id, name, task, project_id)`, `Image(id, dataset_id, path)`, `Run(id: str PK, model_id, dataset_id, status="pending")`, `Model(id, run_id, path)`.
  - `make_engine(url: str = "sqlite:///./workspace/db.sqlite")` — returns a SQLAlchemy engine (with `check_same_thread=False`).
  - `init_db(engine)` — creates all tables.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_db.py`:
```python
from sqlmodel import Session

from visionsuite_api.db import Run, init_db, make_engine


def test_run_roundtrip():
    engine = make_engine("sqlite://")  # in-memory
    init_db(engine)
    with Session(engine) as s:
        s.add(Run(id="r1", model_id="dummy/echo", dataset_id="d1"))
        s.commit()
    with Session(engine) as s:
        got = s.get(Run, "r1")
    assert got is not None and got.status == "pending"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_db.py -v`
Expected: FAIL (ModuleNotFoundError: visionsuite_api.db).

- [ ] **Step 3: Implement `db.py`**

```python
from sqlmodel import Field, SQLModel, create_engine


class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    task: str
    project_id: int | None = Field(default=None, foreign_key="project.id")


class Image(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id")
    path: str


class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    model_id: str
    dataset_id: str
    status: str = "pending"


class Model(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    path: str


def make_engine(url: str = "sqlite:///./workspace/db.sqlite"):
    return create_engine(url, connect_args={"check_same_thread": False})


def init_db(engine) -> None:
    SQLModel.metadata.create_all(engine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api/tests/test_db.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): SQLite schema via SQLModel"
```

---

### Task 6: API core — app factory, health route, job manager

**Files:**
- Create: `packages/api/visionsuite_api/jobs.py`
- Create: `packages/api/visionsuite_api/routes/health.py`
- Create: `packages/api/visionsuite_api/main.py`
- Test: `packages/api/tests/test_jobs.py`, `packages/api/tests/test_health.py`

**Interfaces:**
- Consumes: `visionsuite_core.backends` (`LocalBackend`, `RunSpec`, `RunEvent`, `RunStatus`, `TrainingBackend`).
- Produces:
  - `jobs.Job` — attrs `spec: RunSpec`, `events: list[RunEvent]`, `status: RunStatus`.
  - `jobs.JobManager(backend: TrainingBackend | None = None)` with `async submit(spec) -> Job`, `get(run_id) -> Job | None`. Runs one job at a time (internal lock); appends each event to `job.events` and mirrors terminal status onto `job.status`.
  - `main.create_app(engine=None, manager=None) -> FastAPI` — builds the app, initializes the DB, stores `engine`/`manager` on `app.state`, includes routers. Module-level `app = create_app()`.
  - `GET /api/health -> {"status": "ok"}`.

- [ ] **Step 1: Write the failing tests**

`packages/api/tests/test_jobs.py`:
```python
import asyncio

from visionsuite_api.jobs import JobManager
from visionsuite_core.backends import RunSpec, RunStatus


async def test_job_runs_to_done():
    jm = JobManager()
    job = await jm.submit(RunSpec(run_id="r1", model_id="m", dataset_id="d"))
    for _ in range(200):
        if job.status == RunStatus.DONE:
            break
        await asyncio.sleep(0.01)
    assert job.status == RunStatus.DONE
    assert any(e.type == "progress" and e.progress == 1.0 for e in job.events)
    assert jm.get("r1") is job
```

`packages/api/tests/test_health.py`:
```python
from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def test_health_ok():
    client = TestClient(create_app(engine=make_engine("sqlite://")))
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/api/tests/test_jobs.py packages/api/tests/test_health.py -v`
Expected: FAIL (ModuleNotFoundError: visionsuite_api.jobs / .main).

- [ ] **Step 3: Implement the three modules**

`packages/api/visionsuite_api/jobs.py`:
```python
import asyncio

from visionsuite_core.backends import LocalBackend, RunEvent, RunSpec, RunStatus, TrainingBackend

TERMINAL = {RunStatus.DONE, RunStatus.FAILED, RunStatus.CANCELLED}


class Job:
    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.events: list[RunEvent] = []
        self.status: RunStatus = RunStatus.PENDING


class JobManager:
    def __init__(self, backend: TrainingBackend | None = None) -> None:
        self.backend = backend or LocalBackend()
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def submit(self, spec: RunSpec) -> Job:
        job = Job(spec)
        self._jobs[spec.run_id] = job
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: Job) -> None:
        async with self._lock:  # one run at a time
            job.status = RunStatus.RUNNING
            async for event in self.backend.stream(job.spec):
                job.events.append(event)
                if event.status is not None:
                    job.status = event.status

    def get(self, run_id: str) -> Job | None:
        return self._jobs.get(run_id)
```

`packages/api/visionsuite_api/routes/health.py`:
```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

`packages/api/visionsuite_api/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db import init_db, make_engine
from .jobs import JobManager
from .routes import health


def create_app(engine=None, manager=None) -> FastAPI:
    app = FastAPI(title="VisionSuite API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine or make_engine()
    init_db(app.state.engine)
    app.state.manager = manager or JobManager()
    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/api -v`
Expected: PASS (db + jobs + health).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): app factory, health route, one-at-a-time job manager"
```

---

### Task 7: Runs REST routes

**Files:**
- Create: `packages/api/visionsuite_api/routes/runs.py`
- Modify: `packages/api/visionsuite_api/main.py` (include the runs router)
- Test: `packages/api/tests/test_runs_rest.py`

**Interfaces:**
- Consumes: `app.state.engine`, `app.state.manager`; `visionsuite_api.db.Run`; `visionsuite_core.backends.RunSpec`.
- Produces:
  - `CreateRunRequest` (pydantic): `model_id: str = "dummy/echo"`, `dataset_id: str = "dummy"`.
  - `POST /api/runs` → inserts a `Run` row, submits a `RunSpec` to the manager, returns `{"run_id": str, "status": "pending"}`.
  - `GET /api/runs/{run_id}` → `{"run_id": str, "status": str}` or 404 if unknown.
  - `runs.router` (APIRouter).

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_runs_rest.py`:
```python
import time

from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app(engine=make_engine("sqlite://")))


def test_create_run_returns_id():
    client = _client()
    r = client.post("/api/runs", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending" and len(body["run_id"]) > 0


def test_get_run_reaches_done():
    client = _client()
    run_id = client.post("/api/runs", json={}).json()["run_id"]
    for _ in range(200):
        status = client.get(f"/api/runs/{run_id}").json()["status"]
        if status == "done":
            break
        time.sleep(0.01)
    assert status == "done"


def test_get_unknown_run_404():
    assert _client().get("/api/runs/nope").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_runs_rest.py -v`
Expected: FAIL (404 on POST — route not registered).

- [ ] **Step 3: Implement `runs.py` and register it**

`packages/api/visionsuite_api/routes/runs.py`:
```python
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session

from visionsuite_core.backends import RunSpec

from ..db import Run

router = APIRouter()


class CreateRunRequest(BaseModel):
    model_id: str = "dummy/echo"
    dataset_id: str = "dummy"


@router.post("/api/runs")
async def create_run(request: Request, body: CreateRunRequest) -> dict:
    run_id = uuid4().hex
    with Session(request.app.state.engine) as session:
        session.add(Run(id=run_id, model_id=body.model_id, dataset_id=body.dataset_id))
        session.commit()
    await request.app.state.manager.submit(
        RunSpec(run_id=run_id, model_id=body.model_id, dataset_id=body.dataset_id)
    )
    return {"run_id": run_id, "status": "pending"}


@router.get("/api/runs/{run_id}")
def get_run(request: Request, run_id: str) -> dict:
    job = request.app.state.manager.get(run_id)
    if job is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "status": job.status.value}
```

In `main.py`, add the import and registration:
```python
from .routes import health, runs
```
```python
    app.include_router(health.router)
    app.include_router(runs.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api/tests/test_runs_rest.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): create/get run REST endpoints"
```

---

### Task 8: WebSocket event stream + walking-skeleton integration test

**Files:**
- Modify: `packages/api/visionsuite_api/routes/runs.py` (add WS endpoint)
- Test: `packages/api/tests/test_runs_ws.py`

**Interfaces:**
- Consumes: `app.state.manager`; `Job.events`, `Job.status`; `visionsuite_core.backends.RunStatus`.
- Produces: `WS /api/runs/{run_id}/events` — streams each `RunEvent` as JSON `{"type","message","progress","status"}` (status serialized as its string value or null), replaying already-buffered events, then closes when the job reaches a terminal status. Unknown run → close with code 4404.

- [ ] **Step 1: Write the failing test**

`packages/api/tests/test_runs_ws.py`:
```python
from fastapi.testclient import TestClient

from visionsuite_api.db import make_engine
from visionsuite_api.main import create_app


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/api/tests/test_runs_ws.py -v`
Expected: FAIL (WS route not found → connection rejected).

- [ ] **Step 3: Add the WS endpoint to `runs.py`**

Add imports at the top of `runs.py`:
```python
import asyncio

from fastapi import WebSocket

from ..jobs import TERMINAL
```

Append the endpoint:
```python
@router.websocket("/api/runs/{run_id}/events")
async def run_events(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    job = websocket.app.state.manager.get(run_id)
    if job is None:
        await websocket.close(code=4404)
        return
    sent = 0
    while True:
        while sent < len(job.events):
            event = job.events[sent]
            sent += 1
            await websocket.send_json(
                {
                    "type": event.type,
                    "message": event.message,
                    "progress": event.progress,
                    "status": event.status.value if event.status else None,
                }
            )
        if job.status in TERMINAL:
            break
        await asyncio.sleep(0.05)
    await websocket.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest packages/api -v`
Expected: PASS (all API tests — this is the backend walking-skeleton proof).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(api): websocket run-event stream (backend walking skeleton)"
```

---

### Task 9: Frontend scaffold + typed API client

**Files:**
- Create: `web/` via Vite (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`)
- Create: `web/src/lib/api.ts`
- Test: `web/src/lib/api.test.ts`
- Modify: `web/vite.config.ts` (dev proxy + vitest config)

**Interfaces:**
- Consumes: the API from Tasks 7–8.
- Produces:
  - `api.createRun(body?: {model_id?: string; dataset_id?: string}) -> Promise<{run_id: string; status: string}>`.
  - `api.runEventsUrl(runId: string) -> string` — a `ws(s)://…/api/runs/{runId}/events` URL derived from `VITE_API_BASE` (or `window.location.origin`).

- [ ] **Step 1: Scaffold the Vite app**

Run: `npm create vite@latest web -- --template react-ts` then `cd web && npm install && npm install -D vitest @testing-library/react @testing-library/jest-dom jsdom`
Expected: `web/` created; deps installed.

- [ ] **Step 2: Write the failing test**

`web/src/lib/api.test.ts`:
```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { createRun, runEventsUrl } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("POSTs to /api/runs and returns the body", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ run_id: "r1", status: "pending" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const res = await createRun({});
    expect(res.run_id).toBe("r1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("builds a ws:// events url", () => {
    const url = runEventsUrl("r1");
    expect(url).toMatch(/^wss?:\/\/.*\/api\/runs\/r1\/events$/);
  });
});
```

- [ ] **Step 3: Configure vitest and implement the client**

Replace `web/vite.config.ts`:
```ts
/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": { target: "http://localhost:8000", ws: true } },
  },
  test: { environment: "jsdom", globals: true, setupFiles: [] },
});
```

`web/src/lib/api.ts`:
```ts
const BASE = import.meta.env.VITE_API_BASE ?? "";

export interface RunCreated {
  run_id: string;
  status: string;
}

export async function createRun(
  body: { model_id?: string; dataset_id?: string } = {},
): Promise<RunCreated> {
  const res = await fetch(`${BASE}/api/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json() as Promise<RunCreated>;
}

export function runEventsUrl(runId: string): string {
  const origin = BASE || window.location.origin;
  return origin.replace(/^http/, "ws") + `/api/runs/${runId}/events`;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/lib/api.test.ts`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(web): Vite React scaffold + typed API client"
```

---

### Task 10: Dashboard live run, stub pages, static serving, dev script

**Files:**
- Create: `web/src/routes/Dashboard.tsx`, `web/src/routes/Datasets.tsx`, `web/src/routes/Labeling.tsx`, `web/src/routes/Train.tsx`, `web/src/routes/Models.tsx`
- Modify: `web/src/App.tsx` (router + nav), `web/package.json` (add `react-router-dom`)
- Create: `web/src/routes/Dashboard.test.tsx`
- Modify: `packages/api/visionsuite_api/main.py` (serve built `web/dist` if present)
- Create: `scripts/dev.sh`, `README.md`

**Interfaces:**
- Consumes: `api.createRun`, `api.runEventsUrl`.
- Produces: a Dashboard that starts a dummy run and renders streaming logs + a progress bar to completion; four stub pages; FastAPI serving the built SPA; a one-command dev launcher.

- [ ] **Step 1: Write the failing test**

`web/src/routes/Dashboard.test.tsx`:
```tsx
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Dashboard from "./Dashboard";
import * as api from "../lib/api";

class FakeWS {
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(_url: string) {
    setTimeout(() => {
      this.onmessage?.({ data: JSON.stringify({ type: "log", message: "hi", progress: null, status: null }) });
      this.onmessage?.({ data: JSON.stringify({ type: "progress", message: "", progress: 1, status: null }) });
      this.onmessage?.({ data: JSON.stringify({ type: "status", message: "", progress: null, status: "done" }) });
      this.onclose?.();
    }, 0);
  }
  close() {}
}

afterEach(() => vi.restoreAllMocks());

describe("Dashboard", () => {
  it("starts a run and streams to done", async () => {
    vi.spyOn(api, "createRun").mockResolvedValue({ run_id: "r1", status: "pending" });
    vi.stubGlobal("WebSocket", FakeWS as unknown as typeof WebSocket);

    render(<Dashboard />);
    fireEvent.click(screen.getByRole("button", { name: /start dummy run/i }));

    await waitFor(() => expect(screen.getByText(/status: done/i)).toBeTruthy());
    expect(screen.getByText("hi")).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/routes/Dashboard.test.tsx`
Expected: FAIL (Dashboard module does not exist).

- [ ] **Step 3: Implement Dashboard, stub pages, router, static serving, dev script**

`web/src/routes/Dashboard.tsx`:
```tsx
import { useRef, useState } from "react";
import { createRun, runEventsUrl } from "../lib/api";

interface Line {
  text: string;
}

export default function Dashboard() {
  const [logs, setLogs] = useState<Line[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const wsRef = useRef<WebSocket | null>(null);

  async function start() {
    setLogs([]);
    setProgress(0);
    setStatus("pending");
    const { run_id } = await createRun({});
    const ws = new WebSocket(runEventsUrl(run_id));
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "log") setLogs((prev) => [...prev, { text: ev.message }]);
      if (ev.type === "progress" && ev.progress != null) setProgress(ev.progress);
      if (ev.type === "status" && ev.status) setStatus(ev.status);
    };
    ws.onclose = () => setStatus((s) => (s === "pending" ? "closed" : s));
  }

  return (
    <div>
      <h1>Dashboard</h1>
      <button onClick={start}>Start dummy run</button>
      <p>Status: {status}</p>
      <progress value={progress} max={1} />
      <ul>
        {logs.map((l, i) => (
          <li key={i}>{l.text}</li>
        ))}
      </ul>
    </div>
  );
}
```

`web/src/routes/Datasets.tsx` (repeat the same shape for `Labeling.tsx`, `Train.tsx`, `Models.tsx`, changing the name):
```tsx
export default function Datasets() {
  return <h1>Datasets (coming in Sub-project 1)</h1>;
}
```
`web/src/routes/Labeling.tsx`:
```tsx
export default function Labeling() {
  return <h1>Labeling (coming in Sub-project 2)</h1>;
}
```
`web/src/routes/Train.tsx`:
```tsx
export default function Train() {
  return <h1>Train (coming in Sub-project 3)</h1>;
}
```
`web/src/routes/Models.tsx`:
```tsx
export default function Models() {
  return <h1>Models (coming in Sub-project 4)</h1>;
}
```

Add the router. Run `cd web && npm install react-router-dom`, then `web/src/App.tsx`:
```tsx
import { Link, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Dashboard from "./routes/Dashboard";
import Datasets from "./routes/Datasets";
import Labeling from "./routes/Labeling";
import Models from "./routes/Models";
import Train from "./routes/Train";

export default function App() {
  return (
    <Router>
      <nav style={{ display: "flex", gap: 12 }}>
        <Link to="/">Dashboard</Link>
        <Link to="/datasets">Datasets</Link>
        <Link to="/labeling">Labeling</Link>
        <Link to="/train">Train</Link>
        <Link to="/models">Models</Link>
      </nav>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/labeling" element={<Labeling />} />
        <Route path="/train" element={<Train />} />
        <Route path="/models" element={<Models />} />
      </Routes>
    </Router>
  );
}
```

Serve the built SPA from FastAPI — append to `create_app` in `main.py`, just before `return app`:
```python
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
```

`scripts/dev.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail

uv run uvicorn visionsuite_api.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

cd web && npm run dev
```
Then: `chmod +x scripts/dev.sh`.

`README.md`:
```markdown
# VisionSuite

Local, single-user suite for training Hugging Face vision models (object detection +
image classification) on Apple Silicon. See `docs/superpowers/specs/` for the design.

## Dev quickstart

```bash
uv sync --all-packages          # Python env (core + api)
cd web && npm install && cd ..  # frontend deps
./scripts/dev.sh                # FastAPI :8000 + Vite dev server (proxies /api)
```

Open the Vite URL, click **Start dummy run**, and watch logs stream over WebSocket.

## Tests

```bash
uv run pytest            # backend
cd web && npx vitest run # frontend
```
```

- [ ] **Step 4: Run tests + verify the build**

Run: `cd web && npx vitest run && npm run build`
Expected: Dashboard test PASSES; `npm run build` produces `web/dist/`.
Then: `uv run pytest` from repo root — Expected: all backend tests still PASS.

- [ ] **Step 5: Manual walking-skeleton smoke (record result)**

Run: `./scripts/dev.sh`, open the Vite URL, click **Start dummy run**.
Expected: status goes `pending → done`, 5 log lines appear, progress bar fills to full. Note the outcome in the commit body.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(web): live dummy-run dashboard, stub pages, static serving, dev script"
```

---

## Self-Review

**1. Spec coverage (spec §6 Sub-project 0):**
- Monorepo layout (core/api/web, workspace, scripts) → Tasks 1, 10. ✅
- `ModelAdapter` + `CompatVerdict` + `Dataset` + `TrainingBackend` contracts → Tasks 3, 4. ✅
- Storage model (workspace root + SQLite tables project/dataset/image/run/model) → Tasks 2, 5. ✅
- Job manager (one run at a time) + WebSocket log stream → Tasks 6, 8. ✅
- Walking-skeleton path (POST /runs → row → LocalBackend fake events → WS → dashboard to done) → Tasks 7, 8 (backend), 10 (frontend). ✅
- Dev ergonomics (`scripts/dev.sh`, one command) → Task 10. ✅
- Dependency pins recorded but ML deps not installed → Global Constraints + Task 1. ✅
- Tests: core interfaces via dummy adapter/backend; API health + dummy-run + WS stream → Tasks 3–8; core purity → Task 3. ✅

**2. Placeholder scan:** No "TBD/handle edge cases/similar to Task N". `from_coco`/`from_label_studio` raising `NotImplementedError` are intentional, tested stubs (their real impls are scoped to Sub-projects 1/2), not plan placeholders. ✅

**3. Type consistency:** `RunSpec(run_id, model_id, dataset_id, hyperparams)`, `RunEvent(type, message, progress, status)`, `RunStatus`, `Job(spec, events, status)`, `JobManager.submit/get`, `createRun`/`runEventsUrl` are used identically everywhere they appear (Tasks 4/6/7/8/9/10). `TERMINAL` defined in `jobs.py` (Task 6), imported in `runs.py` (Task 8). ✅

## Notes for the executor
- `asyncio_mode = "auto"` (root `pyproject.toml`) is what lets the `async def test_*` functions run without a decorator — don't remove it.
- Run backend tests with `uv run pytest` from the repo root; frontend tests with `npx vitest run` inside `web/`.
- The core-purity test (Task 3) will fail if any `visionsuite_core` module ever imports fastapi/starlette/uvicorn — that's intended; keep web deps out of core.
