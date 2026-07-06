# VisionSuite Sub-Project 1: Data Pipeline & Ingestion — Design Spec

- **Date:** 2026-07-06
- **Status:** Design (authored by Claude at the user's delegation — no interview; user said "kick off, don't brainstorm with me"). Proceeding to plan + build; user may veto any decision.
- **Depends on:** Sub-project 0 (foundation, merged to `main`).
- **Parent spec:** [`2026-07-06-visionsuite-design.md`](2026-07-06-visionsuite-design.md) §5 (#1).

## 1. Goal

Get real images into VisionSuite and let the user browse them. A **dataset** is created (name + task), then **populated from one or more sources**, with images stored on disk in the internal layout and a manifest in SQLite, browsable as a paginated thumbnail grid. This is the raw material SP2 (labeling) and SP3 (training) consume.

## 2. Scope

**In:**
- **Datasets:** create (name + `VisionTask`), list (with image counts + source summary), delete; delete an individual image.
- **Four ingestion sources**, each a background job with live progress:
  1. **Local folder** — user supplies a filesystem path; backend scans for images (jpg/jpeg/png/webp/bmp), copies them into the dataset, records rows + thumbnails.
  2. **Hugging Face dataset** — user supplies a dataset id (+ optional split/config); backend streams it via the `datasets` library and extracts the image column into the dataset.
  3. **Video → frames** — user uploads a video file; backend extracts frames at a chosen interval (every Nth frame) into the dataset.
  4. **Webcam** — browser captures snapshots via `getUserMedia`; each frame is uploaded and saved as an image.
- **Browse:** paginated image grid with thumbnails; per-dataset image count; image detail (full image, dimensions).
- **Thumbnails** generated on import for fast browsing.
- **Generalized background-job infra:** SP0's `JobManager`/WebSocket stream is extended to run *any* event-producing async job (imports now, training later), keyed by job id — without breaking the existing `/api/runs` behavior.

**Out (SP1 non-goals):**
- Importing **labels/annotations** from HF datasets (only images in SP1; labeling is SP2). A dataset's `class_names` may be set at creation but boxes/classes are added in SP2.
- Editing images, augmentation, dedup, train/val splitting (SP3 concern).
- Segmentation/keypoint data.
- Remote/cloud dataset storage (local `workspace/` only).

## 3. Storage & data model

**On disk**, per dataset (under the resolved workspace root):
```
workspace/datasets/<dataset_id>/
  images/   <image_id>.<ext>       # copied/extracted full images
  thumbs/   <image_id>.webp        # generated thumbnails (max 256px, webp)
```

**SQLite** (extend SP0's tables; SP0 shipped minimal `Dataset(id, name, task, project_id)` and `Image(id, dataset_id, path)`):
- `Dataset`: add `created_at` is out (no timestamps in v1 to stay simple); keep `id, name, task, project_id`. Task stored as the `VisionTask` value string.
- `Image`: extend to `id (int pk), dataset_id (fk), image_id (str, stable), filename, width, height, source (str: "folder"|"hf"|"video"|"webcam"), thumb_path`. `path`/`thumb_path` stored **relative to the workspace root** (never absolute) so the workspace stays relocatable.
- New `ImportJob` row is **not** persisted in SP1 — import jobs live in the in-memory job manager (single-user; a job that dies with the process is acceptable for v1, matching how runs behave). The DB is the durable record of *imported images*, which is what matters.

**Image identity:** `image_id = <sha1 of file bytes, first 16 hex>`; dedup within a dataset by `image_id` (skip re-imports of identical bytes). Filename on disk = `<image_id>.<ext>`.

## 4. Architecture

### Core (`visionsuite_core`) — pure Python, no web deps
New module `ingest.py` with **pure, testable ingestion functions** that yield progress events and return image records — no FastAPI, no DB:
- `IngestedImage` (dataclass): `image_id`, `filename`, `width`, `height`, `source`.
- `iter_folder_images(folder: Path) -> Iterator[Path]` — recursively find supported image files.
- `save_image_bytes(data: bytes, images_dir: Path, thumbs_dir: Path, source: str) -> IngestedImage` — hash → write full image + generate thumbnail (Pillow) → return record. Central helper reused by all four sources.
- `extract_video_frames(video_path: Path, every_n: int) -> Iterator[bytes]` — yield JPEG bytes per sampled frame (imageio).
- `iter_hf_images(dataset_id, split, config, image_column) -> Iterator[bytes]` — yield image bytes from an HF dataset (datasets library, streaming).
These raise clear exceptions on bad input (missing folder, unreadable video, unknown HF id) — callers surface them as failed jobs.

Extend `dataset.py`: implement `Dataset.from_db_rows(...)` helper later (SP3) — **not** in SP1. SP1 leaves `Dataset`/`to_coco` as-is.

### Generalized jobs (`visionsuite_api`)
Refactor `jobs.py` so `JobManager` runs a generic producer:
- `Job` gains `kind: str` ("train"|"import") and keeps `events: list[RunEvent]`, `status`.
- `JobManager.submit_stream(job_id, kind, producer)` where `producer` is a `Callable[[], AsyncIterator[RunEvent]]`; runs one at a time (existing lock), records events + terminal status, retains the task (the SP0 fix stays). The training path (`submit(spec)`) becomes a thin wrapper that calls `submit_stream` with `lambda: self.backend.stream(spec)`.
- The WebSocket endpoint is generalized to `GET /api/jobs/{job_id}/events` (streams any job); the existing `GET /api/runs/{run_id}/events` is kept as a thin alias so SP0 tests/behavior don't regress.

### Ingestion API (`visionsuite_api/routes/datasets.py`)
- `POST /api/datasets` `{name, task}` → create dataset row, return it.
- `GET /api/datasets` → list `{id, name, task, image_count}`.
- `DELETE /api/datasets/{id}` → remove row + on-disk dataset dir.
- `GET /api/datasets/{id}/images?offset=&limit=` → paginated image records (with thumb URLs).
- `GET /api/datasets/{id}/images/{image_id}/thumb` and `.../file` → serve thumbnail / full image bytes.
- `DELETE /api/datasets/{id}/images/{image_id}` → remove row + files.
- **Imports** (each creates a job → returns `{job_id}`, progress over `/api/jobs/{job_id}/events`):
  - `POST /api/datasets/{id}/import/folder` `{path}`
  - `POST /api/datasets/{id}/import/hf` `{dataset_id, split?, config?, image_column?}`
  - `POST /api/datasets/{id}/import/video` (multipart file upload + `every_n`)
  - `POST /api/datasets/{id}/import/webcam` (multipart image upload; one frame per call, no job needed — returns the saved image record directly)
- The import producers call the core `ingest.*` functions, write files under the dataset dir, insert `Image` rows, and yield progress events (`RunEvent(type="progress", progress=i/total)` + logs).

### Frontend (`web`) — the **Datasets** page (replaces the SP0 stub)
- Dataset list with counts + "New dataset" (name + task).
- Dataset detail: source picker (Folder path / HF id / Video upload / Webcam), each launching an import and showing the shared job-progress component (reused from the run stream); a paginated thumbnail grid; delete controls.
- Webcam panel: `getUserMedia` preview + "Capture" button that POSTs each frame.
- A small reusable `useJobStream(jobId)` hook (generalized from the dashboard's WS logic).

## 5. New dependencies (SP1 installs these; NOT the deferred ML stack)
- `pillow` (image I/O + thumbnails), `datasets` (HF), `imageio` + `imageio-ffmpeg` (video frames) — added to `visionsuite-core` (core does the pure I/O).
- `python-multipart` (FastAPI file uploads) — added to `visionsuite-api`.
- These are data-pipeline deps and are fine to install now; the deferred `torch`/`transformers`/etc. remain uninstalled until SP3.

## 6. Testing approach
- **Core `ingest.py`:** unit tests with synthetic inputs — a temp folder of generated PNGs (Pillow), a tiny generated video (imageio writes a few frames), and an HF path mocked/monkeypatched (don't hit the network in tests; test the extraction logic against a fake iterable). Assert `IngestedImage` records, dedup by bytes, thumbnail creation, dimension reading.
- **Generalized jobs:** existing run tests stay green; add a test that `submit_stream` runs an arbitrary producer to a terminal status and that `/api/jobs/{id}/events` streams it.
- **Datasets API:** TestClient — create dataset, folder-import a temp dir of generated images (drive the job to done), list images paginated, fetch a thumbnail, delete an image, delete the dataset (files gone). Webcam = post an image, assert a record.
- **Frontend:** vitest — dataset list renders; import launches a job and the grid updates (mock the api client + WS); webcam capture posts a frame (mock `getUserMedia`).

## 7. Open risks / notes
- `datasets` streaming image column names vary (`image`, `img`, ...) — accept an `image_column` override, default to auto-detecting the first `Image` feature; surface a clear error if none.
- `imageio-ffmpeg` bundles an ffmpeg binary per-platform — verify the wheel installs on both Linux (dev/CI) and Apple Silicon (target). If problematic, fall back to `opencv-python-headless`.
- Large folder imports: stream progress and insert `Image` rows in batches; don't load all images into memory.
- Serving full images through FastAPI is fine for local single-user; no CDN concerns.

## 8. Task outline (finalized in the plan)
0. Add SP1 deps; extend `Image` table.
1. Core `ingest.py`: `save_image_bytes` + thumbnails + dedup (TDD).
2. Core `ingest.py`: `iter_folder_images` + `extract_video_frames` + `iter_hf_images` (TDD, mocked HF).
3. Generalize `JobManager.submit_stream` + `/api/jobs/{id}/events` (keep `/api/runs` alias).
4. Datasets CRUD API (create/list/delete + image list/serve/delete).
5. Folder + webcam import endpoints (+ job wiring).
6. HF + video import endpoints (multipart upload).
7. Frontend Datasets page: list + create + thumbnail grid + delete.
8. Frontend import UIs (folder/HF/video) with shared `useJobStream`.
9. Frontend webcam capture.
10. Whole-branch review + fixes.
