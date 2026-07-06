# VisionSuite Sub-Project 2: Annotation (Label Studio) — Design Spec

- **Date:** 2026-07-06
- **Status:** Design (authored by Claude at user delegation — no interview). Proceeding to plan + build.
- **Depends on:** SP0 (foundation) + SP1 (datasets/images), both merged to `main`.
- **Parent spec:** [`2026-07-06-visionsuite-design.md`](2026-07-06-visionsuite-design.md) §5 (#2). Verified LS specifics: [research brief](../research/2026-07-06-derisk-brief.md) §"#2 Label Studio".

## 1. Goal

Label the images imported in SP1 using **Label Studio** (bounding boxes for detection, classes for classification), track labeling progress in VisionSuite, and pull completed annotations back into an internal **COCO-style** form stored in our DB — the labeled data SP3 trains on. VisionSuite **orchestrates** Label Studio (creates the project, syncs images, pulls annotations) and links out to LS's own labeling UI; it does not rebuild a labeling canvas.

## 2. Scope

**In:**
- **Connect** to a locally-running Label Studio via its Python SDK (one client from stored config); a status/health check.
- **Create a labeling project** for a VisionSuite dataset: generate the LS label config from the dataset's task + a user-provided class list; register the dataset's images via **Local Storage** (no re-upload) and sync; store the LS `project_id` + class names on the dataset.
- **Labeling progress:** show task count + annotated count + a link to open the project in LS.
- **Pull annotations:** export the LS JSON snapshot, convert to COCO **in our own pure converter**, and store per-image annotations in the DB (as a background job).
- **Pure, fully-tested converter** (core): LS label-config generation + LS-JSON→COCO (handles the % coords / result-level dims / rotation gotchas from the research).

**Out (SP2 non-goals):**
- Rebuilding a labeling canvas in-app (LS provides it).
- Model-assisted / pre-labeling (LS ML backend) — later.
- Multi-annotator review/consensus (single-user).
- Segmentation/keypoints (detection boxes + classification only).
- Auto-installing or auto-configuring Label Studio (documented one-time user setup + API key).

## 3. Architecture

### Core (`visionsuite_core`) — pure, no SDK, no web
New `labelstudio_convert.py`:
- `ls_config_for(task: VisionTask, class_names: list[str]) -> str` — the LS labeling-config XML: `<RectangleLabels name="label" toName="image">` with a `<Label>` per class for **detection**; `<Choices name="choice" toName="image" choice="single-radio">` with a `<Choice>` per class for **classification**. (`name`/`toName` become `from_name`/`to_name` in exports.)
- `ls_json_to_coco(tasks: list[dict], class_names: list[str]) -> dict` — convert an LS JSON export to COCO:
  - `categories` = stable ids from `class_names` (index order).
  - **Detection** result (`type=="rectanglelabels"`): `x/y/width/height` are PERCENT; `original_width/original_height` live at the **result level**. COCO bbox `[x_px, y_px, w_px, h_px]` = `value.x/100*ow, value.y/100*oh, value.w/100*ow, value.h/100*oh` (xywh top-left — matches LS directly); `category_id` from `value.rectanglelabels[0]`. If `value.rotation != 0` (or top-level `image_rotation != 0`), compute the axis-aligned enclosing box of the rotated rectangle (LS has no rotated COCO box).
  - **Classification** result (`type=="choices"`): image-level `category_id` from `value.choices[0]`.
  - Skips tasks with no completed annotations; matches each task back to our `image_id` (see identity mapping below).
- These are the high-value, fully-unit-tested functions.

### API (`visionsuite_api`)
- `LabelStudioGateway` (`labelstudio.py`) — the ONLY place the SDK is touched. Thin methods returning plain data: `status()`, `create_project(title, label_config) -> project_id`, `connect_local_storage(project_id, abs_path, regex) + sync`, `project_stats(project_id) -> {total, annotated}`, `export_json(project_id) -> list[dict]` (create→poll `completed`→download). Real impl wraps `from label_studio_sdk import LabelStudio` (pinned `>=2,<3`), calls run in a threadpool (SDK is sync httpx). A module-level `get_gateway(request)` factory reads config from env; **tests monkeypatch it with a fake gateway** so no live LS is needed.
- `routes/labeling.py`:
  - `GET /api/labelstudio/status` → `{connected, version?, url}`.
  - `POST /api/datasets/{ds_id}/labeling/project` `{class_names}` → create LS project (config from dataset task + classes) + connect+sync Local Storage over the dataset's images dir; persist `ls_project_id` + `class_names`; return `{ls_project_id, ls_url}`.
  - `GET /api/datasets/{ds_id}/labeling/status` → `{ls_project_id, total, annotated, ls_url}` (or `configured:false`).
  - `POST /api/datasets/{ds_id}/labeling/pull` → a background **job** (reuse `submit_stream`): `export_json` → `ls_json_to_coco` → upsert `Annotation` rows; progress events; returns `{job_id}`.

### DB additions
- `Dataset`: add `ls_project_id: int | None`, `class_names: str` (JSON list, default `"[]"`).
- New `Annotation(id, dataset_id, image_id, coco_json: str, n_objects: int)` — per-image COCO annotation payload (list of `{bbox, category_id}` for detection, or a single `{category_id}` for classification) as JSON, plus a count for quick UI.

### Image identity mapping (LS task ↔ our image)
LS Local Storage tasks reference images by their `/data/local-files/?d=<rel>` path; the filename is our `<image_id>.<ext>`. The converter/pull derives `image_id` from the task's image filename stem, so annotations map back to the right `Image` row.

### Frontend (`web`) — the **Labeling** page (replaces the SP0 stub)
- LS connection status banner (calls `/api/labelstudio/status`); if disconnected, show the setup hint (start LS, set `LABEL_STUDIO_URL`/`LABEL_STUDIO_API_KEY`).
- Per dataset: a class-names input → "Create labeling project"; once created, an **"Open in Label Studio"** link (new tab to `ls_url`), a status line (`annotated/total`), a **"Pull annotations"** button (launches the pull job, shows progress via the existing `useJobStream`), and the resulting per-class annotation counts.

### Ops
- `scripts/labelstudio.sh` — launches LS locally with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` and `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<abs workspace root>` on :8080; README documents the one-time account + API-key step and setting `LABEL_STUDIO_URL`/`LABEL_STUDIO_API_KEY`.

## 4. Config & dependencies
- Env: `LABEL_STUDIO_URL` (default `http://localhost:8080`), `LABEL_STUDIO_API_KEY` (prefer a **legacy token** so raw REST also works; the SDK accepts both).
- New dep: `label-studio-sdk>=2,<3` (api package). LS server itself is user-installed (`pip install label-studio`), not a project dep. Still NO ML stack.

## 5. Testing approach
- **Core converter (the crux):** unit tests feed the exact LS JSON shapes from the research brief — a detection task (`rectanglelabels`, % coords, result-level `original_width/height`, one rotated box) and a classification task (`choices`) — and assert COCO output: pixel math, top-left xywh, category ids from `class_names`, rotation→axis-aligned box, `image_id` derived from the task filename. Config generator: assert the XML has the right control + a label per class.
- **Gateway/endpoints:** monkeypatch `get_gateway` with a `FakeGateway` (in-memory: records created projects, returns canned stats + a canned export JSON). Test: create-project stores `ls_project_id`+`class_names`; status returns stats; pull job runs export→convert→`Annotation` rows and reports counts. NO live LS.
- **Frontend:** vitest — status banner (mock api), create-project posts classes, pull launches a job (mock `useJobStream`/WS), counts render.
- **Open risk (documented, not blocking):** real LS SDK method signatures may differ slightly from the research; only `LabelStudioGateway` needs adjustment when first run against a live LS on the M5. A `scripts/labelstudio.sh` smoke on the user's machine is the real integration check.

## 6. Task outline (finalized in the plan)
1. DB: `Dataset.ls_project_id`+`class_names`, `Annotation` table; add `label-studio-sdk` dep.
2. Core `ls_config_for` (detection + classification configs).
3. Core `ls_json_to_coco` (detection %→px + rotation guard, classification, category map, image_id from filename).
4. API `LabelStudioGateway` (+ real SDK impl) + `get_gateway` factory + `GET /api/labelstudio/status`.
5. API `POST .../labeling/project` (create + local-storage sync + persist).
6. API `GET .../labeling/status` + `POST .../labeling/pull` (job → convert → Annotation rows).
7. Frontend Labeling page (status, create, open-in-LS, pull w/ progress, counts).
8. Ops: `scripts/labelstudio.sh` + README setup section.
