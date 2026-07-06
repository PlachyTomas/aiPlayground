# VisionSuite — Design Spec (v1)

- **Date:** 2026-07-06
- **Working name:** VisionSuite (rename anytime)
- **Status:** Design approved through architecture + decomposition + Sub-project 0 scope. Awaiting user spec review before writing implementation plans.
- **Research backing:** [de-risking brief](../research/2026-07-06-derisk-brief.md) (6-agent sweep, ~361k tokens, 2026-07-06)

## 1. Goal

A **local, single-user** suite for training vision models from Hugging Face on a **MacBook Air M5 (Apple Silicon, 24GB unified, MPS, fanless)**. Closes a tight loop: **ingest → label → train → evaluate → export → test in-app**. Primary goal is **producing real fine-tuned models**, so reliability of that loop beats breadth.

Driving example: fine-tune an object detector to put bounding boxes on a given object (e.g. from camera frames).

## 2. Scope

**In (v1):**
- Two tasks end-to-end: **object detection** (bounding boxes) and **image classification**.
- **Annotation via Label Studio** (local companion process), integrated through its Python SDK.
- **Four data sources:** local image folders, Hugging Face datasets, webcam capture, video → frames.
- **Model registry:** a curated MPS-safe shortlist **plus paste-any-HF-model** (id or URL) with a live compatibility verdict.
- **Experiment tracking via Trackio** (local-first), surfaced in the dashboard.
- **Export:** native HF format (both tasks) + ONNX (classification clean, detection best-effort).
- **In-app inference:** run a trained/pasted model on sample images and view predictions.

**Out (v1) — explicit non-goals:**
- Live camera / real-time video inference (only still-image test).
- Non-vision modalities (text, audio, multimodal, tabular).
- Multi-user, accounts, auth, label review/consensus.
- Cloud training execution (the `TrainingBackend` interface is built to allow it later; no cloud runner ships in v1).
- Core ML export (deferred; torch-version friction + fragile detection tracing).
- Segmentation / keypoints **training** (the annotation board may grow polygon support later, but nothing trains end-to-end on masks in v1).

## 3. Architecture

Four components around a shared workspace folder, plus Label Studio as a local companion:

```
React + Vite frontend  (the browser app)
  Dashboard · Datasets · Labeling · Train · Models/Test
        │  REST + WebSocket (live logs/progress)
FastAPI backend
  Job manager (1 training run at a time, subprocess) · Data ingestion
  · Label Studio sync (SDK) · Trackio metrics read-back · Inference
        │  imports
Core ML engine  (pure Python package, NO web deps)
  ModelAdapter registry · Dataset format · TrainingBackend
  · Evaluator · Exporter · Inference
        │  reads/writes
workspace/  (db.sqlite · datasets/ · runs/ · models/ · exports/)   ◄──►  Label Studio (local :8080, via SDK)
```

**Design principles:**
- **Core engine is web-framework-free** — runnable from a script/notebook; FastAPI is a thin layer over it. Keeps training/eval testable in isolation and makes the future cloud backend just another `TrainingBackend` implementation.
- **`ModelAdapter` registry** is the heart of "handle various models." Each curated architecture is one adapter; MPS-safety is encoded per adapter. Adding a model = adding an adapter (or pasting an HF id), not touching the app.
- **One internal dataset format** (COCO-style for detection, folder+manifest for classification). Everything converts *to* it: Label Studio exports → internal, HF datasets → internal. Training only ever sees the internal format.
- **SQLite for metadata, filesystem for heavy data.** Single-user → no auth, and a simple one-run-at-a-time job manager (the M5 can't parallelize training anyway).
- **Label Studio is not embedded as a service**; the backend talks to it only via one SDK client.

## 4. Validated technical constraints (from the de-risking sweep)

These are **binding requirements** distilled from the research. Full detail + sources in the [brief](../research/2026-07-06-derisk-brief.md).

**Global / MPS:**
- Require **torch ≥ 2.11** (native `grid_sample` forward on MPS, shipped 23 Mar 2026).
- Set **`PYTORCH_ENABLE_MPS_FALLBACK=1`** in the training-subprocess env (grid_sample *backward* may still fall to CPU; non-negotiable for deformable detectors).
- Use **bf16 autocast** (`torch.autocast('mps', dtype=torch.bfloat16)`), never fp16 (NaN-prone on MPS). Requires macOS 14+ (fine on M5).
- **`torch.compile` OFF** for v1 MPS training (unreliable in `loss.backward()`).
- Registry must **reject deformable-CONVOLUTION** models (`torchvision.ops.deform_conv2d`, DCNv2/v3, InternImage) — no MPS kernel, no fallback. (Deformable *attention* / grid_sample is fine.)
- Fanless throttling (~20–40% below burst on long runs) is the binding limit → checkpoint frequently; heavy detection is a candidate for future HF Jobs offload.

**Curated model shortlist (v1 registry defaults):**
- **Classification (clean, no flag):** timm EfficientNet-B0/B1 or ResNet-50 @224px (default); MobileNetV3 (smallest); ViT-base/16 `google/vit-base-patch16-224`.
- **Detection default (deformable, needs flag+bf16):** `PekingU/rtdetr_v2_r18vd` or `ustc-community/dfine-{nano,small}-coco` @640px.
- **Detection clean fallback (no flag):** `hustvl/yolos-tiny` ≤512px or `facebook/detr-resnet-50` @~640px.
- **Gate behind warning:** R101/large/xlarge variants, ViT-large, native-800px DETR, Deformable DETR (superseded).

**Model import (paste-HF) — 3 stages + verdict:**
- RESOLVE: `HfApi.model_info` (catch Gated/NotFound early) → then full `config.json` via `hf_hub_download`/`AutoConfig` (the `model_info.config` field is truncated).
- CLASSIFY with precedence: **`library_name=="timm"` FIRST** → `transformersInfo.auto_model` → `architectures[0]` suffix → `pipeline_tag` → `model_type` map. (timm has empty config + `auto_model="AutoModel"` — naive detection misclassifies it.)
- LOAD: `AutoModelForObjectDetection` / `AutoModelForImageClassification` + `id2label`/`label2id` + `ignore_mismatched_sizes=True`; pair with `AutoImageProcessor` from the same repo id.
- **Verdict enum:** `KNOWN_GOOD_MPS` (allowlist smoke-tested on M5) · `UNTESTED` (valid task+auto-class, not yet tested; deformable detectors get a grid_sample/fallback warning) · `UNSUPPORTED` (no auto-class/task, `trust_remote_code`, gated/no-token, or timm-detection). Promote UNTESTED→KNOWN_GOOD only after a real 1-step forward+backward passes on-device.

**Detection Trainer specifics (hard-code):** `remove_unused_columns=False`, `eval_do_concat_batches=False`; collator keeps `labels` as a list of per-image dicts (`{class_labels, boxes}`), only `pixel_values`/`pixel_mask` stacked. RT-DETR handles no-object internally (`num_labels` = real classes, no +1).

**Label Studio:** run standalone on :8080 with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` + `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/abs/parent`. Pin **`label-studio-sdk>=2,<3`** (SDK was rewritten twice — all `Client`/`start_project` tutorials are dead; use `from label_studio_sdk import LabelStudio`, resource-namespaced). Sync images via **Local Storage** (`import_storage.local.create+sync`), not re-upload. SDK is sync httpx → call from FastAPI via `run_in_threadpool`. Bbox coords are **percent** with `original_width/height` at the **result level**; **write our own JSON→COCO converter** (`x_px=value.x/100*original_width`, …) rather than trusting LS's COCO exporter; **guard `value.rotation!=0`**.

**Trackio:** `report_to='trackio'` (needs **transformers ≥ 4.54.0**), set `project`+`run_name`. Product path = backend polls **`trackio get metric … --json`** every 2–5s → SSE/WebSocket → render our own charts (Recharts/uPlot). Optional iframe-embed of the Gradio dashboard (`127.0.0.1:7860/?project=…&sidebar=hidden&footer=false`) as a raw view — verify local framing empirically. **Never parse Trackio's SQLite directly.** Pin `trackio==0.29.0`.

**Export / in-app inference:** commit native `save_pretrained` (both tasks) + in-app inference via transformers on the saved checkpoint (`post_process_object_detection(..., target_sizes=[(H,W)])` + PIL boxes; run the single test image **on CPU** to dodge MPS edge cases). Commit ONNX **classification** (`optimum-cli` + `ORTModelForImageClassification`). Best-effort ONNX **detection** (opset ≥16; no `ORTModelForObjectDetection` → bundle a ~30-line post-processor, run onnxruntime on CPUExecutionProvider). Install `optimum[onnxruntime]` + `onnxruntime`. **Defer Core ML.**

## 5. Decomposition & sequencing

Too big for one spec → **five sub-projects**, each its own spec → plan → build cycle. Ordered so something is usable early and each stage de-risks the next.

| # | Sub-project | Done = you can… | De-risks |
|---|-------------|-----------------|----------|
| **0** | **Foundation / walking skeleton** | Launch the app; empty dashboard; FastAPI + SQLite + workspace; core interfaces stubbed; one dummy end-to-end run streams over WebSocket. | Wiring, storage model, dev ergonomics. |
| **1** | **Data pipeline & ingestion** | Import from folders / HF datasets / webcam / video→frames; browse images; datasets in internal format. | The 4 sources; internal dataset format. |
| **2** | **Annotation (Label Studio)** | Push a dataset to LS, label bboxes/classes, pull annotations back; see labeling progress. | LS round-trip + JSON→COCO conversion. |
| **3** | **Training engine** (classification first, then detection) | Pick a model (curated or **pasted HF id**) + dataset → train on MPS → watch live Trackio metrics. | MPS quirks, job manager, live streaming, model import/verdict. |
| **4** | **Eval, export & in-app test** | See mAP/accuracy + prediction viz; export HF/ONNX; test any model on images. | Metrics, export formats, inference panel. |

**Rationale:** #3 (training) is riskiest (MPS), so 0–2 build the data+labeling foundation it needs first, and the training loop is proven on **classification** (clean, fast) before **detection**. By end of #3 you can produce a model; #4 makes it measurable/testable.

**Process:** brainstorm each sub-project in detail when reached (decisions made with real code in hand). Solo-build the foundation (#0) — it's integrative wiring. Later sub-projects (#1 importers, #3 adapters, #4 export/inference) may fan out to reviewed subagents.

## 6. Sub-project 0 — Foundation (detailed)

**Goal:** thinnest end-to-end slice that proves every piece connects, with **zero ML** — so all later work is "swap a stub for the real thing."

**Monorepo layout:**
```
visionsuite/
  pyproject.toml            # uv-managed; workspace of core + api
  packages/
    core/  visionsuite_core/        ← pure Python, NO web deps
      registry.py    ModelAdapter Protocol + registry (one dummy adapter)
      dataset.py     internal dataset format + from_coco/to_coco (stubs)
      backends.py    TrainingBackend Protocol + LocalBackend (stub)
      workspace.py   resolves workspace root + paths
      tests/
    api/   visionsuite_api/          ← thin FastAPI over core
      main.py        app factory, CORS, serves built web
      db.py          SQLite (SQLModel) schema + init
      jobs.py        in-process job manager (1 run) + WebSocket log stream
      routes/        health.py, runs.py (dummy), + stubs
      tests/
  web/                              ← React + Vite (TS)
    src/lib/api.ts                  typed API client
    src/routes/                     Dashboard + stub pages (Datasets/Labeling/Train/Models)
  workspace/                        (runtime, gitignored) db.sqlite · datasets/ runs/ models/ exports/
  scripts/dev.sh                    one command: boot uvicorn + Vite (later + LS + Trackio)
  README.md
```

**Core contracts (interfaces only in #0; real impls land later):**
- `ModelAdapter` (Protocol): `task`, `hf_id`, `load()`, `build_processor()`, `train_config()`, `evaluate()`, `export()`, `mps_compat() -> CompatVerdict`.
- `CompatVerdict` enum: `KNOWN_GOOD_MPS` · `UNTESTED` · `UNSUPPORTED` (+ optional warning text). (Shape fixed now per §4; population is #3's job.)
- `Dataset`: internal representation + `from_coco/to_coco`, `from_label_studio` (stubs raising `NotImplementedError`).
- `TrainingBackend` (Protocol): `submit(run_spec) -> handle`, `stream_logs()`, `status()`, `cancel()`. `LocalBackend` stub emits fake progress.

**Storage model:**
- Single workspace root via `VISIONSUITE_WORKSPACE` (default `./workspace`), resolved in `workspace.py`.
- SQLite tables (initial, minimal): `project`, `dataset`, `image`, `run`, `model`. Created on first boot.
- Filesystem: images `datasets/<id>/images/`, checkpoints `runs/<id>/`, exports `exports/<id>/`.

**Walking-skeleton path (the proof it's wired):**
Dashboard `POST /api/runs` → backend writes a `run` row → `LocalBackend` streams a handful of **fake** log lines + a progress % over WebSocket → dashboard shows a live run ticking to "done." Exercises React↔FastAPI REST **+ WebSocket**, a DB write, the job manager, and a core interface — the exact plumbing #1–#4 reuse.

**Dependency pins baked in now (from §4):** `torch>=2.11`, `transformers>=4.54`, `label-studio-sdk>=2,<3`, `trackio==0.29.0`, `optimum[onnxruntime]`, `onnxruntime`, FastAPI, SQLModel, `uv`. (Heavy ML deps declared but unused until #3.)

**Dev ergonomics:** `uv` for Python; `scripts/dev.sh` runs uvicorn + Vite together (extended later to also launch Label Studio with the local-files env vars, and `trackio show`).

**Testing (#0):** pytest — core interfaces exercised via the dummy adapter/backend; one API test covering `/health`, dummy-run creation, and the WebSocket log stream. TDD applied at plan-execution time.

## 7. Open risks to validate during implementation

Carried from the brief §5 — resolve empirically, don't pre-solve:
- `grid_sampler_2d_backward` MPS status unconfirmed → smoke-test a 1-step train on RT-DETRv2/D-FINE on the actual M5; keep the fallback flag regardless.
- Batch/memory numbers are estimates → probe OOM per model; tune `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.
- MPS may silently miscompute some ops → validate fine-tuned eval vs a short CPU run.
- Trackio: local-iframe embedding + bare-`trackio show` HTTP API are undocumented; port may shift off 7860. The CLI `--json` path avoids all three — prefer it.
- Label Studio: fresh install may only offer JWT tokens; `AsyncLabelStudio` in 2.0.x unverified.
- onnxruntime CoreML-EP/ANE availability is version-dependent → runtime-check providers, never require.
- Fanless sustained throughput unknown until measured → informs any future HF Jobs offload.

## 8. Status & next steps (resumption anchor)

**Decisions locked:** vision-only v1 (detection + classification); local single-user; MacBook Air M5 / MPS; goal = produce real models; Label Studio for annotation; React+Vite + FastAPI + pure-Python core; Trackio tracking; on-device training behind a swappable `TrainingBackend`; export HF+ONNX, in-app still-image test; paste-HF-model import with compatibility verdict.

**Next steps:**
1. User reviews this spec (current gate).
2. On approval → `writing-plans` skill to produce the **Sub-project 0** implementation plan (TDD).
3. Build #0 solo; then brainstorm #1, and so on.

See [`PROGRESS.md`](../../../PROGRESS.md) at repo root for the terse cross-session status.
