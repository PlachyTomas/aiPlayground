# VisionSuite — Progress / Resumption Note

_Terse cross-session status. Last updated: 2026-07-06._

## What this is
A **local, single-user** suite to train Hugging Face **vision** models on a **MacBook Air M5 (MPS, 24GB, fanless)**. Loop: ingest → label → train → eval → export → in-app still-image test. Goal: **produce real fine-tuned models**. Stack: **React+Vite + FastAPI + pure-Python core**, **Label Studio** for annotation, **Trackio** for tracking.

## Where we are (process)
We are in **brainstorming → spec** (Superpowers flow). Currently at the **user-reviews-spec gate**.

- [x] Interview (scope, hardware, goals)
- [x] Architecture approved (4 components + workspace + Label Studio)
- [x] Decomposition approved (5 sub-projects)
- [x] Sub-project 0 (foundation) scope approved
- [x] De-risking research sweep done + persisted
- [x] Master spec written + user-approved
- [x] Git repo initialized (commit 594d22c on `main`)
- [x] Sub-project 0 implementation plan written
- [x] Sub-project 0 BUILT + merged to `main` (10 tasks, TDD; 24 backend + 3 frontend tests green; whole-branch review passed + fixes applied)
- [x] Sub-project 1 (data pipeline) spec + plan written (authored by Claude at user delegation — no interview)
- [ ] **← EXECUTE Sub-project 1 on `feat/subproject-1-data` (10 tasks, subagent-driven Opus)**

## Key documents
- **Design spec:** `docs/superpowers/specs/2026-07-06-visionsuite-design.md` (architecture, scope, validated constraints, decomposition, Sub-project 0 detail).
- **Research brief:** `docs/superpowers/research/2026-07-06-derisk-brief.md` (the expensive 361k-token artifact — MPS/Label Studio/Trackio/model-import/export findings, verified repo IDs, version pins).
- **Sub-project 0 plan:** `docs/superpowers/plans/2026-07-06-subproject-0-foundation.md` (10 TDD tasks, walking skeleton; ML deps deliberately NOT installed until Sub-project 3).
- **Sub-project 1 spec + plan:** `docs/superpowers/specs/2026-07-06-subproject-1-data-design.md` + `docs/superpowers/plans/2026-07-06-subproject-1-data.md` (10 TDD tasks: core ingest, generalized jobs, dataset CRUD, 4 import sources, Datasets UI; adds pillow/datasets/imageio/python-multipart — NOT the ML stack).

## Locked decisions
Vision-only v1 = **object detection + image classification**, end-to-end. Single-user, no auth. On-device MPS training behind a swappable `TrainingBackend` (cloud later, not v1). Annotation = Label Studio (local process, SDK). Tracking = Trackio (local). Model registry = curated MPS-safe shortlist **+ paste-any-HF-model** with a compatibility verdict. Export = HF native + ONNX (classification clean, detection best-effort); Core ML deferred. In-app inference = still images only (no live camera in v1).

## Non-negotiable technical pins (from research)
- `torch>=2.11`, `PYTORCH_ENABLE_MPS_FALLBACK=1` in training env, **bf16** (not fp16), `torch.compile` OFF.
- `transformers>=4.54` (for `report_to='trackio'`), `label-studio-sdk>=2,<3`, `trackio==0.29.0`, `optimum[onnxruntime]`.
- Classify timm by `library_name` first; reject `deform_conv2d` models; write our own LS-JSON→COCO converter (coords are %).

## Deferred from Sub-project 0 review (address in later sub-projects)
- **Cancel wiring** — `RunStatus.CANCELLED` exists but no cancel endpoint/UI yet (wire when training runs get real, SP3).
- **Persist `run.status` to the DB row** — currently the DB `Run.status` stays "pending"; live status is in-memory only (needed for runs surviving restart, SP3).
- **SPA deep-link fallback** — FastAPI static serving 404s on hard-refresh of client routes (e.g. `/datasets`); add a catch-all → `index.html` when polishing prod serving.
- **Lint** — no `ruff` configured; add a tooling pass in SP1.
- Module-level `app = create_app()` has import-time DB side effects (intentional for `uvicorn ...:app`; revisit if it bites tests).

## Next action
Finish the `feat/subproject-0-foundation` branch (merge to `main` or open a PR — single-user, local, no remote yet → fast-forward merge to `main` is fine). Then **brainstorm Sub-project 1 (data pipeline & ingestion)** via the Superpowers flow (brainstorming → writing-plans → subagent-driven execution). Interactive browser smoke of the dashboard still worth doing once on the M5: `./scripts/dev.sh` → click "Start dummy run".
