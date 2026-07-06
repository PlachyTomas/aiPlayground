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
- [x] Sub-project 1 (data pipeline) BUILT + merged to `main` (10 tasks TDD; whole-branch review passed + 2 leak fixes)
- [x] Sub-project 2 (Label Studio annotation) BUILT + merged to `main` (8 tasks TDD; 52 backend + 8 frontend tests green; review passed + 3 fixes: threadpool/bounded-poll/purity)
- [ ] **← Sub-project 3 (training engine) — next; this is where the ML stack finally installs**

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

## Deferred from Sub-project 1 review (address later)
- **No dataset-existence 404** on import/list endpoints → importing into a bogus id makes orphan files/rows (FK unenforced). Add a guard.
- **Import producers abort on one bad item** — a single undecodable image/frame fails the whole job; consider skip-and-log for robustness.
- `delete_image` double DB lookup; `ingest.py` mid-file imports; `Datasets.tsx` exhaustive-deps warning; `save_image_bytes` labels unknown formats `.png` (unreachable in SP1). All cosmetic.

## Deferred from Sub-project 2 review (address later)
- Per-class annotation counts not surfaced (pull logs only a total; UI shows none).
- Pull's delete-then-insert of `Annotation` rows is non-atomic (failure mid-way loses labels).
- `LabelStudioGateway.export_json`/`project_stats` shapes are UNVERIFIED against a live LS — adjust in the gateway when first run on the M5 (the one place SDK reality lands).
- Converter rotation/classification tests are thin; create-project lacks idempotency/empty-class validation; a cancelled pull job can stay RUNNING (CancelledError bypasses JobManager's `except Exception`).

## Next action
**Sub-project 3 — Training engine.** This is where `torch>=2.11` + `transformers>=4.54` + `trackio` finally install (behind the swappable `TrainingBackend` that SP0 stubbed). Wire real MPS training for the curated shortlist (classification first — ViT/timm; then detection — RT-DETRv2/D-FINE with `PYTORCH_ENABLE_MPS_FALLBACK=1` + bf16, torch.compile OFF), the paste-HF-model resolve→classify→load + compatibility verdict, `report_to='trackio'` metrics into the dashboard, and the detection Trainer gotchas (`remove_unused_columns=False`, `eval_do_concat_batches=False`, per-image `labels` dicts). See research brief §"Curated model shortlist" + §"#3". Same spec → plan → subagent-driven-Opus flow. Reminder: the M5 is the real test target — smoke a small classification run on-device early. Smoke the app anytime: `./scripts/dev.sh`.
