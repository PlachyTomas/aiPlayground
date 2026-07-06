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
- [x] Sub-project 1 (data pipeline) BUILT + merged to `main` (10 tasks TDD; 40 backend + 7 frontend tests green; whole-branch review passed + 2 leak fixes)
- [ ] **← Sub-project 2 (Label Studio annotation) — next**

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

## Next action
**Sub-project 2 — Annotation (Label Studio).** Run LS as a local process; FastAPI talks to it via the rewritten SDK (`from label_studio_sdk import LabelStudio`, pin `>=2,<3`); push a dataset's images via Local Storage, pull annotations, convert LS JSON → COCO ourselves (coords are %, `original_width/height` at result level, guard rotation). See the research brief §"#2 Label Studio" for the verified specifics. Follow the same spec → plan → subagent-driven-Opus flow. Smoke the app once on the M5: `./scripts/dev.sh` → Datasets page → import a folder.
