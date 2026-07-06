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
- [x] Master spec written
- [ ] **← USER REVIEWING SPEC (current gate)**
- [ ] `writing-plans` → Sub-project 0 implementation plan
- [ ] Build Sub-project 0 (solo)

## Key documents
- **Design spec:** `docs/superpowers/specs/2026-07-06-visionsuite-design.md` (architecture, scope, validated constraints, decomposition, Sub-project 0 detail).
- **Research brief:** `docs/superpowers/research/2026-07-06-derisk-brief.md` (the expensive 361k-token artifact — MPS/Label Studio/Trackio/model-import/export findings, verified repo IDs, version pins).

## Locked decisions
Vision-only v1 = **object detection + image classification**, end-to-end. Single-user, no auth. On-device MPS training behind a swappable `TrainingBackend` (cloud later, not v1). Annotation = Label Studio (local process, SDK). Tracking = Trackio (local). Model registry = curated MPS-safe shortlist **+ paste-any-HF-model** with a compatibility verdict. Export = HF native + ONNX (classification clean, detection best-effort); Core ML deferred. In-app inference = still images only (no live camera in v1).

## Non-negotiable technical pins (from research)
- `torch>=2.11`, `PYTORCH_ENABLE_MPS_FALLBACK=1` in training env, **bf16** (not fp16), `torch.compile` OFF.
- `transformers>=4.54` (for `report_to='trackio'`), `label-studio-sdk>=2,<3`, `trackio==0.29.0`, `optimum[onnxruntime]`.
- Classify timm by `library_name` first; reject `deform_conv2d` models; write our own LS-JSON→COCO converter (coords are %).

## Next action
User approves/edits the spec → invoke `writing-plans` for the Sub-project 0 plan. Build #0 solo (integrative wiring; don't fan out). Not a git repo yet — offer `git init` before first commit.
