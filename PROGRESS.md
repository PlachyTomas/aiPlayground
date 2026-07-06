# VisionSuite — Progress / Resumption Note

_Committed resume anchor (self-sufficient; does not rely on any out-of-repo memory). Last updated: 2026-07-06._

## ▶ START HERE (new session)
1. Read this file top-to-bottom, then `git log --oneline -20`.
2. Hardware is **resolved** (see below) — SP3 is unblocked.
3. Working style: the user **delegates sub-project design (no interview)** — Claude authors the spec + plan and builds via **subagent-driven-development on Opus** (one implementer subagent per task, Claude reviews each diff in-loop, one whole-branch review at the end, then fast-forward merge to `main`). Keep agent count lean; keep the main model on Opus.
4. Everything needed to resume is committed on `main`. The per-task SDD ledger under `.superpowers/sdd/` is gitignored scratch — this file is the durable record.
5. Smoke the app anytime: `uv sync --all-packages && (cd web && npm install) && ./scripts/dev.sh`.

## ✓ HARDWARE (resolved)
- **Training target = the user's MacBook Air M5 (Apple Silicon / MPS / 24 GB unified).** The MPS design in the research brief + the technical pins below **all apply as written**.
- **This Lenovo notebook (weaker, x86/Linux) is the DEV box only** — used to build + unit-test the device-agnostic plumbing (all green here). It does NOT run real training.
- **Implication for SP3:** unit-test the training engine on the Lenovo with tiny/CPU paths and mocks; **smoke the real MPS training runs on the M5** (the actual target). Keep the concrete local backend MPS-first per the research; the `TrainingBackend` abstraction still allows a future cloud/CUDA runner but v1 targets the M5.

## Status (process)
Sub-projects 0–2 are BUILT, reviewed, and merged to `main`. **SP3 (training engine) is next and unblocked.**

- [x] Interview, architecture, 5-way decomposition, de-risking research sweep (all persisted)
- [x] Git repo (root commit 594d22c on `main`)
- [x] **SP0 — Foundation / walking skeleton** — merged (10 TDD tasks; uv workspace, FastAPI + generic JobManager + `/api/jobs/{id}/events` WS, SQLite, React dashboard)
- [x] **SP1 — Data pipeline & ingestion** — merged (10 TDD tasks; core `ingest`, dataset CRUD + serving, 4 import sources, Datasets page)
- [x] **SP2 — Annotation (Label Studio)** — merged (8 TDD tasks; pure LS→COCO converter, `LabelStudioGateway`, project/status/pull endpoints, Labeling page, launcher)
- [ ] **← SP3 — Training engine** (installs the ML stack; MPS on the M5)
- [ ] SP4 — Eval, export & in-app test

Current test state on the Lenovo/Linux dev box: **52 backend + 8 frontend tests green; `npm run build` OK.** No ML deps installed yet (deferred to SP3).

## Key documents (all committed on `main`)
- **Master design spec:** `docs/superpowers/specs/2026-07-06-visionsuite-design.md`
- **Research brief:** `docs/superpowers/research/2026-07-06-derisk-brief.md` (MPS/Label Studio/Trackio/model-import/export findings, verified repo IDs, version pins — the MPS parts apply, since the M5 is the training target)
- **SP0 plan:** `docs/superpowers/plans/2026-07-06-subproject-0-foundation.md`
- **SP1 spec + plan:** `docs/superpowers/specs/2026-07-06-subproject-1-data-design.md` · `docs/superpowers/plans/2026-07-06-subproject-1-data.md`
- **SP2 spec + plan:** `docs/superpowers/specs/2026-07-06-subproject-2-annotation-design.md` · `docs/superpowers/plans/2026-07-06-subproject-2-annotation.md`

## Locked decisions
Vision-only v1 = **object detection + image classification**, end-to-end. Single-user, no auth. **On-device MPS training on the M5** behind a swappable `TrainingBackend` (cloud/CUDA later). Annotation = Label Studio (local process, SDK). Tracking = Trackio (local). Model registry = curated MPS-safe shortlist **+ paste-any-HF-model** with a compatibility verdict. Export = HF native + ONNX (classification clean, detection best-effort); Core ML deferred. In-app inference = still images only.

## Technical pins from research (apply — M5/Apple-Silicon is the training target)
- `torch>=2.11`, `PYTORCH_ENABLE_MPS_FALLBACK=1` in the training subprocess, **bf16** (not fp16), `torch.compile` OFF.
- `transformers>=4.54` (for `report_to='trackio'`), `trackio==0.29.0`, `optimum[onnxruntime]`.
- Already installed + applying: `label-studio-sdk>=2,<3`; classify timm by `library_name` first; reject `deform_conv2d` models; our own LS-JSON→COCO converter (coords are %).
- Curated shortlist (research brief §"Curated model shortlist"): classification = timm EfficientNet-B0/B1, ResNet-50, MobileNetV3, ViT-base; detection default = RT-DETRv2-R18 / D-FINE-nano-small @640; clean fallback = YOLOS-tiny / DETR-R50.

## Deferred review findings (address in later sub-projects)
**SP0:** cancel wiring unbuilt (`RunStatus.CANCELLED` unused); DB `Run.status` never persisted (in-memory only); SPA deep-link 404 on hard-refresh; no `ruff` lint; module-level `app = create_app()` has import-time DB side effects.
**SP1:** import/list endpoints don't 404 on a bogus dataset id (orphan files/rows); import producers abort on one bad item (no skip-and-log); `delete_image` double lookup; `ingest.py` mid-file imports; `Datasets.tsx` exhaustive-deps warning; `save_image_bytes` labels unknown formats `.png`.
**SP2:** per-class annotation counts not surfaced; pull's delete-then-insert is non-atomic; **`LabelStudioGateway.export_json`/`project_stats` SDK shapes UNVERIFIED against a live LS** — adjust in the gateway on first real run; thin converter tests; create-project lacks idempotency/empty-class validation; a cancelled pull job can stay RUNNING (CancelledError bypasses `JobManager`'s `except Exception`).

## Next action
Design **SP3 — Training engine** (same spec → plan → subagent-driven-Opus flow): real training behind the `TrainingBackend`, **MPS device handling on the M5** (fallback flag + bf16, torch.compile off), classification first (ViT/timm) then detection (RT-DETRv2/D-FINE), paste-HF-model resolve→classify→load + compatibility verdict, `report_to='trackio'` metrics streamed into the dashboard, detection Trainer gotchas (`remove_unused_columns=False`, `eval_do_concat_batches=False`, per-image `labels` dicts). Unit-test on the Lenovo (tiny/CPU + mocks); **smoke a small real training run on the M5 early** before trusting the loop.
