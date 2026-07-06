# VisionSuite — Progress / Resumption Note

_Committed resume anchor (self-sufficient; does not rely on any out-of-repo memory). Last updated: 2026-07-06._

## ▶ START HERE (new session)
1. Read this file top-to-bottom, then `git log --oneline -20`.
2. **Resolve the HARDWARE question below before designing Sub-project 3** — it changes SP3's whole design.
3. Working style so far: the user **delegates sub-project design (no interview)** — Claude authors the spec + plan and builds via **subagent-driven-development on Opus** (one implementer subagent per task, Claude reviews each diff in-loop, one whole-branch review at the end, then fast-forward merge to `main`). Keep agent count lean; keep the main model on Opus.
4. Everything needed to resume is committed on `main`. The per-task SDD ledger under `.superpowers/sdd/` is gitignored scratch — this file is the durable record.
5. Smoke the app anytime: `uv sync --all-packages && (cd web && npm install) && ./scripts/dev.sh`.

## ⚠ HARDWARE — must confirm before Sub-project 3
The entire design (research brief, SP3 plan direction) assumes the training machine is a **MacBook Air M5 (Apple Silicon / MPS / 24 GB unified)**. The user has now said they are **running on a Lenovo notebook (weaker)**. This is unresolved and it is load-bearing for SP3:
- If the **Lenovo is the real target** (x86; NVIDIA CUDA GPU, or none): the MPS-specific pins/decisions **do not apply** — no `PYTORCH_ENABLE_MPS_FALLBACK`, `grid_sample`-fallback reasoning, or bf16-on-MPS; instead detect device = CUDA (if an NVIDIA GPU is present) or CPU. A weak/no-GPU Lenovo likely can't train detection locally at reasonable speed → the **cloud fallback (HF Jobs)** the user declined in the M5 interview may need revisiting for detection.
- If the **M5 is still the target** and the Lenovo is just the current dev box: SP3 proceeds as the research brief describes (MPS), and the Lenovo is only for building/testing the non-ML plumbing (which is device-agnostic and already green here on Linux).
- **The `TrainingBackend` abstraction survives either way** (SP0 built it to be swappable); only the concrete local backend's device handling + the feasible-model shortlist change.
- **Action:** ask the user which machine SP3 must run on (and, if Lenovo, its GPU) before writing the SP3 spec. Re-scope device handling + model shortlist accordingly.

## Status (process)
Sub-projects 0–2 are BUILT, reviewed, and merged to `main`. SP3 is next (blocked on the hardware question above).

- [x] Interview, architecture, 5-way decomposition, de-risking research sweep (all persisted)
- [x] Git repo (root commit 594d22c on `main`)
- [x] **SP0 — Foundation / walking skeleton** — merged (10 TDD tasks; uv workspace, FastAPI + generic JobManager + `/api/jobs/{id}/events` WS, SQLite, React dashboard)
- [x] **SP1 — Data pipeline & ingestion** — merged (10 TDD tasks; core `ingest`, dataset CRUD + serving, 4 import sources, Datasets page)
- [x] **SP2 — Annotation (Label Studio)** — merged (8 TDD tasks; pure LS→COCO converter, `LabelStudioGateway`, project/status/pull endpoints, Labeling page, launcher)
- [ ] **← SP3 — Training engine** (installs the ML stack; MPS **or** CUDA/CPU per hardware answer)
- [ ] SP4 — Eval, export & in-app test

Current test state on this Linux box: **52 backend + 8 frontend tests green; `npm run build` OK.** No ML deps installed yet (deferred to SP3).

## Key documents (all committed on `main`)
- **Master design spec:** `docs/superpowers/specs/2026-07-06-visionsuite-design.md`
- **Research brief:** `docs/superpowers/research/2026-07-06-derisk-brief.md` (MPS/Label Studio/Trackio/model-import/export findings, verified repo IDs, version pins — **note: the MPS parts assume Apple Silicon**)
- **SP0 plan:** `docs/superpowers/plans/2026-07-06-subproject-0-foundation.md`
- **SP1 spec + plan:** `docs/superpowers/specs/2026-07-06-subproject-1-data-design.md` · `docs/superpowers/plans/2026-07-06-subproject-1-data.md`
- **SP2 spec + plan:** `docs/superpowers/specs/2026-07-06-subproject-2-annotation-design.md` · `docs/superpowers/plans/2026-07-06-subproject-2-annotation.md`

## Locked decisions
Vision-only v1 = **object detection + image classification**, end-to-end. Single-user, no auth. Local training behind a **swappable `TrainingBackend`** (cloud later). Annotation = Label Studio (local process, SDK). Tracking = Trackio (local). Model registry = curated shortlist **+ paste-any-HF-model** with a compatibility verdict. Export = HF native + ONNX (classification clean, detection best-effort); Core ML deferred. In-app inference = still images only. **Caveat:** "on-device MPS training" was decided assuming the M5 — revisit per the hardware question.

## Technical pins from research (⚠ MPS-specific — conditional on Apple-Silicon target)
- `torch>=2.11`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, **bf16** (not fp16), `torch.compile` OFF — **all MPS-only**; ignore/replace if the target is CUDA/CPU.
- `transformers>=4.54` (for `report_to='trackio'`), `trackio==0.29.0`, `optimum[onnxruntime]` — device-agnostic, still apply.
- Already installed + applying: `label-studio-sdk>=2,<3`; classify timm by `library_name` first; reject `deform_conv2d` models; our own LS-JSON→COCO converter (coords are %).

## Deferred review findings (address in later sub-projects)
**SP0:** cancel wiring unbuilt (`RunStatus.CANCELLED` unused); DB `Run.status` never persisted (in-memory only); SPA deep-link 404 on hard-refresh; no `ruff` lint; module-level `app = create_app()` has import-time DB side effects.
**SP1:** import/list endpoints don't 404 on a bogus dataset id (orphan files/rows); import producers abort on one bad item (no skip-and-log); `delete_image` double lookup; `ingest.py` mid-file imports; `Datasets.tsx` exhaustive-deps warning; `save_image_bytes` labels unknown formats `.png`.
**SP2:** per-class annotation counts not surfaced; pull's delete-then-insert is non-atomic; **`LabelStudioGateway.export_json`/`project_stats` SDK shapes UNVERIFIED against a live LS** — adjust in the gateway on first real run; thin converter tests; create-project lacks idempotency/empty-class validation; a cancelled pull job can stay RUNNING (CancelledError bypasses `JobManager`'s `except Exception`).

## Next action
1. **Resolve the HARDWARE question** (M5 vs Lenovo + its GPU) — ask the user.
2. Then design **SP3 — Training engine**: real training behind the `TrainingBackend`, device handling per the answer, classification first (ViT/timm) then detection (RT-DETRv2/D-FINE), paste-HF-model resolve→classify→load + compat verdict, `report_to='trackio'` metrics into the dashboard, detection Trainer gotchas (`remove_unused_columns=False`, `eval_do_concat_batches=False`, per-image `labels` dicts). Same spec → plan → subagent-driven-Opus flow. Smoke a small **real** training run on the actual target machine early.
