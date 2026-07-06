# VisionSuite De-Risking Sweep — Synthesis Brief

> Source: parallel multi-agent research sweep (6 agents, ~361k tokens), 2026-07-06.
> Verifies the riskiest technical assumptions behind the VisionSuite plan: MPS vision
> training, Label Studio SDK round-trip, Trackio embedding, HF model import/compat, export.
> Full per-topic findings + sources archived at the end of this file.

## 1. Confirmed

- **Classification trains clean on MPS, no flag.** ViT (`google/vit-base-patch16-224`), timm ResNet/MobileNetV3/EfficientNet use only native-kernel ops (conv, BN, GELU/SiLU/hardswish, matmul, softmax, interpolate). Fastest, lowest-memory, best fanless fit — the local-training core.
- **Deformable detectors do NOT hard-fail on MPS.** RT-DETR/v2, D-FINE, Deformable DETR: the fast CUDA deformable kernel is gated behind `@use_kernel_forward_from_hub`, so on MPS HF auto-routes to a pure-PyTorch `grid_sample` path. Models instantiate and run — failure mode is a CPU-fallback warning, not a crash.
- **DETR (`facebook/detr-resnet-50`) and YOLOS (`hustvl/yolos-tiny`) are grid_sample-free** and train cleanly without the fallback flag — the guaranteed-clean detection fallbacks.
- **bf16 autocast works on MPS** (`torch.autocast('mps', dtype=torch.bfloat16)`), halves activation memory, requires macOS 14+.
- **Label Studio local round-trip is fully supported and stable** (with the rewritten SDK).
- **Trackio is real/current (v0.29.0), genuinely local-first, no cloud account.** `report_to="trackio"` logs from Trainer; both iframe-embed and programmatic read-back exist.
- **resolve→classify→load flow matches official docs.** Head-swap via `AutoModelForObjectDetection`/`AutoModelForImageClassification` + `id2label`/`label2id` + `ignore_mismatched_sizes=True` works across RT-DETR/DETR/YOLOS/ViT/timm.
- **Native `save_pretrained` + transformers in-app inference is guaranteed** for both tasks. **ONNX via Optimum covers the exact families** VisionSuite trains.

## 2. Corrections (plan was optimistic/wrong → corrected fact)

- **grid_sample on MPS is only partly native.** PyTorch 2.11 (23 Mar 2026) made the *forward* native; the *backward* (exercised in training) is **unconfirmed** and may still fall to CPU. → **Require torch>=2.11 AND keep `PYTORCH_ENABLE_MPS_FALLBACK=1` set for training** (not optional for deformable detectors on any version).
- **`torch.compile` on MPS is unreliable for training** (fails in `loss.backward()`, even ResNet-18). → **OFF for v1.**
- **fp16 autocast on MPS is NaN-prone.** → **Standardize on bf16.**
- **Label Studio SDK was rewritten twice.** All `Client`/`start_project`/`project.import_tasks` tutorials are dead. → Use `from label_studio_sdk import LabelStudio`, resource-namespaced methods; **pin `label-studio-sdk>=2,<3`** + server version.
- **Bbox coords are PERCENTAGES (0–100), not pixels**, and `original_width`/`original_height` live at the **result level, not inside `value`**. Forgetting this yields boxes ~100× too small. **Convert JSON→COCO yourself** — don't trust LS's COCO exporter (re-resolves `/data/local-files` URLs, common failure point).
- **timm breaks naive task-detection**: empty `config`, `auto_model="AutoModel"`, `library_name="timm"`. → **Gate on `library_name=="timm"` FIRST**, else every timm model misclassifies. timm = classification/backbone only; never route to `AutoModelForObjectDetection`.
- **`model_info` `config` is truncated** (only `architectures`+`model_type`). → Fetch full `config.json` (`hf_hub_download`/`AutoConfig`) for `num_labels`/`id2label`/`image_size` — still no weights.
- **ONNX exporter split into `optimum-onnx`** package → install `optimum[onnxruntime]` + `onnxruntime`.
- **No `ORTModelForObjectDetection` exists.** ONNX detection needs a hand-written post-processor; classification gets `ORTModelForImageClassification` free.
- **Core ML is not a v1 target** — coremltools tops out ~torch 2.7–2.8 (needs a second pinned torch env) and traces transformer detectors poorly.
- **Every DETR-family loss uses scipy `linear_sum_assignment` (CPU Hungarian matcher)** → a fixed GPU→CPU sync per step. Not a kernel gap; budget the overhead.
- **Trackio local-iframe embedding is undocumented** (only Space-hosted is). `report_to='trackio'` needs **transformers>=4.54.0**.

## 3. Curated model shortlist

**Classification (all clean, no flag — primary workhorses):**
- Default: timm **EfficientNet-B0/B1** or **ResNet-50** @224px, batch 32–128 bf16.
- **MobileNetV3** — smallest/coolest footprint.
- **ViT-base/16** (`google/vit-base-patch16-224`) — transformer option, batch 32–64.

**Detection — modern default (deformable, needs flag + torch>=2.11 + bf16):**
- **RT-DETRv2-R18** (`PekingU/rtdetr_v2_r18vd`) or **D-FINE-nano/small** (`ustc-community/dfine-nano-coco`, `dfine-small-coco`) @640px, batch 2–4.

**Detection — guaranteed-clean fallback (no grid_sample, no flag):**
- **YOLOS-tiny** (`hustvl/yolos-tiny`) ≤512px, or **DETR-R50** (`facebook/detr-resnet-50`) @~640px. Offer on any grid_sample/MPS issue.

**Gate behind a warning (OOM/thermal/slow or superseded):** RT-DETR-R101, D-FINE-large/xlarge, DETR-R101, ViT-large (reduce batch/res or push to HF Jobs); Deformable DETR (`SenseTime/deformable-detr` — works but strictly superseded by RT-DETR, deprioritize); DETR at native 800px (force downscale).

**Avoid:** fp16 autocast on MPS, `torch.compile` on MPS, and any **deformable-CONVOLUTION** model (DCNv2/v3, InternImage, `torchvision.ops.deform_conv2d`) — no MPS kernel, no graceful fallback. (Distinct from deformable *attention*, which is fine.) v1 registry must guard against `deform_conv2d`.

## 4. Spec impacts

**Architecture spec (global):**
- Pin **torch>=2.11**; set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the training-subprocess env; bf16 autocast (guard macOS 14+); `torch.compile` OFF for v1.
- Model registry: guard against `deform_conv2d`; classify timm by `library_name` first.
- Fanless throttling is the binding limit (~20–40% below burst on long runs): checkpoint frequently; offer offloading heavy detection to HF Jobs while keeping classification + small detection local.

**#2 Label Studio:**
- Run as standalone local process on :8080, launched with `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` + `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/abs/parent`. Do not embed.
- FastAPI talks to it **only via one `LabelStudio(base_url, api_key)` client**; SDK is sync httpx → wrap in `run_in_threadpool`.
- Share on-disk images via **Local Storage** (`ls.import_storage.local.create(...)` + `sync(...)`), not re-upload.
- Configs: `RectangleLabels` (detection), `Choices` `choice="single-radio"` (classification); config `name`/`toName` become `from_name`/`to_name`.
- Export: `exports.create` → poll `.status=='completed'` → `download(export_type='JSON')`; convert to COCO in your own converter: `x_px=value.x/100*original_width` (etc.), COCO bbox `[x,y,w,h]` xywh top-left matches LS directly; build `category_id` map from config `<Label>` values; **guard `value.rotation!=0`** (forbid `canRotate` or compute axis-aligned enclosing box).
- Store token type; prefer a **legacy token** for any raw REST fallback.

**#3 training + model import/compat + Trackio:**
- **Import pipeline (3 stages):** RESOLVE (`HfApi.model_info` to catch Gated/NotFound early, then full `config.json`) → CLASSIFY with precedence `library_name=="timm"` → `transformersInfo.auto_model` → `architectures[0]` suffix → `pipeline_tag` → `model_type` map → LOAD (`AutoModelFor{ObjectDetection,ImageClassification}` + `id2label`/`label2id` + `ignore_mismatched_sizes=True`, paired with `AutoImageProcessor` from same repo id).
- **Compat verdict enum:** KNOWN-GOOD-MPS = allowlist smoke-tested on M5 (`vit`, timm `resnet`/`mobilenet`/`efficientnet`, `detr`, `yolos`); UNTESTED = valid task+auto-class not yet tested — includes `rt_detr`/`rt_detr_v2`/`deformable_detr` + a "requires `PYTORCH_ENABLE_MPS_FALLBACK=1`, grid_sample may fall to CPU" warning; UNSUPPORTED = no auto-class/task, `trust_remote_code` needed, gated/no-token, or timm-detection. Promote UNTESTED→KNOWN-GOOD only after a real 1-step forward+backward passes on-device.
- **Detection training gotchas (hard-code into Trainer setup):** `remove_unused_columns=False`, `eval_do_concat_batches=False`, collator keeps `labels` as a list of per-image dicts (`{class_labels, boxes}`) — only `pixel_values`/`pixel_mask` stacked. RT-DETR handles no-object internally (`num_labels`=real classes, no +1).
- **Trackio:** `report_to='trackio'` (transformers>=4.54.0), set `project`+`run_name`. Phase 0: iframe-embed `http://127.0.0.1:7860/?project=<p>&sidebar=hidden&footer=false&metrics=...` (verify local framing empirically). Phase 1 (product path): backend polls **`trackio get metric --project P --run R --metric M --json`** every 2–5s → SSE/WebSocket → Recharts/uPlot in React. **Never parse the SQLite file directly** (internal, pre-release). Pin `trackio==0.29.0`. Reserve `space_id` for optional share only.

**#4 eval/export/inference:**
- **Commit (guaranteed):** native `save_pretrained` (model + image_processor) for both tasks; in-app sample inference via transformers directly on the saved checkpoint — `AutoImageProcessor` + `post_process_object_detection(outputs, threshold, target_sizes=[(H,W)])` with PIL box drawing (main), `pipeline(...)` as 3-line fallback (already returns original-pixel coords). **Run the single test image on CPU** to dodge all MPS op-coverage edge cases.
- **Commit (clean):** ONNX **classification** via `optimum-cli export onnx --task image-classification` + `ORTModelForImageClassification` (pipeline-compatible).
- **Commit-with-caveat:** ONNX **detection** (DETR/RT-DETR/RT-DETRv2/YOLOS/D-FINE) — export works on Apple Silicon but needs **opset≥16** for grid_sample models, and since there's no `ORTModelForObjectDetection` you bundle a ~30-line post-processor (sigmoid/softmax, cxcywh→xyxy, rescale) and run `onnxruntime` on **CPUExecutionProvider**. Label it "ONNX (detection, CPU)". Query dim is static (DETR/YOLOS ~100, RT-DETR 300); only batch axis is dynamic — don't promise dynamic image H/W.
- **Defer Core ML** entirely for v1. Treat onnxruntime CoreML-EP/ANE as best-effort only — check `onnxruntime.get_available_providers()` at runtime, never require it.

## 5. Open risks (validate during implementation)

- **`grid_sampler_2d_backward` MPS status unconfirmed** — run a real 1-step forward+backward on the M5 for RT-DETRv2/D-FINE before trusting; keep the fallback flag regardless.
- **Batch/memory table is estimates, not benchmarks** — probe OOM per model on 24GB (~14–18GB usable); tune `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.
- **MPS can silently produce wrong results on some ops** — validate a fine-tuned model's eval metrics against a short CPU run, especially deformable detectors on a fresh PyTorch.
- **Trackio unknowns:** local-iframe embedding not documented; whether bare `trackio show` exposes the POST `/api/{tool}` JSON endpoints (may need `trackio[mcp]` + `--mcp-server`); port may not be 7860 if taken — capture the printed URL. CLI `--json` path avoids all of these.
- **Label Studio token availability** — a fresh install may offer only JWT personal-access tokens (org setting); confirm a legacy token is obtainable if raw REST is needed. `AsyncLabelStudio` existence in the pinned 2.0.x unverified.
- **onnxruntime CoreML-EP/ANE availability** in the stock arm64 wheel is version-dependent — verify at runtime; community wheels (`onnxruntime-silicon`) may be needed if ANE is later wanted.
- **Fanless sustained throughput** unknown until measured on long runs — informs whether heavy detection must go to HF Jobs.
- **Core ML** — if demand appears later, expect a separate pinned-torch conversion env and classification-only support.

---

## Appendix — key verified specifics (for implementers)

**MPS / training env:**
- `deformable ATTENTION` (grid_sample, used by RT-DETR/D-FINE) ≠ `deformable CONVOLUTION` (DCNv2/v3, `torchvision.ops.deform_conv2d`, no MPS kernel). Registry must reject the latter.
- Memory/batch guidance (24GB unified, ~14–18GB usable — estimates, probe per model): Detection @640px — D-FINE-nano/RT-DETR-R18 batch 4–8; D-FINE-small/RT-DETR-R34 batch 2–4; D-FINE-medium/RT-DETR-R50 batch 1–2. Classification @224px — MobileNetV3/EfficientNet-B0 64–128; ResNet-50 32–64; ViT-base 32–64.

**Label Studio bbox result JSON (verbatim shape):**
```json
{
  "type": "rectanglelabels", "from_name": "label", "to_name": "image",
  "original_width": 600, "original_height": 403, "image_rotation": 0,
  "value": { "rotation": 0, "x": 4.98, "y": 12.82, "width": 32.52, "height": 44.91,
             "rectanglelabels": ["Airplane"] }
}
```
x/y/w/h are PERCENT of original_width/height; x,y = top-left. Classification result: `{"type":"choices","from_name":"choice","to_name":"image","value":{"choices":["Airbus"]}}`.

**Verified repo IDs:** RT-DETRv2 `PekingU/rtdetr_v2_r18vd` (r34/r50/r101 variants); RT-DETR v1 `PekingU/rtdetr_r18vd`,`rtdetr_r50vd`; D-FINE `ustc-community/dfine-{nano,small,medium,large,xlarge}-coco`; DETR `facebook/detr-resnet-50`; Deformable DETR `SenseTime/deformable-detr`; YOLOS `hustvl/yolos-tiny`,`yolos-small`; ViT `google/vit-base-patch16-224`.

**Key sources:** transformers `perf_train_special`, `tasks/object_detection`, `tasks/image_classification`, `model_doc/rt_detr_v2`, `timm_wrapper`; PyTorch 2.11 release blog + issues #97606/#139386/#161905; labelstud.io guide (sdk, storage_local, export, task_format, tags/rectanglelabels, tags/choices); huggingface.co/docs/trackio (index, deploy_embed, transformers_integration, api_mcp_server); huggingface.co/docs/optimum-onnx/onnx/overview; coremltools 9.0 release.

_Confidence: MPS-training medium (grid_sample backward unconfirmed); all other topics high._
