# VisionSuite

Local, single-user suite for training Hugging Face vision models (object detection +
image classification) on Apple Silicon. See `docs/superpowers/specs/` for the design.

## Dev quickstart

```bash
uv sync --all-packages          # Python env (core + api)
cd web && npm install && cd ..  # frontend deps
./scripts/dev.sh                # FastAPI :8000 + Vite dev server (proxies /api)
```

Open the Vite URL, click **Start dummy run**, and watch logs stream over WebSocket.

## Tests

```bash
uv run pytest            # backend
cd web && npx vitest run # frontend
```

## Labeling (Label Studio)

VisionSuite uses a local Label Studio for annotation.

1. Install it once: `pip install label-studio` (or `uv tool install label-studio`).
2. Start it (serves local files from your workspace): `./scripts/labelstudio.sh`
3. On first run, create an account, open **Account & Settings**, copy the **legacy** access token, and:
   `export LABEL_STUDIO_API_KEY=<token>` (and `LABEL_STUDIO_URL` if not on :8080), then start the app.
4. In VisionSuite → **Labeling**: pick a dataset, enter classes, **Create labeling project** (this
   creates the LS project and syncs the dataset's images via Local Storage), click **Open in Label
   Studio** to label, then **Pull annotations** to import them back as COCO.
