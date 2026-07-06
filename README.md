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
