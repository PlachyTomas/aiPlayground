#!/usr/bin/env bash
set -euo pipefail

uv run uvicorn visionsuite_api.main:app --reload --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

cd web && npm run dev
