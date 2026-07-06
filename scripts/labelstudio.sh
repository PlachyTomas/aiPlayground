#!/usr/bin/env bash
set -euo pipefail

WS="${VISIONSUITE_WORKSPACE:-$(pwd)/workspace}"
mkdir -p "$WS"
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(cd "$WS" && pwd -P)"

echo "Launching Label Studio on http://localhost:8080"
echo "Document root: $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"
echo "First run: create an account, then Account & Settings -> copy the (legacy) token,"
echo "and export it for VisionSuite:  export LABEL_STUDIO_API_KEY=<token>"
exec label-studio start
