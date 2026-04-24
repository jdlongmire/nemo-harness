#!/usr/bin/env bash
##
## Download all Python dependencies as wheels for air-gapped deployment.
##
## Usage (run on a connected machine):
##   ./scripts/vendor-deps.sh
##
## This creates vendor/ with all .whl files. Copy the entire nemo-harness
## directory to the air-gapped machine and docker compose up.
##

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$PROJECT_DIR/vendor"

echo "Downloading dependencies to $VENDOR_DIR..."
mkdir -p "$VENDOR_DIR"

pip download \
    -r "$PROJECT_DIR/requirements.txt" \
    -d "$VENDOR_DIR" \
    --platform manylinux2014_x86_64 \
    --python-version 311 \
    --only-binary=:all: \
    2>/dev/null || \
pip download \
    -r "$PROJECT_DIR/requirements.txt" \
    -d "$VENDOR_DIR"

echo ""
echo "Vendored $(ls "$VENDOR_DIR"/*.whl 2>/dev/null | wc -l) wheel files to $VENDOR_DIR/"
echo ""
echo "Next steps for air-gapped deployment:"
echo "  1. Pre-pull Docker images:"
echo "     docker pull gitea/gitea:1.22"
echo "     docker pull ollama/ollama:latest"
echo "     docker pull python:3.11-slim"
echo "     docker save -o nemo-images.tar gitea/gitea:1.22 ollama/ollama:latest python:3.11-slim"
echo ""
echo "  2. Pre-download Ollama models:"
echo "     ollama pull nomic-embed-text"
echo "     ollama pull nemotron-3-nano-4b  (or your preferred model)"
echo "     # Models stored in ~/.ollama/models/"
echo ""
echo "  3. Copy everything to air-gapped machine and run:"
echo "     docker load -i nemo-images.tar"
echo "     docker compose up -d"
