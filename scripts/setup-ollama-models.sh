#!/usr/bin/env bash
##
## Pull required Ollama models for the Nemo Harness.
## Run this on a connected machine before air-gapped deployment.
##

set -euo pipefail

echo "Pulling embedding model..."
ollama pull nomic-embed-text

echo "Pulling inference model..."
ollama pull nemotron-3-nano-4b

echo ""
echo "Models ready. For air-gapped transfer:"
echo "  Ollama models are stored in ~/.ollama/models/"
echo "  Copy that directory to the target machine's ollama_data volume."
