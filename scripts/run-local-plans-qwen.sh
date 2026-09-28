#!/usr/bin/env bash
# Existing Qwen vLLM server; local program search and one reflective reader.
set -euo pipefail
cd "$(dirname "$0")/.."
export LLM=qwen PROFILE=witnessrag-local
export EMBED_DEVICE=${EMBED_DEVICE:-cuda}
# This is a distinct method, not a fifth cell of the reflection ablation.
unset WRAG_REFLECTION_STUDY_ROOT WRAG_CONTROLLED_ROOT
exec bash scripts/run-witness-proof-locomo.sh "${1:-runs/witnessrag-local-qwen}"
