#!/usr/bin/env bash
# Existing Qwen vLLM server; local program search and one reflective reader.
set -euo pipefail
cd "$(dirname "$0")/.."
export LLM=qwen PROFILE=witnessrag-local
export GPU=${GPU:-7}
export EMBED_DEVICE=${EMBED_DEVICE:-cuda}
# Match the completed GPT-4o-mini standard run, including its MiniLM reranker.
export EXTRA_FLAGS="--fact-budget ${FACT_BUDGET:-40} --fact-rerank ${FACT_RERANK:-cross-encoder/ms-marco-MiniLM-L6-v2} ${EXTRA_FLAGS:-}"
unset WRAG_REFLECTION_STUDY_ROOT WRAG_CONTROLLED_ROOT WRAG_FROZEN_MEMORY_SOURCE
exec bash scripts/run-witness-proof-locomo.sh "${1:-runs/witnessrag-local-qwen}"
