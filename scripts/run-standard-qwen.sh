#!/usr/bin/env bash
# Standard local-v2, explicit fact budget, GPU embeddings/reranker, existing Qwen vLLM.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-locomo}
case "$TASK" in
  locomo|memoryagentbench) ;;
  *) echo "Usage: bash scripts/run-standard-qwen.sh {locomo|memoryagentbench} [output]" >&2; exit 2 ;;
esac
export FACT_BUDGET=${FACT_BUDGET:-40}
if [[ ! "$FACT_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "FACT_BUDGET must be a positive integer." >&2; exit 2
fi
OUTPUT=${2:-runs/standard-$TASK-qwen14b-facts$FACT_BUDGET}
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_PYTHON" ]] || { echo "Benchmark Python not found: $BENCH_PYTHON" >&2; exit 1; }
export GPU=${GPU:-7} PORT=${PORT:-8095}
export MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
export MODEL_REVISION=${MODEL_REVISION:-}
export EMBED_DEVICE=cuda EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
export FACT_RERANK=${FACT_RERANK:-cross-encoder/ms-marco-MiniLM-L6-v2}
export BENCH_PYTHON
export CUDA_VISIBLE_DEVICES=${EMBED_GPU:-$GPU} CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Select one GPU for embeddings with EMBED_GPU; the vLLM server may use several GPUs." >&2
  exit 2
fi
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY=${VLLM_API_KEY:-local-pilot} OPENAI_MODEL="$MODEL"
export WRAG_MODEL_REVISION="$MODEL_REVISION"
export WRAG_AZURE_CONCURRENCY=${CONCURRENCY:-8}
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
unset WRAG_REFLECTION_STUDY_ROOT WRAG_CONTROLLED_ROOT WRAG_FROZEN_MEMORY_SOURCE
"$BENCH_PYTHON" scripts/check-qwen-server.py --base-url "$OPENAI_BASE_URL" \
  --model "$MODEL" --check-cuda --output "$OUTPUT/cache/server.json"

if [[ "$TASK" == locomo ]]; then
  # pilot remaps --gpu to cuda:0 in its worker; use the embedding GPU here.
  export GPU="$CUDA_VISIBLE_DEVICES"
  export CACHE_DIR=${CACHE_DIR:-runs/.cache/standard-locomo-qwen14b}
  export LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-all}
  exec bash scripts/run-local-plans-qwen.sh "$OUTPUT"
fi

CACHE=${CACHE_DIR:-runs/.cache/memoryagentbench-qwen14b}
"$BENCH_PYTHON" -m benchmarks.memoryagentbench prepare --cache "$CACHE"
exec "$BENCH_PYTHON" -m benchmarks.memoryagentbench run \
  --backend vllm --base-url "$OPENAI_BASE_URL" --model "$MODEL" --model-revision "$MODEL_REVISION" \
  --embed-device cuda --embed-model "$EMBED_MODEL" --fact-budget "$FACT_BUDGET" --fact-rerank "$FACT_RERANK" \
  --conflict-recency-weight "${CONFLICT_RECENCY_WEIGHT:-0.65}" \
  --protocol "${PROTOCOL:-paper}" --suite "${SUITE:-all}" \
  --cache "$CACHE" --output "$OUTPUT" --resume
