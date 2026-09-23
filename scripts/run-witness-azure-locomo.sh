#!/usr/bin/env bash
# LoCoMo on the Petrobras Azure deployment. No local vLLM server is started.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY in .env or the environment}"
if [[ -z "${AZURE_OPENAI_BASE_URL:-}" && -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
  echo "Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT." >&2
  exit 1
fi

BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
OUTPUT=${1:-runs/witness-suite-locomo-azure-evidence}
GPU=${GPU:-1}
CONVERSATION=${LOCOMO_CONVERSATION:-all}
DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
CACHE_DIR=${WRAG_CACHE_DIR:-"$OUTPUT/cache"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_LLM_BACKEND=azure
export WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT"
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}
export WRAG_AZURE_CONCURRENCY=${WRAG_AZURE_CONCURRENCY:-4}
export WRAG_CONTINUE_ON_CONTENT_FILTER=1
export WRAG_EMBED_BACKEND=st
export WRAG_EMBED_MODEL=${WRAG_EMBED_MODEL:-BAAI/bge-m3}
export WRAG_EMBED_DEVICE=${WRAG_EMBED_DEVICE:-cpu}
export WRAG_CACHE_DIR="$CACHE_DIR"
export WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42

"$BENCH_PYTHON" -m wrag.cli diag-azure
RESUME=()
[[ -f "$OUTPUT/pilot.json" ]] && RESUME+=(--resume)
IMPROVEMENTS=()
[[ "${EVIDENCE_READER:-1}" == 1 ]] && IMPROVEMENTS+=(--evidence-reader)
[[ "${GAP_CONTEXT_RESCUE:-1}" == 1 ]] && IMPROVEMENTS+=(--gap-context-rescue)
"$BENCH_PYTHON" -m wrag.pilot --backend azure --gpu "$GPU" \
  --model "$DEPLOYMENT" --tokenizer-model "$TOKENIZER_MODEL" \
  --dataset locomo --locomo-conversation "$CONVERSATION" --methods witnessrag \
  --embed-model "$WRAG_EMBED_MODEL" --embed-device "$WRAG_EMBED_DEVICE" \
  --concurrency "$WRAG_AZURE_CONCURRENCY" \
  --locomo-chunk-tokens 2048 --locomo-ie-window-tokens 512 --top-k 5 \
  --witness-candidate-pool 20 --binding-aware-grounding --answer-set \
  --vocab-compile --hybrid-fallback --dialogue-ie --query-plans \
  --max-query-plans 5 --active-frontier --active-obligations --active-context \
  --active-operators --soft-obligations --temporal-memory \
  --complementary-context --temporal-annotations "${IMPROVEMENTS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --hours "${HOURS:-72}" --output "$OUTPUT" "${RESUME[@]}"
