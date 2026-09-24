#!/usr/bin/env bash
# LoCoMo on the Petrobras Azure deployment. No local vLLM server is started.
set -euo pipefail
cd "$(dirname "$0")/.."

# LLM=azure (default): gpt-4-1-mini-petrobras through .env.
# LLM=qwen: an already-running vLLM server on 127.0.0.1:$PORT
#           (scripts/serve-qwen-vllm.sh), MODEL=Qwen/Qwen2.5-14B-Instruct.
LLM=${LLM:-azure}
case "$LLM" in
  azure)
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
    ;;
  qwen) ;;
  *) echo "LLM must be azure or qwen." >&2; exit 2 ;;
esac

BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
if [[ "$LLM" == qwen ]]; then OUTPUT=${1:-runs/witness-suite-locomo-qwen-${METHOD:-witnessrag}}; else OUTPUT=${1:-runs/witness-suite-locomo-azure-selective}; fi
GPU=${GPU:-1}
CONVERSATION=${LOCOMO_CONVERSATION:-all}
DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
METHOD=${METHOD:-witnessrag}
TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
# The cache is intentionally independent of OUTPUT. A new experimental output
# must not repay the unchanged OpenIE/index prompts for all ten conversations.
# Reader budget (scripts/run-locomo-budget.sh): chunks handed to the reader
# and chunk size. The registered configuration is 5 x 2048 tokens.
TOP_K=${TOP_K:-5}
CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/witness-$LLM"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_CONTINUE_ON_CONTENT_FILTER=1
export WRAG_EMBED_BACKEND=st
export WRAG_EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
export WRAG_EMBED_DEVICE=${EMBED_DEVICE:-cpu}
export WRAG_CACHE_DIR="$CACHE_DIR"
export WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42

if [[ "$LLM" == qwen ]]; then
  PORT=${PORT:-8095}
  MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
  if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "No vLLM server at http://127.0.0.1:$PORT/v1; start scripts/serve-qwen-vllm.sh first." >&2
    exit 1
  fi
  BACKEND_FLAGS=(--backend vllm --existing-server --port "$PORT" --model "$MODEL"
    --concurrency "${CONCURRENCY:-16}")
else
  export WRAG_LLM_BACKEND=azure
  export WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT"
  export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}
  # Pin the registered profile instead of letting unrelated legacy WRAG_* values
  # from .env silently change the experimental condition.
  export WRAG_AZURE_CONCURRENCY=${AZURE_CONCURRENCY:-4}
  "$BENCH_PYTHON" -m wrag.cli diag-azure
  BACKEND_FLAGS=(--backend azure --model "$DEPLOYMENT" --concurrency "$WRAG_AZURE_CONCURRENCY")
fi
RESUME=()
[[ -f "$OUTPUT/pilot.json" ]] && RESUME+=(--resume)
METHOD_FLAGS=()
case "$METHOD" in
  witnessrag)
    METHOD_FLAGS=(--binding-aware-grounding --vocab-compile --hybrid-fallback
      --dialogue-ie --selective-witness --gap-context-rescue)
    ;;
  hybrid)
    # Paired ablation: same corpus, embedder, reader and top-k, without OpenIE,
    # compilation or graph intervention.
    ;;
  *)
    echo "METHOD must be witnessrag or hybrid." >&2
    exit 2
    ;;
esac
"$BENCH_PYTHON" -m wrag.pilot "${BACKEND_FLAGS[@]}" --gpu "$GPU" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --dataset locomo --locomo-conversation "$CONVERSATION" --methods "$METHOD" \
  --embed-model "$WRAG_EMBED_MODEL" --embed-device "$WRAG_EMBED_DEVICE" \
  --locomo-chunk-tokens "$CHUNK_TOKENS" --locomo-ie-window-tokens 512 --top-k "$TOP_K" --qa-max-tokens 128 \
  --witness-candidate-pool 20 --answer-set --temporal-annotations --evidence-reader \
  "${METHOD_FLAGS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --hours "${HOURS:-72}" --output "$OUTPUT" "${RESUME[@]}"
