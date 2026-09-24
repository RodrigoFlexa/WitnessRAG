#!/usr/bin/env bash
# LoCoMo on the Petrobras Azure deployment, category-agnostic WitnessRAG.
#
# PROFILE=agnostic         planner (evidence contract) decides the route; no
#                          benchmark label is read. Same composition core,
#                          reader, top-k and cache as the labelled selective run.
# PROFILE=agnostic-lenses  the same plus memory lenses chosen by the contract
#                          (recency/stability, salience, corroborated confidence).
# PROFILE=lens-temporal|lens-salience|lens-confidence  single-lens ablations.
# PROFILE=compose-all      ablation: plan, then send every question to COMPOSE.
# PROFILE=direct-all       ablation: plan, then send every question to DIRECT.
#
# The default CACHE_DIR is the one used by run-witness-azure-locomo.sh. Sharing
# it is deliberate: when the contract route agrees with the labelled route the
# compile and reader prompts are byte-identical, so they are served from cache
# and the answers are identical. Only contract calls and disagreements are new.
# Compare afterwards with:
#   python scripts/router-report.py --agnostic <this output> \
#       --labelled runs/witness-suite-locomo-azure-selective
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
PROFILE=${PROFILE:-agnostic}
OUTPUT=${1:-runs/witness-suite-locomo-azure-$PROFILE}
GPU=${GPU:-1}
CONVERSATION=${LOCOMO_CONVERSATION:-all}
DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
# The cache is intentionally independent of OUTPUT. A new experimental output
# must not repay the unchanged OpenIE/index prompts for all ten conversations.
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/witness-azure"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_LLM_BACKEND=azure
export WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT"
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}
# Pin the registered profile instead of letting unrelated legacy WRAG_* values
# from .env silently change the experimental condition.
export WRAG_AZURE_CONCURRENCY=${AZURE_CONCURRENCY:-4}
export WRAG_CONTINUE_ON_CONTENT_FILTER=1
export WRAG_EMBED_BACKEND=st
export WRAG_EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
export WRAG_EMBED_DEVICE=${EMBED_DEVICE:-cpu}
export WRAG_CACHE_DIR="$CACHE_DIR"
export WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42

"$BENCH_PYTHON" -m wrag.cli diag-azure
RESUME=()
[[ -f "$OUTPUT/pilot.json" ]] && RESUME+=(--resume)
METHOD_FLAGS=(--binding-aware-grounding --vocab-compile --hybrid-fallback
  --dialogue-ie --selective-witness --gap-context-rescue --agnostic-router)
case "$PROFILE" in
  agnostic) ;;
  agnostic-lenses) METHOD_FLAGS+=(--memory-lenses --lens-max-swaps "${LENS_MAX_SWAPS:-1}") ;;
  # Single-lens ablations.
  lens-temporal|lens-salience|lens-confidence)
    METHOD_FLAGS+=(--memory-lenses --lenses "${PROFILE#lens-}"
      --lens-max-swaps "${LENS_MAX_SWAPS:-1}") ;;
  # Routing ablations: the planner still runs, the route is forced.
  compose-all) METHOD_FLAGS+=(--route-override compose) ;;
  direct-all) METHOD_FLAGS+=(--route-override direct) ;;
  *)
    echo "PROFILE must be agnostic, agnostic-lenses, lens-temporal, lens-salience," \
      "lens-confidence, compose-all or direct-all." >&2
    exit 2
    ;;
esac
"$BENCH_PYTHON" -m wrag.pilot --backend azure --gpu "$GPU" \
  --model "$DEPLOYMENT" --tokenizer-model "$TOKENIZER_MODEL" \
  --dataset locomo --locomo-conversation "$CONVERSATION" --methods witnessrag \
  --embed-model "$WRAG_EMBED_MODEL" --embed-device "$WRAG_EMBED_DEVICE" \
  --concurrency "$WRAG_AZURE_CONCURRENCY" \
  --locomo-chunk-tokens 2048 --locomo-ie-window-tokens 512 --top-k 5 --qa-max-tokens 128 \
  --witness-candidate-pool 20 --answer-set --temporal-annotations --evidence-reader \
  "${METHOD_FLAGS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --hours "${HOURS:-72}" --output "$OUTPUT" "${RESUME[@]}"
