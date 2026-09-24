#!/usr/bin/env bash
# LoCoMo on the Petrobras Azure deployment with the proof controller (design v3):
#   Buscar -> Planejar (with evidence) -> Provar -> Verificar -> Responder,
# for every question and without the LoCoMo category label.
#
# PROFILE=proof             the registered configuration (default).
# PROFILE=proof-no-verify   ablation: usable proofs change the context unverified.
# PROFILE=proof-partial     ablation: agreement probes / gap rescue in the last
#                           slot when a connected plan has no proof (the
#                           selective controller's multi-hop fallback).
# PROFILE=proof-1cycle      ablation: a single plan, no replanning.
#
# CACHE_DIR defaults to the cache of run-witness-azure-locomo.sh. Sharing it is
# deliberate: OpenIE/index prompts are unchanged, and every question whose
# context equals the hybrid context sends a byte-identical reader prompt, so the
# answer comes from the cache and is identical. Only planning, verification and
# changed contexts are new calls. Compare afterwards with:
#   python scripts/proof-report.py --proof <this output> \
#       --reference runs/witness-suite-locomo-azure-selective \
#       [--hybrid runs/witness-suite-locomo-azure-hybrid]
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
PROFILE=${PROFILE:-proof}
OUTPUT=${1:-runs/witness-suite-locomo-azure-$PROFILE}
GPU=${GPU:-1}
CONVERSATION=${LOCOMO_CONVERSATION:-all}
DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
# Reader budget (scripts/run-locomo-budget.sh): chunks handed to the reader
# and chunk size. The registered configuration is 5 x 2048 tokens.
TOP_K=${TOP_K:-5}
CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/witness-azure"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_LLM_BACKEND=azure
export WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT"
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}
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
# The graph flags are those of the selective run, so the memory (OpenIE, entity
# clusters, relation families) is the same; only the controller differs.
METHOD_FLAGS=(--binding-aware-grounding --vocab-compile --hybrid-fallback
  --dialogue-ie --gap-context-rescue --proof-controller)
case "$PROFILE" in
  proof) ;;
  proof-no-verify) METHOD_FLAGS+=(--no-proof-verify) ;;
  proof-partial) METHOD_FLAGS+=(--partial-evidence) ;;
  proof-1cycle) METHOD_FLAGS+=(--proof-cycles 1) ;;
  *)
    echo "PROFILE must be proof, proof-no-verify, proof-partial or proof-1cycle." >&2
    exit 2
    ;;
esac
"$BENCH_PYTHON" -m wrag.pilot --backend azure --gpu "$GPU" \
  --model "$DEPLOYMENT" --tokenizer-model "$TOKENIZER_MODEL" \
  --dataset locomo --locomo-conversation "$CONVERSATION" --methods witnessrag \
  --embed-model "$WRAG_EMBED_MODEL" --embed-device "$WRAG_EMBED_DEVICE" \
  --concurrency "$WRAG_AZURE_CONCURRENCY" \
  --locomo-chunk-tokens "$CHUNK_TOKENS" --locomo-ie-window-tokens 512 --top-k "$TOP_K" --qa-max-tokens 128 \
  --witness-candidate-pool 20 --answer-set --temporal-annotations --evidence-reader \
  "${METHOD_FLAGS[@]}" \
  --cache-dir "$CACHE_DIR" \
  --hours "${HOURS:-72}" --output "$OUTPUT" "${RESUME[@]}"
