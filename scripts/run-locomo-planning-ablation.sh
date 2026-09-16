#!/usr/bin/env bash
# Run one complete LoCoMo conversation with the two WitnessRAG planning modes,
# then print a strict paired comparison.  Outputs must be new directories.
# Use LOCOMO_CONVERSATION=all para executar as 10 conversas (categorias 1 e 4).
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT_OUT=${1:-"runs/locomo-planning-ablation-$STAMP"}
SIMPLE_OUT="${ROOT_OUT}-simple"
MULTI_OUT="${ROOT_OUT}-multi"

GPU=${GPU:-4}
PORT=${PORT:-8089}
SIMPLE_PORT=${SIMPLE_PORT:-$PORT}
MULTI_PORT=${MULTI_PORT:-$((PORT + 1))}
SIMPLE_HOURS=${SIMPLE_HOURS:-2}
MULTI_HOURS=${MULTI_HOURS:-6}
LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-0}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/locomo-witness-canonical-conv0/cache"}
CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
IE_WINDOW_TOKENS=${IE_WINDOW_TOKENS:-512}
TOP_K=${TOP_K:-5}
WITNESS_CANDIDATE_POOL=${WITNESS_CANDIDATE_POOL:-20}
MAX_QUERY_PLANS=${MAX_QUERY_PLANS:-5}
VERIFY_WITNESSES=${VERIFY_WITNESSES:-1}

if [ "$SIMPLE_PORT" = "$MULTI_PORT" ]; then
  echo "erro: SIMPLE_PORT e MULTI_PORT não podem ser iguais (porta $SIMPLE_PORT)." >&2
  exit 1
fi

verify_args=()
if [ "$VERIFY_WITNESSES" = "1" ]; then
  verify_args+=(--verify-witnesses)
fi

common=(
  --dataset locomo --locomo-conversation "$LOCOMO_CONVERSATION"
  --methods witnessrag
  --model Qwen/Qwen2.5-14B-Instruct
  --embed-model BAAI/bge-m3
  --locomo-chunk-tokens "$CHUNK_TOKENS"
  --locomo-ie-window-tokens "$IE_WINDOW_TOKENS"
  --top-k "$TOP_K" --witness-candidate-pool "$WITNESS_CANDIDATE_POOL"
  --binding-aware-grounding
  --answer-set --vocab-compile --hybrid-fallback --dialogue-ie
  --gpu "$GPU"
  --vllm-python "$PWD/.venv-vllm/bin/python"
  --cache-dir "$CACHE_DIR"
  --max-query-plans "$MAX_QUERY_PLANS"
  "${verify_args[@]}"
)

echo "== planejamento simples =="
.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --port "$SIMPLE_PORT" \
  --hours "$SIMPLE_HOURS" --output "$SIMPLE_OUT"

echo "== planejamento adaptativo (até cinco planos) =="
.venv-bench/bin/python -m wrag.pilot "${common[@]}" --query-plans \
  --port "$MULTI_PORT" \
  --hours "$MULTI_HOURS" --output "$MULTI_OUT"

echo "== comparação pareada =="
.venv-bench/bin/python scripts/compare-planning.py "$SIMPLE_OUT" "$MULTI_OUT"

echo
echo "Rodadas salvas em:"
echo "  $SIMPLE_OUT"
echo "  $MULTI_OUT"
