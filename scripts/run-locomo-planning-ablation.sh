#!/usr/bin/env bash
# Run one complete LoCoMo conversation with the two WitnessRAG planning modes,
# then print a strict paired comparison.  Outputs must be new directories.
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
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/locomo-witness-canonical-conv0/cache"}

if [ "$SIMPLE_PORT" = "$MULTI_PORT" ]; then
  echo "erro: SIMPLE_PORT e MULTI_PORT não podem ser iguais (porta $SIMPLE_PORT)." >&2
  exit 1
fi

common=(
  --dataset locomo --locomo-conversation 0
  --methods witnessrag
  --model Qwen/Qwen2.5-14B-Instruct
  --embed-model BAAI/bge-m3
  --locomo-chunk-tokens 2048
  --locomo-ie-window-tokens 512
  --top-k 5 --witness-candidate-pool 20
  --binding-aware-grounding --verify-witnesses
  --answer-set --vocab-compile --hybrid-fallback --dialogue-ie
  --gpu "$GPU"
  --vllm-python "$PWD/.venv-vllm/bin/python"
  --cache-dir "$CACHE_DIR"
)

echo "== planejamento simples =="
.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --port "$SIMPLE_PORT" \
  --hours "$SIMPLE_HOURS" --output "$SIMPLE_OUT"

echo "== múltiplos planos independentes =="
.venv-bench/bin/python -m wrag.pilot "${common[@]}" --query-plans \
  --port "$MULTI_PORT" \
  --hours "$MULTI_HOURS" --output "$MULTI_OUT"

echo "== comparação pareada =="
.venv-bench/bin/python scripts/compare-planning.py "$SIMPLE_OUT" "$MULTI_OUT"

echo
echo "Rodadas salvas em:"
echo "  $SIMPLE_OUT"
echo "  $MULTI_OUT"
