#!/usr/bin/env bash
# Component ablation on all ten LoCoMo conversations through the Petrobras
# Azure gateway (gpt-4.1-mini). No vLLM server is needed.
#
#   bash scripts/run-locomo-ablation-azure.sh
#
# Two stages, both resumable (run the same command again after an interruption):
#   1. ANTES: the v2 configuration (no planner requirements, no member scan),
#      complete variant only; it is the "before" reference printed by stage 2.
#   2. Ablation with proof requirements + member scan: full, no-witness,
#      no-time-reference, no-time-model, no-reflection (docs/requisitos-prova.md).
#
# Environment knobs:
#   FACTS=10            fact budget delivered to the reader (10 or 40)
#   DEPLOYMENT=gpt-4-1-mini-petrobras
#   CONCURRENCY=4       parallel Azure requests
#   EMBED_DEVICE=auto   cuda when nvidia-smi works, otherwise cpu
#   GPU=0               physical GPU for embeddings/reranker (when cuda)
#   GATE=0              1: skip the other variants if 'full' does not beat ANTES
#   CACHE_DIR=runs/.cache/witness-azure   (reuses earlier Azure prompts by content)
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
export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}

PY=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$PY" ]] || PY=${BENCH_PYTHON:-"$PWD/venv/bin/python"}
[[ -x "$PY" ]] || PY=${BENCH_PYTHON:-"$PWD/.venv/bin/python"}
FACTS=${FACTS:-10}
DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
CONCURRENCY=${CONCURRENCY:-4}
GPU=${GPU:-0}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/witness-azure"}
OUT=${OUT:-"$PWD/runs/locomo-ablation-azure-facts$FACTS"}
EMBED_DEVICE=${EMBED_DEVICE:-auto}
if [[ "$EMBED_DEVICE" == auto ]]; then
  if nvidia-smi >/dev/null 2>&1; then EMBED_DEVICE=cuda; else EMBED_DEVICE=cpu; fi
fi
mkdir -p "$OUT" "$CACHE_DIR"
LOG="$OUT/progress.log"

WRAG_LLM_BACKEND=azure WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT" "$PY" -m wrag.cli diag-azure

COMMON=(--conversation all --backend azure --deployment "$DEPLOYMENT" --concurrency "$CONCURRENCY"
        --embed-device "$EMBED_DEVICE" --gpu "$GPU" --fact-budget "$FACTS" --cache "$CACHE_DIR" --hours 168)

echo "=================== ANTES: configuracao v2, $FACTS fatos, 10 conversas ===================" | tee -a "$LOG"
"$PY" -u scripts/run-locomo-ablation-conv.py "${COMMON[@]}" --variants full --no-gate --no-requirements \
  --output "$OUT/antes-v2" 2>&1 | tee -a "$LOG"

GATE_FLAG=(--no-gate)
[[ "${GATE:-0}" == 1 ]] && GATE_FLAG=()
echo "=================== ABLACAO: requisitos + varredura de membros, $FACTS fatos ===================" | tee -a "$LOG"
"$PY" -u scripts/run-locomo-ablation-conv.py "${COMMON[@]}" --member-scan "${GATE_FLAG[@]}" \
  --before "$OUT/antes-v2/full" --output "$OUT/ablacao" 2>&1 | tee -a "$LOG"

echo "Resumo: $OUT/ablacao/summary.md" | tee -a "$LOG"
