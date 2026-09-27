#!/usr/bin/env bash
# Cascata com o Qwen no servidor (docs/cascata.md). Três rodadas com a mesma
# memória, o mesmo leitor e o mesmo cache, e o relatório pareado:
#   witnessrag-robust    sempre planeja (referência)
#   robust-no-plan       nunca planeja (mesma entrega)
#   witnessrag-cascade   uma chamada curta decide PLAN ou DIRECT
# A cascata roda por último: suas chamadas de plano e de leitor são as mesmas
# das outras duas rodadas e saem do cache; só o roteador é chamada nova.
#
#   bash scripts/serve-qwen-vllm.sh          # em outro terminal, se não estiver de pé
#   bash scripts/run-cascade-qwen.sh         # 10 conversas
#   LOCOMO_CONVERSATION=5 bash scripts/run-cascade-qwen.sh runs/qwen-cascade-conv05
#
# Variáveis: PORT (8095), GPU (1), EMBED_DEVICE (cuda), LOCOMO_CONVERSATION
# (all), VARIANTS (as três acima), ROOT (1º argumento; runs/qwen-cascade).
# Repetir o comando retoma cada rodada de onde parou.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=${1:-runs/qwen-cascade}
VARIANTS=${VARIANTS:-"witnessrag-robust robust-no-plan witnessrag-cascade"}
export LLM=qwen EMBED_DEVICE=${EMBED_DEVICE:-cuda} GPU=${GPU:-1} PORT=${PORT:-8095}
mkdir -p "$ROOT"
for variant in $VARIANTS; do
  echo "=== $(date '+%F %T') $variant -> $ROOT/$variant"
  PROFILE=$variant bash scripts/run-witness-proof-locomo.sh "$ROOT/$variant" \
    2>&1 | tee -a "$ROOT/$variant.log"
  echo "=== $(date '+%F %T') $variant terminou (código ${PIPESTATUS[0]})"
done
PY=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$PY" ]] || PY=python3
"$PY" scripts/paired-report.py "$ROOT" witnessrag-robust robust-no-plan witnessrag-cascade
"$PY" scripts/cascade-report.py "$ROOT"
