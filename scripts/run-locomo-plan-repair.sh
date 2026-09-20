#!/usr/bin/env bash
# Resume-safe pilot for the versioned plan-repair controller.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_PYTHON" ]] || { echo "Python ausente: $BENCH_PYTHON" >&2; exit 1; }
export PYTHONHASHSEED=42
export WRAG_LLM_CACHE=1
export WRAG_EMBED_CACHE=1
output=${1:-runs/locomo-plan-repair-01}
source_dir=${SOURCE:-runs/locomo-controlled-01}
[[ -d "$source_dir/controlled" ]] || { echo "Memória congelada ausente: $source_dir/controlled" >&2; exit 1; }
export WRAG_FROZEN_MEMORY_SOURCE="$(cd "$source_dir" && pwd)"
gpu=${GPU:-4}
port=${PORT:-$("$BENCH_PYTHON" -c 'from wrag.controlled import free_port; print(free_port(8097))')}
extra=()
[[ -f "$output/pilot.json" ]] && extra+=(--resume)
exec "$BENCH_PYTHON" -m wrag.pilot \
  --dataset locomo --locomo-conversation all --methods witnessrag \
  --model Qwen/Qwen2.5-14B-Instruct --embed-model BAAI/bge-m3 \
  --locomo-chunk-tokens 2048 --locomo-ie-window-tokens 512 \
  --top-k 5 --witness-candidate-pool 20 --binding-aware-grounding \
  --answer-set --vocab-compile --hybrid-fallback --dialogue-ie \
  --query-plans --max-query-plans 5 --active-frontier \
  --active-obligations --active-context --plan-repair \
  --gpu "$gpu" --port "$port" --hours "${HOURS:-18}" \
  --vllm-python "${VLLM_PYTHON:-$PWD/.venv-vllm/bin/python}" \
  --cache-dir "${CACHE_DIR:-$output/cache}" --output "$output" \
  "${extra[@]}" "${@:2}"
