#!/usr/bin/env bash
# Paired replan-only evaluation: frozen standard-20 memory and initial context.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-full}
[[ "$MODE" == full || "$MODE" == pilot ]] || { echo "Use full or pilot" >&2; exit 2; }
PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
SOURCE=${SOURCE_RUN:-"$PWD/runs/standard-locomo-qwen14b/facts20"}
CACHE=${CACHE_DIR:-"$PWD/runs/.cache/standard-locomo-qwen14b"}
OUTPUT=${OUTPUT_DIR:-"$PWD/runs/replan4-frozen20-$MODE"}
export CUDA_VISIBLE_DEVICES=${GPU:-7} CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Select one GPU" >&2; exit 2; }
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8095/v1}
export OPENAI_API_KEY=${VLLM_API_KEY:-local-pilot}
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
mkdir -p "$OUTPUT"
"$PYTHON" scripts/check-qwen-server.py --base-url "$OPENAI_BASE_URL" --model "$MODEL" \
  --check-cuda --output "$OUTPUT/server.json"
for conv in $(seq 0 9); do
  printf -v name 'conv%02d' "$conv"
  extra=()
  if [[ "$MODE" == full ]]; then
    extra+=(--all-questions)
  elif [[ "$conv" == 0 ]]; then
    extra+=(--include-qids locomo:conv-26:qa11)
  elif [[ "$conv" == 9 ]]; then
    extra+=(--include-qids locomo:conv-50:qa40 locomo:conv-50:qa56)
  fi
  "$PYTHON" -u scripts/test-reflection-replan.py --source-run "$SOURCE" --cache-dir "$CACHE" \
    --model "$MODEL" --embed-device cuda:0 --expected-fact-budget 20 --conversation "$conv" \
    --output "$OUTPUT/$name" --resume "${extra[@]}"
done
"$PYTHON" scripts/analyze-replan-frozen.py --results "$OUTPUT" \
  --old-replan "${OLD_REPLAN_RUN:-$PWD/runs/replan2-locomo-qwen14b/facts20}"
