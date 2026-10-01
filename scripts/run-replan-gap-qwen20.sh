#!/usr/bin/env bash
# All 152 questions of conversation zero, with model-controlled gap coverage.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_BIN=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
SOURCE=${SOURCE_RUN:-"$PWD/runs/standard-locomo-qwen14b/facts20"}
CACHE=${CACHE_DIR:-"$PWD/runs/.cache/standard-locomo-qwen14b"}
OUTPUT=${OUTPUT_DIR:-"$PWD/runs/replan5-gap-conv00-final/facts20"}
export CUDA_VISIBLE_DEVICES=${GPU:-7} CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Select one GPU" >&2; exit 2; }
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8095/v1}
export OPENAI_API_KEY=${VLLM_API_KEY:-local-pilot}
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
mkdir -p "$OUTPUT"
# Prevent two resumed writers from duplicating the same question records.
exec 9>"$OUTPUT/launch.lock"
flock -n 9 || { echo "This output is already running: $OUTPUT" >&2; exit 2; }
"$BENCH_BIN" scripts/check-qwen-server.py --base-url "$OPENAI_BASE_URL" --model "$MODEL" \
  --check-cuda --output "$OUTPUT/server.json"
"$BENCH_BIN" -u scripts/test-reflection-replan.py --source-run "$SOURCE" --cache-dir "$CACHE" \
  --model "$MODEL" --embed-device cuda:0 --expected-fact-budget 20 --conversation 0 \
  --all-questions --output "$OUTPUT" --resume
