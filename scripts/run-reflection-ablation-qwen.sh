#!/usr/bin/env bash
# Existing Qwen2.5-14B vLLM server; four cascade variants, resume + reports.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_OUTPUT=${1:-runs/cascade-reflection-ablation-v2}
if (( $# )); then shift; fi
TASK_PYTHON=${BENCH_PYTHON:-}
if [[ -z "$TASK_PYTHON" ]]; then
  for candidate in .venv-bench/bin/python .venv/bin/python venv/bin/python; do
    if [[ -x "$candidate" ]]; then TASK_PYTHON=$candidate; break; fi
  done
fi
TASK_PYTHON=${TASK_PYTHON:-python3}
exec "$TASK_PYTHON" scripts/run-reflection-ablation.py \
  --output "$TASK_OUTPUT" --conversation "${LOCOMO_CONVERSATION:-all}" \
  --port "${PORT:-8095}" --gpu "${GPU:-1}" --device "${EMBED_DEVICE:-cuda}" \
  --model "${MODEL:-Qwen/Qwen2.5-14B-Instruct}" \
  --concurrency "${CONCURRENCY:-8}" "$@"
