#!/usr/bin/env bash
# Accurate Retrieval only; sources are ordered SH -> MH -> LME -> EventQA.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_BIN=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
OUTPUT=${OUTPUT_DIR:-"$PWD/runs/adapted-memoryagentbench-ar-qwen14b/facts40"}
CACHE=${CACHE_DIR:-"$PWD/runs/.cache/memoryagentbench-qwen14b"}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
export CUDA_VISIBLE_DEVICES=${EMBED_GPU:-0} CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Select one embedding GPU" >&2; exit 2; }
export WRAG_EMBED_STRICT_DEVICE=1
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8095/v1}
export OPENAI_API_KEY=${VLLM_API_KEY:-local-pilot}
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
unset WRAG_REFLECTION_STUDY_ROOT WRAG_CONTROLLED_ROOT WRAG_FROZEN_MEMORY_SOURCE
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/launch.lock"
flock -n 9 || { echo "AR already running: $OUTPUT" >&2; exit 2; }
"$BENCH_BIN" scripts/check-qwen-server.py --base-url "$OPENAI_BASE_URL" \
  --model "$MODEL" --check-cuda --output "$OUTPUT/server.json"
"$BENCH_BIN" -m benchmarks.memoryagentbench prepare --cache "$CACHE"
"$BENCH_BIN" -u -m benchmarks.memoryagentbench run \
  --backend vllm --base-url "$OPENAI_BASE_URL" --model "$MODEL" \
  --embed-device cuda --embed-model BAAI/bge-m3 --fact-budget 40 \
  --fact-rerank cross-encoder/ms-marco-MiniLM-L6-v2 \
  --adaptation ar-source-v2 --source-excerpt-chars 2400 \
  --protocol paper --suite paper --splits Accurate_Retrieval \
  --sources ruler_qa1_197K ruler_qa2_421K 'longmemeval_s*' eventqa_full \
  --cache "$CACHE" --output "$OUTPUT" --resume
"$BENCH_BIN" -m benchmarks.memoryagentbench report --output "$OUTPUT"
echo "AR generation complete. LongMemEval official judge is pending."
