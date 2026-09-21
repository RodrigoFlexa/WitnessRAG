#!/usr/bin/env bash
# Reproducible Qwen2.5-14B benchmark/ablation launcher. Physical GPU defaults to 1.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCHMARK=${1:-locomo}
OUTPUT=${2:-"runs/witness-suite-$BENCHMARK-$(date +%Y%m%d-%H%M%S)"}
GPU=${GPU:-1}
PORT=${PORT:-8095}
MODEL=Qwen/Qwen2.5-14B-Instruct
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
VLLM_PYTHON=${VLLM_PYTHON:-"$PWD/.venv-vllm/bin/python"}
CACHE_DIR=${CACHE_DIR:-"$OUTPUT/cache"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export WRAG_LLM_BACKEND=vllm OPENAI_MODEL="$MODEL"
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=local-vllm
export WRAG_EMBED_BACKEND=st WRAG_EMBED_MODEL=BAAI/bge-m3 WRAG_EMBED_DEVICE=cuda:0
export WRAG_CACHE_DIR="$CACHE_DIR" WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42

"$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 --tensor-parallel-size 1 --generation-config vllm \
  >"$OUTPUT/vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "vLLM did not become ready; inspect $OUTPUT/vllm.log" >&2; exit 1; }

profile_flags() {
  case "$1" in
    base) echo "" ;;
    soft) echo "--soft-obligations" ;;
    temporal) echo "--soft-obligations --temporal-memory" ;;
    full) echo "--soft-obligations --temporal-memory --complementary-context" ;;
    *) echo "unknown profile: $1" >&2; exit 2 ;;
  esac
}

if [[ "$BENCHMARK" == locomo ]]; then
  # Default is conv00: establish a paired gain before LOCOMO_CONVERSATION=all.
  CONVERSATION=${LOCOMO_CONVERSATION:-0}
  for PROFILE in base soft temporal full; do
    FLAGS=$(profile_flags "$PROFILE")
    # shellcheck disable=SC2086
    "$BENCH_PYTHON" -m wrag.pilot --existing-server --gpu "$GPU" --port "$PORT" \
      --dataset locomo --locomo-conversation "$CONVERSATION" --methods witnessrag \
      --model "$MODEL" --embed-model BAAI/bge-m3 --locomo-chunk-tokens 2048 \
      --locomo-ie-window-tokens 512 --top-k 5 --witness-candidate-pool 20 \
      --binding-aware-grounding --answer-set --vocab-compile --hybrid-fallback \
      --dialogue-ie --query-plans --max-query-plans 5 --active-frontier \
      --active-obligations --active-context --active-operators $FLAGS \
      --cache-dir "$CACHE_DIR" --hours "${HOURS:-24}" --output "$OUTPUT/$PROFILE"
  done
  "$BENCH_PYTHON" scripts/compare-witness-suite.py "$OUTPUT"
elif [[ "$BENCHMARK" == hotpotqa ]]; then
  for TOKENS in 56000 224000 448000; do
    "$BENCH_PYTHON" -m wrag.pilot --existing-server --gpu "$GPU" --port "$PORT" \
      --dataset hotpotqa --methods witnessrag-lite --questions "${HOTPOT_QUESTIONS:-1000}" \
      --model "$MODEL" --embed-model BAAI/bge-m3 --top-k 5 --max-passages 448000 \
      --corpus-passages "$TOKENS" --binding-aware-grounding --answer-set \
      --vocab-compile --hybrid-fallback --query-plans --max-query-plans 5 \
      --active-frontier --active-obligations --active-context --soft-obligations \
      --temporal-memory --complementary-context --cache-dir "$CACHE_DIR" \
      --hours "${HOURS:-48}" --output "$OUTPUT/${TOKENS}"
  done
elif [[ "$BENCHMARK" == ruler ]]; then
  : "${RULER_DIR:?set RULER_DIR to JSON/JSONL files named retrieval, mt, agg and qa}"
  for TASK in retrieval mt agg qa; do
    INPUT=$(find "$RULER_DIR" -maxdepth 1 -type f \( -name "${TASK}.json" -o -name "${TASK}.jsonl" \) | head -n1)
    [[ -n "$INPUT" ]] || { echo "missing RULER task $TASK in $RULER_DIR" >&2; exit 1; }
    "$BENCH_PYTHON" -m wrag.long_benchmark --benchmark ruler --input "$INPUT" \
      --output "$OUTPUT/$TASK" --task "$TASK" --profile full --engine witnessrag-lite --model "$MODEL" \
      --context-tokens 131072 --chunk-tokens 1024 --chunk-overlap 128 --top-k 5 --resume
  done
elif [[ "$BENCHMARK" == narrativeqa ]]; then
  : "${NARRATIVEQA_FILE:?set NARRATIVEQA_FILE to the evaluation JSON/JSONL}"
  "$BENCH_PYTHON" -m wrag.long_benchmark --benchmark narrativeqa --input "$NARRATIVEQA_FILE" \
    --output "$OUTPUT/full" --profile full --engine witnessrag-lite --model "$MODEL" \
    --context-tokens 0 --chunk-tokens 1024 --chunk-overlap 128 --top-k 5 --resume
else
  echo "benchmark must be locomo, hotpotqa, ruler, or narrativeqa" >&2
  exit 2
fi

echo "Reports are under $OUTPUT (report.json/report.md and live_report.json)."
