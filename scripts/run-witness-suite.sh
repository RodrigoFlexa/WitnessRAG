#!/usr/bin/env bash
# Reproducible benchmark client. The Qwen vLLM server is managed separately.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCHMARK=${1:-locomo}
OUTPUT=${2:-"runs/witness-suite-$BENCHMARK-$(date +%Y%m%d-%H%M%S)"}
GPU=${GPU:-1}
PORT=${PORT:-8095}
MODEL=Qwen/Qwen2.5-14B-Instruct
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
CACHE_DIR=${CACHE_DIR:-"$OUTPUT/cache"}
EMBED_DEVICE=${EMBED_DEVICE:-cpu}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_LLM_BACKEND=vllm OPENAI_MODEL="$MODEL"
# The OpenAI-compatible client validates that this variable is non-empty.  The
# local vLLM endpoint does not authenticate or send traffic to OpenAI.
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=local-vllm
export WRAG_EMBED_BACKEND=st WRAG_EMBED_MODEL=BAAI/bge-m3 WRAG_EMBED_DEVICE="$EMBED_DEVICE"
export WRAG_CACHE_DIR="$CACHE_DIR" WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42

if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "No vLLM server at http://127.0.0.1:$PORT/v1." >&2
  echo "Start Qwen2.5-14B in the server tmux, then rerun this command." >&2
  exit 1
fi
echo "Using existing Qwen vLLM server on port $PORT; embeddings: $EMBED_DEVICE."

profile_flags() {
  case "$1" in
    base) echo "" ;;
    soft) echo "--soft-obligations" ;;
    temporal|temporal-v2) echo "--soft-obligations --temporal-memory" ;;
    full|full-v2) echo "--soft-obligations --temporal-memory --complementary-context" ;;
    reform) echo "--soft-obligations --temporal-memory --complementary-context --temporal-annotations" ;;
    reform-low-fallback) echo "--soft-obligations --temporal-memory --complementary-context --temporal-annotations --admit-provisional-witnesses" ;;
    *) echo "unknown profile: $1" >&2; exit 2 ;;
  esac
}

if [[ "$BENCHMARK" == locomo ]]; then
  # Default is conv00: establish a paired gain before LOCOMO_CONVERSATION=all.
  CONVERSATION=${LOCOMO_CONVERSATION:-0}
  # Space-separated subset, useful when resuming a long suite.  The default
  # preserves the registered four-arm experiment.
  LOCOMO_PROFILES=${LOCOMO_PROFILES:-"base soft temporal full"}
  for PROFILE in $LOCOMO_PROFILES; do
    profile_flags "$PROFILE" >/dev/null  # validate before touching output
    STATUS_FILE="$OUTPUT/$PROFILE/status.json"
    if [[ -f "$STATUS_FILE" ]] && "$BENCH_PYTHON" - "$STATUS_FILE" <<'PY'
import json, sys
try:
    complete = json.load(open(sys.argv[1], encoding="utf-8")).get("status") == "complete"
except (OSError, ValueError, TypeError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
    then
      echo "Skipping completed LoCoMo profile: $PROFILE"
      continue
    fi
    FLAGS=$(profile_flags "$PROFILE")
    RESUME=()
    [[ -f "$OUTPUT/$PROFILE/pilot.json" ]] && RESUME+=(--resume)
    # shellcheck disable=SC2086
    "$BENCH_PYTHON" -m wrag.pilot --existing-server --gpu "$GPU" --port "$PORT" \
      --dataset locomo --locomo-conversation "$CONVERSATION" --methods witnessrag \
      --model "$MODEL" --embed-model BAAI/bge-m3 --embed-device "$EMBED_DEVICE" --locomo-chunk-tokens 2048 \
      --locomo-ie-window-tokens 512 --top-k 5 --witness-candidate-pool 20 \
      --binding-aware-grounding --answer-set --vocab-compile --hybrid-fallback \
      --dialogue-ie --query-plans --max-query-plans 5 --active-frontier \
      --active-obligations --active-context --active-operators $FLAGS \
      --cache-dir "$CACHE_DIR" --hours "${HOURS:-24}" --output "$OUTPUT/$PROFILE" \
      "${RESUME[@]}"
  done
  # A comparison is meaningful only for arms that exist; the reporter already
  # pairs on completed question IDs and labels partial arms explicitly.
  "$BENCH_PYTHON" scripts/compare-witness-suite.py "$OUTPUT"
elif [[ "$BENCHMARK" == multihopqa ]]; then
  QUESTIONS=${MULTIHOP_QUESTIONS:-1000}
  for DATASET in musique 2wikimultihopqa hotpotqa; do
    OUT="$OUTPUT/$DATASET"
    RESUME=()
    [[ -f "$OUT/pilot.json" ]] && RESUME+=(--resume)
    "$BENCH_PYTHON" -m wrag.pilot --existing-server --gpu "$GPU" --port "$PORT" \
      --dataset "$DATASET" --methods witnessrag --questions "$QUESTIONS" \
      --model "$MODEL" --embed-model BAAI/bge-m3 --embed-device "$EMBED_DEVICE" \
      --top-k 5 --max-passages 100000 --full-corpus --witness-candidate-pool 20 \
      --binding-aware-grounding --vocab-compile --hybrid-fallback --query-plans \
      --max-query-plans 5 --active-frontier --active-obligations --active-context \
      --soft-obligations --cache-dir "$CACHE_DIR" --hours "${HOURS:-48}" \
      --output "$OUT" "${RESUME[@]}"
  done
elif [[ "$BENCHMARK" == hotpotqa ]]; then
  for TOKENS in 56000 224000 448000; do
    RESUME=()
    [[ -f "$OUTPUT/${TOKENS}/pilot.json" ]] && RESUME+=(--resume)
    "$BENCH_PYTHON" -m wrag.pilot --existing-server --gpu "$GPU" --port "$PORT" \
      --dataset hotpotqa --methods witnessrag-lite --questions "${HOTPOT_QUESTIONS:-1000}" \
      --model "$MODEL" --embed-model BAAI/bge-m3 --embed-device "$EMBED_DEVICE" \
      --top-k 5 --max-passages 100000 --corpus-token-budget "$TOKENS" \
      --witness-candidate-pool 20 --hybrid-fallback --complementary-context --cache-dir "$CACHE_DIR" \
      --hours "${HOURS:-48}" --output "$OUTPUT/${TOKENS}" "${RESUME[@]}"
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
  echo "benchmark must be locomo, multihopqa, hotpotqa, ruler, or narrativeqa" >&2
  exit 2
fi

echo "Reports are under $OUTPUT (report.json/report.md and live_report.json)."
