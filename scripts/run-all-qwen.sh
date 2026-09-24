#!/usr/bin/env bash
# The whole evaluation on a GPU server with Qwen2.5-14B-Instruct (vLLM).
#
#   tmux new -s qwen
#   bash scripts/run-all-qwen.sh 2>&1 | tee runs/qwen-all.log
#
# Steps (STEPS="locomo budget gam" by default, in this order):
#   locomo  LoCoMo, design v3, all ten conversations (run-witness-proof-locomo.sh)
#   budget  LoCoMo reader-budget sweep: v3 with k=4..1 and the hybrid search
#           with k=5..1 (run-locomo-budget.sh; the k=5 hybrid is the paired
#           baseline of the locomo step)
#   gam     HotpotQA 56K/224K/448K, RULER 128K and NarrativeQA under the GAM
#           protocol, engines rag and witnessrag (run-gam-bench.sh)
# Every step resumes: run the same command again after an interruption.
#
# LLM_GPU=0        physical GPU of the vLLM server
# EMBED_GPU=0      physical GPU of BGE-M3 (the same GPU is fine on 80 GB)
# VLLM_PYTHON=     Python of the vLLM environment (default ./.venv-vllm)
# BENCH_PYTHON=    Python of the benchmark environment (default ./venv)
# MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.80 MAX_MODEL_LEN=32768 CONCURRENCY=16
set -euo pipefail
cd "$(dirname "$0")/.."

export LLM=qwen
export PORT=${PORT:-8095}
export MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
export CONCURRENCY=${CONCURRENCY:-16}
LLM_GPU=${LLM_GPU:-0}
EMBED_GPU=${EMBED_GPU:-$LLM_GPU}
STEPS=${STEPS:-"locomo budget gam"}
if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in "$PWD/venv/bin/python" "$PWD/.venv-bench/bin/python" "$PWD/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then BENCH_PYTHON="$candidate"; break; fi
  done
fi
export BENCH_PYTHON=${BENCH_PYTHON:-python3}
mkdir -p runs

if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Starting vLLM ($MODEL) on GPU $LLM_GPU, port $PORT; log: runs/vllm-qwen.log"
  GPU=$LLM_GPU MAX_NUM_SEQS=${MAX_NUM_SEQS:-16} \
    GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80} \
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768} \
    nohup bash scripts/serve-qwen-vllm.sh > runs/vllm-qwen.log 2>&1 &
  for _ in $(seq 1 360); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 10
  done
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || {
    echo "vLLM did not come up in one hour; see runs/vllm-qwen.log" >&2
    exit 1
  }
fi
echo "vLLM ready at http://127.0.0.1:$PORT/v1"

for STEP in $STEPS; do
  echo "===== $STEP ====="
  case "$STEP" in
    locomo)
      GPU=$EMBED_GPU EMBED_DEVICE=cuda PROFILE=proof bash scripts/run-witness-proof-locomo.sh ;;
    budget)
      GPU=$EMBED_GPU EMBED_DEVICE=cuda bash scripts/run-locomo-budget.sh ;;
    gam)
      GPU=$EMBED_GPU EMBED_DEVICE=cuda WITNESS_ON_RULER=1 bash scripts/run-gam-bench.sh ;;
    *)
      echo "unknown step $STEP (locomo, budget, gam)" >&2
      exit 2 ;;
  esac
done

echo "Done. LoCoMo: runs/witness-suite-locomo-qwen-proof/locomo_agregado.json"
echo "      budget: runs/locomo-budget-qwen/budget_report.md"
echo "      GAM table: runs/gam-qwen-$(basename "$MODEL")/gam_table.md"
