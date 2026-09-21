#!/usr/bin/env bash
# Long-lived local Qwen server for the benchmark suite.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-1}
PORT=${PORT:-8095}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
VLLM_PYTHON=${VLLM_PYTHON:-"$PWD/.venv-vllm/bin/python"}

[[ -x "$VLLM_PYTHON" ]] || {
  echo "vLLM Python not found: $VLLM_PYTHON" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# The host exposes an older /usr/bin/nvcc. FlashInfer 0.6.18 adds
# --compress-mode=size during sampler JIT compilation, which that nvcc rejects.
# Greedy decoding at temperature zero does not require the FlashInfer sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Serving $MODEL on physical GPU $GPU at http://127.0.0.1:$PORT/v1"
echo "FlashInfer sampler disabled (host nvcc compatibility)."

exec "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
  --served-model-name "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-16384}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --tensor-parallel-size 1 \
  --generation-config vllm
