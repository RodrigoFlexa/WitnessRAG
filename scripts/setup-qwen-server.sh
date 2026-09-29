#!/usr/bin/env bash
# Linux NVIDIA: isolated runtimes, CUDA-aware wheels, recorded dependency versions.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA driver/nvidia-smi not found." >&2; exit 1; }
if [[ ! -x .venv-vllm/bin/python ]]; then uv venv --python "${PYTHON_VERSION:-3.12}" --seed .venv-vllm; fi
if [[ ! -x .venv-bench/bin/python ]]; then uv venv --python "${PYTHON_VERSION:-3.12}" --seed .venv-bench; fi
VLLM_PACKAGE=vllm
if [[ -n "${VLLM_VERSION:-}" ]]; then VLLM_PACKAGE="vllm==$VLLM_VERSION"; fi
uv pip install --python .venv-vllm/bin/python --torch-backend auto "$VLLM_PACKAGE"
uv pip install --python .venv-bench/bin/python --torch-backend auto \
  -r requirements-pilot.txt -r requirements-memoryagentbench.txt
mkdir -p runs/server-setup
uv pip freeze --python .venv-vllm/bin/python > runs/server-setup/vllm-requirements.txt
uv pip freeze --python .venv-bench/bin/python > runs/server-setup/benchmark-requirements.txt
nvidia-smi > runs/server-setup/nvidia-smi.txt
CUDA_VISIBLE_DEVICES="${GPU:-7}" .venv-bench/bin/python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
echo "Setup complete. Start scripts/serve-qwen-vllm.sh, then run scripts/run-standard-qwen-variants.sh."
