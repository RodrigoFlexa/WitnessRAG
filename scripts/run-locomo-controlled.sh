#!/usr/bin/env bash
# One command; the pilot owns its server and the controller owns checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_PYTHON" ]] || { echo "Python ausente: $BENCH_PYTHON" >&2; exit 1; }
export PYTHONHASHSEED=42
exec "$BENCH_PYTHON" -m wrag.controlled "${1:-runs/locomo-controlled-01}" "${@:2}"
