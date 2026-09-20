#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_PYTHON" ]] || { echo "Python ausente: $BENCH_PYTHON" >&2; exit 1; }
export PYTHONHASHSEED=42
exec "$BENCH_PYTHON" -m wrag.reading_experiment "${1:-runs/locomo-reading-01}" --source "${SOURCE:-runs/locomo-controlled-01}" "${@:2}"
