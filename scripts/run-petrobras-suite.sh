#!/usr/bin/env bash
# WSL2/Linux, Azure OpenAI configured in .env. No vLLM or dataset transfer.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_BIN=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_BIN" ]] || { echo "Missing benchmark Python: $BENCH_BIN; see docs/petrobras-suite.md" >&2; exit 1; }
exec "$BENCH_BIN" -u scripts/run-petrobras-suite.py "$@"
