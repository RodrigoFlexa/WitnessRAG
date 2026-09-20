#!/usr/bin/env bash
# One command: two independent vLLM servers, shared frozen source, disjoint qids.
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
[[ -x "$BENCH_PYTHON" ]] || { echo "Python ausente: $BENCH_PYTHON" >&2; exit 1; }
export PYTHONHASHSEED=42
output=${1:-runs/locomo-gap-01}
source_dir=${SOURCE:-runs/locomo-controlled-01}
shift || true
"$BENCH_PYTHON" -m wrag.gap_experiment "$output" --source "$source_dir" --check
"$BENCH_PYTHON" -m wrag.gap_experiment "$output" --source "$source_dir" --prepare
if [[ -n "${GPU_B:-}" ]]; then
  GPU=${GPU_A:-4} PORT=${PORT_A:-8091} "$BENCH_PYTHON" -m wrag.gap_experiment \
    "$output" --source "$source_dir" --shards 2 --shard 0 "$@" &
  first=$!
  GPU=$GPU_B PORT=${PORT_B:-8093} "$BENCH_PYTHON" -m wrag.gap_experiment \
    "$output" --source "$source_dir" --shards 2 --shard 1 "$@" &
  second=$!
  trap 'kill "$first" "$second" 2>/dev/null || true' INT TERM
  set +e
  wait "$first"; one=$?
  wait "$second"; two=$?
  set -e
  trap - INT TERM
  if (( one != 0 || two != 0 )); then
    echo "Falha em um shard (status $one/$two). Consulte os logs e retome com o mesmo comando." >&2
    exit 1
  fi
  "$BENCH_PYTHON" -m wrag.gap_experiment "$output" --source "$source_dir" --report
else
  GPU=${GPU_A:-${GPU:-4}} PORT=${PORT_A:-8091} exec "$BENCH_PYTHON" -m wrag.gap_experiment \
    "$output" --source "$source_dir" "$@"
fi
