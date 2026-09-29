#!/usr/bin/env bash
# Run 20 and 40 facts sequentially against the same memory/model/configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:-locomo}
case "$TASK" in
  locomo|memoryagentbench) ;;
  *) echo "Usage: bash scripts/run-standard-qwen-variants.sh {locomo|memoryagentbench} [output-root]" >&2; exit 2 ;;
esac
OUTPUT_ROOT=${2:-runs/standard-$TASK-qwen14b}
for budget in 20 40; do
  echo "Running $TASK with $budget facts (GPU ${GPU:-7})."
  FACT_BUDGET=$budget bash scripts/run-standard-qwen.sh "$TASK" "$OUTPUT_ROOT/facts$budget"
done
