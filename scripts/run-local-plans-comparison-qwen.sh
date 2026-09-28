#!/usr/bin/env bash
# Two local multi-plan versions, same existing Qwen server and extraction cache.
set -euo pipefail
cd "$(dirname "$0")/.."
OUTPUT=${1:-runs/local-plans-comparison-qwen}
BASE_EXTRA=${EXTRA_FLAGS:-}
for VERSION in v1 v2; do
  if [[ "$VERSION" == v1 ]]; then
    LIMITS="--local-plan-beam 12 --local-plan-candidates 48"
  else
    LIMITS="--local-plan-beam 32 --local-plan-candidates 96"
  fi
  EXTRA_FLAGS="$BASE_EXTRA --local-plan-version $VERSION $LIMITS" \
    bash scripts/run-local-plans-qwen.sh "$OUTPUT/$VERSION"
done
"${BENCH_PYTHON:-$PWD/.venv-bench/bin/python}" scripts/compare-local-plans.py \
  --before "$OUTPUT/v1" --after "$OUTPUT/v2" --output "$OUTPUT/comparison.md"
