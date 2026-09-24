#!/usr/bin/env bash
# Reader-budget sweep on LoCoMo: same memory, same controller, fewer tokens
# handed to the reader. The reference is the registered run (5 chunks of 2048
# tokens, ~10K reader tokens per question).
#
#   bash scripts/run-locomo-budget.sh
#
# CONVERSATIONS=all      or a subset, e.g. "0,1,2" (the first three)
# BUDGETS="4 3 2 1"      chunks of 2048 tokens given to the reader (k=5 is the
#                        reference run, restricted to the same questions)
# METHODS="proof hybrid" proof = design v3 (run-witness-proof-locomo.sh);
#                        hybrid = the same search without the controller, the
#                        paired baseline at each budget (also run at k=5)
# CHUNKS=""              optional, e.g. "1024 512": k=5 with smaller chunks.
#                        This rebuilds the memory (new extraction and planning),
#                        so it costs like a full run; the k sweep does not.
# REFERENCE=runs/witness-suite-locomo-$LLM-proof    the k=5 proof run
# LLM=azure | qwen       passed to both run scripts (qwen: vLLM on $PORT)
#
# Cost: with k<5 the index, the entity graph and almost every planning prompt
# are identical to the reference run, so they come from the shared cache. The
# new calls are the reader (one per question) and the few verifications of
# proofs that now change the smaller context.
set -euo pipefail
cd "$(dirname "$0")/.."

CONVERSATIONS=${CONVERSATIONS:-all}
BUDGETS=${BUDGETS:-"4 3 2 1"}
METHODS=${METHODS:-"proof hybrid"}
CHUNKS=${CHUNKS:-""}
export LLM=${LLM:-azure}
REFERENCE=${REFERENCE:-runs/witness-suite-locomo-$LLM-proof}
SUFFIX=""
[[ "$LLM" != azure ]] && SUFFIX="-$LLM"
if [[ "$CONVERSATIONS" == all ]]; then
  ROOT=${ROOT:-"runs/locomo-budget$SUFFIX"}
else
  ROOT=${ROOT:-"runs/locomo-budget$SUFFIX-c${CONVERSATIONS//,/}"}
fi
export LOCOMO_CONVERSATION="$CONVERSATIONS"
if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in "$PWD/venv/bin/python" "$PWD/.venv-bench/bin/python"; do
    if [[ -x "$candidate" ]]; then BENCH_PYTHON="$candidate"; break; fi
  done
fi
export BENCH_PYTHON=${BENCH_PYTHON:-python3}
mkdir -p "$ROOT"

run_point() {  # method k chunk
  local method=$1 k=$2 chunk=$3 out
  out="$ROOT/$method-k$k"
  [[ "$chunk" != 2048 ]] && out="$out-c$chunk"
  echo "=== $method, $k x $chunk tokens -> $out ==="
  case "$method" in
    proof) TOP_K=$k CHUNK_TOKENS=$chunk bash scripts/run-witness-proof-locomo.sh "$out" ;;
    hybrid) METHOD=hybrid TOP_K=$k CHUNK_TOKENS=$chunk bash scripts/run-witness-azure-locomo.sh "$out" ;;
    *) echo "unknown method $method" >&2; exit 2 ;;
  esac
}

for M in $METHODS; do
  [[ "$M" == hybrid ]] && run_point hybrid 5 2048
done
for K in $BUDGETS; do
  for M in $METHODS; do
    run_point "$M" "$K" 2048
  done
done
for C in $CHUNKS; do
  for M in $METHODS; do
    run_point "$M" 5 "$C"
  done
done

RUNS=("proof-k5=$REFERENCE")
for D in "$ROOT"/*/; do
  RUNS+=("$(basename "$D")=${D%/}")
done
"$BENCH_PYTHON" scripts/budget-report.py --output "$ROOT/budget_report" "${RUNS[@]}"
echo "Tabela: $ROOT/budget_report.md"
