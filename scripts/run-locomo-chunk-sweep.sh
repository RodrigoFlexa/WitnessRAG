#!/usr/bin/env bash
# Chunk-size sweep on LoCoMo: how much does the size of a chunk (the unit of
# retrieval and reading) change what similarity search and the proof deliver?
#
#   bash scripts/run-locomo-chunk-sweep.sh
#
# Two regimes, both paired with the hybrid baseline at the same point:
#   k5      k = 5 chunks of s tokens. The reader budget shrinks with s.
#   budget  k = round(10240 / s): the reader always sees ~10K tokens, only the
#           granularity changes (20 x 512, 40 x 256). The proof may use a
#           proportional share of the slots (--proof-edit-fraction 0.4, which
#           reproduces k_W = 2/1 at k = 5) and the candidate pool is 2k.
# 2048 x 5 is the registered point of both regimes and runs once.
#
# Variables:
#   LLM=openai | azure | qwen   openai: run-witness-openai-locomo.sh (key in .env)
#   CHUNKS="2048 1024 512 256"
#   REGIMES="k5 budget"
#   METHODS="hybrid proof proof-v4"   hybrid, or any PROFILE of scripts/proof-profiles.sh
#   CONVERSATIONS=all | 0 | 0,1,2
#   QUESTIONS=                   optional cap per conversation (smoke tests)
#   READER_FLAGS=                appended to every run, e.g. --yesno-rationale
#   ROOT=runs/locomo-chunk-sweep-$LLM
#
# Caveat (declare it when reporting): the OpenIE windows are cut inside each
# chunk, so a new chunk size also re-extracts the facts. The proof rows measure
# the whole system at that size; the hybrid rows do not use the graph and are
# a pure retrieval effect. Extraction at a new size costs about one full run.
set -euo pipefail
cd "$(dirname "$0")/.."

LLM=${LLM:-openai}
CHUNKS=${CHUNKS:-"2048 1024 512 256"}
REGIMES=${REGIMES:-"k5 budget"}
METHODS=${METHODS:-"hybrid proof proof-v4"}
CONVERSATIONS=${CONVERSATIONS:-all}
ROOT=${ROOT:-runs/locomo-chunk-sweep-$LLM}
BUDGET_TOKENS=${BUDGET_TOKENS:-10240}
export LOCOMO_CONVERSATION="$CONVERSATIONS"
[[ -n "${QUESTIONS:-}" ]] && export QUESTIONS
mkdir -p "$ROOT"

if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in "$PWD/venv/bin/python" "$PWD/.venv-bench/bin/python" "$PWD/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then BENCH_PYTHON="$candidate"; break; fi
  done
fi
export BENCH_PYTHON=${BENCH_PYTHON:-python3}

run_point() {  # method chunk k extra_flags
  local method=$1 chunk=$2 k=$3 extra=$4 out pool
  out="$ROOT/$method-c$chunk-k$k"
  pool=$(( 2 * k > 20 ? 2 * k : 20 ))
  echo "=== $method: $k x $chunk tokens -> $out ==="
  local flags="${READER_FLAGS:-} $extra"
  if [[ "$LLM" == openai ]]; then
    if [[ "$method" == hybrid ]]; then
      METHOD=hybrid TOP_K=$k CHUNK_TOKENS=$chunk POOL=$pool EXTRA_FLAGS="${READER_FLAGS:-}" \
        bash scripts/run-witness-openai-locomo.sh "$out"
    else
      METHOD=proof PROFILE=$method TOP_K=$k CHUNK_TOKENS=$chunk POOL=$pool EXTRA_FLAGS="$flags" \
        bash scripts/run-witness-openai-locomo.sh "$out"
    fi
  else
    if [[ "$method" == hybrid ]]; then
      METHOD=hybrid TOP_K=$k CHUNK_TOKENS=$chunk POOL=$pool EXTRA_FLAGS="${READER_FLAGS:-}" \
        bash scripts/run-witness-azure-locomo.sh "$out"
    else
      PROFILE=$method TOP_K=$k CHUNK_TOKENS=$chunk POOL=$pool EXTRA_FLAGS="$flags" \
        bash scripts/run-witness-proof-locomo.sh "$out"
    fi
  fi
}

RUNS=()
declare -A DONE=()
for CHUNK in $CHUNKS; do
  for REGIME in $REGIMES; do
    case "$REGIME" in
      k5) K=5; EXTRA="" ;;
      budget) K=$(( (BUDGET_TOKENS + CHUNK / 2) / CHUNK )); EXTRA="--proof-edit-fraction 0.4" ;;
      *) echo "REGIMES aceita k5 e budget" >&2; exit 2 ;;
    esac
    for M in $METHODS; do
      key="$M-c$CHUNK-k$K"
      [[ -n "${DONE[$key]:-}" ]] && continue
      DONE[$key]=1
      # At k = 5 the proportional k_W equals the fixed one; keep the registered flags.
      [[ "$K" == 5 ]] && EXTRA=""
      run_point "$M" "$CHUNK" "$K" "$EXTRA"
      RUNS+=("$key=$ROOT/$key")
    done
  done
done

# Reference: the hybrid at the registered point when it is in the sweep.
REF=()
for item in "${RUNS[@]}"; do
  [[ "$item" == hybrid-c2048-k5=* ]] && REF=("$item")
done
ORDERED=("${REF[@]}")
for item in "${RUNS[@]}"; do
  [[ "$item" == hybrid-c2048-k5=* ]] || ORDERED+=("$item")
done
"$BENCH_PYTHON" scripts/budget-report.py --output "$ROOT/sweep_report" "${ORDERED[@]}"
echo "Tabela: $ROOT/sweep_report.md"
