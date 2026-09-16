#!/usr/bin/env bash
# One command runs the full paired LoCoMo matrix and writes a strict comparison.
# Default: all ten conversations; LOCOMO_CONVERSATION=0 is a smaller smoke run.
# Re-run with the same output directory to resume completed variants/conversations.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT_OUT=${1:-"runs/locomo-proof-ablation-$STAMP"}
if [[ "$ROOT_OUT" != /* ]]; then
  ROOT_OUT="$PWD/$ROOT_OUT"
fi
GPU=${GPU:-4}
PORT=${PORT:-8089}
HOURS=${HOURS:-18}
MAX_RESUMES=${MAX_RESUMES:-5}
LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-all}
CACHE_DIR=${CACHE_DIR:-"$ROOT_OUT/cache"}
CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
IE_WINDOW_TOKENS=${IE_WINDOW_TOKENS:-512}
TOP_K=${TOP_K:-5}
MAX_QUERY_PLANS=${MAX_QUERY_PLANS:-5}
WITNESS_CANDIDATE_POOL=${WITNESS_CANDIDATE_POOL:-20}
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
VLLM_PYTHON=${VLLM_PYTHON:-"$PWD/.venv-vllm/bin/python"}
EXISTING_SERVER=${EXISTING_SERVER:-0}

if [[ ! -x "$BENCH_PYTHON" ]]; then
  echo "Python do benchmark ausente: $BENCH_PYTHON" >&2
  exit 1
fi
if [[ "$EXISTING_SERVER" != 1 && ! -x "$VLLM_PYTHON" ]]; then
  echo "Python do vLLM ausente: $VLLM_PYTHON" >&2
  exit 1
fi
if [[ "$LOCOMO_CONVERSATION" != all && ! "$LOCOMO_CONVERSATION" =~ ^[0-9]+$ ]]; then
  echo "LOCOMO_CONVERSATION deve ser all ou um índice inteiro." >&2
  exit 1
fi
mkdir -p "$ROOT_OUT" "$CACHE_DIR"

common=(
  --dataset locomo --locomo-conversation "$LOCOMO_CONVERSATION"
  --methods witnessrag --model Qwen/Qwen2.5-14B-Instruct
  --embed-model BAAI/bge-m3
  --locomo-chunk-tokens "$CHUNK_TOKENS"
  --locomo-ie-window-tokens "$IE_WINDOW_TOKENS"
  --top-k "$TOP_K" --witness-candidate-pool "$WITNESS_CANDIDATE_POOL"
  --binding-aware-grounding --answer-set --vocab-compile
  --hybrid-fallback --dialogue-ie
  --gpu "$GPU" --port "$PORT" --vllm-python "$VLLM_PYTHON"
  --cache-dir "$CACHE_DIR" --max-query-plans "$MAX_QUERY_PLANS"
  --hours "$HOURS"
)
if [[ "$EXISTING_SERVER" == 1 ]]; then
  common+=(--existing-server)
fi

variant_done() {
  "$BENCH_PYTHON" - "$1" "$LOCOMO_CONVERSATION" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); conv = sys.argv[2]
try:
    status = json.loads((p/'status.json').read_text())['status']
    progress = json.loads((p/'conversations.json').read_text())['conversas']
    expected = 10 if conv == 'all' else 1
    done = len(progress) == expected
    sys.exit(0 if status == 'complete' and done else 1)
except (OSError, KeyError, ValueError):
    sys.exit(1)
PY
}

run_variant() {
  local name=$1
  shift
  local out="$ROOT_OUT/$name"
  local attempt=0
  if variant_done "$out"; then
    echo "== $name: concluído anteriormente =="
    return
  fi
  while (( attempt < MAX_RESUMES )); do
    attempt=$((attempt + 1))
    local resume=()
    if [[ -f "$out/pilot.json" ]]; then
      resume+=(--resume)
    fi
    echo "== $name: tentativa $attempt/$MAX_RESUMES =="
    "$BENCH_PYTHON" -m wrag.pilot "${common[@]}" "$@" \
      --output "$out" "${resume[@]}" || true
    if variant_done "$out"; then
      return
    fi
    "$BENCH_PYTHON" - "$out" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    state = json.loads((p/'status.json').read_text())['status']
except (OSError, KeyError, ValueError):
    state = 'unknown'
if state not in {'time_limit', 'interrupted', 'complete'}:
    print(f'{p}: falha {state}; veja benchmark.log, worker-error.txt e error.txt', file=sys.stderr)
    sys.exit(1)
PY
  done
  echo "$name não concluiu após $MAX_RESUMES tentativas; use o mesmo diretório para retomar." >&2
  exit 1
}

# The seven primary rows isolate search, sufficiency, context, operators and
# textual checking. The last two test dependence on acquisition and replanning.
run_variant control --query-plans
run_variant frontier --query-plans --active-frontier
run_variant obligations --query-plans --active-obligations
run_variant frontier-obligations --query-plans --active-frontier --active-obligations
run_variant evidence --query-plans --active-frontier --active-obligations --active-context
run_variant operators --query-plans --active-frontier --active-obligations --active-context --active-operators
run_variant verified --query-plans --active-frontier --active-obligations --active-context --active-operators --verify-witnesses
run_variant no-acquisition --query-plans --active-frontier --active-obligations --active-context --active-operators --no-acquisition
run_variant one-plan --active-frontier --active-obligations --active-context --active-operators

"$BENCH_PYTHON" scripts/compare-active-research.py "$ROOT_OUT"
echo "Resultados e comparação: $ROOT_OUT/ablation.md"
