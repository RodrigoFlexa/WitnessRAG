#!/usr/bin/env bash
# Paired follow-up: evidence baseline versus graded obligations + proof reader.
# All ten conversations by default; re-run with the same output to resume.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d-%H%M%S)
ROOT_OUT=${1:-"runs/locomo-soft-proof-$STAMP"}
if [[ "$ROOT_OUT" != /* ]]; then ROOT_OUT="$PWD/$ROOT_OUT"; fi
GPU=${GPU:-4}
PORT=${PORT:-8089}
HOURS=${HOURS:-18}
MAX_RESUMES=${MAX_RESUMES:-8}
LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-all}
BENCH_PYTHON=${BENCH_PYTHON:-"$PWD/.venv-bench/bin/python"}
VLLM_PYTHON=${VLLM_PYTHON:-"$PWD/.venv-vllm/bin/python"}
EXISTING_SERVER=${EXISTING_SERVER:-0}
if [[ -z "${CACHE_DIR:-}" ]]; then
  if [[ -d "$PWD/runs/locomo-proof-ablation-01/cache" ]]; then
    CACHE_DIR="$PWD/runs/locomo-proof-ablation-01/cache"
  else
    CACHE_DIR="$ROOT_OUT/cache"
  fi
fi
[[ -x "$BENCH_PYTHON" ]] || { echo "Python do benchmark ausente: $BENCH_PYTHON" >&2; exit 1; }
[[ "$EXISTING_SERVER" == 1 || -x "$VLLM_PYTHON" ]] || {
  echo "Python do vLLM ausente: $VLLM_PYTHON" >&2; exit 1;
}
[[ "$LOCOMO_CONVERSATION" == all || "$LOCOMO_CONVERSATION" =~ ^[0-9]+$ ]] || {
  echo "LOCOMO_CONVERSATION deve ser all ou um índice inteiro" >&2; exit 1;
}
mkdir -p "$ROOT_OUT" "$CACHE_DIR"

common=(
  --dataset locomo --locomo-conversation "$LOCOMO_CONVERSATION"
  --methods witnessrag --model Qwen/Qwen2.5-14B-Instruct
  --embed-model BAAI/bge-m3 --locomo-chunk-tokens 2048
  --locomo-ie-window-tokens 512 --top-k 5 --witness-candidate-pool 20
  --binding-aware-grounding --answer-set --vocab-compile
  --hybrid-fallback --dialogue-ie --query-plans --max-query-plans 5
  --active-frontier --active-obligations --active-context
  --gpu "$GPU" --port "$PORT" --vllm-python "$VLLM_PYTHON"
  --cache-dir "$CACHE_DIR" --hours "$HOURS"
)
if [[ "$EXISTING_SERVER" == 1 ]]; then common+=(--existing-server); fi

variant_done() {
  "$BENCH_PYTHON" - "$1" "$LOCOMO_CONVERSATION" <<'PY'
import json, pathlib, sys
p, conv = pathlib.Path(sys.argv[1]), sys.argv[2]
try:
    status = json.loads((p/'status.json').read_text())['status']
    if status != 'complete':
        sys.exit(1)
    if conv == 'all':
        progress = json.loads((p/'conversations.json').read_text())['conversas']
        if len(progress) != 10 or not all(
                (pathlib.Path(item['run_dir'])/'run.json').exists() for item in progress):
            sys.exit(1)
    elif not list((p/'benchmark').glob('*/run.json')):
        sys.exit(1)
except (OSError, KeyError, ValueError):
    sys.exit(1)
PY
}

run_variant() {
  local name=$1; shift
  local out="$ROOT_OUT/$name"
  local attempt=0
  if variant_done "$out"; then echo "== $name: concluído anteriormente =="; return; fi
  while (( attempt < MAX_RESUMES )); do
    attempt=$((attempt + 1))
    local resume=()
    if [[ -f "$out/pilot.json" ]]; then resume+=(--resume); fi
    echo "== $name: tentativa $attempt/$MAX_RESUMES =="
    "$BENCH_PYTHON" -m wrag.pilot "${common[@]}" "$@" \
      --output "$out" "${resume[@]}" || true
    if variant_done "$out"; then return; fi
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
  echo "$name incompleto após $MAX_RESUMES tentativas; retome com o mesmo diretório" >&2
  exit 1
}

run_variant evidence
run_variant soft-proof --soft-obligations --proof-reader
"$BENCH_PYTHON" scripts/compare-active-research.py "$ROOT_OUT" \
  --variants evidence soft-proof
echo "Comparação pronta: $ROOT_OUT/ablation.md"
