#!/usr/bin/env bash
# Toda a avaliação do desenho v4 no LoCoMo (10 conversas) com Qwen2.5-14B local.
# Pré-requisito: o servidor vLLM já está de pé (scripts/serve-qwen-vllm.sh).
#
#   bash scripts/run-v4-qwen-suite.sh            # tudo, em ordem de prioridade
#   STAGES="main ablations" bash scripts/run-v4-qwen-suite.sh
#
# Etapas (cada rodada vai para um diretório próprio em runs/v4-qwen/ e é
# retomada sozinha se o comando for repetido com o MESMO código):
#   main       híbrido, prova v3 e prova v4, 5 x 2048
#   ablations  v4-typed, v4-mixed, v4-abductive, v4-no-types
#   reader     --yesno-rationale no híbrido e no v4 (muda o leitor de ambos)
#   sweep      trechos 1024/512/256, regimes k5 e budget, híbrido/v3/v4
# Uma rodada que falha não interrompe as seguintes; veja runs/v4-qwen/*.log.
set -uo pipefail
cd "$(dirname "$0")/.."

export LLM=qwen
export LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-all}
ROOT=${ROOT:-runs/v4-qwen}
STAGES=${STAGES:-"main ablations reader sweep"}
mkdir -p "$ROOT"

if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in "$PWD/venv/bin/python" "$PWD/.venv/bin/python" "$PWD/.venv-bench/bin/python"; do
    if [[ -x "$candidate" ]]; then BENCH_PYTHON="$candidate"; break; fi
  done
fi
export BENCH_PYTHON=${BENCH_PYTHON:-python3}
export PORT=${PORT:-8095}
if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Sem servidor vLLM em 127.0.0.1:$PORT. Rode antes: bash scripts/serve-qwen-vllm.sh" >&2
  exit 1
fi

run() {  # nome, comando...
  local name=$1; shift
  echo "=== $(date '+%F %T') $name ==="
  if "$@" > "$ROOT/$name.log" 2>&1; then
    echo "    ok"
  else
    echo "    FALHOU (veja $ROOT/$name.log)"
  fi
}
hybrid() { METHOD=hybrid bash scripts/run-witness-azure-locomo.sh "$ROOT/$1"; }
proof() { PROFILE=$2 bash scripts/run-witness-proof-locomo.sh "$ROOT/$1"; }

for STAGE in $STAGES; do
  case "$STAGE" in
    main)
      run hybrid hybrid hybrid
      run proof-v3 proof proof-v3 proof
      run proof-v4 proof proof-v4 proof-v4
      ;;
    ablations)
      for P in v4-typed v4-mixed v4-abductive v4-no-types; do
        run "$P" proof "$P" "$P"
      done
      ;;
    reader)
      run hybrid-yn env EXTRA_FLAGS=--yesno-rationale METHOD=hybrid \
        bash scripts/run-witness-azure-locomo.sh "$ROOT/hybrid-yn"
      run proof-v4-yn env EXTRA_FLAGS=--yesno-rationale PROFILE=proof-v4 \
        bash scripts/run-witness-proof-locomo.sh "$ROOT/proof-v4-yn"
      ;;
    sweep)
      run sweep env CHUNKS="1024 512 256" REGIMES="k5 budget" \
        METHODS="hybrid proof proof-v4" CONVERSATIONS="$LOCOMO_CONVERSATION" \
        ROOT="$ROOT/sweep" bash scripts/run-locomo-chunk-sweep.sh
      ;;
    *) echo "etapa desconhecida: $STAGE" >&2 ;;
  esac
done

# Tabela principal: híbrido como referência.
RUNS=()
for name in hybrid proof-v3 proof-v4 v4-typed v4-mixed v4-abductive v4-no-types hybrid-yn proof-v4-yn; do
  [[ -d "$ROOT/$name" ]] && RUNS+=("$name=$ROOT/$name")
done
for dir in "$ROOT"/sweep/*-c*-k*/; do
  [[ -d "$dir" ]] && RUNS+=("$(basename "$dir")=${dir%/}")
done
"$BENCH_PYTHON" scripts/budget-report.py --output "$ROOT/report" "${RUNS[@]}" \
  || echo "relatório falhou; rode scripts/budget-report.py à mão"
echo "Relatório: $ROOT/report.md"
