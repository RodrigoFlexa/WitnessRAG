#!/usr/bin/env bash
# Estudo de ablação do WitnessRAG no LoCoMo (10 conversas), docs/ablacao.md.
# Roda o método completo PRIMEIRO e depois cada ablação de um componente,
# com a mesma memória, o mesmo leitor e o mesmo orçamento. No fim, gera a
# tabela pareada contra o completo.
#
#   LLM=openai bash scripts/run-ablation.sh           # gpt-4o-mini (chave no .env)
#   LLM=qwen   bash scripts/run-ablation.sh           # vLLM já de pé (serve-qwen-vllm.sh)
#
# Variáveis: CHUNK_TOKENS (2048), TOP_K (5), POOL (2*TOP_K; mínimo 20), VARIANTS
# ("full no-plan no-proof no-verify no-temporal-score"), LOCOMO_CONVERSATION
# (all), EMBED_DEVICE (cpu|cuda), GPU, ROOT (runs/ablation-$LLM-c$CHUNK-k$K).
# Cada variante vai para $ROOT/<variante>; repetir o comando retoma de onde parou.
set -uo pipefail
cd "$(dirname "$0")/.."

LLM=${LLM:-openai}
export CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
export TOP_K=${TOP_K:-5}
export POOL=${POOL:-$(( 2 * TOP_K > 20 ? 2 * TOP_K : 20 ))}
export LOCOMO_CONVERSATION=${LOCOMO_CONVERSATION:-all}
VARIANTS=${VARIANTS:-"full no-plan no-proof no-verify no-temporal-score"}
ROOT=${ROOT:-runs/ablation-$LLM-c$CHUNK_TOKENS-k$TOP_K}
mkdir -p "$ROOT"

for variant in $VARIANTS; do
  if [[ "$variant" == full ]]; then profile=witnessrag; else profile="abl-$variant"; fi
  out="$ROOT/$variant"
  echo "=== $(date '+%F %T') $variant (perfil $profile) -> $out"
  case "$LLM" in
    openai)
      METHOD=proof PROFILE=$profile bash scripts/run-witness-openai-locomo.sh "$out" \
        2>&1 | tee "$ROOT/$variant.log" ;;
    qwen)
      LLM=qwen PROFILE=$profile bash scripts/run-witness-proof-locomo.sh "$out" \
        2>&1 | tee "$ROOT/$variant.log" ;;
    *) echo "LLM deve ser openai ou qwen" >&2; exit 2 ;;
  esac
  echo "=== $(date '+%F %T') $variant terminou (código ${PIPESTATUS[0]})"
done

PY=${BENCH_PYTHON:-python3}
for c in .venv-bench/bin/python .venv/bin/python venv/bin/python; do
  [[ -x "$c" && -z "${BENCH_PYTHON:-}" ]] && { PY=$c; break; }
done
"$PY" scripts/ablation-report.py "$ROOT"
