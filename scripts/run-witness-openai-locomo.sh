#!/usr/bin/env bash
# LoCoMo (10 conversas) com WitnessRAG usando a API pública da OpenAI.
# Mesmo perfil registrado de scripts/run-witness-azure-locomo.sh; só o LLM muda.
#
#   OPENAI_API_KEY no .env (ou no ambiente), depois:
#   bash scripts/run-witness-openai-locomo.sh [saida]
#
# Variáveis úteis: MODEL (gpt-4o-mini), LOCOMO_CONVERSATION (all | 0 | 0,3,7),
# METHOD (witnessrag | hybrid | proof), CONCURRENCY (8), EMBED_DEVICE (cpu | cuda),
# QUESTIONS (vazio = todas; use p.ex. 5 para um smoke test).
# METHOD=proof usa o controlador de prova com PROFILE (o mesmo conjunto de
# perfis de scripts/run-witness-proof-locomo.sh: proof, proof-v4, v4-typed,
# v4-excerpts, v4-abductive, ...). TOP_K, CHUNK_TOKENS, POOL (20) e
# EXTRA_FLAGS (por exemplo "--yesno-rationale" ou "--proof-edit-fraction 0.4")
# servem à varredura de scripts/run-locomo-chunk-sweep.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
: "${OPENAI_API_KEY:?Defina OPENAI_API_KEY no .env ou no ambiente}"

if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in .venv-bench/bin/python .venv-bench/Scripts/python.exe .venv/bin/python .venv/Scripts/python.exe; do
    [[ -x "$candidate" ]] && { BENCH_PYTHON="$PWD/$candidate"; break; }
  done
  BENCH_PYTHON=${BENCH_PYTHON:-python}
fi

MODEL=${MODEL:-gpt-4o-mini}
METHOD=${METHOD:-witnessrag}
PROFILE=${PROFILE:-proof}
if [[ "$METHOD" == proof ]]; then
  OUTPUT=${1:-runs/witness-suite-locomo-openai-${MODEL}-${PROFILE}}
else
  OUTPUT=${1:-runs/witness-suite-locomo-openai-${MODEL}-${METHOD}}
fi
CONVERSATION=${LOCOMO_CONVERSATION:-all}
TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
TOP_K=${TOP_K:-5}
CHUNK_TOKENS=${CHUNK_TOKENS:-2048}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/witness-openai"}
mkdir -p "$OUTPUT" "$CACHE_DIR"

export WRAG_LLM_BACKEND=openai OPENAI_MODEL="$MODEL"
export WRAG_CONTINUE_ON_CONTENT_FILTER=1
export WRAG_EMBED_BACKEND=st
export WRAG_EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
export WRAG_EMBED_DEVICE=${EMBED_DEVICE:-cpu}
export WRAG_CACHE_DIR="$CACHE_DIR"
export WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42
export WRAG_AZURE_CONCURRENCY=${CONCURRENCY:-8}

"$BENCH_PYTHON" -m wrag.cli diag-openai --model "$MODEL"

RESUME=()
[[ -f "$OUTPUT/pilot.json" ]] && RESUME+=(--resume)
QUESTION_FLAGS=()
[[ -n "${QUESTIONS:-}" ]] && QUESTION_FLAGS+=(--questions "$QUESTIONS")
RUN_METHOD=$METHOD
case "$METHOD" in
  witnessrag)
    METHOD_FLAGS=(--binding-aware-grounding --vocab-compile --hybrid-fallback
      --dialogue-ie --selective-witness --gap-context-rescue)
    ;;
  hybrid) METHOD_FLAGS=() ;;
  proof)
    RUN_METHOD=witnessrag
    # shellcheck disable=SC1091
    source scripts/proof-profiles.sh
    proof_profile_flags "$PROFILE"
    METHOD_FLAGS=("${PROOF_FLAGS[@]}")
    ;;
  *) echo "METHOD deve ser witnessrag, hybrid ou proof." >&2; exit 2 ;;
esac
read -r -a EXTRA <<< "${EXTRA_FLAGS:-}"

"$BENCH_PYTHON" -m wrag.pilot --backend openai --model "$MODEL" \
  --concurrency "$WRAG_AZURE_CONCURRENCY" --gpu "${GPU:-0}" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --dataset locomo --locomo-conversation "$CONVERSATION" --methods "$RUN_METHOD" \
  "${QUESTION_FLAGS[@]}" \
  --embed-model "$WRAG_EMBED_MODEL" --embed-device "$WRAG_EMBED_DEVICE" \
  --locomo-chunk-tokens "$CHUNK_TOKENS" --locomo-ie-window-tokens 512 --top-k "$TOP_K" --qa-max-tokens 128 \
  --witness-candidate-pool "${POOL:-20}" --answer-set --temporal-annotations --evidence-reader \
  "${METHOD_FLAGS[@]}" "${EXTRA[@]}" \
  --cache-dir "$CACHE_DIR" \
  --hours "${HOURS:-72}" --output "$OUTPUT" "${RESUME[@]}"
