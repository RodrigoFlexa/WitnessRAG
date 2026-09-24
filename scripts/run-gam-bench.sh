#!/usr/bin/env bash
# Long-context benchmarks under the GAM protocol (arXiv:2511.18423, Table 1b):
# HotpotQA 56K/224K/448K (F1), RULER 128K Retri./MT/AGG./QA (accuracy),
# NarrativeQA (F1). Data, pages, top-5, reader prompt and metrics follow the
# official GAM evaluation code; see wrag/gambench/protocol.py and
# docs/gam-benchmarks.md.
#
#   bash scripts/run-gam-bench.sh [OUTPUT_ROOT]
#
# Environment (all optional):
#   LLM=azure | qwen            azure: gpt-4-1-mini-petrobras via .env (default)
#                               qwen: an already-running vLLM server (PORT)
#   ENGINES="rag witnessrag"    rag = GAM's RAG baseline (calibrates the harness)
#                               witnessrag = our method (design v3)
#                               also: hybrid, witnessrag-lite
#   BENCHMARKS="hotpotqa narrativeqa ruler"
#   HOTPOT_SPLITS=all           or e.g. 56k,224k
#   RULER_TASKS=all             or e.g. niah_single_1,vt
#   WITNESS_ON_RULER=1          0 skips witnessrag on RULER (~4M extraction calls;
#                               python -m wrag.gambench estimate shows the size)
#   END_IDX=                    smoke test: only the first N samples per split
#                               (outside the protocol; the table marks it with *)
#   DATA_DIR=data/gam           prepared on first use (downloads ~2.8 GB)
#   EMBED_DEVICE=cuda|cpu       default: cuda when nvidia-smi works
#   GPU=                        CUDA_VISIBLE_DEVICES for the embedder
#   EMBED_BATCH=8               BGE-M3 batch for 2048-token pages
#   BATCH=8                     answers requested in parallel
#   AZURE_CONCURRENCY=8 DEPLOYMENT=gpt-4-1-mini-petrobras
#   PORT=8095 MODEL=Qwen/Qwen2.5-14B-Instruct      (LLM=qwen)
#   BENCH_PYTHON=               default: ./venv, ./.venv-bench, then python3
set -euo pipefail
cd "$(dirname "$0")/.."

LLM=${LLM:-azure}
ENGINES=${ENGINES:-"rag witnessrag"}
BENCHMARKS=${BENCHMARKS:-"hotpotqa narrativeqa ruler"}
HOTPOT_SPLITS=${HOTPOT_SPLITS:-all}
RULER_TASKS=${RULER_TASKS:-all}
WITNESS_ON_RULER=${WITNESS_ON_RULER:-1}
DATA_DIR=${DATA_DIR:-data/gam}
BATCH=${BATCH:-8}

if [[ -z "${BENCH_PYTHON:-}" ]]; then
  for candidate in "$PWD/venv/bin/python" "$PWD/.venv-bench/bin/python" "$PWD/.venv/bin/python"; do
    if [[ -x "$candidate" ]]; then BENCH_PYTHON="$candidate"; break; fi
  done
  BENCH_PYTHON=${BENCH_PYTHON:-python3}
fi
echo "Python: $BENCH_PYTHON"

if [[ -z "${EMBED_DEVICE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    EMBED_DEVICE=cuda
  else
    EMBED_DEVICE=cpu
  fi
fi
if [[ "$EMBED_DEVICE" == cuda && -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi
if [[ "$EMBED_DEVICE" == cpu ]]; then
  echo "AVISO: BGE-M3 na CPU. O protocolo embute ~480 mil páginas de 2.048 tokens;" >&2
  echo "       na CPU isso leva dias. Use EMBED_DEVICE=cuda se houver GPU livre." >&2
fi

case "$LLM" in
  azure)
    if [[ -f .env ]]; then
      set -a
      # shellcheck disable=SC1091
      source .env
      set +a
    fi
    : "${AZURE_OPENAI_API_KEY:?Set AZURE_OPENAI_API_KEY in .env or the environment}"
    if [[ -z "${AZURE_OPENAI_BASE_URL:-}" && -z "${AZURE_OPENAI_ENDPOINT:-}" ]]; then
      echo "Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT." >&2
      exit 1
    fi
    DEPLOYMENT=${DEPLOYMENT:-gpt-4-1-mini-petrobras}
    export WRAG_LLM_BACKEND=azure
    export WRAG_AZURE_DEPLOYMENT="$DEPLOYMENT"
    export AZURE_OPENAI_API_VERSION=${AZURE_OPENAI_API_VERSION:-2024-10-21}
    export WRAG_AZURE_CONCURRENCY=${AZURE_CONCURRENCY:-8}
    MODEL_LABEL="$DEPLOYMENT (Azure)"
    TAG="azure-$DEPLOYMENT"
    ;;
  qwen)
    PORT=${PORT:-8095}
    MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
    export WRAG_LLM_BACKEND=vllm OPENAI_MODEL="$MODEL"
    export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=local-vllm
    export WRAG_AZURE_CONCURRENCY=${CONCURRENCY:-16}
    if ! curl -fsS "$OPENAI_BASE_URL/models" >/dev/null 2>&1; then
      echo "No vLLM server at $OPENAI_BASE_URL. Start it first (scripts/serve-qwen-vllm.sh)." >&2
      exit 1
    fi
    MODEL_LABEL="$MODEL (vLLM)"
    TAG="qwen-$(basename "$MODEL")"
    ;;
  *)
    echo "LLM must be azure or qwen" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT=${1:-"runs/gam-$TAG"}
CACHE_DIR=${CACHE_DIR:-"$PWD/runs/.cache/gam-$TAG"}
mkdir -p "$OUTPUT_ROOT" "$CACHE_DIR"
export WRAG_CONTINUE_ON_CONTENT_FILTER=1
export WRAG_EMBED_BACKEND=st
export WRAG_EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}
export WRAG_EMBED_DEVICE="$EMBED_DEVICE"
export WRAG_EMBED_BATCH_SIZE=${EMBED_BATCH:-8}
export WRAG_EMBED_MAX_SEQ_LENGTH=${EMBED_MAX_SEQ_LENGTH:-0}
export WRAG_TOKENIZER_MODEL=${WRAG_TOKENIZER_MODEL:-Qwen/Qwen2.5-14B-Instruct}
export WRAG_CACHE_DIR="$CACHE_DIR"
export WRAG_LLM_CACHE=1 WRAG_EMBED_CACHE=1 PYTHONHASHSEED=42
unset WRAG_CONTROLLED_ROOT WRAG_FROZEN_MEMORY_SOURCE

if [[ "$LLM" == azure ]]; then
  "$BENCH_PYTHON" -m wrag.cli diag-azure
fi

# shellcheck disable=SC2086
"$BENCH_PYTHON" -m wrag.gambench prepare --data-dir "$DATA_DIR" \
  --benchmarks "$(echo $BENCHMARKS | tr ' ' ',')"

RANGE=()
[[ -n "${END_IDX:-}" ]] && RANGE+=(--end-idx "$END_IDX")

# Benchmark by benchmark, both engines each: HotpotQA finishes (rag and
# witnessrag) before NarrativeQA starts, and RULER, the longest, comes last.
for BENCHMARK in $BENCHMARKS; do
  for ENGINE in $ENGINES; do
    case "$BENCHMARK" in
      hotpotqa) SPLITS="$HOTPOT_SPLITS" ;;
      ruler) SPLITS="$RULER_TASKS" ;;
      narrativeqa) SPLITS=all ;;
      *) echo "unknown benchmark $BENCHMARK" >&2; exit 2 ;;
    esac
    if [[ "$ENGINE" == witnessrag && "$BENCHMARK" == ruler && "$WITNESS_ON_RULER" != 1 ]]; then
      echo "Pulando witnessrag no RULER (WITNESS_ON_RULER=0)."
      continue
    fi
    echo "=== $ENGINE / $BENCHMARK ($SPLITS) ==="
    "$BENCH_PYTHON" -m wrag.gambench run --data-dir "$DATA_DIR" --benchmark "$BENCHMARK" \
      --splits "$SPLITS" --engine "$ENGINE" --output-root "$OUTPUT_ROOT" --batch "$BATCH" \
      --model-label "$MODEL_LABEL" --resume "${RANGE[@]}"
  done
done

"$BENCH_PYTHON" -m wrag.gambench report --output-root "$OUTPUT_ROOT" \
  --engines "$(echo $ENGINES | tr ' ' ',')" --model-label "$MODEL_LABEL"
echo "Tabela: $OUTPUT_ROOT/gam_table.md"
