#!/usr/bin/env bash
# Piloto completo: 100 perguntas por dataset, cinco sistemas, mais as ablações.
#
# A ordem importa. A primeira rodada paga a indexação (extração + grafo) e
# preenche o cache; as ablações depois reusam tudo e custam só as chamadas de
# consulta que mudaram.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N:-100}
DATASETS=${DATASETS:-musique,2wikimultihopqa,hotpotqa}

echo "== diagnóstico do gateway =="
python -m wrag.cli diag-azure

echo "== rodada principal =="
python -m wrag.cli run --datasets "$DATASETS" \
  --methods dense,bm25,graphrag,hipporag,hipporag2,witnessrag \
  -n "$N" --tag principal

echo "== teto de compilação (condição de consulta oracular) =="
python -m wrag.cli run --datasets 2wikimultihopqa,musique \
  --methods witnessrag-oracle -n "$N" --tag oraculo

echo "== ablação: sem aquisição adaptativa =="
python -m wrag.cli run --datasets "$DATASETS" --methods witnessrag \
  -n "$N" --no-acquisition --tag sem-aquisicao

echo "== curva de orçamento de memória =="
for B in 0.50 0.25; do
  python -m wrag.cli run --datasets "$DATASETS" --methods witnessrag \
    -n "$N" --budget "$B" --tag "orcamento-$B"
done

echo "pronto. Relatórios em runs/*/report.md"
