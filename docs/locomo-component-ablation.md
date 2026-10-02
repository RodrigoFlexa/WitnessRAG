# Estudo de componentes no LoCoMo

Cinco execuções novas, sem misturar resultados históricos: Qwen2.5-14B-Instruct,
dez conversas, 1.540 perguntas por variante, 40 fatos, BGE-M3, MiniLM,
chunks de 2.048 tokens, janelas de extração de 512, top-k 5, leitor com teto
de 128 tokens, temperatura zero e semente 42. O replanejamento experimental
fica desligado. Os planos locais v2 não fazem chamadas generativas.

| ID | Alteração |
|---|---|
| `full` | Padrão local-v2, incluindo reflexão conjunta no leitor. |
| `no-witness` | Fatos selecionados individualmente pela pergunta, com o mesmo reranker, limites e deduplicação. Sem gerar/executar planos, pacotes, instruções do contrato ou resgate de fontes; o leitor recebe somente fatos, ainda datados. Reflexão permanece. |
| `no-time-reference` | Zera a proximidade temporal no aterramento, ranqueamento de testemunhas e resgate de fontes. Preserva intervalos, operadores e filtros explícitos, modalidade, proveniência e importância. |
| `no-time-model` | Reaproveita a aquisição textual, descarta campos `time` e `kind` antes de construir vetores/grafo novos, desativa resolução de intervalos, operadores, referência temporal e agrupamento cronológico dos fatos. Preserva frases, fontes e expressões temporais literais. Leitor usa descrição de memória sem datas resolvidas. |
| `no-reflection` | Mesmo contexto do completo, sem o wrapper de reflexão conjunta. O leitor e suas regras básicas de resposta/proveniência permanecem. |

`no-witness` corresponde ao baseline de **somente fatos ranqueados** solicitado.
Também remove o contexto documental acrescentado pela busca local e seu contrato:
seu delta mede esse caminho de recuperação como um conjunto, não apenas a operação
de junção isoladamente. `no-reflection` mede a instrução opcional de reflexão;
não impede raciocínio interno do modelo nem elimina as regras básicas do leitor.
Não se usa a flag antiga `--ablation`, que pertence a outro controlador.

A extração-base é compartilhada entre variantes por cache e nunca é alterada
em disco. A memória sem tempo é reconstruída a partir de cópias dos fatos; não
é uma nova extração com prompt diferente. Assim evitamos confundir qualidade do
extrator com a representação temporal. Expressões de datas que são parte do texto
original continuam disponíveis; o tratamento estruturado é que é removido.

## Servidor compartilhado

Servidor Qwen exclusivo da ablação na GPU física 0, porta 8096, bfloat16,
contexto de 32.768, reserva de 50% da VRAM e no máximo duas sequências. Embeddings
e reranker também usam GPU 0, com batches menores. Isso não muda os orçamentos
científicos nem o dtype/pesos do modelo. A run AR da GPU 7 permanece independente.
Não usar estes resultados para comparar latência, porque a GPU 0 é compartilhada.

O lançamento real usa um checkout congelado em `.audit-code`, um servidor
protegido com nohup e uma sessão tmux de acompanhamento. A fila usa lock de SO,
registra os PIDs e recusa um segundo escritor. Para retomar, usar exatamente o
mesmo código, cache, porta e pasta. Falhas param a fila; não repetimos um erro
determinístico em loop.

```bash
BENCH_PYTHON=/home/rodrigo.flexa/WitnessRAG/.venv-bench/bin/python
"$BENCH_PYTHON" scripts/run-locomo-component-ablation.py \
  --output /home/rodrigo.flexa/WitnessRAG/runs/locomo-ablation-qwen14b/facts40 \
  --cache /home/rodrigo.flexa/WitnessRAG/runs/.cache/locomo-ablation-qwen14b \
  --gpu 0 --port 8096 --concurrency 2
```

## Acompanhamento

```bash
tmux attach -t locomo-ablation
```

Ou, de qualquer terminal no repositório:

```bash
.venv-bench/bin/python scripts/watch-locomo-ablation.py \
  --output runs/locomo-ablation-qwen14b/facts40 --watch
tail -f runs/locomo-ablation-qwen14b/facts40/progress.log
tail -f runs/locomo-ablation-qwen14b/facts40/full/benchmark.log
```

A fila escreve `progress.md`, `progress.json` e `progress.log` a cada minuto.
O painel mostra F1 **oficial**, total concluído e F1/n por categoria.
Os parciais usam apenas questões salvas; podem ter denominadores diferentes.
`progress.json` inclui também a comparação com o completo restrita aos mesmos
IDs já respondidos. Os IDs são qualificados pela conversa para evitar colisões.
Na conclusão, a fila exige 1.540 IDs únicos por variante e o mesmo conjunto
de IDs em todas as cinco. Linhas JSONL parcialmente escritas não são contadas;
duplicatas e métricas inválidas são erros, não removidas silenciosamente.
