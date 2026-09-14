# LoCoMo: primeira conversa, somente WitnessRAG

O piloto usa a conversa de índice **0**, `conv-26`, do arquivo oficial
`locomo10.json`. Inclui todas as **102 perguntas** das categorias **4 = single-hop
(70)** e **1 = multi-hop (32)**. Exclui categorias 2 (temporal), 3 (open-domain)
e 5 (adversarial). O filtro segue as categorias anotadas, sem reclassificar pela
redação da pergunta ou pelo número de evidências.

Referências: [dataset e documentação](https://github.com/snap-research/locomo) e
[avaliador oficial](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py).
O download usa o commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, preservado no código,
e registra SHA-256 do arquivo. Não acompanha alterações futuras de `main` automaticamente.

## Executar no servidor Linux

Depois de sincronizar o código, na pasta WitnessRAG:

```bash
.venv-bench/bin/python -m wrag.pilot \
  --dataset locomo --locomo-conversation 0 \
  --methods witnessrag \
  --binding-aware-grounding --verify-witnesses \
  --gpu 3 --vllm-python "$PWD/.venv-vllm/bin/python" --port 8087 \
  --hours 2 --output runs/locomo-first-witness
```

Esse comando ativa as duas opções experimentais da revisão. Para medir a versão
anterior do executor, omita `--binding-aware-grounding --verify-witnesses` e use
uma saída diferente. Nenhum comparador é executado. O fallback denso interno
continua sendo parte do WitnessRAG.

O padrão LoCoMo roda todas as 102 perguntas; **não precisa passar `-n`**.
Um `-n 10` explícito sorteia dez perguntas com a semente configurada, mantendo
a conversa inteira como corpus. Para outros datasets, o padrão continua sendo
100 perguntas e os sete métodos. A GPU/porta precisam estar disponíveis, ou use
`--existing-server` com um servidor compatível já ativo. A pasta de saída deve
ser nova ou vazia.

Para evitar download, acrescente `--locomo-file /caminho/locomo10.json`. Para
repetir sobre a mesma extração/embeddings, use
`--cache-dir "$PWD/runs/locomo-first-witness/cache"`, mantendo modelo, endpoint,
revisão e segmentação idênticos. Este corpus requer sua própria extração inicial;
os fatos extraídos do 2Wiki não podem ser reaproveitados como fatos do LoCoMo.

## Corpus e avaliação

- Toda a primeira conversa permanece disponível: 19 sessões, 419 falas.
- Blocos contíguos de até oito falas, sem cruzar sessões: 61 passagens. O tamanho
  é configurável com `--locomo-turns-per-passage`, sem usar perguntas ou gabaritos.
- Cada passagem preserva participante, identificador da fala, data e texto.
  Legendas BLIP já distribuídas no dataset entram como legendas automáticas;
  imagens não são baixadas/processadas. Resumos, observações e QA não entram no índice.
- Evidências `D...` são mapeadas aos blocos que as contêm somente para avaliação.
  Referências compostas como `D8:6; D9:17` são separadas. Referência desconhecida
  ou ausente causa erro explícito, em vez de reduzir silenciosamente o denominador.
- O leitor e os demais componentes do Witness continuam os mesmos. `top_k=5`
  significa cinco blocos, não cinco falas nem cinco sessões.
- O relatório traz uma tabela por **categoria oficial**. O campo genérico de
  número de hops do harness é uma aproximação pelo número de blocos de apoio;
  não deve substituir a categoria single-hop/multi-hop.
- **F1/EM são as métricas existentes do harness, não as métricas oficiais do
  LoCoMo.** O avaliador oficial usa stemming e tratamento particular de respostas
  múltiplas. Recall aqui mede blocos de evidência recuperados. Não comparar esses
  números diretamente com tabelas publicadas do LoCoMo.

Excluir perguntas temporais não significa apagar datas da conversa: elas podem
ser contexto necessário às demais perguntas.

## Saídas

`runs/locomo-first-witness/benchmark.log` mostra execução e cache.
`data_selection.json` registra seleção, origem, hash, contagens e mapeamento das
evidências. `benchmark/<rodada>/report.md` inclui resultados globais e por categoria;
`benchmark/<rodada>/locomo/witnessrag.jsonl` contém cada pergunta e seus diagnósticos.

Para conferir os dados sem GPU e sem chamadas de modelo:

```bash
.venv-bench/bin/python -m wrag.locomo --output runs/locomo-data-check
```

Isso apenas prepara dados. Use uma pasta diferente para a execução do piloto.
A conversão do arquivo real foi conferida localmente; os testes de integração
usam LLM stub/TF-IDF e não demonstram qualidade com Qwen.
A suíte local passou com 84 testes (143 avisos de deprecação do PuLP).
