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
  --answer-set --vocab-compile --hybrid-fallback --dialogue-ie --top-k 10 \
  --gpu 3 --vllm-python "$PWD/.venv-vllm/bin/python" --port 8087 \
  --hours 3 --output runs/locomo-conjunto
```

Esse comando ativa as opções experimentais. Nenhum comparador é executado; o
fallback interno continua sendo parte do WitnessRAG. Todas são **desligadas por
padrão**, e omitir todas reproduz o comportamento das rodadas anteriores — é
assim que as rodadas já medidas continuam comparáveis.

| opção | o que muda | por que |
|---|---|---|
| `--answer-set` | a resposta estrutural passa a ser o conjunto de atribuições certas, com prova por item; `aggregation="count"` vira executável, como o tamanho do conjunto; a verificação cobre uma resposta distinta por vez; o contexto leva uma prova de cada resposta antes de provas extras da mesma; o leitor recebe a variante ciente de conjunto | uma testemunha certifica UMA atribuição, e boa parte da categoria 1 do LoCoMo pede o conjunto ("o que X já fez", "onde X acampou"). Responder um item de um gabarito de três limita o F1 a ~1/3 por construção |
| `--vocab-compile` | as relações e entidades do grafo mais próximas da pergunta entram no prompt de compilação | o compilador inventava predicados que nenhum fato instancia (`identity`, `destress method`) e o átomo morria no aterramento |
| `--hybrid-fallback` | o fallback passa a ser fusão recíproca de postos entre denso e BM25 | respostas conversacionais dependem de nomes próprios raros que o vetor de um bloco de oito falas dilui |
| `--dialogue-ie` | extração adaptada a diálogo: o falante vira o sujeito das falas em primeira pessoa, correferência é resolvida dentro do bloco e o fato ganha um escopo temporal | sem isso o sujeito da maioria dos fatos é um pronome, e nenhuma junção fecha |
| `--top-k N` | número de blocos entregues ao leitor | com dois a cinco blocos de evidência disputando cinco vagas, `AR@5` exige ranking perfeito; medir em `k` maior separa erro de recuperação de erro do leitor |

`--dialogue-ie` muda a base de fatos `F` compartilhada e exige nova extração;
as outras três reaproveitam a extração em cache. A chave de cache só muda quando
`--dialogue-ie` está ligado, então rodadas antigas não são reextraídas à toa.

O relatório passou a trazer **onde cada pergunta parou**: taxa de disparo da
testemunha, quantas caíram por falta de consulta, por junção que não fechou e por
verificação que rejeitou, além de testemunhas avaliadas e aprovadas. Comece por
essa tabela: com taxa de disparo baixa, a tabela principal mede o fallback, não o
executor, e comparar F1 nesse regime compara outra coisa. A ablação do
`--verify-witnesses` se lê nessa mesma tabela, ligando e desligando a opção.

## As dez conversas

`--locomo-conversation all` roda as dez conversas do `locomo10.json` em sequência,
no mesmo servidor e no mesmo cache: **1123 perguntas** (841 single-hop, 282
multi-hop) sobre 848 blocos.

São **dez corpora separados, uma rodada por conversa** — não um índice único. No
LoCoMo cada conversa é a memória das suas próprias perguntas, e juntar tudo
criaria distratores que o protocolo publicado não tem; "John" fala em três
conversas diferentes, e as perguntas sobre ele ficariam ambíguas.

A saída fica em `<output>/conversations/conv00..conv09/benchmark/<rodada>`, com
`conversations.json` listando o que já terminou. Ao final o piloto imprime uma
tabela agregada e escreve `locomo_agregado.json`; a média é **micro** (toda
pergunta pesa igual, conversa maior pesa mais) e vem acompanhada do detalhe por
conversa, porque a variância entre elas é grande.

O prazo é verificado **entre** conversas: com `--hours` curto, o piloto para
depois da última conversa que coube, em vez de deixar uma rodada parcial com
denominador incomparável. Para reagregar o que existe:

```bash
.venv-bench/bin/python scripts/locomo-aggregate.py runs/<piloto>
```

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
- O relatório traz **as duas colunas de qualidade**: `F1`/`EM` do harness, iguais
  às de MuSiQue e 2Wiki, e `F1 ofic.`/`EM ofic.`, que reproduzem
  `task_eval/evaluation.py` do LoCoMo (stemming de Porter, remoção de "and", F1
  por sub-resposta separada por vírgula na categoria 1 e EM por conjunto de
  tokens). Só a segunda é comparável com tabelas publicadas do LoCoMo, e exige
  `nltk`; sem ele a coluna some, em vez de ser estimada por aproximação.
- Na categoria 1 o avaliador oficial **não penaliza itens previstos a mais**: a
  média é sobre as sub-respostas do gabarito. Uma resposta mais longa nunca perde
  pontos ali, então acompanhe também a precisão antes de concluir que enumerar
  mais itens melhorou o sistema.
- Recall aqui mede blocos de evidência recuperados. `scripts/locomo-official-eval.py`
  recalcula uma rodada já salva, sem chamar modelo, e acrescenta o recall por
  fala (`recall_evidencia`), que é o nível em que o oficial mede:

```bash
.venv-bench/bin/python scripts/locomo-official-eval.py \
  runs/locomo-first-witness/benchmark/<rodada> \
  --data-selection runs/locomo-first-witness/data_selection.json \
  --corpus runs/locomo-first-witness/data/locomo_corpus.json
```

Excluir perguntas temporais não significa apagar datas da conversa: elas podem
ser contexto necessário às demais perguntas.

## Saídas

`runs/locomo-first-witness/benchmark.log` mostra execução e cache.
`data_selection.json` registra seleção, origem, hash, contagens e mapeamento das
evidências. `benchmark/<rodada>/report.md` inclui resultados globais e por categoria;
`benchmark/<rodada>/locomo/witnessrag.jsonl` contém cada pergunta e seus diagnósticos.
Ao terminar, o piloto também imprime no terminal F1, EM, R@5 e AR@5 gerais,
F1/recall por categoria e o caminho do relatório. Execuções interrompidas são
identificadas como parciais, com o número de perguntas avaliadas.

Para conferir os dados sem GPU e sem chamadas de modelo:

```bash
.venv-bench/bin/python -m wrag.locomo --output runs/locomo-data-check
```

Isso apenas prepara dados. Use uma pasta diferente para a execução do piloto.
A conversão do arquivo real foi conferida localmente; os testes de integração
usam LLM stub/TF-IDF e não demonstram qualidade com Qwen.
A suíte local passou com 84 testes (143 avisos de deprecação do PuLP).
