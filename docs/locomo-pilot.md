# LoCoMo: avaliação do WitnessRAG

O protocolo principal usa as dez conversas do `locomo10.json`: **1.123
perguntas**, sendo **841 single-hop** (categoria 4) e **282 multi-hop**
(categoria 1). Exclui categorias 2 (temporal), 3 (open-domain) e 5
(adversarial). O filtro segue as categorias anotadas, sem reclassificar pela
redação da pergunta ou pelo número de evidências.

Referências: [dataset e documentação](https://github.com/snap-research/locomo) e
[avaliador oficial](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py).
O download usa o commit `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, preservado no código,
e registra SHA-256 do arquivo. Não acompanha alterações futuras de `main` automaticamente.

## Executar no servidor Linux

Para a comparação mais próxima do RAG descrito no ZeroMem, rode as dez
conversas do zero. `dense` reproduz o controle sem memória; `hybrid` isola o
fallback que o Witness usa; `witnessrag` mede o ganho estrutural. Os três usam o
mesmo leitor e recebem exatamente cinco chunks.

```bash
.venv-bench/bin/python -m wrag.pilot \
  --dataset locomo --locomo-conversation all \
  --methods dense,hybrid,witnessrag \
  --model Qwen/Qwen2.5-14B-Instruct --embed-model BAAI/bge-m3 \
  --locomo-chunk-tokens 2048 --locomo-ie-window-tokens 512 \
  --top-k 5 --witness-candidate-pool 20 \
  --binding-aware-grounding --verify-witnesses \
  --answer-set --vocab-compile --hybrid-fallback --dialogue-ie \
  --gpu 3 --vllm-python "$PWD/.venv-vllm/bin/python" --port 8087 \
  --hours 12 --output runs/locomo-fair-top5
```

Não reutilize o cache da rodada de oito falas: a unidade documental, o embedding
e a extração mudaram. O pool de 20 é interno; somente cinco chunks chegam ao
leitor. OpenIE processa janelas de 512 tokens com overlap de 64, mas cada fato
mantém a proveniência do chunk pai. Assim o teto de 40 triplas vale por janela,
sem aumentar o orçamento da resposta.

| opção | o que muda | por que |
|---|---|---|
| `--answer-set` | a resposta estrutural passa a ser o conjunto de atribuições certas, com prova por item; `aggregation="set"` e `aggregation="count"` viram executáveis; numa consulta `set`, o verificador decide se cada candidato é um membro correto, sem exigir que ele sozinho seja a lista inteira; o contexto leva uma prova de cada resposta antes de provas extras da mesma; o leitor recebe a variante ciente de conjunto | uma testemunha certifica UMA atribuição, e boa parte da categoria 1 do LoCoMo pede o conjunto ("o que X já fez", "onde X acampou"). Responder um item de um gabarito de três limita o F1 a ~1/3 por construção |
| `--vocab-compile` | as relações e entidades do grafo mais próximas da pergunta entram no prompt de compilação | o compilador inventava predicados que nenhum fato instancia (`identity`, `destress method`) e o átomo morria no aterramento |
| `--query-plans` | gera até três interpretações em uma chamada, testa todas no índice e escolhe a primeira que fecha; se nenhuma fechar, aquisição roda só no plano parcial que mais avançou | trata a saída do LLM como hipótese e reduz a dependência de uma única compilação sem multiplicar chamadas de compilação |
| `--hybrid-fallback` | o fallback passa a ser fusão recíproca de postos entre denso e BM25 | respostas conversacionais dependem de nomes próprios raros que o vetor de um bloco de oito falas dilui |
| `--dialogue-ie` | extração adaptada a diálogo: o falante vira o sujeito das falas em primeira pessoa, correferência é resolvida dentro do bloco e o fato ganha um escopo temporal | sem isso o sujeito da maioria dos fatos é um pronome, e nenhuma junção fecha |
| `--top-k N` | número de chunks entregues ao leitor | a comparação principal fixa `N=5`; use outro valor somente como ablação declarada |
| `--witness-candidate-pool N` | profundidade explorada antes da seleção por prova | preserva cobertura interna sem ampliar o contexto final |
| `--locomo-ie-window-tokens N` | granularidade interna de NER/OpenIE | evita resumir um chunk longo inteiro no teto de 40 triplas; não cria documentos extras para o leitor |
| `--no-relation-family-merge` | mantém `paint`, `painted` e `painting` como relações distintas no índice, usando a mesma extração | ablação da normalização morfológica sem misturá-la com uma nova rodada de OpenIE |

`--dialogue-ie` muda a base de fatos `F` compartilhada e exige nova extração.
O prompt atual pede predicados canônicos reutilizáveis e o índice agrupa somente
flexões com a mesma assinatura, preservando palavras e preposições. Assim,
`paint`/`painted` compartilham uma relação, mas `work at`/`work for` não. A forma
original continua no fato e nas citações. Esta revisão do prompt gera uma nova
chave e não reutiliza automaticamente a extração anterior.

As outras opções experimentais reaproveitam a extração em cache. A chave inclui
o prompt efetivamente usado; esta revisão invalida apenas o cache do modo de
diálogo, enquanto extrações legadas continuam válidas.

O relatório passou a trazer **onde cada pergunta parou**: taxa de disparo da
testemunha, quantas caíram por falta de consulta, por junção que não fechou e por
verificação que rejeitou, além de testemunhas avaliadas e aprovadas. Ele também
separa disparo de contexto alterado e agrega os tipos de rejeição. Comece por
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
pergunta pesa igual, conversa maior pesa mais). A saída final mostra somente o
agregado global e por categoria, sem tabela por conversa.

O prazo é verificado **entre** conversas: com `--hours` curto, o piloto para
depois da última conversa que coube, em vez de deixar uma rodada parcial com
denominador incomparável. Para reagregar o que existe:

```bash
.venv-bench/bin/python scripts/locomo-aggregate.py runs/<piloto>
```

O padrão LoCoMo roda todas as perguntas elegíveis da conversa selecionada;
**não precisa passar `-n`**. A conversa zero tem 102 perguntas.
Um `-n 10` explícito sorteia dez perguntas com a semente configurada, mantendo
a conversa inteira como corpus. Para outros datasets, o padrão continua sendo
100 perguntas e os sete métodos. A GPU/porta precisam estar disponíveis, ou use
`--existing-server` com um servidor compatível já ativo. A pasta de saída deve
ser nova ou vazia. Uma rodada parcial compatível pode ser retomada com
`--resume`; mudanças de modelo, embedding, segmentação, métodos ou orçamento são
recusadas para não misturar protocolos no agregado.

Para evitar download, acrescente `--locomo-file /caminho/locomo10.json`. Para
repetir sobre a mesma extração/embeddings, use
`--cache-dir "$PWD/runs/locomo-first-witness/cache"`, mantendo modelo, endpoint,
revisão e segmentação idênticos. Este corpus requer sua própria extração inicial;
os fatos extraídos do 2Wiki não podem ser reaproveitados como fatos do LoCoMo.

## Corpus e avaliação

- Toda a primeira conversa permanece disponível: 19 sessões, 419 falas.
- O modo legado usa blocos contíguos de até oito falas. O modo comparável usa
  chunks contíguos de até 2.048 tokens: são 129 chunks nas dez conversas.
- Cada passagem preserva participante, identificador da fala, data e texto.
  Legendas BLIP já distribuídas no dataset entram como legendas automáticas;
  imagens não são baixadas/processadas. Resumos, observações e QA não entram no índice.
- Evidências `D...` são mapeadas aos blocos que as contêm somente para avaliação.
  Referências compostas como `D8:6; D9:17` são separadas. Quatro erros conhecidos
  do snapshot oficial têm correções explícitas e auditadas; qualquer outro id
  desconhecido continua causando erro.
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
A suíte local passou com 121 testes e 2 testes opcionais ignorados.
