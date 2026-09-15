# Resultados — 20260915-000328-591091-conv00

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-base-en-v1.5`)
- **top-k**: 15 · **perguntas por dataset**: 102 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

102 perguntas · 61 passagens · 1.4 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-base-en-v1.5`.

Comparação pareada: 102 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 67.4 | 82.4 | 73.5 | 22.5 | 50.6 | 3.9 | 0 | 6.63 |
| hybrid | 66.7 | 80.4 | 72.5 | 21.6 | 51.5 | 3.9 | 0 | 0.01 |
| dense | 57.8 | 68.4 | 60.8 | 23.5 | 53.0 | 7.8 | 0 | 0.01 |
| bm25 | 65.5 | 76.2 | 72.5 | 25.5 | 52.0 | 3.9 | 0 | 0.00 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [59.8, 75.3] | [76.3, 88.5] | [64.7, 81.4] | [14.7, 31.4] | [43.5, 57.8] |
| hybrid | [58.5, 74.9] | [74.1, 87.1] | [63.7, 81.4] | [13.7, 30.4] | [44.0, 58.5] |
| dense | [49.5, 67.0] | [60.3, 76.5] | [51.0, 70.6] | [15.7, 32.4] | [45.5, 60.1] |
| bm25 | [56.2, 74.0] | [68.4, 84.0] | [63.7, 80.4] | [17.6, 34.3] | [44.4, 59.4] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| hybrid | [-4.3, 1.7] | [0.0, 2.9] |
| dense | [-7.4, 2.5] | [3.9, 21.6] |
| bm25 | [-6.2, 3.1] | [-4.9, 7.8] |

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 70 | 59.2 | 28.6 | 62.5 | 30.0 | 92.9 | 92.9 |
| witnessrag | multi-hop | 32 | 31.7 | 9.4 | 43.3 | 12.5 | 59.5 | 31.2 |
| hybrid | single-hop | 70 | 60.3 | 30.0 | 63.5 | 31.4 | 92.9 | 92.9 |
| hybrid | multi-hop | 32 | 32.4 | 3.1 | 42.4 | 9.4 | 53.2 | 28.1 |
| dense | single-hop | 70 | 58.2 | 30.0 | 61.4 | 31.4 | 82.9 | 82.9 |
| dense | multi-hop | 32 | 41.8 | 9.4 | 56.0 | 15.6 | 36.7 | 12.5 |
| bm25 | single-hop | 70 | 61.3 | 35.7 | 64.1 | 37.1 | 91.4 | 91.4 |
| bm25 | multi-hop | 32 | 31.6 | 3.1 | 36.8 | 3.1 | 43.0 | 31.2 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) | 5 hop(s) |
|---|---|---|---|---|---|
| witnessrag | 58.1 / 91.9 (n=74) | 32.6 / 35.0 (n=20) | 23.2 / 0.0 (n=4) | 33.7 / 0.0 (n=3) | 20.7 / 0.0 (n=1) |
| hybrid | 58.6 / 90.5 (n=74) | 35.3 / 35.0 (n=20) | 25.0 / 0.0 (n=4) | 31.1 / 0.0 (n=3) | 20.7 / 0.0 (n=1) |
| dense | 57.1 / 79.7 (n=74) | 45.1 / 15.0 (n=20) | 35.3 / 0.0 (n=4) | 36.9 / 0.0 (n=3) | 33.3 / 0.0 (n=1) |
| bm25 | 59.5 / 86.5 (n=74) | 36.0 / 45.0 (n=20) | 17.9 / 0.0 (n=4) | 26.0 / 33.3 (n=3) | 25.0 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 39.6 (n=11) | 65.0 (n=19) | 21.7 (n=3) | 40.0 (n=4) | 49.2 (n=34) | 49.9 (n=21) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 102 | 1.0 | 17.6 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 102 | 17.6 | 18 | 10 | 0 | 48 | 26 | 0 | 190 | 14.2 |
| hybrid | 102 | 100.0 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| dense | 102 | 100.0 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| bm25 | 102 | 100.0 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 18 | 1.50 | 38.9 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 11.1 (8.8%) | 11.1 (8.8%) | 1.0 (100.0%) | 1.0 (100.0%) | 1.0 (100.0%) |

### Custo

Indexação compartilhada: 80.35282553732395 s · {"chamadas": 122, "tokens_resposta": 35703, "tokens_prompt_sem_cache": 91733, "tokens_prompt": 91733, "em_cache": 0, "tokens_resposta_sem_cache": 35703, "filtradas": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 646 | 999943 | 33664 | 789.988 |
| hybrid | 102 | 604420 | 1392 | 114.348 |
| dense | 102 | 547206 | 1420 | 105.673 |
| bm25 | 102 | 655454 | 1298 | 119.967 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.023 | 0.000 | 0 |
| hybrid | 0.021 | 0.000 | 0 |
| dense | 0.000 | 0.000 | 0 |
| bm25 | 0.021 | 0.000 | 0 |


## Como ler estes números

1. **Escopo do corpus.** O padrão mantém todas as passagens disponíveis no arquivo
   de corpus. `--subset-corpus` reduz esse universo; a condição é registrada.
2. **Denominador pareado.** Só entram perguntas presentes em todos os métodos,
   excluindo a união dos bloqueios. A tabela estima desempenho nesse subconjunto;
   não mede disponibilidade operacional sobre todas as solicitações.
3. **Comparadores locais.** GraphRAG, HippoRAG e HippoRAG2 são adaptações com extração
   compartilhada, não reproduções fiéis dos sistemas publicados. Aquisição dirigida
   adiciona informação e custo ao WITNESS-RAG; desligue-a para isolar o executor.
   O comparador `relational` executa SQL exato sobre a mesma consulta compilada.
4. **Anotações privilegiadas.** `witnessrag-annotated` e o nome legado
   `witnessrag-oracle` usam traduções heurísticas de anotações, às vezes a resposta
   ouro. São diagnósticos privilegiados, não um teto garantido nem uma ablação pura.
5. **Garantias condicionais.** Exact sem cortes é completo para a consulta e os
   fatos fornecidos. Não certifica a extração ou a compilação do texto. Similaridade
   e scores de risco não calibrados não são probabilidades.
6. **Incerteza experimental.** Use diferenças pareadas, tamanho de efeito e várias
   sementes. Sobreposição de ICs individuais não é um teste de igualdade. Hops por
   número de passagens são apenas um proxy; a fonte está nos registros.
7. **Orçamento.** O ILP otimiza testemunhas enumeradas das demandas disponíveis.
   A máscara de fatos não mede redução física de RAM. Demandas sintetizadas usam
   um prior estrutural amostrado, não uma distribuição validada de perguntas reais.
