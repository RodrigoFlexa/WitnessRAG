# Resultados — 20260915-002222-403822-conv01

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-base-en-v1.5`)
- **top-k**: 15 · **perguntas por dataset**: 55 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

55 perguntas · 53 passagens · 1.29 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-base-en-v1.5`.

Comparação pareada: 55 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 63.3 | 70.6 | 63.6 | 25.5 | 50.5 | 5.5 | 0 | 7.31 |
| hybrid | 61.5 | 70.6 | 63.6 | 25.5 | 50.0 | 5.5 | 0 | 0.01 |
| dense | 52.3 | 66.8 | 60.0 | 23.6 | 43.5 | 7.3 | 0 | 0.01 |
| bm25 | 53.8 | 72.9 | 67.3 | 21.8 | 45.7 | 7.3 | 0 | 0.00 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [50.9, 75.6] | [59.2, 81.7] | [50.9, 78.2] | [14.5, 36.4] | [40.2, 60.9] |
| hybrid | [48.8, 73.8] | [59.4, 82.4] | [50.9, 78.2] | [14.5, 36.4] | [39.7, 60.7] |
| dense | [39.5, 65.9] | [55.0, 78.2] | [47.3, 72.7] | [12.7, 34.5] | [32.6, 53.9] |
| bm25 | [41.2, 67.0] | [62.0, 83.3] | [54.5, 80.0] | [10.9, 32.7] | [35.3, 56.2] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| hybrid | [0.0, 1.5] | [-5.5, 5.5] |
| dense | [-2.4, 16.9] | [-7.3, 14.5] |
| bm25 | [-2.4, 12.8] | [-14.5, 5.5] |

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 44 | 52.3 | 25.0 | 55.2 | 25.0 | 79.5 | 79.5 |
| witnessrag | multi-hop | 11 | 43.3 | 27.3 | 45.7 | 27.3 | 34.8 | 0.0 |
| hybrid | single-hop | 44 | 51.7 | 25.0 | 54.5 | 25.0 | 79.5 | 79.5 |
| hybrid | multi-hop | 11 | 43.3 | 27.3 | 45.7 | 27.3 | 34.8 | 0.0 |
| dense | single-hop | 44 | 43.7 | 22.7 | 49.4 | 22.7 | 75.0 | 75.0 |
| dense | multi-hop | 11 | 42.6 | 27.3 | 44.0 | 27.3 | 34.1 | 0.0 |
| bm25 | single-hop | 44 | 46.4 | 22.7 | 50.7 | 22.7 | 79.5 | 79.5 |
| bm25 | multi-hop | 11 | 42.6 | 18.2 | 43.9 | 27.3 | 46.2 | 18.2 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) |
|---|---|---|---|---|
| witnessrag | 52.3 / 79.5 (n=44) | 36.9 / 0.0 (n=8) | 55.6 / 0.0 (n=1) | 62.8 / 0.0 (n=2) |
| hybrid | 51.7 / 79.5 (n=44) | 36.9 / 0.0 (n=8) | 55.6 / 0.0 (n=1) | 62.8 / 0.0 (n=2) |
| dense | 43.7 / 75.0 (n=44) | 39.1 / 0.0 (n=8) | 44.4 / 0.0 (n=1) | 55.7 / 0.0 (n=2) |
| bm25 | 46.4 / 79.5 (n=44) | 45.4 / 25.0 (n=8) | 100.0 / 0.0 (n=1) | 2.9 / 0.0 (n=2) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | disconnected | intersection | single-hop |
|---|---|---|---|---|---|
| witnessrag | 38.4 (n=5) | 40.1 (n=15) | 35.1 (n=3) | 55.1 (n=18) | 76.4 (n=5) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 55 | 1.8 | 10.9 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 55 | 10.9 | 6 | 9 | 0 | 21 | 19 | 0 | 115 | 9.6 |
| hybrid | 55 | 100.0 | 55 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| dense | 55 | 100.0 | 55 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| bm25 | 55 | 100.0 | 55 | 0 | 0 | 0 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 6 | 1.33 | 16.7 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 20.0 (9.1%) | 20.0 (9.1%) | 1.8 (100.0%) | 1.8 (100.0%) | 1.8 (100.0%) |

### Custo

Indexação compartilhada: 96.25966965127736 s · {"chamadas": 106, "tokens_resposta": 29451, "tokens_prompt_sem_cache": 75395, "tokens_prompt": 75395, "em_cache": 0, "tokens_resposta_sem_cache": 29451, "filtradas": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 342 | 543867 | 20213 | 473.56 |
| hybrid | 55 | 321889 | 1241 | 72.108 |
| dense | 55 | 305642 | 1247 | 69.688 |
| bm25 | 55 | 333067 | 718 | 62.95 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.017 | 0.000 | 0 |
| hybrid | 0.016 | 0.000 | 0 |
| dense | 0.000 | 0.000 | 0 |
| bm25 | 0.015 | 0.000 | 0 |


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
