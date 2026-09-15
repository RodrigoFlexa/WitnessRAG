# Resultados — 20260915-003422-143861-conv02

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-base-en-v1.5`)
- **top-k**: 15 · **perguntas por dataset**: 117 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

117 perguntas · 98 passagens · 1.4 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-base-en-v1.5`.

Comparação pareada: 117 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 67.9 | 81.6 | 72.6 | 36.8 | 67.0 | 5.1 | 0 | 6.80 |
| hybrid | 68.6 | 81.5 | 73.5 | 37.6 | 67.0 | 5.1 | 0 | 0.01 |
| dense | 51.5 | 68.8 | 63.2 | 32.5 | 59.4 | 9.4 | 0 | 0.01 |
| bm25 | 62.5 | 76.6 | 69.2 | 37.6 | 65.7 | 6.8 | 0 | 0.00 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [60.2, 76.1] | [75.4, 87.6] | [64.1, 81.2] | [28.2, 46.2] | [60.5, 73.6] |
| hybrid | [60.7, 76.6] | [75.3, 87.8] | [65.0, 82.1] | [29.1, 47.0] | [60.6, 73.6] |
| dense | [42.6, 59.6] | [61.3, 76.3] | [54.7, 71.8] | [24.8, 41.0] | [52.4, 65.8] |
| bm25 | [53.8, 70.9] | [69.5, 83.4] | [60.7, 77.8] | [29.1, 47.0] | [59.2, 72.4] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| hybrid | [-0.0, 0.0] | [-2.6, 0.0] |
| dense | [1.8, 13.7] | [3.4, 16.2] |
| bm25 | [-4.6, 6.7] | [-2.6, 9.4] |

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 86 | 70.1 | 44.2 | 71.9 | 46.5 | 91.9 | 91.9 |
| witnessrag | multi-hop | 31 | 58.2 | 16.1 | 57.9 | 29.0 | 53.1 | 19.4 |
| hybrid | single-hop | 86 | 70.1 | 44.2 | 71.9 | 46.5 | 91.9 | 91.9 |
| hybrid | multi-hop | 31 | 58.3 | 19.4 | 58.0 | 29.0 | 52.8 | 22.6 |
| dense | single-hop | 86 | 63.6 | 39.5 | 64.6 | 39.5 | 75.6 | 75.6 |
| dense | multi-hop | 31 | 47.6 | 12.9 | 47.9 | 12.9 | 50.2 | 29.0 |
| bm25 | single-hop | 86 | 70.9 | 46.5 | 72.3 | 47.7 | 90.7 | 90.7 |
| bm25 | multi-hop | 31 | 51.3 | 12.9 | 48.2 | 16.1 | 37.5 | 9.7 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) | 5 hop(s) |
|---|---|---|---|---|---|
| witnessrag | 70.1 / 91.9 (n=86) | 59.7 / 27.3 (n=22) | 52.8 / 0.0 (n=3) | 56.7 / 0.0 (n=5) | 50.0 / 0.0 (n=1) |
| hybrid | 70.1 / 91.9 (n=86) | 59.7 / 31.8 (n=22) | 52.8 / 0.0 (n=3) | 57.0 / 0.0 (n=5) | 50.0 / 0.0 (n=1) |
| dense | 63.6 / 75.6 (n=86) | 50.4 / 36.4 (n=22) | 44.4 / 0.0 (n=3) | 37.7 / 20.0 (n=5) | 44.4 / 0.0 (n=1) |
| bm25 | 70.9 / 90.7 (n=86) | 51.5 / 13.6 (n=22) | 66.7 / 0.0 (n=3) | 43.8 / 0.0 (n=5) | 40.0 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 59.3 (n=21) | 68.3 (n=18) | 41.7 (n=2) | 40.7 (n=10) | 76.7 (n=39) | 69.4 (n=11) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 117 | 3.4 | 7.7 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 117 | 7.7 | 9 | 16 | 0 | 58 | 34 | 0 | 196 | 5.6 |
| hybrid | 117 | 100.0 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| dense | 117 | 100.0 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| bm25 | 117 | 100.0 | 117 | 0 | 0 | 0 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 9 | 1.22 | 11.1 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 3.4 (100.0%) | 3.4 (100.0%) | 3.4 (100.0%) | 3.4 (100.0%) | 3.4 (100.0%) |

### Custo

Indexação compartilhada: 98.82325186673552 s · {"chamadas": 196, "tokens_resposta": 55218, "tokens_prompt_sem_cache": 142014, "tokens_prompt": 142014, "em_cache": 0, "tokens_resposta_sem_cache": 55218, "filtradas": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 715 | 1103406 | 39589 | 916.836 |
| hybrid | 117 | 663265 | 1348 | 123.098 |
| dense | 117 | 598083 | 1406 | 112.615 |
| bm25 | 117 | 716128 | 1317 | 129.699 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.054 | 0.000 | 0 |
| hybrid | 0.036 | 0.000 | 0 |
| dense | 0.000 | 0.000 | 0 |
| bm25 | 0.047 | 0.000 | 0 |


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
