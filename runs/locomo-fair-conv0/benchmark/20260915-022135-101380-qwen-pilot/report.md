# Resultados — 20260915-022135-101380-qwen-pilot

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 102 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

102 perguntas · 10 passagens · 1.33 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 102 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| dense | 60.2 | 87.7 | 81.4 | 19.6 | 52.2 | 7.8 | 0 | 0.02 |
| hybrid | 72.4 | 93.0 | 86.3 | 22.5 | 54.1 | 5.9 | 0 | 0.02 |
| witnessrag | 74.2 | 94.1 | 87.3 | 21.6 | 53.8 | 5.9 | 0 | 9.81 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| dense | [52.0, 68.3] | [82.0, 92.5] | [73.5, 88.2] | [11.8, 28.4] | [45.3, 59.8] |
| hybrid | [64.1, 79.6] | [89.2, 96.2] | [79.4, 92.2] | [14.7, 31.4] | [47.5, 61.3] |
| witnessrag | [66.2, 81.2] | [90.8, 96.9] | [80.4, 93.1] | [13.7, 30.4] | [46.5, 61.5] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| dense | [-4.6, 7.7] | [0.0, 12.7] |
| hybrid | [-3.3, 2.5] | [-2.0, 3.9] |

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| dense | single-hop | 70 | 61.1 | 25.7 | 62.9 | 28.6 | 91.4 | 91.4 |
| dense | multi-hop | 32 | 32.8 | 6.2 | 37.5 | 9.4 | 79.7 | 59.4 |
| hybrid | single-hop | 70 | 61.5 | 28.6 | 63.8 | 30.0 | 100.0 | 100.0 |
| hybrid | multi-hop | 32 | 38.1 | 9.4 | 39.8 | 12.5 | 77.6 | 56.2 |
| witnessrag | single-hop | 70 | 60.3 | 28.6 | 62.4 | 30.0 | 100.0 | 100.0 |
| witnessrag | multi-hop | 32 | 39.6 | 6.2 | 41.3 | 12.5 | 81.2 | 59.4 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) |
|---|---|---|---|---|
| dense | 60.3 / 92.1 (n=76) | 25.7 / 63.2 (n=19) | 38.7 / 16.7 (n=6) | 26.7 / 0.0 (n=1) |
| hybrid | 60.6 / 100.0 (n=76) | 34.1 / 52.6 (n=19) | 41.9 / 33.3 (n=6) | 14.3 / 0.0 (n=1) |
| witnessrag | 58.2 / 100.0 (n=76) | 41.0 / 52.6 (n=19) | 44.5 / 50.0 (n=6) | 14.3 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 57.3 (n=9) | 61.2 (n=21) | 43.6 (n=5) | 40.0 (n=9) | 48.2 (n=25) | 56.1 (n=25) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 102 | 2.0 | 26.5 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 102 | 100.0 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| hybrid | 102 | 100.0 | 102 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| witnessrag | 102 | 26.5 | 27 | 8 | 0 | 34 | 33 | 0 | 228 | 22.4 |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 27 | 1.74 | 51.9 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 5.3 (18.6%) | 5.3 (18.6%) | 2.0 (100.0%) | 2.0 (100.0%) | 2.0 (100.0%) |

### Custo

Indexação compartilhada: 77.15754270832986 s · {"tokens_resposta_sem_cache": 33699, "em_cache": 0, "filtradas": 0, "tokens_prompt": 84534, "chamadas": 100, "tokens_resposta": 33699, "tokens_prompt_sem_cache": 84534}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| dense | 102 | 1046018 | 1506 | 166.238 |
| hybrid | 102 | 1045433 | 1418 | 167.556 |
| witnessrag | 660 | 2383956 | 47965 | 1171.214 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| dense | 0.000 | 0.000 | 0 |
| hybrid | 0.021 | 0.000 | 0 |
| witnessrag | 0.021 | 0.000 | 0 |


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
