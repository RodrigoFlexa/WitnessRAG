# Resultados — 20260923-052549-298938-conv08

## Configuração

- **LLM**: `azure` / deployment `gpt-4-1-mini-petrobras`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 156 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

156 perguntas · 11 passagens · 1.53 apoios anotados em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 156 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@2 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 63.8 | 86.4 | 55.8 | 77.6 | 30.8 | 61.3 | 2.6 | 0 | 0.87 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [56.7, 70.1] | [81.6, 90.3] | [47.4, 62.8] | [69.9, 83.3] | [23.1, 38.5] | [55.5, 67.5] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 73 | 70.4 | 42.5 | 70.9 | 66.6 | 46.6 | 90.4 | 90.4 |
| witnessrag | multi-hop | 37 | 42.1 | 16.2 | 41.1 | 37.5 | 16.2 | 79.5 | 54.1 |
| witnessrag | temporal | 33 | 70.9 | 24.2 | 71.7 | 67.3 | 42.4 | 92.4 | 90.9 |
| witnessrag | open-domain | 13 | 40.2 | 23.1 | 40.5 | 37.5 | 23.1 | 68.7 | 38.5 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados | 5 apoios anotados | 6 apoios anotados | 8 apoios anotados |
|---|---|---|---|---|---|---|---|
| witnessrag | 70.0 / 91.2 (n=114) | 42.9 / 64.0 (n=25) | 33.8 / 16.7 (n=6) | 36.5 / 0.0 (n=4) | 30.3 / 0.0 (n=3) | 13.4 / 0.0 (n=3) | 33.3 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|
| witnessrag | 0.0 (n=1) | 57.1 (n=1) | 43.1 (n=2) | 38.1 (n=12) | 45.5 (n=21) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 156 | 0.0 | 0.6 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 156 | 100.0 | 23.1 | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | non_multihop_hybrid | 119 |
| witnessrag | plan_not_compositional_probe | 22 |
| witnessrag | join_incomplete_probe | 7 |
| witnessrag | witness_set_too_broad_probe | 4 |
| witnessrag | answer_set_too_broad_probe | 2 |
| witnessrag | plan_not_compositional | 1 |
| witnessrag | selective_witness | 1 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) |

### Custo

Indexação compartilhada: 205.65403613698436 s · {"chamadas": 108, "tokens_resposta": 20846, "em_cache": 0, "tokens_prompt": 93281, "filtradas": 0, "tokens_prompt_sem_cache": 93281, "tokens_resposta_sem_cache": 20846}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 193 | 1668273 | 3902 | 368.122 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.035 | 0.000 | 0 |


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
