# Resultados — 20260923-043727-612193-conv04

## Configuração

- **LLM**: `azure` / deployment `gpt-4-1-mini-petrobras`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 178 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

178 perguntas · 15 passagens · 1.37 apoios anotados em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 178 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@2 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 76.6 | 89.1 | 69.7 | 83.7 | 33.7 | 60.5 | 5.6 | 0 | 0.71 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [71.1, 82.2] | [85.1, 93.0] | [62.9, 76.4] | [78.1, 88.8] | [27.0, 40.4] | [54.9, 65.9] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 107 | 70.2 | 45.8 | 71.3 | 66.9 | 48.6 | 96.3 | 96.3 |
| witnessrag | multi-hop | 31 | 45.7 | 6.5 | 46.2 | 36.3 | 6.5 | 66.1 | 38.7 |
| witnessrag | temporal | 26 | 68.1 | 34.6 | 68.2 | 64.1 | 38.5 | 94.2 | 92.3 |
| witnessrag | open-domain | 14 | 5.5 | 0.0 | 4.8 | 2.8 | 0.0 | 75.6 | 71.4 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados | 5 apoios anotados | 6 apoios anotados |
|---|---|---|---|---|---|---|
| witnessrag | 66.5 / 94.9 (n=138) | 41.3 / 60.0 (n=25) | 38.7 / 33.3 (n=9) | 15.4 / 0.0 (n=3) | 33.3 / 0.0 (n=1) | 64.9 / 0.0 (n=2) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|
| witnessrag | 47.1 (n=1) | 40.0 (n=1) | 11.1 (n=1) | 47.7 (n=7) | 46.8 (n=21) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 178 | 0.0 | 0.0 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 178 | 100.0 | 16.9 | 178 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | non_multihop_hybrid | 147 |
| witnessrag | plan_not_compositional_probe | 21 |
| witnessrag | join_incomplete_probe | 5 |
| witnessrag | witness_set_too_broad_probe | 3 |
| witnessrag | answer_set_too_broad_probe | 1 |
| witnessrag | plan_not_compositional | 1 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) |

### Custo

Indexação compartilhada: 279.7614164960105 s · {"chamadas": 148, "tokens_resposta": 29857, "em_cache": 0, "tokens_prompt": 128229, "filtradas": 0, "tokens_prompt_sem_cache": 128229, "tokens_resposta_sem_cache": 29857}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 209 | 1878706 | 3458 | 398.456 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.058 | 0.000 | 0 |


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
