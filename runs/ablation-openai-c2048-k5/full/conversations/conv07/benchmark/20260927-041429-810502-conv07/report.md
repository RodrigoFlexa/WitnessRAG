# Resultados — 20260927-041429-810502-conv07

## Configuração

- **LLM**: `openai` / deployment `gpt-4o-mini`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 191 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

191 perguntas · 14 passagens · 1.25 apoios anotados em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 191 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@2 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 71.8 | 90.8 | 68.6 | 86.9 | 30.4 | 56.0 | 6.3 | 0 | 2.77 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [65.5, 77.6] | [87.0, 94.2] | [61.8, 74.9] | [81.7, 91.1] | [24.1, 37.2] | [51.0, 61.2] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 118 | 58.9 | 36.4 | 60.2 | 54.0 | 37.3 | 92.4 | 92.4 |
| witnessrag | multi-hop | 21 | 39.9 | 9.5 | 47.3 | 34.8 | 9.5 | 77.8 | 52.4 |
| witnessrag | temporal | 42 | 59.8 | 26.2 | 59.8 | 53.2 | 31.0 | 96.8 | 95.2 |
| witnessrag | open-domain | 10 | 40.0 | 20.0 | 40.0 | 31.0 | 20.0 | 74.0 | 60.0 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados | 5 apoios anotados |
|---|---|---|---|---|---|
| witnessrag | 57.1 / 93.1 (n=160) | 60.8 / 73.7 (n=19) | 35.4 / 37.5 (n=8) | 41.6 / 0.0 (n=3) | 0.0 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|
| witnessrag | 0.0 (n=1) | 100.0 (n=1) | 51.0 (n=41) | 57.2 (n=22) | 55.6 (n=100) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 191 | 6.3 | 37.7 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 191 | 100.0 | 29.8 | 191 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | respostas_demais | 52 |
| witnessrag | prova_confirmada | 48 |
| witnessrag | join_incompleto | 27 |
| witnessrag | plano_invalido | 26 |
| witnessrag | prova_ja_no_contexto | 24 |
| witnessrag | prova_recusada | 7 |
| witnessrag | testemunhas_demais | 7 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 6.3 (100.0%) | 6.3 (100.0%) | 6.3 (100.0%) | 6.3 (100.0%) | 6.3 (100.0%) |

### Custo

Indexação compartilhada: 125.06575720000546 s · {"tokens_resposta": 51962, "filtradas": 0, "tokens_prompt": 99097, "chamadas": 68, "tokens_prompt_sem_cache": 99097, "tokens_resposta_sem_cache": 51962, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 587 | 1515033 | 34341 | 662.392 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.085 | 0.000 | 0 |


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
