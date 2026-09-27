# Resultados — 20260927-042734-416343-conv08

## Configuração

- **LLM**: `openai` / deployment `gpt-4o-mini`
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
| witnessrag | 64.4 | 87.6 | 56.4 | 79.5 | 28.2 | 51.5 | 3.8 | 0 | 3.04 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [57.5, 70.7] | [82.8, 91.5] | [48.1, 63.5] | [73.1, 85.3] | [21.2, 35.9] | [45.6, 58.0] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 73 | 64.0 | 43.8 | 65.4 | 60.9 | 43.8 | 90.4 | 90.4 |
| witnessrag | multi-hop | 37 | 36.1 | 13.5 | 35.9 | 30.3 | 13.5 | 82.9 | 59.5 |
| witnessrag | temporal | 33 | 44.1 | 9.1 | 44.2 | 37.0 | 18.2 | 95.5 | 93.9 |
| witnessrag | open-domain | 13 | 43.9 | 30.8 | 45.0 | 41.1 | 30.8 | 64.9 | 38.5 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados | 5 apoios anotados | 6 apoios anotados | 8 apoios anotados |
|---|---|---|---|---|---|---|---|
| witnessrag | 58.6 / 92.1 (n=114) | 34.3 / 64.0 (n=25) | 34.6 / 50.0 (n=6) | 31.5 / 0.0 (n=4) | 30.7 / 0.0 (n=3) | 23.5 / 0.0 (n=3) | 0.0 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | disconnected | intersection | single-hop |
|---|---|---|---|
| witnessrag | 55.1 (n=45) | 46.6 (n=17) | 51.2 (n=77) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 156 | 7.7 | 42.3 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 156 | 100.0 | 25.6 | 156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | respostas_demais | 51 |
| witnessrag | prova_confirmada | 44 |
| witnessrag | prova_ja_no_contexto | 22 |
| witnessrag | join_incompleto | 18 |
| witnessrag | plano_invalido | 17 |
| witnessrag | prova_recusada | 2 |
| witnessrag | busca_truncada | 1 |
| witnessrag | testemunhas_demais | 1 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 7.7 (100.0%) | 7.7 (100.0%) | 7.7 (100.0%) | 7.7 (100.0%) | 7.7 (100.0%) |

### Custo

Indexação compartilhada: 90.14113150001504 s · {"tokens_resposta": 37862, "filtradas": 0, "tokens_prompt": 78980, "chamadas": 54, "tokens_prompt_sem_cache": 78980, "tokens_resposta_sem_cache": 37862, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 468 | 1204943 | 27056 | 586.723 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.069 | 0.000 | 0 |


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
