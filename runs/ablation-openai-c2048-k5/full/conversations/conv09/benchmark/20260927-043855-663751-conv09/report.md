# Resultados — 20260927-043855-663751-conv09

## Configuração

- **LLM**: `openai` / deployment `gpt-4o-mini`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 158 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

158 perguntas · 14 passagens · 1.27 apoios anotados em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 158 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@2 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 64.8 | 92.1 | 59.0 | 87.2 | 27.8 | 51.5 | 9.5 | 0 | 2.85 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [58.1, 71.9] | [88.0, 95.5] | [51.3, 66.7] | [81.4, 92.3] | [21.5, 34.8] | [44.9, 57.9] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 87 | 56.6 | 35.6 | 58.6 | 53.6 | 36.8 | 96.6 | 96.6 |
| witnessrag | multi-hop | 32 | 33.4 | 15.6 | 37.8 | 30.0 | 15.6 | 75.8 | 53.1 |
| witnessrag | temporal | 32 | 59.4 | 18.8 | 60.5 | 55.2 | 37.5 | 100.0 | 100.0 |
| witnessrag | open-domain | 7 | 34.3 | 28.6 | 52.4 | 50.0 | 28.6 | 70.0 | 60.0 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados |
|---|---|---|---|---|
| witnessrag | 55.1 / 96.8 (n=126) | 31.1 / 54.2 (n=24) | 56.1 / 50.0 (n=6) | 54.5 / 0.0 (n=2) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | disconnected | intersection | single-hop |
|---|---|---|---|---|
| witnessrag | 100.0 (n=1) | 51.8 (n=37) | 52.6 (n=17) | 51.8 (n=91) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 158 | 4.4 | 41.8 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 158 | 100.0 | 38.0 | 158 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | respostas_demais | 50 |
| witnessrag | prova_confirmada | 45 |
| witnessrag | prova_ja_no_contexto | 21 |
| witnessrag | join_incompleto | 19 |
| witnessrag | plano_invalido | 12 |
| witnessrag | testemunhas_demais | 8 |
| witnessrag | prova_recusada | 2 |
| witnessrag | busca_truncada | 1 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 4.4 (100.0%) | 4.4 (100.0%) | 4.4 (100.0%) | 4.4 (100.0%) | 4.4 (100.0%) |

### Custo

Indexação compartilhada: 175.01146149999113 s · {"tokens_resposta": 47164, "filtradas": 0, "tokens_prompt": 99339, "chamadas": 68, "tokens_prompt_sem_cache": 99339, "tokens_resposta_sem_cache": 47164, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 478 | 1223895 | 27471 | 559.762 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.096 | 0.000 | 0 |


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
