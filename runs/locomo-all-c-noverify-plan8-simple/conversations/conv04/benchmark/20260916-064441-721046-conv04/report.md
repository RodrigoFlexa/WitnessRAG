# Resultados — 20260916-064441-721046-conv04

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 138 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

138 perguntas · 15 passagens · 1.38 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 138 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 69.3 | 85.9 | 80.4 | 36.2 | 58.7 | 10.9 | 0 | 2.27 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [61.6, 75.7] | [80.2, 90.8] | [73.9, 87.0] | [28.3, 44.2] | [51.9, 65.3] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 107 | 66.3 | 45.8 | 66.8 | 46.7 | 91.6 | 91.6 |
| witnessrag | multi-hop | 31 | 32.4 | 3.2 | 35.3 | 3.2 | 66.1 | 41.9 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) | 5 hop(s) | 6 hop(s) |
|---|---|---|---|---|---|---|
| witnessrag | 65.7 / 91.7 (n=108) | 37.9 / 50.0 (n=18) | 21.5 / 14.3 (n=7) | 22.2 / 50.0 (n=2) | 38.1 / 100.0 (n=1) | 44.3 / 0.0 (n=2) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 50.0 (n=4) | 66.4 (n=12) | 66.7 (n=3) | 60.5 (n=14) | 59.8 (n=42) | 56.3 (n=63) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 138 | 0.7 | 55.1 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 138 | 55.1 | 44.9 | 76 | 14 | 0 | 48 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 76 | 6.30 | 76.3 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 4.0 (18.1%) | 2.0 (36.2%) | 1.3 (54.3%) | 0.7 (100.0%) | 0.7 (100.0%) |

### Custo

Indexação compartilhada: 149.80199682246894 s · {"tokens_resposta": 38863, "filtradas": 0, "tokens_prompt_sem_cache": 134889, "em_cache": 0, "chamadas": 148, "tokens_resposta_sem_cache": 38863, "tokens_prompt": 134889}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 549 | 2197244 | 18327 | 576.727 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.032 | 0.000 | 0 |


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
