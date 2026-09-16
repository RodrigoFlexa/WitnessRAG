# Resultados — 20260916-024219-873097-conv02

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 117 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

117 perguntas · 15 passagens · 1.31 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 117 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 57.8 | 75.2 | 70.1 | 32.5 | 57.3 | 16.2 | 0 | 2.47 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [49.4, 66.2] | [67.5, 83.5] | [61.5, 79.5] | [24.8, 41.0] | [50.4, 65.6] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 86 | 63.8 | 39.5 | 64.2 | 40.7 | 84.9 | 84.9 |
| witnessrag | multi-hop | 31 | 39.4 | 12.9 | 41.3 | 16.1 | 48.4 | 29.0 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) |
|---|---|---|---|---|
| witnessrag | 62.4 / 82.6 (n=92) | 40.6 / 27.8 (n=18) | 54.0 / 33.3 (n=3) | 18.4 / 0.0 (n=4) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 30.0 (n=5) | 52.1 (n=5) | 100.0 (n=1) | 35.1 (n=13) | 61.0 (n=40) | 62.3 (n=53) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 117 | 2.6 | 53.0 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 117 | 53.0 | 45.3 | 62 | 9 | 0 | 46 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 62 | 5.31 | 69.4 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 9.5 (17.9%) | 7.0 (36.8%) | 5.7 (45.3%) | 2.6 (100.0%) | 2.6 (100.0%) |

### Custo

Indexação compartilhada: 147.44377750344574 s · {"tokens_resposta": 44151, "chamadas": 148, "tokens_prompt": 133646, "tokens_resposta_sem_cache": 44151, "filtradas": 0, "tokens_prompt_sem_cache": 133646, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 483 | 1884403 | 12680 | 498.64 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.030 | 0.000 | 0 |


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
