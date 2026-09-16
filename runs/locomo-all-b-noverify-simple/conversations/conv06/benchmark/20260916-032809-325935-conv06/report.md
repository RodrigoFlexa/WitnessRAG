# Resultados — 20260916-032809-325935-conv06

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 103 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

103 perguntas · 14 passagens · 1.22 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 103 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 67.3 | 84.4 | 79.6 | 35.9 | 57.8 | 10.7 | 0 | 1.98 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [58.6, 75.2] | [77.9, 90.3] | [71.8, 87.4] | [27.2, 45.6] | [49.6, 65.8] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 83 | 62.6 | 44.6 | 63.8 | 45.8 | 86.7 | 86.7 |
| witnessrag | multi-hop | 20 | 37.7 | 0.0 | 41.0 | 5.0 | 74.5 | 50.0 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 5 hop(s) |
|---|---|---|---|---|
| witnessrag | 60.6 / 86.2 (n=87) | 50.0 / 54.5 (n=11) | 26.7 / 25.0 (n=4) | 21.1 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 97.0 (n=3) | 47.0 (n=6) | 0.0 (n=1) | 50.2 (n=13) | 64.4 (n=31) | 55.7 (n=49) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 103 | 3.9 | 68.0 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 103 | 68.0 | 54.4 | 70 | 5 | 0 | 28 | 0 | 0 | 0 | — |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 70 | 5.77 | 74.3 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 15.8 (18.4%) | 10.8 (35.9%) | 6.1 (64.1%) | 6.1 (64.1%) | 3.9 (100.0%) |

### Custo

Indexação compartilhada: 132.13312455359846 s · {"tokens_resposta": 37296, "chamadas": 138, "tokens_prompt": 124698, "tokens_resposta_sem_cache": 37296, "filtradas": 0, "tokens_prompt_sem_cache": 124698, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 359 | 1527075 | 9469 | 391.225 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.029 | 0.000 | 0 |


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
