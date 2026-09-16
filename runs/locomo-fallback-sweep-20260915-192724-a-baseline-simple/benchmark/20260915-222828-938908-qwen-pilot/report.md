# Resultados — 20260915-222828-938908-qwen-pilot

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
| witnessrag | 71.7 | 94.6 | 89.2 | 22.5 | 55.3 | 5.9 | 0 | 0.05 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [63.2, 79.4] | [91.2, 97.4] | [82.4, 94.1] | [14.7, 31.4] | [48.4, 62.5] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 70 | 62.6 | 28.6 | 65.0 | 30.0 | 100.0 | 100.0 |
| witnessrag | multi-hop | 32 | 39.4 | 9.4 | 42.6 | 18.8 | 82.8 | 65.6 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) |
|---|---|---|---|---|
| witnessrag | 61.7 / 100.0 (n=76) | 37.8 / 63.2 (n=19) | 36.9 / 50.0 (n=6) | 14.3 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | branching | chain | cyclic | disconnected | intersection | single-hop |
|---|---|---|---|---|---|---|
| witnessrag | 63.4 (n=3) | 77.1 (n=7) | 0.0 (n=1) | 44.3 (n=10) | 55.5 (n=21) | 55.1 (n=60) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 102 | 2.9 | 35.3 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 102 | 35.3 | 8.8 | 36 | 12 | 0 | 26 | 28 | 0 | 240 | 28.7 |

#### Motivos de rejeição do verificador

Contagem por testemunha avaliada; uma pergunta pode produzir várias rejeições.

| método | motivo | n |
|---|---|---|
| witnessrag | wrong_answer_type | 88 |
| witnessrag | wrong_identity | 37 |
| witnessrag | not_explicit | 22 |
| witnessrag | missing_atom | 14 |
| witnessrag | invalid_evidence | 7 |
| witnessrag | wrong_relation | 3 |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 36 | 1.92 | 52.8 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 0.0 (18.6%) | 3.6 (27.5%) | 3.6 (27.5%) | 2.9 (100.0%) | 2.9 (100.0%) |

### Custo

Indexação compartilhada: 0.7961075594648719 s · {"chamadas": 0, "filtradas": 0, "tokens_prompt_sem_cache": 0, "tokens_resposta_sem_cache": 0, "tokens_resposta": 0, "tokens_prompt": 0, "em_cache": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 612 | 2212311 | 44231 | 182.621 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.020 | 0.000 | 0 |


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
