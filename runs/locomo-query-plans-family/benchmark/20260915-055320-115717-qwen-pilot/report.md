# Resultados — 20260915-055320-115717-qwen-pilot

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 102 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

102 perguntas · 10 passagens · 1.33 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 61 perguntas presentes em todos os métodos.

> Rodada incompleta: 41 perguntas sem resultado em algum método.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| witnessrag | 71.3 | 94.0 | 86.9 | 23.0 | 57.7 | 6.6 | 0 | 13.28 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [60.7, 81.7] | [89.6, 97.5] | [77.0, 95.1] | [13.1, 32.8] | [48.6, 66.0] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 45 | 67.9 | 28.9 | 70.4 | 31.1 | 100.0 | 100.0 |
| witnessrag | multi-hop | 16 | 28.9 | 6.2 | 33.1 | 6.2 | 77.1 | 50.0 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) |
|---|---|---|---|
| witnessrag | 65.9 / 100.0 (n=48) | 25.3 / 40.0 (n=10) | 34.4 / 33.3 (n=3) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | disconnected | intersection | single-hop |
|---|---|---|---|---|
| witnessrag | 26.0 (n=5) | 81.8 (n=1) | 75.0 (n=14) | 55.7 (n=39) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 61 | 3.3 | 42.6 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 61 | 42.6 | 11.5 | 26 | 2 | 0 | 17 | 16 | 0 | 138 | 37.7 |

#### Motivos de rejeição do verificador

Contagem por testemunha avaliada; uma pergunta pode produzir várias rejeições.

| método | motivo | n |
|---|---|---|
| witnessrag | wrong_answer_type | 50 |
| witnessrag | wrong_identity | 21 |
| witnessrag | invalid_evidence | 5 |
| witnessrag | not_explicit | 4 |
| witnessrag | incomplete_answer | 2 |
| witnessrag | missing_atom | 2 |
| witnessrag | wrong_relation | 2 |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 26 | 2.00 | 46.2 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 9.1 (18.0%) | 9.1 (36.1%) | 9.1 (36.1%) | 3.3 (100.0%) | 3.3 (100.0%) |

### Custo

Indexação compartilhada: 127.556216054596 s · {"filtradas": 0, "tokens_prompt_sem_cache": 89785, "tokens_resposta": 35082, "tokens_resposta_sem_cache": 35082, "chamadas": 100, "em_cache": 0, "tokens_prompt": 89785}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 365 | 1281053 | 32040 | 914.742 |

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
