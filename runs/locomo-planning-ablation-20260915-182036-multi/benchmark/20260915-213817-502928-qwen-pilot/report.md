# Resultados — 20260915-213817-502928-qwen-pilot

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
| witnessrag | 71.5 | 93.5 | 87.3 | 22.5 | 54.1 | 5.9 | 0 | 7.66 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [63.2, 78.6] | [90.0, 96.6] | [80.4, 93.1] | [14.7, 31.4] | [47.5, 61.2] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 70 | 61.4 | 28.6 | 63.8 | 30.0 | 100.0 | 100.0 |
| witnessrag | multi-hop | 32 | 38.3 | 9.4 | 39.6 | 15.6 | 79.4 | 59.4 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) | 2 hop(s) | 3 hop(s) | 4 hop(s) |
|---|---|---|---|---|
| witnessrag | 60.6 / 100.0 (n=76) | 35.3 / 52.6 (n=19) | 38.3 / 50.0 (n=6) | 17.4 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | intersection | single-hop |
|---|---|---|---|
| witnessrag | 100.0 (n=1) | 54.9 (n=10) | 53.5 (n=91) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 102 | 3.9 | 32.4 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 102 | 32.4 | 8.8 | 33 | 1 | 0 | 13 | 55 | 0 | 171 | 38.6 |

#### Motivos de rejeição do verificador

Contagem por testemunha avaliada; uma pergunta pode produzir várias rejeições.

| método | motivo | n |
|---|---|---|
| witnessrag | wrong_answer_type | 51 |
| witnessrag | wrong_identity | 19 |
| witnessrag | not_explicit | 17 |
| witnessrag | invalid_evidence | 8 |
| witnessrag | missing_atom | 5 |
| witnessrag | wrong_relation | 4 |
| witnessrag | incomplete_answer | 1 |

### Planejamento adaptativo

O replanejador recebe somente a pergunta, vocabulário do grafo, lacunas, rejeições e fatos candidatos adquiridos; não recebe respostas ouro.

| método | n | planos médios | chamadas médias | replanejadas | revisão escolhida | revisão com prova |
|---|---|---|---|---|---|---|
| witnessrag | 102 | 1.16 | 1.94 | 80 | 8 | 5 |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 33 | 2.00 | 57.6 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 5.3 (18.6%) | 7.1 (27.5%) | 7.1 (27.5%) | 3.9 (100.0%) | 3.9 (100.0%) |

### Custo

Indexação compartilhada: 96.32256545126438 s · {"tokens_resposta_sem_cache": 24419, "tokens_prompt": 90198, "tokens_prompt_sem_cache": 62315, "em_cache": 50, "chamadas": 100, "filtradas": 0, "tokens_resposta": 26351}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 582 | 2004457 | 36134 | 956.352 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.021 | 0.000 | 0 |


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
