# Resultados — 20260922-163548-987315-qwen-pilot

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-m3`)
- **top-k**: 5 · **perguntas por dataset**: 152 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## locomo

152 perguntas · 15 passagens · 1.28 apoios anotados em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-m3`.

Comparação pareada: 152 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@2 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 59.7 | 83.6 | 54.7 | 77.3 | 25.7 | 54.3 | 8.6 | 0 | 5.84 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@2 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|---|
| witnessrag | [52.0, 67.4] | [78.4, 88.5] | [46.7, 62.7] | [71.3, 84.0] | [19.1, 32.9] | [47.8, 59.8] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### LoCoMo: categorias oficiais

Adaptação textual de uma conversa. Recall é sobre blocos de diálogo. O número de blocos de apoio não define a categoria de hops.

| método | categoria | n | F1 | EM | F1 ofic. | BLEU-1 | EM ofic. | R@5 | AR@5 |
|---|---|---|---|---|---|---|---|---|---|
| witnessrag | single-hop | 70 | 63.3 | 32.9 | 65.8 | 58.4 | 34.3 | 95.7 | 95.7 |
| witnessrag | multi-hop | 32 | 33.0 | 6.2 | 37.2 | 28.2 | 9.4 | 64.3 | 37.5 |
| witnessrag | temporal | 37 | 61.9 | 32.4 | 62.3 | 56.8 | 32.4 | 78.4 | 78.4 |
| witnessrag | open-domain | 13 | 36.3 | 15.4 | 53.0 | 44.9 | 15.4 | 80.3 | 72.7 |

Avaliador oficial: `https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py`.


### Por número de apoios anotados (F1 / all-recall@5)

| método | 1 apoios anotados | 2 apoios anotados | 3 apoios anotados | 4 apoios anotados | 5 apoios anotados |
|---|---|---|---|---|---|
| witnessrag | 60.3 / 89.0 (n=120) | 31.8 / 44.0 (n=25) | 27.0 / 0.0 (n=5) | 29.6 / 0.0 (n=1) | 60.0 / 0.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | disconnected | intersection | single-hop |
|---|---|---|---|---|
| witnessrag | 100.0 (n=2) | 57.1 (n=2) | 41.7 (n=3) | 52.1 (n=118) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 152 | 1.3 | 82.2 |

### Onde a pergunta parou (taxa de disparo da testemunha)

Uma taxa de disparo baixa significa que a maior parte das respostas veio do fallback, não do executor: nesse regime a tabela principal mede o recuperador de reserva. Aprovação é sobre as testemunhas efetivamente verificadas.

| método | n | disparo | mudou contexto | testemunha usada | sem consulta | compilação bloqueada | junção não fechou | verificação rejeitou | prova insuficiente | outro | verificadas | aprovadas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| witnessrag | 152 | 41.4 | 61.8 | 63 | 0 | 0 | 0 | 0 | 27 | 62 | 0 | — |

#### Etapa de parada por pergunta

A classificação detalhada usa os planos registrados; categorias são mutuamente exclusivas.

| método | etapa | perguntas |
|---|---|---:|
| witnessrag | proof_accepted | 63 |
| witnessrag | provisional_context | 62 |
| witnessrag | no_novel_plan_or_acquisition | 27 |

### Planejamento adaptativo

O replanejador recebe somente a pergunta, vocabulário do grafo, lacunas, rejeições e fatos candidatos adquiridos; não recebe respostas ouro.

| método | n | planos médios | chamadas médias | replanejadas | revisão escolhida | revisão com prova |
|---|---|---|---|---|---|---|
| witnessrag | 152 | 1.26 | 2.34 | 126 | 16 | 7 |

### Conjunto de respostas certas

Itens por pergunta em que o executor provou alguma atribuição. A completude do conjunto NÃO é certificada: nada limita o que ficou de fora por falha de extração, compilação ou corte.

| método | n | itens por pergunta | com mais de um item |
|---|---|---|---|
| witnessrag | 125 | 5.27 | 73.6 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 3.6 (18.4%) | 3.6 (36.2%) | 3.6 (36.2%) | 1.3 (100.0%) | 1.3 (100.0%) |

### Custo

Indexação compartilhada: 300.3866159282625 s · {"chamadas": 150, "tokens_resposta": 48553, "em_cache": 0, "tokens_prompt": 136814, "filtradas": 0, "tokens_prompt_sem_cache": 136814, "tokens_resposta_sem_cache": 48553}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 959 | 2817551 | 38752 | 1219.514 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| witnessrag | 0.022 | 0.000 | 0 |


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
