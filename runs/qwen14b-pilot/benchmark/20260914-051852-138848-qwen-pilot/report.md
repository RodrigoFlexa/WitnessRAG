# Resultados — 20260914-051852-138848-qwen-pilot

## Configuração

- **LLM**: `openai` / deployment `Qwen/Qwen2.5-14B-Instruct`
- **Embeddings**: `st` (`st-BAAI_bge-base-en-v1.5`)
- **top-k**: 5 · **perguntas por dataset**: 100 · **seed**: 42
- **Feixe da junção**: 400 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}


## 2wikimultihopqa

100 perguntas · 1059 passagens · 2.46 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido/piloto: True · embedding ajustado: `st-BAAI_bge-base-en-v1.5`.

Comparação pareada: 100 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| dense | 68.0 | 75.2 | 47.0 | 41.0 | 44.6 | 31.0 | 0 | 0.00 |
| bm25 | 56.0 | 67.2 | 36.0 | 34.0 | 36.9 | 42.0 | 0 | 0.01 |
| graphrag | 53.8 | 69.0 | 45.0 | 39.0 | 44.6 | 33.0 | 0 | 0.02 |
| hipporag | 75.5 | 92.8 | 82.0 | 51.0 | 57.4 | 18.0 | 0 | 0.43 |
| hipporag2 | 76.8 | 94.0 | 89.0 | 53.0 | 59.9 | 15.0 | 0 | 1.38 |
| relational | 68.0 | 75.2 | 47.0 | 41.0 | 44.6 | 31.0 | 0 | 2.02 |
| witnessrag | 53.5 | 61.8 | 40.0 | 40.0 | 42.9 | 37.0 | 0 | 0.75 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| dense | [63.2, 73.2] | [70.8, 80.2] | [37.0, 57.0] | [31.0, 51.0] | [35.0, 54.3] |
| bm25 | [51.2, 61.3] | [61.8, 72.8] | [27.0, 46.0] | [25.0, 44.0] | [28.0, 46.2] |
| graphrag | [47.2, 60.0] | [62.7, 75.5] | [36.0, 55.0] | [30.0, 49.0] | [35.6, 54.2] |
| hipporag | [70.5, 80.5] | [89.8, 95.8] | [75.0, 89.0] | [41.0, 61.0] | [48.2, 67.0] |
| hipporag2 | [71.2, 82.0] | [90.0, 97.2] | [82.0, 95.0] | [44.0, 63.0] | [51.0, 68.6] |
| relational | [63.2, 73.2] | [70.8, 80.2] | [37.0, 57.0] | [31.0, 51.0] | [35.0, 54.3] |
| witnessrag | [45.0, 62.0] | [54.2, 69.2] | [30.0, 50.0] | [31.0, 50.0] | [33.6, 52.3] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| dense | [-5.7, 1.8] | [-15.0, 0.0] |
| bm25 | [-0.7, 13.1] | [-4.0, 11.0] |
| graphrag | [-9.4, 6.5] | [-14.0, 4.0] |
| hipporag | [-22.6, -7.2] | [-53.0, -31.0] |
| hipporag2 | [-25.6, -7.7] | [-60.0, -38.0] |
| relational | [-5.7, 1.8] | [-15.0, 0.0] |

### Por número de hops (F1 / all-recall@5)

| método | 2 hop(s) | 4 hop(s) |
|---|---|---|
| dense | 37.1 / 59.7 (n=77) | 69.6 / 4.3 (n=23) |
| bm25 | 29.4 / 45.5 (n=77) | 62.0 / 4.3 (n=23) |
| graphrag | 37.1 / 57.1 (n=77) | 69.6 / 4.3 (n=23) |
| hipporag | 52.5 / 87.0 (n=77) | 73.9 / 65.2 (n=23) |
| hipporag2 | 57.1 / 90.9 (n=77) | 69.6 / 82.6 (n=23) |
| relational | 37.1 / 59.7 (n=77) | 69.6 / 4.3 (n=23) |
| witnessrag | 34.9 / 50.6 (n=77) | 69.6 / 4.3 (n=23) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | intersection | single-hop |
|---|---|---|---|
| relational | 11.9 (n=44) | 100.0 (n=1) | 33.7 (n=7) |
| witnessrag | 10.6 (n=44) | 0.0 (n=1) | 31.3 (n=7) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| relational | 100 | 0.0 | 0.0 |
| witnessrag | 100 | 5.0 | 16.0 |

### Evidência e resposta estrutural

Cobertura lexical de triplas dirigidas, sensível ao vocabulário. Não certifica fidelidade semântica ao texto. Só contam provas completas no contexto entregue.

| método | cobertura de triplas | cobertura = 100% | EM estrutural |
|---|---|---|---|
| relational | 0.0 | 0.0 | 0.0 |
| witnessrag | 1.0 | 0.0 | 5.0 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| relational | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) | 0.0 (100.0%) |
| witnessrag | 0.0 (9.0%) | 0.0 (9.0%) | 5.0 (100.0%) | 5.0 (100.0%) | 5.0 (100.0%) |

### Custo

Indexação compartilhada: 963.0463205212727 s · {"tokens_prompt": 602090, "em_cache": 0, "tokens_prompt_sem_cache": 602090, "filtradas": 0, "tokens_resposta_sem_cache": 326348, "chamadas": 2118, "tokens_resposta": 326348}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| dense | 100 | 86836 | 965 | 31.151 |
| bm25 | 100 | 79712 | 971 | 29.731 |
| graphrag | 100 | 113211 | 995 | 36.036 |
| hipporag | 200 | 116861 | 2698 | 75.286 |
| hipporag2 | 200 | 145265 | 7254 | 169.411 |
| relational | 200 | 142896 | 12166 | 232.574 |
| witnessrag | 416 | 204759 | 14852 | 104.341 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| dense | 0.000 | 0.000 | 0 |
| bm25 | 0.126 | 0.000 | 0 |
| graphrag | 47.893 | 0.000 | 111084 |
| hipporag | 0.006 | 0.000 | 0 |
| hipporag2 | 0.000 | 0.000 | 0 |
| relational | 0.086 | 0.000 | 0 |
| witnessrag | 0.027 | 0.000 | 0 |


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
