# Resultados — 20260910-230140-529391-review-offline

## Configuração

- **LLM**: `stub` / deployment `—`
- **Embeddings**: `tfidf` (`tfidf-v2-bc2839d221bdd0d0`)
- **top-k**: 5 · **perguntas por dataset**: 1 · **seed**: 42
- **Feixe da junção**: 0 · **rodadas de aquisição**: 2 · **orçamento de memória**: 1.0
- **Bloqueios por filtro de conteúdo**: 0 {}

> **Atenção**: a rodada usou o embedder TF-IDF de fallback. Ele existe para verificar o encanamento offline e **não** é um retriever de artigo. Nenhum número desta rodada deve ser comparado com valores publicados.


## sample

1 perguntas · 3 passagens · 1.0 hops em média · 0 pergunta(s) excluída(s) por filtro de conteúdo em algum método

Corpus reduzido: False · embedding ajustado: `tfidf-v2-812b9be6a30ae2af`.

Comparação pareada: 1 perguntas presentes em todos os métodos.

### Tabela principal (mesmo denominador para todos)

| método | R@2 | R@5 | AR@5 | EM | F1 | abst. | filtradas | lat. (s) |
|---|---|---|---|---|---|---|---|---|
| dense | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| bm25 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| graphrag | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.01 |
| hipporag | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| hipporag2 | 0.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| relational | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| witnessrag | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |
| witnessrag-annotated | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0 | 0.00 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| dense | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| bm25 | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| graphrag | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| hipporag | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| hipporag2 | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| relational | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| witnessrag | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |
| witnessrag-annotated | [—, —] | [—, —] | [—, —] | [—, —] | [—, —] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|
| dense | [—, —] | [—, —] |
| bm25 | [—, —] | [—, —] |
| graphrag | [—, —] | [—, —] |
| hipporag | [—, —] | [—, —] |
| hipporag2 | [—, —] | [—, —] |
| relational | [—, —] | [—, —] |
| witnessrag-annotated | [—, —] | [—, —] |

### Por número de hops (F1 / all-recall@5)

| método | 1 hop(s) |
|---|---|
| dense | 0.0 / 100.0 (n=1) |
| bm25 | 0.0 / 100.0 (n=1) |
| graphrag | 0.0 / 100.0 (n=1) |
| hipporag | 0.0 / 100.0 (n=1) |
| hipporag2 | 0.0 / 100.0 (n=1) |
| relational | 0.0 / 100.0 (n=1) |
| witnessrag | 0.0 / 100.0 (n=1) |
| witnessrag-annotated | 0.0 / 100.0 (n=1) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain |
|---|---|
| relational | 0.0 (n=1) |
| witnessrag | 0.0 (n=1) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| relational | 1 | 0.0 | 0.0 |
| witnessrag | 1 | 0.0 | 0.0 |
| witnessrag-annotated | 1 | 0.0 | 0.0 |

### Custo

Indexação compartilhada: 0.025390800001332536 s · {"tokens_resposta_sem_cache": 460, "tokens_resposta": 460, "filtradas": 0, "chamadas": 6, "em_cache": 0, "tokens_prompt_sem_cache": 744, "tokens_prompt": 744}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| dense | 1 | 394 | 17 | 0.002 |
| bm25 | 1 | 394 | 14 | 0.0 |
| graphrag | 1 | 394 | 17 | 0.012 |
| hipporag | 2 | 408 | 30 | 0.004 |
| hipporag2 | 2 | 666 | 90 | 0.003 |
| relational | 2 | 408 | 95 | 0.001 |
| witnessrag | 2 | 408 | 95 | 0.001 |
| witnessrag-annotated | 1 | 394 | 17 | 0.001 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
| dense | 0.000 | 0.000 | 0 |
| bm25 | 0.002 | 0.000 | 0 |
| graphrag | 0.554 | 0.000 | 444 |
| hipporag | 0.000 | 0.000 | 0 |
| hipporag2 | 0.000 | 0.000 | 0 |
| relational | 0.001 | 0.000 | 0 |
| witnessrag | 0.000 | 0.000 | 0 |
| witnessrag-annotated | 0.000 | 0.000 | 0 |


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
