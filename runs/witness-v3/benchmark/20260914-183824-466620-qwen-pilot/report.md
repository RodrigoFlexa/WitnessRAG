# Resultados — 20260914-183824-466620-qwen-pilot

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
| witnessrag | 52.0 | 62.7 | 44.0 | 41.0 | 43.3 | 34.0 | 0 | 0.10 |

IC95% por bootstrap (1000 reamostragens):

| método | R@2 IC95% | R@5 IC95% | AR@5 IC95% | EM IC95% | F1 IC95% |
|---|---|---|---|---|---|
| witnessrag | [44.0, 60.2] | [55.0, 70.5] | [34.0, 54.0] | [31.0, 51.0] | [33.7, 52.6] |

### Diferenças pareadas: WITNESS-RAG menos comparador

IC95% exploratório da diferença, sem correção por múltiplas comparações.

| comparador | ΔF1 IC95% | ΔAR@5 IC95% |
|---|---|---|

### Por número de hops (F1 / all-recall@5)

| método | 2 hop(s) | 4 hop(s) |
|---|---|---|
| witnessrag | 35.4 / 55.8 (n=77) | 69.6 / 4.3 (n=23) |

### Por forma da consulta compilada (só métodos que compilam)

| método | chain | intersection | single-hop |
|---|---|---|---|
| witnessrag | 10.0 (n=44) | 0.0 (n=1) | 40.8 (n=7) |

### Resposta do executor antes do leitor

| método | n | EM estrutural | contexto com testemunha completa |
|---|---|---|---|
| witnessrag | 100 | 4.0 | 16.0 |

### Evidência e resposta estrutural

Cobertura lexical de triplas dirigidas, sensível ao vocabulário. Não certifica fidelidade semântica ao texto. Só contam provas completas no contexto entregue.

| método | cobertura de triplas | cobertura = 100% | EM estrutural |
|---|---|---|---|
| witnessrag | 0.5 | 0.0 | 4.0 |

### Seletividade da resposta estrutural (score não calibrado)

EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.

| método | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| witnessrag | 0.0 (9.0%) | 0.0 (9.0%) | 4.0 (100.0%) | 4.0 (100.0%) | 4.0 (100.0%) |

### Custo

Indexação compartilhada: 1.8062858022749424 s · {"tokens_prompt_sem_cache": 0, "em_cache": 0, "tokens_resposta_sem_cache": 0, "tokens_resposta": 0, "chamadas": 0, "filtradas": 0, "tokens_prompt": 0}

Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão contabilizados em tokens; estes números não representam custo financeiro total.

| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |
|---|---|---|---|---|
| witnessrag | 404 | 208691 | 14827 | 41.895 |

| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |
|---|---|---|---|
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
