# Tipos por GLiNER no controlador de prova (25/09/2026)

## O que foi feito

`--type-model gliner` (exige `--typed-variables`; desligado por padrão). Depois
da junção, cada resposta candidata de um plano tipado (`?x : "martial art"`)
recebe uma nota de tipo: o GLiNER (`urchade/gliner_medium-v2.1`) roda na fala
de origem com o tipo pedido e quatro contrastes (person, location,
organization, date), multi-rótulo; a nota é a do rótulo pedido no trecho que
contém a resposta (resposta não localizada = sem nota).

- `--type-mode rank` (padrão): nada é removido. Se alguma candidata recebe o
  tipo com nota >= 0,5 (tipo "nomeável" nesta memória), as candidatas são
  reordenadas por suporte x nota (0,5 sem nota). Senão, a ordem fica igual à
  do v4. Nos conjuntos item a item, a nota substitui o cosseno resposta x tipo.
- `--type-mode veto` (ablação): remove as localizadas com nota < `--type-min-score`.

Código: `wrag/witness/typing.py`; `WitnessRAGRetriever._type_filter`,
`_answer_kind`, `_candidate_turns`, `_log_candidates` (log offline via
`WRAG_TYPE_LOG`); flags em `wrag/pilot.py`, campos em `WitnessConfig`.

Rodar: `METHOD=proof PROFILE=proof-v4 EXTRA_FLAGS="--type-model gliner" bash scripts/run-witness-openai-locomo.sh`
(requer `pip install gliner`).

## Teste offline (conversas 1-3, 2400 candidatas de 211 perguntas tipadas)

Rótulo aproximado: candidata "certa" se compartilha palavra com o ouro.

| sinal | AUC certa > errada |
|---|---:|
| cosseno bge-m3 (resposta x tipo) | 0,612 |
| GLiNER na fala de origem | 0,735 |

Como veto (0,3): corta 85% das erradas, mas mantém só 114/222 certas;
perguntas com alguma certa caem de 124 para 76. Tipos abstratos ("inspiration",
"plan", "happy memory") e respostas-frase não são rotulados por NER.

## LoCoMo completo, gpt-4o-mini, proof-v4 com e sem `--type-model gliner` (ordenação)

1538 perguntas pareadas, 10 conversas, 5 x 2048, mesmo leitor, cache de LLM
compartilhado. F1 oficial; IC 95% por bootstrap de conversas.

| categoria | n | sem | com | delta | IC 95% |
|---|---:|---:|---:|---:|---|
| todas | 1538 | 58,71 | 58,80 | +0,09 | [-0,08; +0,27] |
| single-hop | 841 | 67,90 | 67,72 | -0,18 | [-0,35; -0,06] |
| multi-hop | 280 | 44,44 | 45,44 | +0,99 | [-0,07; +2,15] |
| temporal | 321 | 58,15 | 58,32 | +0,17 | [-0,11; +0,59] |
| open-domain | 96 | 21,64 | 21,17 | -0,47 | [-3,23; +2,39] |

O tipo ordenou em 399 perguntas; contexto diferente em 144, resposta diferente
em 77; F1 melhorou em 21 e piorou em 25.

Leitura: efeito nulo no total. Ordenar não ataca o gargalo medido (paradas
por "respostas demais", que continuam iguais, porque o teste de seletividade
conta as respostas antes de qualquer ordem). O ganho possível em multi-hop é
pequeno e dentro do ruído; single-hop perde pouco, mas de forma consistente.
