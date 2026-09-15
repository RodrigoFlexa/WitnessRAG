# Como o WITNESS-RAG funciona hoje

Estado do método depois das mudanças de 14/09/2026. Descreve o caminho real de
uma pergunta no código, o que cada etapa garante e o que ela não garante, e os
números medidos no piloto LoCoMo (primeira conversa, 102 perguntas, Qwen2.5-14B).

Para a proposta original, ver [proposta.md](proposta.md); para as limitações
anteriores, [revisao-implementacao.md](revisao-implementacao.md); para rodar,
[locomo-pilot.md](locomo-pilot.md).

## O caminho de uma pergunta

```
                 ┌─ offline, uma vez por corpus ──────────────────┐
                 │  NER → OpenIE → grafo (fatos, entidades,       │
                 │  relações canônicas, vetores)                  │
                 └───────────────────────────────────────────────┘
pergunta
   │
   ├─1─ compilação em consulta conjuntiva        (LLM, 1 chamada)
   │       relações e entidades do grafo entram no prompt
   │
   ├─2─ aterramento dos átomos em fatos          (embeddings, 0 chamadas)
   │
   ├─3─ junção com atribuição consistente        (busca, 0 chamadas)
   │       ├── fechou  → 4
   │       └── não fechou → aquisição dirigida (LLM) e tenta de novo → 4
   │
   ├─4─ verificação das testemunhas no texto     (LLM, ≤5 chamadas)
   │       ├── alguma aprovada → 5
   │       └── nenhuma aprovada → fallback
   │
   ├─5─ conjunto de respostas certas + proveniência por item
   │       contexto = passagens das demonstrações, completado pelo fallback
   │
   └─6─ leitura final                            (LLM, 1 chamada)
```

Quando qualquer etapa falha, o contexto vem inteiro do **fallback** — hoje a
fusão recíproca de postos entre denso e BM25. O fallback é parte do método, não
um comparador: é ele que responde a maior parte das perguntas.

## As etapas

### 1. Compilação

A pergunta vira uma consulta conjuntiva positiva com uma variável de resposta
`?x`: single-hop é um átomo, cadeia liga átomos por variável intermediária,
interseção repete a mesma variável. Com `--vocab-compile`, as relações e
entidades do grafo mais próximas da pergunta entram no prompt como **sugestão**.

O que isso corrigiu: o compilador inventava predicados que nenhum fato
instanciava (`identity`, `destress method`) e o átomo morria no aterramento.
Depois da mudança, perguntas com alguma testemunha proposta subiram de 41 para
48 em 102, e `destress method` virou `destress`.

Fora do fragmento: negação, comparação, `max`/`min`. Essas caem no fallback com
o motivo registrado. `count` é executável desde que `--answer-set` esteja ligado,
porque contar exige enumerar o conjunto.

### 2. Aterramento

Cada átomo recebe fatos candidatos. Em `exact`, casamento simbólico de relação e
constantes. Em `semantic` (padrão), similaridade de relação, de argumento e da
verbalização, com limiar. Similaridade **não demonstra equivalência lógica** e
não autoriza inverter a relação: uma relação inversa precisa ser outro predicado
explícito.

A identidade de entidade casa por **cluster canônico**, não por grafia, senão
"Juan Courten" e "Juan de Courten" quebram a junção.

### 3. Junção

Caminhada por camadas com atribuição consistente das variáveis compartilhadas: a
mesma variável em dois átomos precisa receber o mesmo cluster. É o que separa o
método de um ranking por relevância — recuperar um funcionário da Atlas e outra
pessoa que pesquisa Óptica não responde a interseção.

Saem **testemunhas mínimas por inclusão**. Cortes (candidatos por átomo, feixe,
número de testemunhas) são registrados, e a completude só é declarada quando não
houve corte algum, em modo `exact`.

Quando a junção não fecha, a **lacuna** é conhecida: qual átomo, sobre qual
entidade já ligada. Isso alimenta a aquisição dirigida — uma releitura de
passagens procurando exatamente aquela relação, com os fatos novos entrando numa
`MemoryView` reversível, desfeita ao fim da pergunta para que a resposta da
pergunta 300 não dependa das 299 anteriores.

### 4. Verificação

Cada testemunha candidata vai ao LLM com a consulta, as ligações e as passagens.
Ele precisa citar um trecho **verbatim** por átomo; a citação é conferida
localmente contra o texto da passagem. Falha fechada: saída inválida, bloqueada
ou sem citação não promove prova.

Com `--answer-set`, o orçamento de verificação faz rodízio entre respostas
distintas, em vez de gastar as cinco chamadas em provas da mesma resposta.

A verificação é o freio de qualidade do executor, e ela funciona: quando a
extração degenerou, a aprovação caiu de 16,2% para 6,2% e as justificativas de
rejeição estavam certas (bindings do tipo "Melanie destressa por Caroline").
Aprovação do LLM **não é certificado lógico**.

### 5. Conjunto de respostas

Uma testemunha certifica **uma** atribuição. A resposta de uma consulta
conjuntiva é o **conjunto** das atribuições certas, cada uma com a própria
demonstração. Com `--answer-set`:

- as testemunhas são agrupadas por ligação de `?x` com símbolo canônico;
- cada item leva as suas passagens e os seus fatos;
- `count` responde o tamanho do conjunto;
- o contexto leva uma demonstração de cada resposta antes de provas extras da
  mesma, senão a melhor resposta consome as `k` vagas sozinha.

**A completude do conjunto não é certificada.** Ele contém o que a memória prova;
nada limita o que ficou de fora por falha de extração, de compilação ou de corte.
`count` herda essa limitação — é uma contagem sobre o provado, não sobre o mundo.

### 6. Leitura

O mesmo leitor para todos os métodos, com as mesmas `k` passagens e o mesmo
formato: é o que faz a coluna de F1 medir recuperação e não engenharia de prompt.
Com `--answer-set`, a variante ciente de conjunto pede todos os itens
**sustentados pelas passagens** quando a pergunta pede um conjunto. A regra muda
para todos os métodos da rodada, então a comparação entre métodos continua
válida; a comparação com rodadas antigas, não.

O método também produz uma **resposta estrutural** (do executor, sem o leitor),
relatada em separado. A coluna principal continua sendo a do leitor comum.

## Extração

A base de fatos `F` é compartilhada por GraphRAG, HippoRAG, HippoRAG 2 e
WITNESS-RAG: um extrator por método tornaria qualquer diferença entre métodos
inseparável da diferença entre extratores.

`--dialogue-ie` troca o prompt por um adaptado a transcrições: o falante vira o
sujeito das falas em primeira pessoa, a correferência é resolvida dentro do
bloco, parentescos são nomeados pelo dono, e o fato ganha um escopo temporal
(data da sessão ou a data explícita no texto). **Só faz sentido em corpus
conversacional.**

A qualidade do vocabulário extraído é agora medida, porque foi por ali que uma
regressão passou despercebida:

| | prompt geral | diálogo v1 (quebrado) | diálogo v2 |
|---|---|---|---|
| fatos | 1257 | 1093 | 1023 |
| relações distintas | 657 | 949 | 431 |
| relações que aparecem uma vez | 494 | 882 | 286 |
| palavras por relação | 2,02 | 4,83 | 1,51 |
| objetos distintos | 898 | 390 | 692 |

Na v1 o prompt deixava a fala inteira virar predicado e o interlocutor virar
objeto — `("Melanie", "said running is a great way to destress", "Caroline")`.
Com 81% das relações aparecendo uma única vez, nenhum átomo alcança fatos
suficientes e a variável compartilhada nunca casa. `relacoes_por_fato` perto de 1
é o sinal barato desse colapso, e agora sai em `summary.json`.

## Onde o método está

Piloto LoCoMo, primeira conversa, 102 perguntas, F1 do avaliador oficial, IC95%
por bootstrap pareado contra a linha de base:

| | baseline | v2 | Δ | IC95% |
|---|---|---|---|---|
| total (n=102) | 46,60 | **58,09** | +11,49 | [+5,5, +17,6] |
| single-hop (n=70) | 54,00 | **63,91** | +9,91 | [+2,6, +18,0] |
| multi-hop (n=32) | 30,41 | **45,35** | +14,94 | [+6,7, +24,1] |

De onde veio o ganho, em ordem de tamanho: contexto (fallback híbrido e `k`
maior levaram o `AR@k` entregue no multi-hop de 15,6% para 59,4%), leitor ciente
de conjunto (itens certos por pergunta de 40% para 58%, com a precisão por item
parada em 46% — são itens que faltavam, não enchimento) e, por último, o
executor.

**O executor ainda responde pouco: dispara em 15,7% das perguntas.** As outras
84% são o fallback. Onde ele dispara no multi-hop, o F1 oficial é 67,1 contra
38,1 do fallback; no single-hop é 41,5 contra 66,8 — o conjunto dilui respostas
de frase única. Os dois números vêm de 8 perguntas cada e têm seleção: são
indício, não medida.

## O que não está resolvido

- **Taxa de disparo.** 15,7% é o número que precisa subir antes de qualquer
  outra coisa. Enquanto ele for baixo, a tabela principal mede o fallback.
- **Completude do conjunto**, e portanto a contagem, sem limite de erro.
- **Scores não calibrados.** `risco` e `score` não são probabilidades, e a
  testemunha de menor limite é escolhida depois de olhar os dados.
- **Aquisição** continua uma heurística de similaridade menos custo, agora com o
  ganho normalizado pelo topo da consulta. Não é estimativa de valor da
  informação.
- **Uma conversa, uma semente.** Os IC95% acima têm largura de 6 a 9 pontos.
  Nada aqui se generaliza sem os outros corpora e mais sementes.
