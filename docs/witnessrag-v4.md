# WitnessRAG v4: tipos, conjuntos item a item, prova como falas e premissas

Estado de 25/09/2026. O v4 estende o controlador de prova (v3,
`--proof-controller`) com quatro opções independentes e uma opção do leitor.
**Todas ficam desligadas por padrão**: uma rodada sem elas envia exatamente os
mesmos prompts e monta exatamente os mesmos contextos do v3 (há teste para
isso em `tests/test_v4.py`).

| opção | o que muda | ataca |
|---|---|---|
| `--typed-variables` | o plano declara o tipo pedido pela pergunta: `?x : "martial art"` | multi-hop (e single-hop) com `respostas_demais` |
| `--item-set-proofs` | planos de conjunto são provados item a item | multi-hop de agregação |
| `--witness-delivery excerpts` | a prova vai ao leitor como as falas de origem, sem trocar trechos | gargalo de k_W; risco de trocar trechos bons |
| `--abductive-premises` | perguntas "Would X...?" recebem premissas da vizinhança de X no grafo | open-domain |
| `--yesno-rationale` (leitor) | sim/não com a justificativa curta | open-domain (forma da resposta) |
| `--proof-edit-fraction f` | k_W proporcional a k | varredura com orçamento fixo |

Perfis prontos em `scripts/proof-profiles.sh`: `proof` (v3), `proof-v4` (as
quatro opções do controlador), `v4-typed`, `v4-excerpts`, `v4-abductive`,
`v4-no-types`.

---

## 1. Variáveis tipadas (átomos unários)

**Problema medido.** Na rodada v3 do Azure, 58% das multi-hop e 57% das
single-hop pararam em `respostas_demais`. Exemplo: "What martial arts has John
done?" virava `does(John, ?x)` e casava com tudo o que John faz. O tipo que a
pergunta pedia ("martial arts") era descartado.

**O que o plano ganha.** Um campo opcional `types`:

```json
{"answer_var": "x",
 "atoms": [{"relation": "practice", "subject": "John", "object": "?x"}],
 "aggregation": "set", "types": {"x": "martial art"}}
```

Formalmente, $Q(?x) \leftarrow \textsf{practice}(\text{John}, ?x) \wedge \textsf{Type}_{\text{martial art}}(?x)$.
A consulta continua conjuntiva; o átomo unário não muda a junção.

**Como o tipo é conferido.** Medimos a similaridade de cosseno do BGE-M3 entre
respostas e tipos, e ela **não separa** o que é do tipo do que não é:
"kickboxing"/"martial art" = 0,69 contra "basketball"/"martial art" = 0,65;
"Luna the cat"/"pet" = 0,56 contra "family"/"pet" = 0,61. Um limiar de
similaridade seria arbitrário. Por isso:

1. a afinidade com o tipo só **ordena** os itens (suporte × afinidade);
2. quem decide é o **Verificar**, que já existia e já tinha o motivo
   `wrong_type`: o plano passado a ele agora diz "?x must be a martial art", e
   ele julga item a item com a fala de origem de cada fato.

No diagnóstico, um item aceito assim tem o tipo "verificado", não "provado":
o átomo unário é conferido por julgamento sobre a fonte, como a verificação v3.

## 2. Conjuntos provados item a item

**Problema.** O teste de seletividade v3 exige no máximo 3 respostas e 3
testemunhas. Para uma pergunta de conjunto ("What hobbies does Evan pursue?",
7 itens no ouro), muitas respostas é o resultado certo, não um plano ruim.

**Regra v4 (só para planos de conjunto).**

$$y = 1 \iff A^\star \neq \emptyset,\qquad A^\star = \{a : \exists W_a,\ \text{casamento}(W_a)\cdot\text{conf}(W_a) \ge 0{,}45\}$$

- com tipo: os `proof_set_max_items` (12) melhores itens por suporte ×
  afinidade vão ao Verificar, que decide item a item;
- sem tipo e com mais de 12 itens: o plano não é seletivo, vale a recusa v3;
- cortes da busca (candidatos, feixe) não recusam a prova: eles limitam a
  completude, e uma prova de conjunto nunca declara completude.

Planos de um valor mantêm exatamente as regras v3.

## 3. A prova entregue como falas de origem

**Problema.** Com `pages`, a prova entra trocando até k_W = 2 trechos de 2048
tokens da cauda. Sete itens em sete sessões não cabem, e cada troca tira um
trecho que a busca julgou relevante.

**v4 (`--witness-delivery excerpts`).** Os k trechos da busca ficam intactos.
A prova **verificada** entra como um bloco curto no fim do contexto, perto da
pergunta:

```
[6] Proved evidence: facts found in the memory, with their source turns
- answer: kickboxing
  fact: John | practice | kickboxing (event date: 8 May 2023)
    (session date: 8 May 2023)
    [D1:3] John: I started kickboxing classes last month...
```

- teto de `excerpt_max_chars` = 2400 caracteres (~600 tokens, ~6% do
  orçamento de 10 mil tokens do leitor);
- só provas verificadas geram texto. Uma prova de valor que já estava nos
  trechos não é verificada (nada mudaria) e, por coerência, não gera bloco;
- para conjuntos, a verificação acontece sempre que haverá bloco.

Garantia: o contexto de trechos é **idêntico** ao da busca; um erro de prova
custa no máximo o bloco, nunca um trecho.

## 4. Premissas abdutivas (open-domain)

**Problema.** "Would Caroline likely have Dr. Seuss books on her bookshelf?"
não tem testemunha: a resposta não decorre de fatos armazenados, e sim de
fatos + conhecimento de mundo, $S \cup K \models H$. A similaridade com a
pergunta também falha ("Dr. Seuss" contra "classic children's books").

**v4 (`--abductive-premises`).** Para perguntas de hipótese, o planejador
escreve, na mesma chamada de planejamento:

```json
"hypothesis": {"about": "Caroline",
               "concepts": ["children's books", "book collecting", "reading habits",
                            "Caroline's library"]}
```

Os conceitos nomeiam fatos que **apoiariam ou contradiriam** a hipótese. O
sistema pega a vizinhança $N(e)$ da pessoa no grafo (fatos em que ela é sujeito
ou objeto), ordena por $0{,}7\cdot\max_c \cos(f, c) + 0{,}3\cdot\cos(f, q)$ e
entrega as até 8 melhores falas de origem, com data, num bloco "Statements
about Caroline that may bear on the question". É suporte, não prova: não troca
trechos e não chama o modelo.

## 5. Leitor: sim/não com justificativa (`--yesno-rationale`)

O ouro do open-domain mistura veredito e razão ("Likely no; though she likes
reading, she wants to be a counselor"), e o leitor v3 devolve só "likely no".
Com a opção, o leitor de evidências responde "likely no; she wants to be a
counselor" (até 15 palavras) e a canonicalização de sim/não é desligada.

**Integridade.** Isso muda a forma da resposta para se aproximar da métrica,
não a evidência. É legítimo apenas porque (i) é decidido pelo texto da
pergunta, nunca pela categoria, e (ii) vale para **todos os métodos**, inclusive
o híbrido. Reporte sempre as duas versões, com e sem a opção.

## 6. Varredura de tamanho de trecho

`scripts/run-locomo-chunk-sweep.sh`, com LLM = openai | azure | qwen:

| regime | k | k_W da prova | pool |
|---|---|---|---|
| `k5` | 5 | 2 / 1 (v3) | 20 |
| `budget` | round(10240 / s): 10, 20, 40 | `--proof-edit-fraction 0.4`: 4/2, 8/4, 16/8 | 2k |

O ponto 5 × 2048 é o mesmo nos dois regimes e roda uma vez. O relatório
(`budget-report.py`) usa o híbrido 5 × 2048 como referência, com ΔF1 pareado e
IC 95% por bootstrap de conversas.

**Ressalva a declarar.** As janelas da OpenIE são cortadas dentro de cada
trecho, então mudar o tamanho do trecho também reextrai os fatos. As linhas do
controlador de prova medem o sistema inteiro naquele tamanho; as linhas do
híbrido não usam o grafo e medem só a recuperação.

## 7. Como rodar

```bash
# testes (sem servidor)
python -m pytest -q tests/test_v4.py

# OpenAI (chave no .env), três conversas, v3 contra v4 e híbrido
LOCOMO_CONVERSATION=0,1,2 METHOD=proof PROFILE=proof    bash scripts/run-witness-openai-locomo.sh runs/t-proof
LOCOMO_CONVERSATION=0,1,2 METHOD=proof PROFILE=proof-v4 bash scripts/run-witness-openai-locomo.sh runs/t-v4
LOCOMO_CONVERSATION=0,1,2 METHOD=hybrid                  bash scripts/run-witness-openai-locomo.sh runs/t-hybrid
python scripts/budget-report.py --output runs/t-report \
  hybrid=runs/t-hybrid proof=runs/t-proof proof-v4=runs/t-v4

# Windows
powershell -ExecutionPolicy Bypass -File scripts\run-witness-openai-locomo.ps1 -Method proof -Profile proof-v4 -Conversation 0,1,2

# Azure / Qwen: os mesmos perfis
PROFILE=proof-v4 bash scripts/run-witness-proof-locomo.sh

# varredura (padrão: hybrid, proof e proof-v4; 2048/1024/512/256; k5 e budget)
LLM=openai CONVERSATIONS=0,1,2 bash scripts/run-locomo-chunk-sweep.sh
```

Com a chave da OpenAI, o limite de tokens por minuto (200 mil no gpt-4o-mini)
é atingido com três rodadas em paralelo; rode no máximo duas ao mesmo tempo ou
aumente `WRAG_AZURE_MAX_RETRIES`.

## 8. Mapa conceito ↔ código

| conceito | onde |
|---|---|
| tipos no plano | `wrag/witness/query.py::_attach_types`, `ConjunctiveQuery.types` |
| hipótese no plano | `wrag/witness/plan.py::parse_hypothesis`, `ProofPlan.hypothesis` |
| extensões do prompt | `wrag/prompts.py::PLAN_TYPES_EXTENSION`, `PLAN_HYPOTHESIS_EXTENSION` |
| prova de conjunto item a item | `wrag/methods/witnessrag.py::_prove_items` |
| k_W proporcional | `WitnessRAGRetriever._edit_limit` |
| bloco de falas da prova | `WitnessRAGRetriever._proof_excerpts` |
| premissas abdutivas | `WitnessRAGRetriever._abductive_premises` |
| blocos no leitor | `wrag/eval/reader.py::read(extra_passages=...)` |
| sim/não com razão | `wrag/prompts.py::qa_evidence_template` |
| perfis | `scripts/proof-profiles.sh` |
| varredura | `scripts/run-locomo-chunk-sweep.sh` |

## 9. Resultados do piloto (gpt-4o-mini, 25/09/2026)

Conversas 0, 1 e 2 do LoCoMo (385 perguntas), 5 × 2048, mesmo leitor. A
conversa 0 é a de desenvolvimento. F1 oficial; IC 95% por bootstrap de
conversas (com 3 conversas, os intervalos são largos).

| rodada | F1 | ΔF1 vs híbrido | single | multi | temporal | open |
|---|---:|---|---:|---:|---:|---:|
| híbrido | 59,40 | — | 61,96 | 49,53 | 67,79 | 33,74 |
| prova v3 | 59,76 | +0,37 [−0,96; +1,76] | 63,48 | 49,66 | 65,87 | 33,74 |
| prova v4 | 60,28 | +0,88 [−1,96; +2,75] | 62,77 | 51,78 | 66,98 | 37,75 |

Primeira versão da entrega "excerpts" (com linhas "answer:" e "fact:") piorou
3 pontos na conversa 0: o leitor copiava a resposta extraída. Daí o formato
atual (só diálogo) e o modo "mixed". Varredura parcial (híbrido, conversas 1-2):
5 × 512 contra 5 × 2048 = −1,88 F1 [−2,48; −1,57], R@5 88,6 → 78,5.
