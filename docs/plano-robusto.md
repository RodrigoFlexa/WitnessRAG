# Plano robusto e seleção de fatos (27/09/2026)

> Revisão posterior de 27/09: os resultados abaixo pertencem ao código anterior
> às correções de verificação e seleção descritas em
> [revisao-plano-robusto.md](revisao-plano-robusto.md). O contador de provas por
> outra leitura foi corrigido de 7 para 3: quatro leituras finais não tinham
> prova aceita. Os prompts e as predições históricas foram preservados; não há
> ainda uma nova medição de F1 do código corrigido. `robust-no-plan` agora é
> também um perfil executável nos scripts Windows e Bash.

Cinco mudanças no WitnessRAG, todas desligadas por padrão. Sem as opções, os
prompts e os contextos eram os mesmos do perfil `witnessrag` na implementação
original destas opções (há teste para
isso em `tests/test_robust_plan.py`).

| opção | o que muda | parte do método |
|---|---|---|
| `--relation-alternatives` | cada átomo pode ter até 3 frases de relação alternativas | plano / Executor |
| `--plan-readings N` | o planejador escreve até N leituras alternativas da pergunta na mesma chamada | plano / Executor / Refletor |
| `--fact-fill question` | os fatos que completam o orçamento vêm só da pergunta, não dos átomos | entrega |
| `--fact-rerank [modelo]` | cross-encoder (bge-reranker-v2-m3) no lugar da similaridade na seleção | entrega |
| `--fact-time both` | a data do fato como o falante disse, ancorada na sessão, e na precisão dita | entrega / leitor |

Perfis (`scripts/proof-profiles.sh` e `run-witness-openai-locomo.ps1`):
`witnessrag-robust` (tudo), `wr-no-rerank` (tudo menos o reranker),
`wr-plan` (alternativas + leituras), `wr-alt`, `wr-readings`, `wr-fill`,
`wr-rerank`, `wr-time`.

## Por que

A ablação de 26/09 mostrou duas coisas:

1. **A prova é o componente que rende.** "Sem prova" foi a única ablação com
   intervalo fora do zero (−1,0 F1), e o ganho estava em multi-hop e temporal.
2. **"Sem prova" ficou abaixo de "sem plano".** Os átomos do plano, usados só
   como consultas de similaridade, pioram a seleção dos fatos (−2 em
   multi-hop). Ou seja: o plano ajuda quando prova, e atrapalha quando só
   orienta a similaridade.

Daí dois alvos: fazer o plano **provar mais vezes** (itens 1 e 2) e impedir que
ele **atrapalhe onde não prova** (item 3). Os itens 4 e 5 atacam o que o leitor
recebe.

## 1. Átomos disjuntivos (relações alternativas)

O erro de plano mais comum é de vocabulário: a pergunta diz "live in", a memória
diz "move to". Antes, isso custava um ciclo inteiro (a junção quebra, o
Refletor explica a lacuna, o Planejador reescreve). Agora o átomo carrega as
frases alternativas:

```json
{"relation": "live in", "alternatives": ["move to", "reside in"],
 "subject": "?y", "object": "?x"}
```

Formalmente, o átomo $r_i(t_i, t'_i)$ vira a disjunção
$\bigvee_{j} r_{ij}(t_i, t'_i)$, e a consulta passa de conjuntiva (CQ) a uma
**união de consultas conjuntivas** (UCQ) que compartilha uma única junção:

$$Q(?x) \leftarrow \bigwedge_{i=1}^{n} \Big(\bigvee_{j=0}^{m_i} r_{ij}\Big)(t_i, t'_i),\qquad m_i \le 3 .$$

Continua sendo lógica de primeira ordem positiva, e a testemunha continua sendo
um fato por átomo. Na execução, a similaridade de relação do átomo com um fato
é a melhor entre as frases:

$$\operatorname{sim}_R(a, f) = \max_j \operatorname{sim}_R(r_{aj}, r_f),$$

(1 se alguma frase for da mesma família do verbo do fato). O resto da nota de
casamento, os vetos e a caminhada não mudam. Uma frase alternativa não pode
conter argumentos nem variáveis, e alternativas escritas com a opção desligada
são descartadas antes da junção.

## 2. Outras leituras da pergunta

Uma pergunta pode estar guardada de mais de um jeito: "Andrew and his
girlfriend" como sujeito, ou só "Andrew" (quem fala); "a mentora de Caroline"
como cadeia, ou como o nome já resolvido; dar ou receber. O planejador agora
escreve, **na mesma chamada**, a leitura literal e até N outras:

```json
{"atoms": [{"relation": "go to", "alternatives": ["visit"],
            "subject": "Kai and his wife", "object": "?x"}],
 "aggregation": "set", ...,
 "other_readings": [{"answer_var": "x", "aggregation": "set",
   "atoms": [{"relation": "go to", "alternatives": ["visit"],
              "subject": "Kai", "object": "?x"}]}]}
```

Cada leitura herda o período e os pesos do plano e troca só a consulta
(átomos, agregação, tipos). A regra de uso é conservadora:

- a leitura principal vale sempre que prova;
- se ela não prova (junção incompleta, respostas demais, suporte baixo), o
  Executor roda as outras leituras **sem nova chamada ao LLM** e fica com a de
  maior suporte entre as utilizáveis;
- se a prova da principal é **recusada pela verificação**, as outras leituras
  são executadas e a melhor é verificada, antes de qualquer replanejamento;
- se nada prova, o motivo de cada leitura vai ao Planejador do próximo ciclo.

Formalmente, o plano é um conjunto ordenado de UCQs
$\mathcal{Q} = (Q^{(0)}, Q^{(1)}, \dots, Q^{(N)})$ e a resposta é a testemunha
de maior suporte da primeira leitura que prova (a principal tem precedência).
Uma leitura inválida é descartada sem invalidar o plano; se a principal for
inválida e outra não, a primeira válida é promovida.

**Por que os exemplos mudaram.** Na primeira versão os campos eram opcionais e
o gpt-4o-mini não escreveu nenhuma alternativa nem leitura: o planejador pequeno
copia os exemplos e ignora instruções. No modo robusto os dois campos são
obrigatórios e **todos** os exemplos do prompt os mostram
(`prompts._robust_examples`, nomes sintéticos).

## 3. Completar o orçamento só pela pergunta

A entrega por fatos escolhe, nesta ordem: fatos da prova verificada, fatos das
melhores respostas do plano, e o resto por relevância. No v4 essa relevância era
o máximo da similaridade com a pergunta **e com cada átomo**. Com
`--fact-fill question`, o resto vem só da pergunta. A prova e as respostas do
plano continuam com prioridade.

## 4. Reranker

Com `--fact-rerank`, os 120 primeiros candidatos (`--fact-rerank-pool`) são
reordenados por um cross-encoder (BAAI/bge-reranker-v2-m3) que lê a pergunta
junto com a frase do fato e a fala de origem. A prioridade da prova e das
respostas do plano é mantida; o reranker só substitui a similaridade. Não vê
resposta nem rótulo. As notas ficam em cache (`<cache>/rerank/`). Custo: cerca
de 13 s por pergunta em 2 núcleos de CPU; em GPU é desprezível.

## 5. Datas bitemporais

Na entrega por fatos, o leitor recebia só o intervalo resolvido
(`event: 1 August 2023 - 31 August 2023`) e o copiava na resposta, quando o
diálogo dizia "in August" ou "last weekend". Com `--fact-time both`, o fato que
foi datado por uma expressão mostra:

- o tempo **como o falante disse, ancorado no dia da sessão**:
  "last weekend" dito em 24/10/2023 vira `the weekend before 24 October 2023`
  (`timeline.anchored_phrase`: last X, X ago, yesterday, next X, in N X, this X);
- as datas resolvidas **na precisão da expressão**: um mês inteiro é
  `August 2023`, um ano inteiro é `2023` (`timeline.natural_interval`).

Exemplo: `- Andrew hiked with his girlfriend. (event: the weekend before 24 August 2023; 19 August 2023 - 20 August 2023)`.

O leitor de fatos recebe a regra de responder na precisão da nota.
**Integridade:** isto não traz informação nova. O leitor por trechos já via
a data da sessão e o "last weekend" na própria fala; aqui a nota volta a ter o
que a resolução tinha jogado fora. Mas muda a forma da resposta, e o F1 do
LoCoMo é sensível a ela. Por isso é uma opção separada e deve ser reportada com e
sem, como o `--yesno-rationale`.

## Resultados: conv05 (conv-44, 123 perguntas)

A conv05 é a conversa com mais evidência faltando no contexto (39 perguntas com
R@5 < 1). gpt-4o-mini; memória em proposições (`--ie-style memory`); trechos de
2048 tokens, k = 5; mesmo leitor; cache de LLM compartilhado, então onde o
contexto não muda a resposta é idêntica. Referência: o WitnessRAG completo
(`witnessrag`, o mesmo da ablação). IC 95% por bootstrap de **perguntas** (uma
conversa só).

| variante | n | F1 | Δ F1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| witnessrag | 123 | 51.50 |  |  | 64.8 | 41.9 | 43.0 | 4.1 | 1,816 | 8,150 | 1.72 | 35.0% | 0 |
| wr-plan | 123 | 53.65 | +2.15 [-1.00; +5.86] | 13/12 | 66.3 | 44.8 | 45.7 | 7.1 | 1,818 | 9,268 | 1.59 | 46.3% | 3 |
| wr-fill | 123 | 54.98 | +3.48 [-0.55; +8.03] | 15/9 | 69.8 | 43.8 | 41.9 | 16.3 | 1,817 | 8,152 | 1.72 | 35.0% | 0 |
| wr-time | 123 | 53.99 | +2.49 [+0.49; +4.83] | 15/5 | 65.6 | 43.0 | 52.4 | 4.1 | 1,935 | 8,270 | 1.72 | 35.0% | 0 |
| wr-rerank | 123 | 58.68 | +7.18 [+3.16; +11.57] | 23/6 | 69.7 | 56.6 | 45.6 | 14.3 | 1,879 | 8,213 | 1.72 | 35.0% | 0 |
| wr-no-rerank | 123 | 58.32 | +6.82 [+2.13; +11.68] | 25/12 | 72.8 | 44.3 | 50.7 | 15.8 | 1,942 | 9,391 | 1.59 | 46.3% | 3 |
| witnessrag-robust | 123 | 62.96 | +11.46 [+6.62; +16.62] | 36/11 | 73.4 | 58.6 | 54.2 | 19.7 | 2,016 | 9,465 | 1.59 | 46.3% | 3 |
| robust-no-plan | 123 | 61.63 | +10.13 [+5.28; +15.26] | 31/11 | 74.5 | 54.3 | 50.0 | 19.0 | 2,006 | 2,006 | 0.0 | 0.0% | 0 |
| hybrid-chunks | 123 | 52.93 | +1.43 [-5.07; +7.98] | 38/33 | 71.3 | 42.4 | 34.0 | 0.0 | 10,263 | 10,263 | 0.0 | 0.0% | 0 |

`robust-no-plan` = o perfil `abl-no-plan` com `--fact-fill question --fact-rerank
--fact-time both`: a mesma entrega do `witnessrag-robust`, sem Planejador,
Executor e Refletor. `hybrid-chunks` = busca híbrida com 5 trechos de 2048
tokens e o mesmo leitor de evidências (sem grafo).

Pareado contra os trechos: `witnessrag-robust` − `hybrid-chunks` = **+10,02 F1
[+3,71; +16,38]**, com 2,0 mil tokens no leitor em vez de 10,3 mil.

### Leitura

- **Tudo junto: +11,5 F1** (51,50 → 62,96), 36 perguntas melhores e 11 piores.
  Multi-hop 41,9 → 58,6; temporal 43,0 → 54,2; single-hop 64,8 → 73,4.
- **O reranker é a peça que mais rende sozinha** (+7,2), sobretudo em multi-hop
  (+14,7): perguntas de conjunto cujos itens estão espalhados, e que a
  similaridade de embedding não traz todos para dentro dos 40 fatos.
- **O plano robusto faz o que promete no mecanismo**: perguntas com prova
  aceita sobem de 35% para 46%, os planos por pergunta caem de 1,72 para 1,59,
  e 3 provas aceitas vieram de uma leitura alternativa (2 verificadas e 1 com
  verificação dispensada). Sozinho, +2,2 F1 (IC cruza o zero).
- **As datas ancoradas ajudam o temporal** (+9,4 na categoria). A primeira
  versão (só o "said as") não mudava nada: o leitor copiava o intervalo
  resolvido. O que mudou o resultado foi apresentar o mês como mês e o tempo
  relativo ancorado na sessão.
- **Quanto o plano contribui dentro do robusto: +1,3 F1** (robusto contra
  robust-no-plan, IC [−1,8; +4,4]), concentrado em multi-hop (+4,3) e temporal
  (+4,2), com −1,1 em single-hop, ao custo de 7,5 mil tokens a mais por pergunta.
  A maior parte do ganho de hoje vem da entrega (reranker, datas,
  preenchimento), que também melhora um sistema sem plano. É o mesmo padrão da
  ablação de 26/09: o plano rende onde há composição e tempo, e o LoCoMo tem
  poucas perguntas assim.

## Ressalvas

- **Uma conversa.** 123 perguntas, e a conv05 já tinha sido usada para testar
  hipóteses em 26/09: aqui ela é conversa de desenvolvimento, não de teste. O
  número que vale para o artigo é o das 10 conversas.
- **Datas bitemporais mudam a forma da resposta.** Reportar com e sem.
- **Tokens.** As leituras e alternativas aumentam a saída do planejador; os
  replanejamentos caem. O reranker não gasta tokens de LLM, só CPU/GPU.

## Como rodar

```bash
# testes (sem servidor)
python -m pytest -q tests/test_robust_plan.py

# uma conversa, referência e variantes (Linux / servidor)
LOCOMO_CONVERSATION=5 METHOD=proof PROFILE=witnessrag bash scripts/run-witness-openai-locomo.sh runs/conv05/witnessrag
LOCOMO_CONVERSATION=5 METHOD=proof PROFILE=witnessrag-robust bash scripts/run-witness-openai-locomo.sh runs/conv05/witnessrag-robust
python scripts/paired-report.py runs/conv05 witnessrag wr-plan wr-fill wr-time wr-rerank wr-no-rerank witnessrag-robust

# Windows
powershell -ExecutionPolicy Bypass -File scripts\run-witness-openai-locomo.ps1 -Method proof -Profile witnessrag-robust -Conversation 5

# Qwen no servidor: os mesmos perfis
LLM=qwen PROFILE=witnessrag-robust bash scripts/run-witness-proof-locomo.sh
```

## Mapa conceito ↔ código

| conceito | onde |
|---|---|
| átomo disjuntivo | `wrag/witness/query.py::Atom.alternatives`, `_parse_atoms` |
| similaridade de relação disjuntiva | `wrag/witness/search.py::_encode_relations`, `_relation_similarity` |
| leituras no plano | `wrag/witness/plan.py::ProofPlan.readings`, `_attach_readings` |
| execução das leituras | `wrag/methods/witnessrag.py::_try_readings`, `_retrieve_proof` |
| prompt robusto | `wrag/prompts.py::_PLAN_PART_ALTERNATIVES`, `_PLAN_PART_READINGS`, `_robust_examples` |
| preenchimento pela pergunta | `WitnessRAGRetriever._fact_context` (`fact_fill`) |
| reranker | `wrag/witness/rerank.py`, `WitnessRAGRetriever._rerank_facts` |
| datas bitemporais | `wrag/witness/timeline.py::anchored_phrase`, `natural_interval`; `prompts.QA_FACTS_*_BITEMPORAL` |
| relatório pareado | `scripts/paired-report.py` |

## Correção incluída

`scripts/proof-profiles.sh`: a linha `[[ "$profile" == abl-* ]] && ...` era o
último comando da função; para o perfil `witnessrag` o teste é falso, a
função devolvia 1 e, com `set -e`, `run-witness-openai-locomo.sh` saía sem
mensagem logo depois do diagnóstico. Afetava os perfis `witnessrag` e `abl-*` no
Linux (o script PowerShell não tinha o problema).
