# Controlador de prova (v3): avaliação offline sem LLM

Data: 24/09/2026. Script: `scripts/proof-offline-eval.py`.

## O que foi medido e o que não foi

Este relatório mede **só a recuperação**: R@5 (fração das passagens de evidência
anotadas que estão no contexto de 5 trechos) e AR@5 (todas estão), além de
quantos contextos mudam em relação ao híbrido. **Não há F1**: nenhuma resposta
foi gerada. O F1 precisa da rodada no Azure (`scripts/run-witness-proof-locomo.sh`).

Insumos, todos já existentes no repositório:

- memórias congeladas das 10 conversas (`runs/locomo-controlled-01/controlled`):
  fatos extraídos pelo Qwen2.5-14B com `--dialogue-ie`, vetores BGE-M3;
- trechos de 2048 tokens, idênticos aos da rodada do Azure;
- para o replay dos controladores, as consultas compiladas pelo Qwen na rodada
  `locomo-all-b-noverify-simple` (categorias single-hop e multi-hop).

A rodada do Azure (GPT-4.1-mini) usa fatos e planos do GPT, não do Qwen. Os
números abaixo são uma aproximação dessa rodada, não uma reprodução.

## 1. O harness reproduz a linha de base

Busca híbrida (BGE-M3 + BM25, RRF), top-5, sem intervenção:

| categoria | n | R@5 | AR@5 | rodada Azure (selective-v1) |
|---|---:|---:|---:|---|
| single-hop | 841 | 94.41 | 94.41 | 94.41 / 94.41 |
| multi-hop | 282 | 69.30 | 44.68 | 68.62 / 42.55 (contexto alterado em 91%) |
| temporal | 320 | 89.69 | 88.44 | 89.72 / 88.47 |
| open-domain | 92 | 70.29 | 56.52 | 70.29 / 56.52 |

Onde o selective-v1 não muda o contexto (single-hop, temporal e open-domain), o
harness reproduz os números da rodada do Azure. A diferença de uma pergunta no
temporal vem de uma evidência que o mapeamento do harness não localiza.

## 2. O que o selective-v1 fez no multi-hop

Replay do `selective-v1` com os fatos e as consultas do Qwen:

| | R@5 | AR@5 | contexto alterado | ganhos / perdas de revocação |
|---|---:|---:|---:|---|
| híbrido | 69.30 | 44.68 | 0% | |
| selective-v1 (replay) | 68.52 | 41.84 | 83% | 32 / 38 |
| selective-v1 (Azure, real) | 68.62 | 42.55 | 91% | |

O replay acompanha a rodada real. Das 282 perguntas multi-hop, 230 mudaram de
contexto pelas **sondas de concordância**, 5 por testemunha e 47 ficaram com o
híbrido. As sondas trocam a quinta passagem e, em média, **reduzem** a
revocação da evidência anotada.

Não sabemos o F1 do híbrido puro com o GPT-4.1-mini. Sem esse número, não dá para
dizer se o F1 multi-hop de 46,98 veio das sondas ou existiria sem elas. A
rodada `METHOD=hybrid` de `scripts/run-witness-azure-locomo.sh` responde isso.

## 3. Calibração dos pesos da pontuação

A pontuação é `w_sim·sim + w_imp·imp + w_tempo·prox`.

- **Período "hoje" (plano padrão).** Com qualquer peso de tempo ou importância
  entre 0,05 e 0,2, de 12% a 72% dos contextos mudaram sem ganho de revocação.
  Algumas combinações perderam até 1,4 ponto no single-hop. Conclusão: o nível
  "normal" deixa todo o peso na similaridade.
- **Janela citada na pergunta.** Aqui o tempo ajuda em todas as categorias
  (janela extraída da pergunta por expressão regular, como proxy do planejador).

| peso do tempo | single R@5 | multi R@5 | temporal R@5 | open R@5 | contextos alterados |
|---|---:|---:|---:|---:|---:|
| 0 (híbrido) | 94.41 | 69.30 | 89.69 | 70.29 | 0% |
| 0,2 | 95.01 | 69.47 | 90.57 | 71.38 | 1–5% |
| **0,3 (nível "strong")** | **95.01** | **69.47** | **90.57** | **71.38** | 1–7% |
| 0,5 | 95.01 | 69.47 | 90.26 | 71.38 | 1–11% |

- **Primeira vez (período "início").** Temporal +0,3.
- **Mais recente.** Nenhuma mudança de revocação.
- **Importância.** Efeito líquido nulo. Fica com peso pequeno (0,1) e só quando
  o plano pede ("strong").

## 4. Replay do controlador de prova

Planos vindos das consultas do Qwen. O período vem da pergunta, pela mesma
expressão regular da calibração. Um ciclo.

Duas políticas de verificação: `all` aceita toda prova utilizável, é o limite
inferior. `oracle` aceita só se um fato vem de uma passagem de evidência, é o
limite superior. O verificador real fica entre as duas.

| configuração | single R@5 | single alterado | single G/P | multi R@5 | multi AR@5 | multi alterado | multi G/P |
|---|---:|---:|---|---:|---:|---:|---|
| híbrido | 94.41 | 0% | | 69.30 | 44.68 | 0% | |
| selective-v1 | 94.41 | 0% | | 68.52 | 41.84 | 83% | 32/38 |
| **prova v3, verify=all** | **94.89** | 9,9% | 4/0 | 69.24 | 44.33 | 4,3% | 2/2 |
| **prova v3, verify=oracle** | **94.89** | 7,7% | 4/0 | **69.59** | **45.04** | 1,4% | 2/0 |
| prova v3 + sondas (verify=all) | 93.46 | 48% | 7/15 | 67.28 | 42.20 | 34% | 7/20 |
| prova v3 + sondas (verify=oracle) | 93.46 | 47% | 7/15 | 67.57 | 42.55 | 32% | 8/19 |

(G/P = perguntas cuja revocação subiu / desceu entre as que mudaram de contexto.)

## Decisões tomadas a partir destes números

1. **Pesos.** "normal" = só similaridade. "strong" = tempo 0,3 e importância 0,1.
   O planejador liga o "strong" quando a pergunta tem data, "primeira vez",
   "mais recente" ou algo marcante.
2. **Sem sondas por padrão.** A evidência parcial (sondas de concordância e
   resgate de lacuna) mudou metade dos contextos de perguntas simples e perdeu
   revocação nas duas categorias. Fica como ablação (`--partial-evidence`,
   `PROFILE=proof-partial`).
3. **Prova com limites.**
   - Uma prova confirmada de plano composto pode trazer até 2 trechos novos.
   - Um plano de um fato e um valor pode trazer 1.
   - O prefixo é protegido.
   - A verificação só é chamada quando a prova mudaria o contexto.
   - Nesses limites, a prova não perdeu revocação em nenhuma das duas políticas.
4. **Onde o contexto não muda, a resposta é idêntica à da rodada anterior.** O
   prompt do leitor é byte a byte o mesmo e vem do cache. Isso vale para a maior
   parte das perguntas.

## Limites desta avaliação

- **Fatos e planos são do Qwen.** Com o GPT-4.1-mini, a extração e os planos
  devem ser melhores. O número de provas utilizáveis deve subir.
- **O planejador real escreve o plano vendo as evidências.** O replay usa a
  consulta compilada sem evidências e detecta o período por expressão regular.
- **Revocação não é F1.** O efeito das sondas no F1 do multi-hop só aparece na
  rodada paga. Recomendação: rodar `PROFILE=proof` e `PROFILE=proof-partial`,
  mais o híbrido, com o mesmo cache.
