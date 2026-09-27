# Memória v2 (proposições) e entrega por fatos (26/09/2026)

Objetivo: chegar perto (ou passar) do leitor com 5 trechos de 2048 tokens
gastando pouco, com o plano escolhendo o que o leitor lê.

## O que mudou

**Extração (`--ie-style memory`).** Um passo por janela de 512 tokens (sem
NER). Cada fala informativa vira de 1 a 4 proposições atômicas (a unidade
do Dense X Retrieval, Chen et al. 2023), cada uma com:
tripla (sujeito, relação canônica curta, objeto mais específico possível),
frase autocontida com as palavras do falante, fala de origem declarada
("D4:6"), expressão de tempo e tipo (past, plan, ongoing, said). Conselhos,
opiniões, razões, descrições, significados e fotos compartilhadas também viram
memória. A frase é o que o embedding do fato indexa; a tripla continua sendo o
que o plano percorre.

**Datas.** A fala de origem declarada substitui a heurística de sobreposição.
Planos apontam para frente: "on Friday" dito numa terça é a sexta seguinte;
"in two weeks" é contado a partir da sessão.

**Entrega por fatos (`--fact-delivery facts|facts+summary`, `--fact-budget 40`).**
Fatos da prova verificada primeiro, depois os das melhores respostas do plano,
depois por relevância = max(similaridade com a pergunta, com cada átomo do
plano); no máximo 4 por fala; agrupados por sessão, com data do evento ou
"planned for". B acrescenta resumos de até 6 trechos (escritos uma vez).
`--fact-no-plan` é a ablação que escolhe só pela pergunta.

## Resultados (gpt-4o-mini, proof-v4)

Conversa 1 (81 perguntas):

| variante | F1 | single | multi | temporal | tokens do leitor | total / pergunta |
|---|---:|---:|---:|---:|---:|---:|
| trechos 5 x 2048 | 58,33 | 45,5 | 43,4 | 86,3 | 10.385 | 15.751 |
| fatos, extração antiga (A) | 38,62 | 23,8 | 35,8 | 64,9 | 1.054 | 6.418 |
| fatos + resumo, extração antiga (B) | 41,80 | 29,7 | 38,3 | 63,7 | 1.521 | 6.886 |
| **A, memória v2** | 60,06 | 48,5 | 55,7 | 81,5 | 1.281 | 6.784 |
| A, memória v2, sem plano | 56,63 | 45,0 | 47,2 | 80,3 | 1.292 | 6.796 |
| **B, memória v2** | **61,46** | 50,9 | 56,1 | 81,5 | 1.756 | 7.260 |

Conversa 5 (123 perguntas):

| variante | F1 | single | multi | temporal | open | tokens do leitor | total / pergunta |
|---|---:|---:|---:|---:|---:|---:|---:|
| trechos 5 x 2048 | 56,25 | 72,3 | 44,2 | 46,4 | 0,0 | 10.258 | 16.579 |
| **A, memória v2** | 54,94 | 69,7 | 47,2 | 42,1 | 1,9 | 1.316 | 7.526 |
| A, memória v2, sem plano | 52,74 | 69,2 | 39,0 | 39,4 | 11,3 | 1.313 | 7.523 |
| **B, memória v2** | **57,54** | 69,8 | 50,2 | 47,3 | 15,2 | 1.827 | 8.037 |

Resumos (B): 15.522 tokens (conv 1) e 29.258 (conv 5), uma vez por conversa.

## Leitura

- A memória v2 é o que viabiliza a entrega por fatos: na conversa 1, a
  resposta está na memória em 42 de 55 perguntas não temporais (antes 31), e
  a seleção guiada pelo plano entrega quase todas.
- B com memória v2 supera os trechos nas duas conversas (+3,1 e +1,3 F1) com
  5,6 a 5,9 vezes menos tokens no leitor.
- O plano faz diferença neste regime: +3,4 e +2,2 F1 sobre escolher os fatos só
  pela pergunta, e +8 pontos em multi-hop nas duas conversas.
- Onde ainda perde: single-hop na conversa 5 (-2,5) e temporal (-5 na conversa 1).
- O custo total por pergunta agora é dominado pelo planejamento e pela
  verificação (~5-6 mil tokens), não pelo leitor: é o próximo lugar para economizar.
