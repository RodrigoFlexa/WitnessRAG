# Entrega por fatos: o leitor recebe fatos em vez de trechos (26/09/2026)

Opção `--fact-delivery facts|facts+summary` (desligada por padrão; `--fact-budget N`,
padrão 40). Conversa 1 do LoCoMo (81 perguntas), gpt-4o-mini, proof-v4 5 x 2048
(mesma memória e mesmo controlador; só a entrega ao leitor muda).

## Como funciona

1. O controlador roda igual (busca, plano, prova, verificação).
2. Seleção de até 40 fatos: primeiro os fatos da prova verificada; depois os
   das melhores respostas do plano (mesmo sem prova aceita); o resto por
   relevância = max(cosseno com a pergunta, cosseno com cada átomo do plano),
   com bônus pequeno para fatos dos 5 melhores trechos. No máximo 4 fatos por
   fala de origem; sem triplas repetidas.
3. Renderização por sessão, da mais antiga para a mais recente, cada fato com
   a data do evento já resolvida: `- Melanie | camp at | mountains (event: 19 June 2023 - 25 June 2023)`.
4. B (`facts+summary`): mais um resumo de até 3 frases de cada um dos 6 trechos
   que mais contribuíram com fatos. O resumo é escrito uma vez por trecho, sem
   ver pergunta alguma (custo de construção da memória, em cache).
5. Leitor: o mesmo prompt de evidência, com cabeçalho que explica as notas e a
   regra de tempo trocada para "use a data do evento do fato".

Código: `WitnessRAGRetriever._fact_context`, `_chunk_summary`;
`prompts.qa_facts_template`, `SUMMARY_TEMPLATE`; `reader.read(facts_mode=...)`.

## Resultados (conversa 1)

| variante | F1 | single-hop (44) | multi-hop (11) | temporal (26) | tokens do leitor / pergunta | total / pergunta | resumos (uma vez) |
|---|---:|---:|---:|---:|---:|---:|---:|
| trechos 5 x 2048 (base) | 58,33 | 45,5 | 43,4 | 86,3 | 10.385 | 15.751 | 0 |
| A: só fatos | 38,62 | 23,8 | 35,8 | 64,9 | 1.054 | 6.418 | 0 |
| B: fatos + resumos | 41,80 | 29,7 | 38,3 | 63,7 | 1.521 | 6.886 | 15.522 |

## Por que perde

- Extração: nas 55 perguntas não temporais, a resposta está em algum fato da
  memória em só 31 (56%). Detalhes como "Marley flooring", "The Lean Startup"
  ou "contemporary" nunca viraram fato. Muitas triplas são vagas
  ("Gina | reminds | myself about it").
- Seleção: dos 31 casos em que o fato existe, os 40 fatos entregues contêm a
  resposta em 19 (61%).
- Tempo: planos e intenções recebem a data da sessão ("planning to open the
  studio" datado em 13/6, quando o plano era 20/6). No texto, o leitor lia a data
  certa na própria fala.
- O resumo ajuda um pouco (+3,2 F1, +467 tokens por pergunta), sobretudo em
  single-hop, mas não compensa o que a extração perde.
