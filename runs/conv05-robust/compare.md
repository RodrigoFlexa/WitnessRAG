# Comparação pareada (conv05)

F1 oficial do LoCoMo; Δ contra `witnessrag` nas mesmas perguntas; IC 95% por bootstrap de perguntas (4000 reamostragens). Perguntas por categoria: single-hop 62, multi-hop 30, temporal 24, open-domain 7. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| witnessrag | 123 | 51.50 |  |  | 64.8 | 41.9 | 43.0 | 4.1 | 1,816 | 8,150 | 1.72 | 35.0% | 0 |
| wr-plan | 123 | 53.65 | +2.15 [-1.00; +5.86] | 13/12 | 66.3 | 44.8 | 45.7 | 7.1 | 1,818 | 9,268 | 1.59 | 46.3% | 7 |
| wr-fill | 123 | 54.98 | +3.48 [-0.55; +8.03] | 15/9 | 69.8 | 43.8 | 41.9 | 16.3 | 1,817 | 8,152 | 1.72 | 35.0% | 0 |
| wr-time | 123 | 53.99 | +2.49 [+0.49; +4.83] | 15/5 | 65.6 | 43.0 | 52.4 | 4.1 | 1,935 | 8,270 | 1.72 | 35.0% | 0 |
| wr-rerank | 123 | 58.68 | +7.18 [+3.16; +11.57] | 23/6 | 69.7 | 56.6 | 45.6 | 14.3 | 1,879 | 8,213 | 1.72 | 35.0% | 0 |
| wr-no-rerank | 123 | 58.32 | +6.82 [+2.13; +11.68] | 25/12 | 72.8 | 44.3 | 50.7 | 15.8 | 1,942 | 9,391 | 1.59 | 46.3% | 7 |
| witnessrag-robust | 123 | 62.96 | +11.46 [+6.62; +16.62] | 36/11 | 73.4 | 58.6 | 54.2 | 19.7 | 2,016 | 9,465 | 1.59 | 46.3% | 7 |
| robust-no-plan | 123 | 61.63 | +10.13 [+5.28; +15.26] | 31/11 | 74.5 | 54.3 | 50.0 | 19.0 | 2,006 | 2,006 | 0.0 | 0.0% | 0 |
| hybrid-chunks | 123 | 52.93 | +1.43 [-5.07; +7.98] | 38/33 | 71.3 | 42.4 | 34.0 | 0.0 | 10,263 | 10,263 | 0.0 | 0.0% | 0 |
