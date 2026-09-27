# Comparação pareada (conv05-robust)

F1 oficial do LoCoMo; Δ contra `witnessrag` nas mesmas perguntas; IC 95% por bootstrap de perguntas (4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas com penalidade de brevidade, normalização e stemming; não é uma métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: single-hop 62, multi-hop 30, temporal 24, open-domain 7. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| witnessrag | 123 | 51.50 |  | 45.98 |  |  | 64.8 | 41.9 | 43.0 | 4.1 | 1,816 | 8,150 | 1.72 | 35.0% | 0 |
| wr-plan | 123 | 53.65 | +2.15 [-1.24; +5.90] | 47.93 | +1.95 [-1.33; +5.43] | 13/12 | 66.3 | 44.8 | 45.7 | 7.1 | 1,818 | 9,268 | 1.59 | 46.3% | 3 |
| wr-fill | 123 | 54.98 | +3.48 [-0.49; +7.78] | 49.43 | +3.45 [-0.53; +7.79] | 15/9 | 69.8 | 43.8 | 41.9 | 16.3 | 1,817 | 8,152 | 1.72 | 35.0% | 0 |
| wr-time | 123 | 53.99 | +2.49 [+0.38; +4.90] | 48.84 | +2.86 [+0.40; +5.70] | 15/5 | 65.6 | 43.0 | 52.4 | 4.1 | 1,935 | 8,270 | 1.72 | 35.0% | 0 |
| wr-rerank | 123 | 58.68 | +7.18 [+2.99; +11.69] | 53.22 | +7.24 [+3.11; +11.74] | 23/6 | 69.7 | 56.6 | 45.6 | 14.3 | 1,879 | 8,213 | 1.72 | 35.0% | 0 |
| wr-no-rerank | 123 | 58.32 | +6.82 [+2.22; +11.49] | 53.18 | +7.20 [+2.48; +12.01] | 25/12 | 72.8 | 44.3 | 50.7 | 15.8 | 1,942 | 9,391 | 1.59 | 46.3% | 3 |
| witnessrag-robust | 123 | 62.96 | +11.46 [+6.62; +16.64] | 57.42 | +11.44 [+6.53; +16.63] | 36/11 | 73.4 | 58.6 | 54.2 | 19.7 | 2,016 | 9,465 | 1.59 | 46.3% | 3 |
| robust-no-plan | 123 | 61.63 | +10.13 [+5.17; +15.33] | 56.47 | +10.49 [+5.28; +15.99] | 31/11 | 74.5 | 54.3 | 50.0 | 19.0 | 2,006 | 2,006 | 0.0 | 0.0% | 0 |
| hybrid-chunks | 123 | 52.93 | +1.43 [-5.22; +8.01] | 47.76 | +1.78 [-5.02; +8.43] | 38/33 | 71.3 | 42.4 | 34.0 | 0.0 | 10,263 | 10,263 | 0.0 | 0.0% | 0 |
