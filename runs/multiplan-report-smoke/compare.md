# Comparação pareada (conv05-robust)

F1 oficial do LoCoMo; Δ contra `witnessrag-robust` nas mesmas perguntas; IC 95% por bootstrap de perguntas (4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas com penalidade de brevidade, normalização e stemming; não é uma métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: single-hop 62, multi-hop 30, temporal 24, open-domain 7. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| witnessrag-robust | 123 | 62.96 |  | 57.42 |  |  | 73.4 | 58.6 | 54.2 | 19.7 | 2,016 | 9,465 | 1.59 | 46.3% | 3 |
| robust-no-plan | 123 | 61.63 | -1.32 [-4.45; +1.66] | 56.47 | -0.95 [-3.65; +1.92] | 6/12 | 74.5 | 54.3 | 50.0 | 19.0 | 2,006 | 2,006 | 0.0 | 0.0% | 0 |

BLEU-1 local por categoria:

| variante | single-hop | multi-hop | temporal | open-domain |
|---|---:|---:|---:|---:|
| witnessrag-robust | 69.33 | 46.35 | 51.52 | 19.64 |
| robust-no-plan | 70.02 | 43.75 | 48.31 | 19.05 |
