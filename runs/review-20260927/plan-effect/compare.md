# Comparação pareada (conv05-robust)

F1 oficial do LoCoMo; Δ contra `robust-no-plan` nas mesmas perguntas; IC 95% por bootstrap de perguntas (4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas com penalidade de brevidade, normalização e stemming; não é uma métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: single-hop 62, multi-hop 30, temporal 24, open-domain 7. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| robust-no-plan | 123 | 61.63 |  | 56.47 |  |  | 74.5 | 54.3 | 50.0 | 19.0 | 2,006 | 2,006 | 0.0 | 0.0% | 0 |
| witnessrag-robust | 123 | 62.96 | +1.32 [-1.66; +4.45] | 57.42 | +0.95 [-1.92; +3.65] | 12/6 | 73.4 | 58.6 | 54.2 | 19.7 | 2,016 | 9,465 | 1.59 | 46.3% | 3 |
