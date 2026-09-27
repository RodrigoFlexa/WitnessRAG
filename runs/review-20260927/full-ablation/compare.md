# Comparação pareada (ablation-openai-c2048-k5)

F1 oficial do LoCoMo; Δ contra `full` nas mesmas perguntas; IC 95% por bootstrap de conversas (4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas com penalidade de brevidade, normalização e stemming; não é uma métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: single-hop 841, multi-hop 282, temporal 321, open-domain 96. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 1540 | 55.35 |  | 49.59 |  |  | 62.7 | 45.4 | 53.6 | 25.9 | 1,796 | 7,807 | 1.65 | 39.9% | 0 |
| no-plan | 1540 | 54.86 | -0.50 [-1.39; +0.41] | 48.94 | -0.65 [-1.77; +0.38] | 156/155 | 62.3 | 45.6 | 52.2 | 26.1 | 1,796 | 1,796 | 0.0 | 0.0% | 0 |
| no-proof | 1540 | 54.31 | -1.04 [-1.69; -0.33] | 48.70 | -0.90 [-1.48; -0.29] | 104/117 | 62.1 | 43.6 | 52.0 | 25.3 | 1,784 | 5,432 | 1.14 | 0.0% | 0 |
| no-verify | 1540 | 55.00 | -0.35 [-0.78; +0.06] | 49.34 | -0.26 [-0.63; +0.10] | 27/34 | 62.1 | 45.7 | 53.4 | 25.8 | 1,800 | 7,063 | 1.63 | 42.1% | 0 |
