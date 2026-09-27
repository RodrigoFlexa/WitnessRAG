# Ablação do WitnessRAG (ablation-openai-c2048-k5)

F1 oficial do LoCoMo; Δ contra o completo nas mesmas perguntas; IC 95% por bootstrap de conversas; tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | single-hop | multi-hop | temporal | open-domain | tokens leitor/perg. | tokens total/perg. |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| WitnessRAG (completo) | 1540 | 55.35 |  | 62.7 | 45.4 | 53.6 | 25.9 | 1,796 | 7,807 |
| sem plano | 1540 | 54.86 | -0.50 [-1.40; +0.41] | 62.3 | 45.6 | 52.2 | 26.1 | 1,796 | 1,796 |
| sem prova | 1540 | 54.31 | -1.04 [-1.68; -0.34] | 62.1 | 43.6 | 52.0 | 25.3 | 1,784 | 5,432 |
| sem verificação | 1540 | 55.00 | -0.35 [-0.77; +0.07] | 62.1 | 45.7 | 53.4 | 25.8 | 1,800 | 7,063 |
| sem nota temporal | 146 | 53.03 | +0.16 [+0.16; +0.16] | 57.6 | 52.3 | 53.1 | 30.1 | 1,803 | 7,663 |
