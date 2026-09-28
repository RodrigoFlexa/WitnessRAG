# Ablação do WitnessRAG (ablation-qwen-c2048-k5)

F1 oficial do LoCoMo; Δ contra o completo nas mesmas perguntas; IC 95% por bootstrap de conversas; tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | single-hop | multi-hop | temporal | open-domain | tokens leitor/perg. | tokens total/perg. |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| WitnessRAG (completo) | 1540 | 49.41 |  | 55.4 | 41.1 | 49.6 | 20.1 | 1,684 | 7,953 |
| sem plano | 1540 | 50.16 | +0.76 [-0.29; +1.78] | 56.4 | 41.8 | 50.2 | 20.3 | 1,704 | 1,704 |
| sem prova | 1540 | 47.84 | -1.57 [-1.92; -1.15] | 54.3 | 39.5 | 46.5 | 20.3 | 1,674 | 5,420 |
| sem verificação | 1540 | 49.23 | -0.17 [-1.28; +1.03] | 55.2 | 40.8 | 49.2 | 21.9 | 1,690 | 7,323 |
| sem nota temporal | 1540 | 49.08 | -0.33 [-0.87; +0.13] | 55.2 | 41.1 | 48.8 | 19.7 | 1,685 | 7,960 |
