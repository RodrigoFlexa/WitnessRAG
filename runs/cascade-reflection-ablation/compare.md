# Comparação pareada (cascade-reflection-ablation)

F1 oficial do LoCoMo; Δ contra `cascade` nas mesmas perguntas; IC 95% por bootstrap de conversas (4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas com penalidade de brevidade, normalização e stemming; não é uma métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: single-hop 841, multi-hop 282, temporal 321, open-domain 96. Tokens por pergunta sem a construção da memória.

| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | single-hop | multi-hop | temporal | open-domain | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cascade | 1540 | 53.69 |  | 46.99 |  |  | 58.9 | 47.8 | 54.3 | 22.8 | 1,913 | 6,736 | 0.97 | 19.3% | 61 |
| summary-reflection | 1540 | 54.43 | +0.74 [-0.30; +1.85] | 47.88 | +0.90 [-0.50; +2.33] | 262/246 | 60.0 | 46.6 | 55.6 | 25.0 | 3,576 | 8,358 | 0.96 | 19.6% | 58 |
| reader-reflection | 1540 | 51.41 | -2.28 [-3.26; -0.89] | 43.50 | -3.49 [-4.18; -2.58] | 186/240 | 54.4 | 48.1 | 55.2 | 21.8 | 2,259 | 7,082 | 0.97 | 19.3% | 61 |
| both | 1540 | 51.39 | -2.30 [-3.60; -0.88] | 43.53 | -3.46 [-4.68; -2.36] | 260/328 | 54.5 | 47.0 | 55.7 | 22.4 | 3,923 | 8,704 | 0.96 | 19.6% | 58 |

BLEU-1 local por categoria:

| variante | single-hop | multi-hop | temporal | open-domain |
|---|---:|---:|---:|---:|
| cascade | 53.14 | 35.46 | 49.18 | 19.59 |
| summary-reflection | 54.31 | 34.67 | 50.33 | 22.19 |
| reader-reflection | 47.96 | 30.83 | 50.41 | 18.52 |
| both | 48.04 | 30.06 | 50.80 | 19.34 |
