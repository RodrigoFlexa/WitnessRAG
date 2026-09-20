# Ablação da pesquisa de provas

Rodadas: `/home/rodrigo.flexa/WitnessRAG/runs/locomo-soft-proof-01`

| Condição | n | F1 oficial | F1 multi | Δ multi | AR@5 multi | F1 contagem | Acerto numérico¹ | Disparo multi | Prova provisória | Busca multi (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evidence | 1123 | 0.5986 | 0.4136 | +0.0000 | 0.4255 | 0.1319 | 0.4706 (17) | 0.2837 | 0.0000 | 0.05 |
| soft-only | 1123 | 0.6010 | 0.4242 | +0.0107 | 0.4504 | 0.1319 | 0.4118 (17) | 0.2979 | 0.6028 | 5.79 |
| proof-only | 1123 | 0.5847 | 0.3997 | -0.0138 | 0.4220 | 0.0625 | 0.4667 (15) | 0.2908 | 0.0000 | 5.57 |
| soft-proof | 1123 | 0.5540 | 0.3796 | -0.0340 | 0.4504 | 0.0417 | 0.3077 (13) | 0.2695 | 0.6454 | 6.25 |

¹ Diagnóstico adicional: só conta respostas/gabaritos com um número único reconhecível; não substitui a métrica oficial.

Comparação pareada (multi-hop, contra control):

- `soft-only`: 54 ganhos, 45 perdas, 183 empates.
- `proof-only`: 22 ganhos, 36 perdas, 224 empates.
- `soft-proof`: 58 ganhos, 78 perdas, 146 empates.
