# Ablação da pesquisa de provas

Rodadas: `/home/rodrigo.flexa/WitnessRAG/runs/locomo-proof-ablation-01`

| Condição | n | F1 oficial | F1 multi | Δ multi | AR@5 multi | F1 contagem | Acerto numérico¹ | Disparo multi | Busca multi (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1123 | 0.5597 | 0.3831 | +0.0000 | 0.3688 | 0.0833 | 0.4444 (18) | 0.8936 | 2.43 |
| frontier | 1123 | 0.5668 | 0.3998 | +0.0167 | 0.4043 | 0.1250 | 0.5000 (18) | 0.8936 | 3.23 |
| obligations | 1123 | 0.5952 | 0.3914 | +0.0083 | 0.4433 | 0.0833 | 0.4000 (15) | 0.2589 | 2.59 |
| frontier-obligations | 1123 | 0.5953 | 0.4037 | +0.0206 | 0.4397 | 0.1042 | 0.4706 (17) | 0.2837 | 0.57 |
| evidence | 1123 | 0.5986 | 0.4136 | +0.0305 | 0.4255 | 0.1319 | 0.4706 (17) | 0.2837 | 0.03 |
| operators | 1123 | 0.5885 | 0.4053 | +0.0223 | 0.4255 | 0.0417 | 0.5000 (18) | 0.2837 | 0.03 |
| verified | 1123 | 0.5885 | 0.4062 | +0.0231 | 0.4468 | 0.0417 | 0.5000 (18) | 0.1915 | 3.10 |
| no-acquisition | 1123 | 0.5846 | 0.3989 | +0.0159 | 0.4255 | 0.0417 | 0.5000 (18) | 0.2589 | 0.02 |
| one-plan | 1123 | 0.5861 | 0.4008 | +0.0178 | 0.4362 | 0.0417 | 0.4444 (18) | 0.1986 | 2.22 |

¹ Diagnóstico adicional: só conta respostas/gabaritos com um número único reconhecível; não substitui a métrica oficial.

Comparação pareada (multi-hop, contra control):

- `frontier`: 32 ganhos, 18 perdas, 232 empates.
- `obligations`: 44 ganhos, 38 perdas, 200 empates.
- `frontier-obligations`: 61 ganhos, 48 perdas, 173 empates.
- `evidence`: 72 ganhos, 53 perdas, 157 empates.
- `operators`: 71 ganhos, 53 perdas, 158 empates.
- `verified`: 72 ganhos, 55 perdas, 155 empates.
- `no-acquisition`: 66 ganhos, 55 perdas, 161 empates.
- `one-plan`: 69 ganhos, 60 perdas, 153 empates.
