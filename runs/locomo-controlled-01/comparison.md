# LoCoMo controlado

| Condição | n | F1 geral | F1 single | F1 multi | Busca média (s) |
|---|---:|---:|---:|---:|---:|
| evidence/common | 1123 | 0.5982 | 0.6618 | 0.4083 | 5.28 |
| evidence/proof | 1123 | 0.5881 | 0.6540 | 0.3916 | 5.28 |
| soft-v2/common | 1123 | 0.5914 | 0.6519 | 0.4112 | 5.75 |
| soft-v2/proof | 1123 | 0.5544 | 0.6176 | 0.3660 | 5.75 |

Auditoria: 80 recuperações com replay idêntico; 65 também idênticas na repetição com inferência nova.

Custos de busca são compartilhados pelos dois leitores; não somar duas vezes.
Busca usa um cache novo compartilhado. Leitura usa chamadas novas; hits são registrados.
Custo frio e quente é medido separadamente na amostra da auditoria (comparison.json).
Intervalos por conversa, tokens e custos por célula estão em comparison.json.
Os resultados são de desenvolvimento; estas conversas já orientaram a proposta.
