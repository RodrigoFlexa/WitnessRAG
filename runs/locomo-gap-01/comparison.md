# Busca dirigida na memória congelada

Mesmas cinco passagens integrais no leitor comum; apenas a recuperação varia.

| Condição | F1 geral | F1 single | F1 multi | AR@5 multi | F1 contagem | acerto numérico | tokens leitor |
|---|---:|---:|---:|---:|---:|---:|---:|
| common | 0.5983 | 0.6629 | 0.4056 | 0.4397 | 0.1448 | 0.4000 (25) | 11518642 |
| gap-lexical | 0.6003 | 0.6629 | 0.4136 | 0.4681 | 0.1263 | 0.3200 (25) | 11518275 |
| gap-verified | 0.6006 | 0.6629 | 0.4148 | 0.5000 | 0.1263 | 0.4000 (25) | 11516490 |
| gap-two | 0.5967 | 0.6629 | 0.3993 | 0.4291 | 0.0741 | 0.2800 (25) | 11514105 |
| count-full | 0.5968 | 0.6629 | 0.3999 | 0.4397 | 0.0857 | 0.4400 (25) | 11517373 |

gap-lexical − common: +0.0081; IC95% por conversa [-0.0068, +0.0238] (n=282).

gap-verified − common: +0.0093; IC95% por conversa [-0.0042, +0.0243] (n=282).

gap-two − common: -0.0063; IC95% por conversa [-0.0218, +0.0130] (n=282).

count-full − common: -0.0591; IC95% por conversa [-0.1044, +0.0000] (n=27).

Elegíveis: 199; contexto alterado: lexical 151, verificado 81, dois novos 151.
Custos reais de chamadas novas e do verificador estão em comparison.json; tokens por condição incluem as respostas reutilizadas.
Correspondência literal ausente não equivale a item sem suporte semântico.
F1 oficial permanece a métrica principal; acerto numérico é secundário.
