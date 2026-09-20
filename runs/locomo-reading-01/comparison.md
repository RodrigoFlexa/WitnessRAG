# Leitura sobre recuperação congelada

Mesmos pids nas três condições; leitor de provas desligado.

| Condição | F1 geral | F1 single | F1 multi | F1 contagem | acerto numérico | tokens prompt |
|---|---:|---:|---:|---:|---:|---:|
| common | 0.5960 | 0.6593 | 0.4074 | 0.1448 | 0.4000 (25) | 11518642 |
| focused | 0.5384 | 0.6062 | 0.3363 | 0.0594 | 0.2800 (25) | 3342846 |
| focused-count | 0.5381 | 0.6062 | 0.3353 | 0.0487 | 0.2800 (25) | 3341577 |

focused − common, multi: -0.0711; IC95% por conversa [-0.1086, -0.0384].

focused-count − common, multi: -0.0721; IC95% por conversa [-0.1091, -0.0402].

F1 oficial sem alteração; acerto numérico usa somente gabaritos numéricos inequívocos e conta respostas não numéricas como erro.
A resposta common histórica é registrada por pergunta para medir deriva entre execuções, sem servir de controle pareado.
