# LoCoMo com gpt-4o-mini, proof-v4, trechos de 512 e k=20 (26/09/2026)

Configuração: `METHOD=proof PROFILE=proof-v4 CHUNK_TOKENS=512 TOP_K=20 POOL=40
EXTRA_FLAGS="--proof-edit-fraction 0.4" bash scripts/run-witness-openai-locomo.sh`
(regime "budget" da varredura v4: pool 2k, k_W proporcional a k: 8 trocas em
planos compostos, 4 em simples). A OpenIE foi refeita para trechos de 512.

A rodada parou na conversa 8 (117 de 156 perguntas) por falta de créditos da
API (erro 429 "no credits remaining"); a conversa 9 não rodou. Os registros
até a parada estão completos. Para terminar: adicionar créditos e repetir o
mesmo comando com a mesma saída (retoma sozinho).

Comparação pareada com o proof-v4 em 5 x 2048 (mesmas perguntas, F1 oficial,
IC 95% por bootstrap de conversas):

| categoria | n | 5 x 2048 | 20 x 512 | delta | IC 95% |
|---|---:|---:|---:|---:|---|
| todas | 1343 | 58,59 | 61,40 | +2,81 | [+0,91; +5,12] |
| single-hop | 741 | 68,26 | 70,65 | +2,39 | [+0,04; +5,31] |
| multi-hop | 237 | 43,66 | 49,66 | +6,00 | [+1,70; +11,62] |
| temporal | 279 | 57,53 | 58,01 | +0,49 | [−1,29; +1,92] |
| open-domain | 86 | 19,89 | 24,98 | +5,09 | [−0,83; +11,19] |

Sem a conversa parcial (1226 perguntas): 58,47 contra 61,40.

Ressalva: o ganho mistura granularidade da recuperação (vale para qualquer
método) e o efeito da prova com mais vagas. Para isolar o método, falta o
híbrido em 20 x 512 (`METHOD=hybrid CHUNK_TOKENS=512 TOP_K=20 POOL=40`).
