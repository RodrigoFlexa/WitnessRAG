# Hipóteses de plano e grafo testadas em uma conversa (26/09/2026)

gpt-4o-mini, perfil proof-v4, 5 x 2048, cache de LLM compartilhado. Conversa de
teste: conv05 (123 perguntas, 30 multi-hop, a com mais evidência faltando no
contexto: 39 perguntas com R@5 < 1). Confirmação em conv08 (156 perguntas).
Todas as opções ficam desligadas por padrão.

| hipótese | flag | F1 conv05 | Δ | multi-hop | R@5 | respostas demais | junção incompleta | contexto mudou | F1 conv08 (Δ) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base proof-v4 | | 56,25 | | 44,2 | 79,5 | 43 | 13 | | 56,05 |
| dominância (1,05) | `--proof-dominance 1.05` | 56,25 | 0,00 | 44,2 | 79,5 | 25 | 13 | 4 | |
| relações das âncoras (RoG) | `--anchor-relations 12` | 55,96 | −0,29 | 47,8 | 78,1 | 39 | 13 | 30 | |
| relação mais frouxa (0,55) | `--relation-threshold 0.55` | 56,46 | +0,21 | 45,3 | 79,1 | 51 | 6 | 16 | 56,34 (+0,29) |
| união em conjuntos | `--set-union` | 56,29 | +0,04 | 44,4 | 78,9 | 43 | 9 | 8 | |
| combinação (0,55 + dom. + união) | | 55,48 | −0,77 | 44,4 | 78,1 | 28 | 3 | 15 | |
| plano para o leitor | `--plan-to-reader` | 57,72 | +1,47 | 49,2 | 79,5 | 43 | 13 | 0 | 54,24 (−1,81) |

## Leitura

1. As hipóteses de plano e grafo fazem o que prometem no mecanismo
   (dominância: respostas demais 43 → 25; relação frouxa: junção incompleta
   13 → 6), mas o F1 não se move: a prova aceita aponta quase sempre para
   trechos que já estavam nos 5.
2. Quando a prova muda o contexto, a revocação cai (ΔR@5 negativo em todas
   as variantes na conv05): a troca pela cauda tira trechos certos, sobretudo
   em perguntas de conjunto cujas respostas estão espalhadas em mais de 5 trechos.
3. Na conv05, 30 das 39 perguntas com evidência faltando têm a resposta em
   fatos do trecho certo: o grafo tem a informação; o que falta é o plano
   alcançá-la e o contexto comportá-la.
4. Plano para o leitor: +5 em multi-hop na conv05, mas o planejador marca
   perguntas singulares como conjunto e o leitor passa a listar; na conv08 o
   saldo é −1,8. Não é robusto.
5. Única mudança com sinal consistente nas duas conversas: relação 0,55
   (+0,2/+0,3), dentro do ruído.

Conclusão: com 84% das perguntas já com toda a evidência nos 5 trechos, o
controlador de prova edita no máximo 2 vagas de um contexto que raramente
precisa de edição. Mudanças de plano e travessia não têm onde render F1 neste
regime; o ganho possível está em (a) perguntas de conjunto espalhadas (entrega
por falas, sem trocar trechos) e (b) na leitura.
