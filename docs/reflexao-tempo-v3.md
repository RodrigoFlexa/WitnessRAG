# Referência temporal e reflexão (v3)

Mudanças motivadas pela ablação v2 (10 conversas, Qwen2.5-14B, 40 fatos), em
que "sem referência temporal" (−0,20) e "sem reflexão" (+0,14) não mexeram no
F1. A análise pareada mostrou:

- a busca já entrega a fala-ouro nos 40 fatos em 91% das perguntas; o gargalo
  é a leitura (F1 temporal 58,9 mesmo com a evidência toda presente);
- a referência temporal só ligava em 250 de 1540 perguntas (44 de 321
  temporais): para "When did X…?" o contrato marcava `period=now` e peso 0;
- a reflexão era uma instrução no prompt do leitor, sem chamada nem raciocínio
  visível: mudava 30% das respostas com 185 ganhos e 159 perdas; escolher a
  melhor entre "com" e "sem" daria 62,45 (+3,1).

## Iterações e o que ficou

| versão | conversa | completo | antes (v2) | o que mostrou |
|---|---|---|---|---|
| v3 | conv03, 40 fatos | 53,81 | 53,49 | referência mudando o plano: −9,6 nas 17 perguntas tocadas; tempos ao leitor: +4,7 em 20; descrição solta ao leitor: −1,5 |
| v3b | conv07, 40 fatos (parcial, 90 perg.) | 65,94 | 68,77 | tempos ao leitor: −22 em 14; leitor simples como padrão perde |
| v3c | conv07, 20 fatos (parcial, 127) | 64,29 | 64,54 | falso "data incompleta" ("when she was 10") e discordância só de forma |
| **v3d** | **conv07, 20 fatos** | **63,78** | **62,43** | +1,35; concordância idêntica à v2 (135/136); ganho só onde a reflexão agiu |

As correções da v3c para a v3d vieram de olhar a conv07; a conv04 (não
inspecionada) é a validação.

## Referência temporal (desenho final)

**Plano** (`plan_reference`). Uma chamada curta lê SÓ a pergunta e a data
presente da memória e declara `target`, `event`, `scope`, `window`,
`relation`, `anchor_event`, `order`, `granularity` (vocabulário fechado;
`order` exige pista literal). Em cache por pergunta.

**Execução.** A referência NÃO altera a busca (operação, período, âncora,
pesos ficam os da gramática). Ela decide se a pergunta pede um tempo
(`asks_time`: operação de data pela gramática, ou `target=date` E pergunta que
começa com when/what date/which month…; uma data dentro da pergunta é escopo,
não alvo) e, nesse caso, o executor projeta os tempos do evento pedido a
partir das testemunhas selecionadas, só tempos que o próprio fato declara.
Esses tempos NÃO vão ao leitor: o contexto do leitor é exatamente o da v2.

**Reflexão.** Os tempos projetados alimentam os sinais de data do verificador.

## Reflexão (desenho final, `reflect-verify-v4`)

1. Duas leituras: com a instrução de reflexão (a leitura da v2, resposta
   padrão) e simples.
2. O verificador só é chamado quando um sinal dispara: as leituras discordam
   (duas formas do MESMO período contam como concordância), alguma se abstém,
   ou a resposta de data é um fragmento ("the 17th", "last Friday"; idade ou
   evento como "when she was 10" não é fragmento) ou não bate com nenhum tempo
   projetado. Sem sinal, a resposta é a da v2, sem chamada extra.
3. O verificador vê a pergunta, a referência, as propostas, os sinais e a mesma
   memória com `[F12]`; analisa e decide `accept`, `revise` ou `search`.
4. `search`, ou abstenção de todas as leituras, dispara UMA busca de lacuna;
   evidência unida sob o mesmo orçamento, nova leitura, segunda verificação sem
   busca.
5. Regras de proteção: o verificador não troca resposta por abstenção; não
   acrescenta qualificadores a resposta de valor único; numa pergunta sim/não
   fica só a polaridade; para tempo, escolhe QUAL tempo, e a forma da leitura
   padrão é mantida quando ela já declara o mesmo período.

Nenhuma chamada vê resposta ouro, passagem anotada ou categoria. O texto do
verificador nunca entra no leitor.

## Ablação

| variante | o que remove |
|---|---|
| `full` | nada (referência temporal + reflexão em laço) |
| `no-witness` | busca planejada; fatos ranqueados um a um (a referência declarada ainda vai à reflexão) |
| `no-time-reference` | referência declarada, ancoragem, tempos candidatos e sinais temporais; pesos de tempo zerados |
| `no-time-model` | tempos resolvidos (implica `no-time-reference`) |
| `no-reflection` | uma leitura simples: sem instrução conjunta, sem verificador, sem busca |

Uma conversa:

```bash
python scripts/run-locomo-ablation-conv.py --conversation 7 --fact-budget 20 \
  --output runs/locomo-ablation-v3d-conv07-facts20 \
  --cache runs/.cache/locomo-ablation-qwen14b \
  --before runs/standard-locomo-qwen14b/facts20 --gpu 0 --port 8096
```

O lançador imprime no início o F1 ANTES das mudanças (rodada `--before`, ou
`<baseline>/full`), compara a cada minuto nas mesmas perguntas e só roda as
demais variantes se o completo novo superar o antes (trava; `--no-gate`
desliga). Saída: `summary.md` e `summary.json` (F1 oficial por categoria, delta pareado
contra o completo, chamadas e tokens por pergunta, decisões da reflexão,
ativação da referência temporal e comparação pareada com a v2).

Testes: `tests/test_reflection_time_v3.py`.
