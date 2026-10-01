# Experimental: suficiência, uma nova busca e cobertura da lacuna

A versão atual é `sufficiency-gap-v5`. A solução padrão mantém planos locais v2,
executor relacional, reranking MiniLM e reflexão na chamada do reader. A variante
acrescenta controle de suficiência antes dessa leitura. Não ativa o reflector
separado das variantes de memória.

## Fluxo e responsabilidade pela relevância

1. Planner e executor recuperam a evidência inicial normalmente.
2. O checker recebe somente a pergunta e os fatos com IDs, enunciados,
   datas de evento/sessão e modalidade. Decide `continue` ou `replan`.
3. Para `continue`, o contexto original segue à reflexão/reader padrão.
   Para `replan`, o checker aponta a lacuna, até duas buscas, fatos úteis
   (`keep`) e fatos irrelevantes (`irrelevant`). Não produz resposta.
4. O executor faz no máximo uma recuperação adicional sobre a mesma memória.
   As sugestões guiam busca e reranking; a pergunta e o contrato temporal
   originais permanecem no planner.
5. Se houver candidatos novos, um segundo checker verifica se cobrem a lacuna.
   Ele recebe os fatos iniciais, os fatos candidatos, a lacuna e até quatro
   citações novas delimitadas. Seleciona fatos iniciais a reter (`retain`),
   unidades novas úteis (`use`), unidades irrelevantes (`irrelevant`) e
   citações novas necessárias (`sources`). Pode reconsiderar a classificação
   inicial de irrelevância quando aparecer uma ligação nova.
6. Somente uma seleção válida com cobertura confirmada substitui o contexto.
   Sem cobertura ou com controle inválido, o contexto original é preservado.
   A reflexão/reader padrão então executa com os mesmos prompts e parâmetros.

A relevância e a cobertura são julgadas pelo modelo. Não há limiar de
similaridade para descartar fatos nem divisão fixa de 12 antigos/8 novos na
v5. A decisão é falível; `covered=true` não constitui uma prova lógica.
Nas respostas negativas do segundo checker, as listas são vazias: esse caminho
registra ausência de cobertura, não classifica cada candidato como irrelevante.

## Validação, cache e custo

O código valida somente controles estruturais: IDs existentes, tipos,
conflitos de seleção, pacotes completos, novidade e orçamento. A união final
contém no máximo **20 fatos únicos** neste experimento. Um pacote novo é
indivisível; não é cortado para caber. Citações adicionais precisam ter sido
mostradas ao checker e efetivamente entregues ao reader. Se uma citação
selecionada não couber no orçamento literal, a seleção inteira é rejeitada
em favor do contexto original. Sugestões nunca viram fatos ou respostas.

A memória retida é exclusiva da pergunta. Não altera a memória persistente.
O checker inicial usa somente os fatos compactos; o checker de cobertura
acrescenta citações limitadas para avaliar informação ausente das triplas.
As duas chamadas de controle têm teto de 384 tokens de saída cada. O reader
Qwen mantém o teto padrão de 128. São duas chamadas lógicas quando não há
candidatos novos e três quando se verifica cobertura após a nova busca.
Hits de cache são registrados e podem evitar execução física de geração.
Bloqueios por filtro preservam a política existente de interrupção da resposta.

## Teste Qwen da v5: primeira conversa, 20 fatos

Foram avaliadas todas as 152 perguntas da conversa zero, sem seleção por F1.
A evidência inicial e a extração foram congeladas a partir da run padrão de
20 fatos. A recuperação adicional usou BGE-M3 e MiniLM na GPU 7 e o mesmo
Qwen2.5-14B-Instruct. Gabaritos foram usados somente na avaliação.

| F1 oficial nas mesmas perguntas | Padrão 20 | Replan v2 | Replan v4 | Replan v5 |
|---|---:|---:|---:|---:|
| Todas, 152 perguntas | 59,42 | 58,66 | 57,21 | 59,42 |
| Multi-hop, 32 perguntas | 53,75 | 50,62 | 50,00 | 53,75 |
| Temporal, 37 perguntas | 65,27 | 65,99 | 62,33 | 65,27 |

Houve 25 buscas adicionais. Nenhuma teve cobertura confirmada: 24 avaliações
negativas e uma saída de controle inválida. Cinco controles iniciais também
foram inválidos. Todos os contextos finais foram preservados; todos os F1
individuais coincidem com a padrão. O checker inicial anotou 239 IDs como
irrelevantes ao longo das perguntas que solicitaram busca, contando repetições
entre perguntas. Essas anotações não são validação humana de irrelevância.

O total lógico de entrada e saída foi 565.411 tokens, contra 331.093 da padrão
(+70,8%) e 514.684 da v4 (+9,9%). Foram 329 chamadas lógicas, incluindo 25 de
cobertura; 245 foram atendidas pelo cache. Extração histórica não foi cobrada
novamente. Os totais não medem tempo físico de GPU nem custo monetário.

A v5 evitou as regressões da v4 nesta conversa, mas não demonstrou melhoria
de recuperação ou economia. O teste real exercitou apenas rejeição de novidades;
a aceitação positiva foi verificada em testes unitários. É necessário investigar
os candidatos e os julgamentos negativos antes de recomendar esta variante
como padrão. Relatório: `runs/replan5-gap-conv00/report.md`.

A run completa v4 foi cancelada a pedido do usuário. Seus resultados parciais
foram preservados; nenhum benchmark ficou em execução após concluir este teste.

## Histórico: piloto Qwen da v4 (20 fatos)

123 perguntas selecionadas nas dez conversas: 60 com F1 zero, 60 controles
perfeitos e três casos adicionais de regressão (podem ter F1 parcial).
É uma amostra de desenvolvimento escolhida por resultados históricos;
os valores abaixo não representam o F1 nas 1.540 perguntas.

| Métrica nas mesmas perguntas | Padrão 20 | Replan v2 de 20 | Replan v4 de 20 |
|---|---:|---:|---:|
| F1 oficial, 123 perguntas | 51,02 | 48,87 | 52,61 |
| F1 multi-hop, 21 perguntas | 36,90 | 24,46 | 34,52 |

A v4 melhora sobre o replan antigo nesta amostra, mas ainda fica abaixo da
padrão no multi-hop. Foram 40 buscas adicionais, nenhum controle inválido,
seis aumentos e duas quedas de F1 em relação à padrão. Nos retries, 13,45 fatos
iniciais foram preservados em média. O custo lógico foi 422.107 tokens contra
267.880 da padrão (+57,6%); extração histórica reutilizada e hits de cache
contabilizados separadamente. Não equivale a tempo físico de GPU.

O caso `conv-26:qa11` conserva o fato sobre Sweden e a fala literal que
identifica o país de origem, porém o Qwen ainda responde `her home country`.
A preservação do contexto corrige um mecanismo de perda sem garantir
inferência correta do leitor. O relatório detalhado fica em
`runs/replan4-analysis-20260930/report.md`, com artefatos da v4 separados do primeiro
protótipo de união da v3.

## Avaliação local da versão v2

O reteste usa as mesmas seis perguntas com F1 zero e seis controles perfeitos
do piloto anterior: GPT-4o-mini, CPU, primeira conversa, orçamento 40.
As triplas históricas e a evidência inicial são verificadas e reutilizadas.
Somente a eventual busca adicional percorre o grafo novamente.
Os resultados validados ficam em `runs/reflection-replan-sufficiency-v2-final/`.

| Grupo | n | F1 histórico | F1 atual | Novas buscas |
|---|---:|---:|---:|---:|
| Erros selecionados | 6 | 0,00 | 18,32 | 4 |
| Controles antes perfeitos | 6 | 100,00 | 100,00 | 2 |

Na pergunta sobre a reunião de adoção, o F1 passou de zero para 95,65 após a
nova recuperação. Há também um crédito lexical de 14,29 na pergunta sobre
pertencimento à comunidade LGBTQ: o reader respondeu `likely yes`, enquanto
o gabarito indica `likely no`. Esse crédito de F1 **não representa uma resposta
semanticamente correta**. Os seis controles mantiveram F1 100, mas dois deles
receberam uma busca adicional que não melhorou o resultado. A releitura do
contexto histórico, com o mesmo reader e sem nova busca, preservou os scores
históricos nas seis perguntas replanejadas.

O checker recebeu em média **2.482 tokens de entrada** com 40 fatos; seu prompt
contém 83 palavras de instruções/formato e 22 no sistema, além da query e
das triplas datadas. Nas 12 perguntas, são 24 chamadas lógicas (12 checker e
12 reader), **61.688 tokens de entrada** e 1.038 de saída. A referência usa
12 chamadas, 30.893 tokens de entrada e 123 de saída: aumento lógico de entrada
de **99,7%**. Os totais incluem respostas reutilizadas do cache; a repetição
final foi integralmente atendida pelo cache, após o primeiro piloto pago.
A extração histórica foi reutilizada e não entra como novo custo de inferência.
Esse teste verifica o novo fluxo, sem demonstrar economia ou ganho no benchmark
completo. Não houve teste real com Qwen ou GPU nesta avaliação local.

## Histórico: piloto da versão anterior

Os números abaixo pertencem à versão `joint-replan-v1`, que combinava
suficiência, reflexão e resposta provisória na primeira chamada. Essa versão
foi substituída; seus resultados não descrevem a implementação atual.

Modelo **GPT-4o-mini**, CPU, primeira conversa do LoCoMo, orçamento 40.
Foram selecionadas antes da geração as primeiras seis perguntas com F1
oficial zero e seis com F1 oficial 1, na ordem da run salva
`local-plans-v2-gpt4omini-all`.

A extração exata da run original foi verificada e reutilizada. O primeiro
contexto foi reproduzido literalmente; apenas o replanejamento executou
uma nova busca no grafo completo. Gabaritos e F1 serviram para seleção e
avaliação, sem entrar nos prompts, no cache ou nas sugestões de busca.

O primeiro piloto revelou respostas excessivamente longas e necessidade de
mostrar datas de origem nas falas. Após corrigir essas duas questões gerais,
o reteste ficou em `runs/reflection-replan-development-v2/`:

| Grupo | n | F1 histórico | F1 da primeira reflexão | F1 final |
|---|---:|---:|---:|---:|
| Erros selecionados | 6 | 0,00 | 33,33 | 38,10 |
| Controles antes perfeitos | 6 | 100,00 | 92,06 | 92,06 |

Dois erros passaram a F1 1 já na primeira reflexão. Dois pedidos de
replanejamento foram executados; um deles mudou a resposta de
`insufficient information` para `likely no`, com F1 0,29. Uma ablação
releu a evidência inicial com o mesmo contrato final, sem recuperar mais
memórias: nesse caso continuou com F1 zero. A outra nova busca não
melhorou a resposta. Quatro dos seis erros seguem sem acerto completo.

Dois controles perderam F1 por diferenças de redação. Este resultado é
de desenvolvimento, sobre uma amostra selecionada por erros e usada para
depuração; não demonstra ganho no benchmark completo ou no Qwen.

Nas mesmas 12 perguntas, as chamadas passaram de 12 para 14; os tokens de
entrada, de **30.893 para 56.768** (+83,8%); os de saída, de **123 para 407**.
Isso inclui o catálogo para retenção, instruções de controle e os dois
contextos adicionais. A ablação de releitura foi registrada separadamente.
A extração histórica não foi cobrada novamente. A variante não demonstrou
economia de tokens; sua justificativa atual é explorar recuperação adaptativa
com custo limitado, preservando a padrão como controle.

Para reproduzir o piloto com a run e o cache locais correspondentes:

```powershell
& "C:\Users\rodri\anaconda3\python.exe" scripts/test-reflection-replan.py `
  --output runs/reflection-replan-sufficiency-novo --resume

& "C:\Users\rodri\anaconda3\python.exe" scripts/test-reflection-replan.py `
  --output runs/reflection-replan-sufficiency-novo --reader-control
```

## Servidor: reproduzir a primeira conversa

Com o vLLM Qwen ativo em `http://127.0.0.1:8095/v1`, execute uma pasta nova:

```bash
mkdir -p runs
OUTPUT_DIR="$PWD/runs/replan5-gap-conv00-repeat/facts20" \
  nohup bash scripts/run-replan-gap-qwen20.sh > runs/replan5-gap-conv00-repeat.log 2>&1 < /dev/null &
tail -f runs/replan5-gap-conv00-repeat.log
```

O lançador executa somente a conversa zero, todas as perguntas, orçamento 20.
Usa GPU 7 para embeddings/reranking (`cuda:0` dentro do processo), verifica o
modelo servido e a fonte de 20 fatos e impede dois escritores na mesma saída.
`--resume` verifica modelo, versão, código, configuração e IDs selecionados.
Para retomar, use a mesma pasta e o mesmo código. `SOURCE_RUN`, `CACHE_DIR`,
`OUTPUT_DIR`, `BENCH_PYTHON`, `MODEL` e `GPU` permitem caminhos/configurações
explícitos. O resultado concluído no servidor está em
`runs/replan5-gap-conv00-final/facts20`; não misture outras versões nessa pasta.

O teste concluído usou código isolado em `.audit-code/replan-gap-v5-final`.
Os históricos v2/v4 e o protótipo v5 interrompido permanecem separados.
Os lançadores `run-replan-frozen-qwen20.sh` são históricos e seus nomes de
saída v4 não identificam a versão carregada: o módulo Python determina a versão.

**A integração experimental está limitada ao LoCoMo nesta proposta.**
MemoryAgentBench mantém a solução padrão e precisa de avaliação própria
para adaptar este controle aos seus formatos de saída e juízes oficiais.

Os traces registram decisões, lacunas, consultas, irrelevância, candidatos,
seleção, novidade e motivo de parada. A contabilidade separa
`memory.sufficiency`, `memory.gap_coverage` e `qa`. Os testes verificam
orçamento, pacotes completos, rejeição de IDs falsos, citações selecionadas,
fallback exato e ausência de gabaritos nos prompts. O reader permanece
idêntico ao da solução padrão.
