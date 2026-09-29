# Experimental: suficiência, um replanejamento e leitura padrão

A solução padrão continua sendo `witnessrag-local`: planos locais v2,
executor relacional, reranker MiniLM, fatos datados e reflexão na chamada
do leitor. A variante acrescenta um checker simples antes dessa leitura,
na versão `sufficiency-first-v2`. Não ativa o antigo reflector separado.

## Fluxo

1. O planner e o executor recuperam o contexto normalmente.
2. Um checker recebe somente a query e as triplas recuperadas, com IDs,
   datas de evento/sessão e origem. Não recebe os trechos literais maiores
   do reader nem produz resposta, inferências escritas ou contexto reformulado.
3. `continue` encaminha o contexto original à reflexão e ao reader padrão.
   É apropriado quando as premissas são minimamente suficientes, inclusive
   quando a resposta exige uma inferência comum ou recência.
4. `replan` identifica a lacuna, propõe até duas consultas direcionadas e
   aponta os IDs dos fatos úteis que devem ser preservados.
5. O executor refaz a busca sobre a mesma memória, guiado pelas consultas.
   O planner mantém a pergunta, a operação e o contrato temporal originais;
   a sugestão influencia busca semântica, reranking e busca de falas.
6. A reflexão e o reader padrão recebem o contexto combinado, respondendo
   ou se abstendo. Seus prompts e parâmetros são os mesmos da solução padrão;
   não recebem a decisão do checker, as sugestões nem uma resposta provisória.
   Não há uma segunda verificação ou possibilidade de replanejar.

A decisão de suficiência é uma avaliação falível do modelo, não uma prova
de que a resposta está completa. Pedidos de busca devem indicar uma lacuna
factual, relacional, temporal ou de referência; incerteza genérica e problemas
de formato são tratados pela própria leitura.

## Cache e orçamento

O cache é exclusivo daquela pergunta. Guarda apenas fatos efetivamente
entregues na primeira busca, com seus IDs, texto, fonte e informações de
tempo. A sugestão não vira uma tripla nem atualiza a memória persistente.
IDs inventados são rejeitados.

Até um quarto do orçamento pode ser reservado a premissas anteriores,
limitado a oito fatos: até cinco no orçamento 20 e até oito no orçamento 40.
O restante fica disponível para a nova busca. Duplicatas são removidas;
o contexto final continua com **no máximo 20 ou 40 fatos**, respectivamente.
As testemunhas da nova busca seguem a regra existente de entrega de
pacotes completos. As premissas retidas permanecem explicitamente não
verificadas e podem ser contraditas por evidência nova.

A chamada do checker tem teto de 384 tokens para controle. No Qwen, a
leitura final mantém o teto padrão de 128 tokens. São **duas chamadas lógicas
por pergunta**, com ou sem replanejamento: checker e reflexão/reader padrão.
O cache de LLM pode evitar novas chamadas ao provedor para prompts repetidos.
Se a nova recuperação não trouxer fatos ou falas adicionais, o reader recebe
o contexto original. JSON de controle inválido não dispara busca, mas também
segue para o reader padrão. Bloqueios por filtro continuam interrompendo a
resposta, conforme a política já existente no projeto.

## Avaliação local da versão atual

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

## Servidor

Com o vLLM Qwen ativo na GPU 7:

```bash
# Controle padrão: 20 e 40 fatos, dez conversas.
GPU=7 bash scripts/run-standard-qwen-variants.sh locomo

# Experimental: 20 e 40 fatos, no máximo um replanejamento.
GPU=7 REFLECTION_REPLAN=1 bash scripts/run-standard-qwen-variants.sh locomo
```

Saídas padrão em `runs/standard-locomo-qwen14b/facts20` e `facts40`;
experimentais em `runs/replan2-locomo-qwen14b/facts20` e `facts40`.
O cache de extração e embeddings é compartilhado. O prompt final do reader
é o padrão; contextos idênticos também podem reutilizar sua resposta em cache.
O flag faz parte da configuração de retomada; não é possível misturar padrão
e experimental na mesma pasta. Não reutilize os checkpoints antigos de
`replan1`: houve uma mudança de método, registrada no código e nos manifestos.

**A integração experimental está limitada ao LoCoMo nesta proposta.**
MemoryAgentBench mantém a solução padrão com seus formatos de saída e
juízes oficiais. A adaptação do controle aos diversos formatos desse
benchmark precisa de uma avaliação própria.

O relatório por pergunta registra decisão, lacuna, consultas, fatos retidos,
fatos novos, motivo de parada e tempo da nova busca. A contabilidade separa
`memory.sufficiency` e `qa`. Os testes verificam o limite de uma nova busca,
isolamento por pergunta, conservação do orçamento, rejeição de IDs falsos,
contrato temporal preservado e ausência de gabaritos nos prompts. Também
comparam literalmente prompt e parâmetros do reader com a solução padrão,
e confirmam que o checker não recebe os trechos grandes do reader.
