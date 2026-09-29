# Proposta experimental: reflexão conjunta com um replanejamento

A solução padrão continua sendo `witnessrag-local`: planos locais v2,
executor relacional, reranker MiniLM, fatos datados e reflexão na chamada
do leitor. A variante abaixo acrescenta controle de recuperação nessa
mesma chamada. Não ativa o antigo experimento com reflector separado.

## Fluxo

1. O planner e o executor recuperam o contexto normalmente.
2. O leitor faz a reflexão e devolve uma resposta curta provisória, mais
   uma decisão: `continue` ou `replan`.
3. `continue` encerra a pergunta. É apropriado quando as premissas são
   minimamente suficientes, inclusive para uma inferência justificável.
4. `replan` identifica a lacuna, propõe até duas consultas direcionadas e
   aponta os IDs dos fatos úteis que devem ser preservados.
5. O executor refaz a busca sobre a mesma memória, guiado pelas consultas.
   O planner mantém a pergunta, a operação e o contrato temporal originais;
   a sugestão influencia busca semântica, reranking e busca de falas.
6. Um leitor final recebe o contexto combinado e responde ou se abstém.
   Não há uma segunda possibilidade de pedir replanejamento.

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

A primeira chamada tem teto de 384 tokens para resposta e controle. No
Qwen, a leitura final mantém o teto padrão de 128 tokens. São uma ou duas
chamadas generativas por pergunta. Se a nova recuperação não trouxer fatos
ou falas adicionais, o controlador preserva a primeira resposta e evita
gastar a segunda chamada de leitura. JSON de controle inválido também não
dispara uma busca.

## Piloto local

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
  --output runs/reflection-replan-novo --resume

& "C:\Users\rodri\anaconda3\python.exe" scripts/test-reflection-replan.py `
  --output runs/reflection-replan-novo --reader-control
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
experimentais em `runs/replan1-locomo-qwen14b/facts20` e `facts40`.
O cache de extração e embeddings é compartilhado, com prompts de leitura
e checkpoints distintos. O flag faz parte da configuração de retomada;
não é possível misturar padrão e experimental na mesma pasta.

**A integração experimental está limitada ao LoCoMo nesta proposta.**
MemoryAgentBench mantém a solução padrão com seus formatos de saída e
juízes oficiais. A adaptação do controle aos diversos formatos desse
benchmark precisa de uma avaliação própria.

O relatório por pergunta registra decisão, lacuna, consultas, fatos retidos,
fatos novos, motivo de parada e tempo da nova busca. A contabilidade separa
`qa.replan_gate` e `qa`. Os testes verificam o limite de uma nova busca,
isolamento por pergunta, conservação do orçamento, rejeição de IDs falsos,
contrato temporal preservado e ausência de gabaritos nos prompts.
