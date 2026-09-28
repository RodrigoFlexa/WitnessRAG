# Ablação de reflexão no cascade

Implementação experimental de 28/09/2026. Todas as células usam o cascade atual, Qwen2.5-14B, o mesmo grafo de observações, os mesmos planos robustos, roteador, seleção de fatos, reranker e datas. A qualidade das novas opções deve ser medida nesta rodada; os testes locais verificam funcionamento e controle experimental, não demonstram ganho no benchmark.

| Pasta / variante | Inferência nos resumos | Reflexão no reader |
|---|---|---|
| `cascade` | não | não |
| `summary-reflection` | sim | não |
| `reader-reflection` | não | sim |
| `both` | sim | sim |

O baseline é reexecutado no próprio estudo. Os resultados antigos não entram como uma quinta célula: poderiam ter outro código, corpus ou contexto.

## Rodar no servidor

Na raiz do repositório, com o Qwen já servido em `http://127.0.0.1:8095`:

```bash
bash scripts/run-reflection-ablation-qwen.sh
```

Esse comando roda as quatro variantes nas dez conversas, em sequência, e gera os relatórios. A configuração padrão é GPU física `1`, embeddings/reranker em CUDA, `Qwen/Qwen2.5-14B-Instruct`, chunks de 2.048 tokens, janelas de extração de 512, 40 fatos, seis resumos e reader com teto de saída configurado de 128 tokens. O corpus permanece restrito à própria conversa; as perguntas das categorias 1 a 4 são avaliadas. Os rótulos são usados no relatório, não nas novas inferências ou na decisão do leitor.

Para iniciar o servidor, se necessário, em outro terminal:

```bash
GPU=1 PORT=8095 bash scripts/serve-qwen-vllm.sh
```

Para deixar o estudo rodando após sair da sessão SSH:

```bash
GPU=1 PORT=8095 nohup bash scripts/run-reflection-ablation-qwen.sh > reflection-ablation.log 2>&1 &
```

O script procura `.venv-bench/bin/python`, `.venv/bin/python` e `venv/bin/python`; `BENCH_PYTHON` permite escolher outro ambiente. O ambiente do benchmark deve ter as dependências que o cascade já usa. O servidor vLLM é reaproveitado; o script não inicia quatro cópias do modelo.

Exemplos de controle:

```bash
# Validar as quatro configurações sem chamar o servidor:
bash scripts/run-reflection-ablation-qwen.sh runs/reflection-preflight --dry-run

# Rodada curta para verificar a integração antes da rodada completa:
LOCOMO_CONVERSATION=5 bash scripts/run-reflection-ablation-qwen.sh runs/reflection-smoke --questions 8

# Outra GPU/porta, mantendo as quatro células comparáveis:
GPU=0 PORT=8096 bash scripts/run-reflection-ablation-qwen.sh runs/reflection-gpu0

# Lista de conversas; ajustar o máximo de memórias por resumo:
bash scripts/run-reflection-ablation-qwen.sh runs/reflection-dev --conversation 0,3,5 --reflection-limit 3

# Apenas reconstruir o relatório de um estudo completo:
bash scripts/run-reflection-ablation-qwen.sh runs/cascade-reflection-ablation --report-only
```

`--questions` seleciona uma amostra por conversa com a semente do piloto, não necessariamente uma amostra de open-domain. A rodada curta verifica a integração. A rodada completa mede todas as categorias e as possíveis regressões.

## Retomada e saídas

Repita o mesmo comando para retomar. Rodadas concluídas são preservadas; rodadas parciais recebem `--resume` do piloto. O estudo registra comandos, fatores e hashes do código em `ablation-manifest.json`. Mudanças de configuração ou código exigem outra pasta de saída, evitando misturar protocolos. Falhas interrompem a sequência, sem imprimir um estudo incompleto como concluído.

Saída padrão: `runs/cascade-reflection-ablation/`.

- `cascade/`, `summary-reflection/`, `reader-reflection/`, `both/`: runs e logs individuais, com o formato já usado pelo projeto;
- `compare.md` / `compare.json`: comparação pareada padrão;
- `reflection-ablation.md` / `.json`: F1, BLEU-1 local e EM gerais e por categoria, deltas e IC95%, efeitos condicionais e interação;
- `reflection-retrieval-control.json`: auditoria de rotas e contextos efetivos;
- `corpus-hashes.json`: conferência de que as variantes usam os mesmos corpora.

O relatório verifica as mesmas perguntas/ouro, rota PLAN/DIRECT e texto dos fatos entre todas as células. Confere também resumos idênticos entre baseline/reader e entre resumos/ambos. Se essas condições falharem, registra exemplos e recusa a comparação controlada: é preciso entender a divergência antes de atribuir o efeito à reflexão.

A interação é `F1(ambos) − F1(resumos) − F1(reader) + F1(baseline)`. Ela mostra se os fatores se complementam ou se ajudam nos mesmos casos. O bootstrap reamostra conversas; com poucas conversas, os intervalos são exploratórios.

## Fator 1: memórias de alto nível nos resumos

O resumo factual original é preservado. Ao construir o resumo de um chunk, uma chamada adicional gera até quatro interpretações úteis e distintas do diálogo: preferências, objetivos, motivações e implicações, incluindo reconhecimento de conceitos a partir de características. O máximo pode variar de um a seis pelo parâmetro do estudo.

Essa chamada recebe somente o texto da conversa. Não recebe pergunta, ouro, categoria ou evidência anotada. É construída na primeira utilização do chunk e armazenada em cache; as duas células com esse fator reaproveitam o mesmo prompt e resultado. A seleção dos chunks continua sendo a do cascade atual: o estudo altera o conteúdo dos resumos entregues, sem adicionar outro índice de recuperação.

Cada interpretação inclui sujeito, inferência, confiança qualitativa (`likely`/`possible`), eventual ponte de conhecimento comum e uma ou duas citações com IDs de fala. O código exige que a citação pertença à fala indicada e descarta referências desconhecidas, paráfrases apresentadas como citações, formato inválido e duplicatas.

Essa validação comprova a origem das premissas, **não a verdade da interpretação**. As memórias são rotuladas como interpretações tentativas; não são inseridas no grafo como observações, não viram átomos de prova e não substituem os fatos originais. Falhas de geração/validação preservam o resumo factual e são registradas.

Os blocos são acrescentados aos resumos finais produzidos por `_fact_context`, tanto em DIRECT quanto em PLAN. O teste de integração confirma que chegam ao contexto final do reader.

## Fator 2: reflexão conjunta no reader

O reader atual produz a resposta final. A opção nova acrescenta a inferência e a revisão da resposta na **mesma chamada**, usando o mesmo contexto recuperado e o mesmo teto de saída configurado. A instrução pede:

1. identificar operação, tipo esperado, opções, tempo e qualificadores;
2. separar observações pessoais, interpretações e conhecimento de mundo;
3. aplicar a ponte necessária quando a conclusão é implícita;
4. confrontar a conclusão com contraevidências e requisitos da pergunta;
5. entregar somente a resposta adequada à operação solicitada.

A reflexão fica interna. A resposta estruturada tem `answer_kind` e `answer`; apenas `answer` vai para F1/BLEU/EM. O tipo é registrado em `reflexao_leitor`, incluindo se o schema foi atendido. Isso também impede que a normalização de sim/não trunque uma opção cujo nome começa com “Yes”. A normalização original é mantida no baseline.

Não há seleção por categoria, regras com respostas do LoCoMo, busca web, outro modelo ou segunda chamada de crítica. O objetivo experimental é verificar se o mesmo Qwen usa melhor as premissas e o conhecimento comum que já possui.

## Interpretar os resultados

Os três contrastes principais são resumos versus baseline, reader versus baseline e ambos versus baseline. Os contrastes condicionais ajudam a entender onde ocorre o efeito: reader com/sem memórias inferidas e memórias inferidas com/sem reader reflexivo.

Os resumos enriquecidos ocupam mais contexto. Por isso, o relatório separa tokens do reader, tokens de inferência e tokens da construção das memórias (`memory.summary` e `memory.reflection`), e informa quantas interpretações foram entregues. Um ganho do fator resumos mede o pacote de enriquecimento; não isola qualidade da inferência de tamanho adicional de contexto. Para sustentar essa distinção num artigo, caberia depois um controle de resumo factual com comprimento equivalente.

As contagens de tokens são lógicas, inclusive para respostas atendidas pelo cache; não representam automaticamente custo físico adicional. O custo da extração/indexação anterior às perguntas também aparece nos logs da rodada.

Nenhuma mudança extra foi aplicada ao controlador para este estudo. As duas opções estão desligadas por padrão e são ativadas somente nas células correspondentes. As conversas/perguntas usadas para ajustar os novos prompts devem ser declaradas como desenvolvimento; resultados nesses mesmos casos não demonstram generalização independente.

## Arquivos

- `scripts/run-reflection-ablation-qwen.sh`: comando do servidor;
- `scripts/run-reflection-ablation.py`: configurações, manifesto e retomada;
- `scripts/reflection-ablation-report.py`: comparação dos quatro fatores e controles;
- `wrag/witness/reflection.py`: geração, validação de origem e apresentação das memórias;
- `wrag/prompts.py`, `wrag/eval/reader.py`: reflexão conjunta opcional;
- `tests/test_reflection_ablation.py`: testes sem modelo ou API.
