# Ablação de reflexão no cascade

Versão 2, de 28/09/2026, após a auditoria da primeira rodada. Mantém exatamente quatro células. Todas usam cascade, Qwen2.5-14B e os planos robustos existentes. O grafo, a recuperação, as rotas, os fatos e os chunks de resumo são congelados e compartilhados entre os processos. A qualidade dos novos prompts deve ser medida nesta rodada; os testes locais verificam funcionamento e controle experimental, não demonstram ganho no benchmark.

Correções comuns às quatro células: espaços entre itens separados por vírgula, namespace de cache independente da porta, gravações atômicas e locks entre processos. O baseline é reexecutado com essas correções comuns. As novas saídas ficam em `runs/cascade-reflection-ablation-v2`; a primeira rodada fica preservada.

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

### Dividir entre duas GPUs

Com servidores do mesmo Qwen/configuração nas portas 8095 e 8096, usando GPUs físicas 1 e 7, execute os dois comandos na mesma raiz do repositório, com a mesma pasta de saída e o mesmo mapeamento completo:

```bash
# GPU 1 / porta 8095: cascade e reader-reflection
nohup bash scripts/run-reflection-ablation-qwen.sh runs/cascade-reflection-ablation-v2 \
  --ports 8095,8096 --gpus 1,7 --only cascade,reader-reflection \
  > reflection-v2-gpu1.log 2>&1 &

# GPU 7 / porta 8096: summary-reflection e both
nohup bash scripts/run-reflection-ablation-qwen.sh runs/cascade-reflection-ablation-v2 \
  --ports 8095,8096 --gpus 1,7 --only summary-reflection,both \
  > reflection-v2-gpu7.log 2>&1 &
```

Os scripts reaproveitam os servidores; não os iniciam. Se necessário, inicie antes um servidor com `GPU=1 PORT=8095` e outro com `GPU=7 PORT=8096`, usando `scripts/serve-qwen-vllm.sh` em terminais separados.

A primeira construção de cada grafo acontece uma vez; o outro processo aguarda o lock dessa conversa. Depois as perguntas podem avançar em paralelo. O primeiro processo que recupera uma pergunta salva seu contexto literal; os demais o reutilizam. As inferências são acrescentadas a esse contexto e também congeladas para summary/both. As chamadas de QA continuam no servidor/GPU de cada célula. `--only` distribui as quatro células, sem criar outras variantes.

O último processo gera os relatórios combinados. Para recalculá-los manualmente depois que ambos terminarem:

```bash
bash scripts/run-reflection-ablation-qwen.sh runs/cascade-reflection-ablation-v2 \
  --ports 8095,8096 --gpus 1,7 --report-only
```

Repita os mesmos comandos para retomar uma interrupção. Não altere código, modelo, GPUs/portas ou parâmetros dentro dessa pasta de estudo. Não use a pasta da v1 para a v2.

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
bash scripts/run-reflection-ablation-qwen.sh runs/cascade-reflection-ablation-v2 --report-only
```

`--questions` seleciona uma amostra por conversa com a semente do piloto, não necessariamente uma amostra de open-domain. A rodada curta verifica a integração. A rodada completa mede todas as categorias e as possíveis regressões.

## Retomada e saídas

Repita o mesmo comando para retomar. Rodadas concluídas são preservadas; rodadas parciais recebem `--resume` do piloto. O estudo registra comandos, fatores e hashes do código em `ablation-manifest.json`. Mudanças de configuração ou código exigem outra pasta de saída, evitando misturar protocolos. Falhas interrompem a sequência, sem imprimir um estudo incompleto como concluído.

Saída padrão: `runs/cascade-reflection-ablation-v2/`.

- `cascade/`, `summary-reflection/`, `reader-reflection/`, `both/`: runs e logs individuais, com o formato já usado pelo projeto;
- `compare.md` / `compare.json`: comparação pareada padrão;
- `reflection-ablation.md` / `.json`: F1, BLEU-1 local e EM gerais e por categoria, deltas e IC95%, efeitos condicionais e interação;
- `reflection-retrieval-control.json`: auditoria de rotas e contextos efetivos;
- `corpus-hashes.json`: conferência de que as variantes usam os mesmos corpora.
- `shared/controlled/`: extração, grafo e vetores congelados, com checksum;
- `shared/retrieval/`: contexto literal por pergunta e contexto acrescido de inferências;
- `shared/summaries/`, `shared/reflections/`: memórias por chunk compartilhadas;
- locks do manifesto, de cada worker e de cada snapshot: evitam duplicação/gravação simultânea.

O relatório verifica as mesmas perguntas/ouro, rota PLAN/DIRECT, texto dos fatos, seleção dos chunks e hashes da memória/recuperação entre todas as células. Confere também resumos idênticos entre baseline/reader e entre resumos/ambos, e preservação integral dos resumos literais nas células enriquecidas. Se essas condições falharem, registra exemplos e recusa a comparação controlada.

A interação é `F1(ambos) − F1(resumos) − F1(reader) + F1(baseline)`. Ela mostra se os fatores se complementam ou se ajudam nos mesmos casos. O bootstrap reamostra conversas; com poucas conversas, os intervalos são exploratórios.

## Fator 1: memórias de alto nível nos resumos

O resumo factual original é preservado integralmente. Uma chamada adicional por chunk gera até quatro interpretações úteis e distintas do diálogo: preferências, objetivos e implicações, priorizando pontes específicas de classificação e reconhecimento. Há exemplos sintéticos externos ao benchmark. O máximo pode variar de um a seis pelo parâmetro do estudo.

Essa chamada recebe somente o texto da conversa. Não recebe pergunta, ouro, categoria ou evidência anotada. É construída na primeira utilização do chunk e armazenada em cache; as duas células reaproveitam o mesmo resultado. A seleção dos chunks é congelada a partir do cascade. Entre suas interpretações, o reranker existente seleciona individualmente até quatro memórias por pergunta, em vez de entregar todas as interpretações dos seis chunks. `--reflection-limit` controla o teto por chunk e o teto total entregue por pergunta; não aumenta o número de variantes.

Cada interpretação inclui sujeito, inferência, confiança qualitativa (`likely`/`possible`), eventual ponte de conhecimento comum e uma ou duas citações com IDs de fala. O código exige que a citação pertença à fala indicada e descarta referências desconhecidas, paráfrases apresentadas como citações, formato inválido e duplicatas.

Essa validação comprova a origem das premissas, **não a verdade da interpretação**. As memórias são rotuladas como interpretações tentativas; não são inseridas no grafo como observações, não viram átomos de prova e não substituem os fatos originais. Interpretações inválidas são descartadas. Falhas de transporte/filtro interrompem o estudo para permitir retomada, sem congelar um contexto incompleto como resultado válido.

Os blocos são acrescentados aos resumos finais produzidos por `_fact_context`, tanto em DIRECT quanto em PLAN. O teste de integração confirma que chegam ao contexto final do reader.

## Fator 2: reflexão conjunta no reader

O reader atual produz a resposta final. A opção nova acrescenta a inferência e a revisão da resposta na **mesma chamada**, usando o mesmo contexto recuperado e o mesmo teto de saída configurado. A instrução pede:

1. identificar operação, tipo esperado, opções, tempo e qualificadores;
2. separar observações pessoais, interpretações e conhecimento de mundo;
3. aplicar a ponte necessária quando a conclusão é implícita;
4. confrontar a conclusão com contraevidências e requisitos da pergunta;
5. entregar somente a resposta adequada à operação solicitada.

A reflexão fica interna. A v2 volta ao JSON simples `{"answer":"..."}`; `answer_kind` não é obrigatório. A instrução de revisão fica depois das evidências, próxima da pergunta, e usa exemplos sintéticos de duração com unidade, escolha e país. O leitor deve preservar a menor resposta completa, evitar trocar idade por ano, devolver a opção de uma escolha e usar a ponte necessária quando uma premissa ainda não responde. `reflexao_leitor.mode=joint-v2` identifica a versão; `schema_valid` verifica o formato, não certifica a semântica. A normalização do reader reflexivo preserva opções começando com “Yes”. A correção de espaços nas listas é comum a todos.

Não há seleção por categoria, regras com respostas do LoCoMo, busca web, outro modelo ou segunda chamada de crítica. O objetivo experimental é verificar se o mesmo Qwen usa melhor as premissas e o conhecimento comum que já possui.

## Interpretar os resultados

Os três contrastes principais são resumos versus baseline, reader versus baseline e ambos versus baseline. Os contrastes condicionais ajudam a entender onde ocorre o efeito: reader com/sem memórias inferidas e memórias inferidas com/sem reader reflexivo.

Os resumos enriquecidos ocupam mais contexto. Por isso, o relatório separa tokens do reader, tokens de inferência e tokens da construção das memórias (`memory.summary` e `memory.reflection`), e informa quantas interpretações foram entregues. Um ganho do fator resumos mede o pacote de enriquecimento; não isola qualidade da inferência de tamanho adicional de contexto. Para sustentar essa distinção num artigo, caberia depois um controle de resumo factual com comprimento equivalente.

As contagens de tokens são lógicas, inclusive para respostas atendidas pelo cache; não representam automaticamente custo físico adicional. Na v2, o total por célula inclui a recuperação compartilhada mais seu reader. A recuperação é construída uma vez para as quatro células; não somar seu custo real quatro vezes. Seu uso original é registrado no snapshot compartilhado; geração dos resumos/inferências permanece separada. O custo da extração/indexação anterior às perguntas também aparece nos logs.

Nenhuma mudança extra foi aplicada ao controlador para este estudo. As duas opções estão desligadas por padrão e são ativadas somente nas células correspondentes. As conversas/perguntas usadas para ajustar os novos prompts devem ser declaradas como desenvolvimento; resultados nesses mesmos casos não demonstram generalização independente.

## Arquivos

- `scripts/run-reflection-ablation-qwen.sh`: comando do servidor;
- `scripts/run-reflection-ablation.py`: configurações, manifesto e retomada;
- `scripts/reflection-ablation-report.py`: comparação dos quatro fatores e controles;
- `wrag/witness/reflection.py`: geração, validação de origem e apresentação das memórias;
- `wrag/prompts.py`, `wrag/eval/reader.py`: reflexão conjunta opcional;
- `tests/test_reflection_ablation.py`: testes sem modelo ou API.
- `wrag/eval/reflection_study.py`: compartilhamento de grafo/contextos e locks;
- `tests/test_reflection_study.py`: dois processos, retomada, corrupção de snapshots, quatro células e mapeamento das GPUs.
