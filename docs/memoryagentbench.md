# MemoryAgentBench no WitnessRAG

A integração está em `benchmarks/memoryagentbench/`, com comandos em
`scripts/run-memoryagentbench-openai.ps1`. Ela usa a solução padrão da execução
`local-plans-v2-gpt4omini-all`: extração `memory`, planner local v2, executor
relacional, reranker MiniLM, 40 fatos e reflexão conjunta no leitor. Não usa a
variante experimental de reflector separado com 10/20 fatos.

Referências: [artigo e código oficial](https://github.com/HUST-AI-HYZ/MemoryAgentBench),
[dataset oficial](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench).
Código fixado em `538026089d1a8a8eff05121d0db89b388f360eba`; dados fixados em
`7ea066982b140a19337e17e60d45d4076e042faf`. Os checksums dos cinco arquivos são
verificados antes das execuções. Prompts e configs oficiais foram incorporados
com a licença MIT em `benchmarks/memoryagentbench/vendor/`.

Para Qwen2.5-14B em vLLM, execução completa em GPU e comparação com 20/40
fatos, veja [servidor-qwen.md](servidor-qwen.md). A CLI aceita `run --backend
vllm --base-url http://127.0.0.1:8095/v1 --model Qwen/Qwen2.5-14B-Instruct`.

## Preparar e conferir os dados

Execute na raiz do WitnessRAG. O ambiente já usado para LoCoMo contém as
dependências; para um ambiente novo, instale `requirements.txt` e
`requirements-memoryagentbench.txt`. A preparação baixa aproximadamente 75 MB
de Parquet e o tokenizador de sentenças NLTK. Não faz chamadas a modelos.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-memoryagentbench-openai.ps1 `
  -Python "C:\Users\rodri\anaconda3\python.exe" `
  -PrepareOnly
```

Para validar todas as perguntas e a fragmentação de todos os contextos:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-memoryagentbench-openai.ps1 `
  -Python "C:\Users\rodri\anaconda3\python.exe" `
  -Protocol paper -Suite all -InspectOnly
```

O release contém **146 contextos e 3.671 perguntas**, em quatro competências.
`-Suite all` inclui todas as tarefas publicadas, as cinco classificações e as
variações de comprimento. `-Suite paper` seleciona as dez colunas da Tabela 3:
**126 contextos e 1.671 perguntas**. As médias principais usam apenas essas dez
colunas; as outras tarefas ficam separadas no relatório.

## Piloto da solução padrão

O piloto abaixo registra todo o contexto de 6K e responde às primeiras cinco
perguntas de resolução de conflitos. Limitar perguntas não reduz a memória.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-memoryagentbench-openai.ps1 `
  -Python "C:\Users\rodri\anaconda3\python.exe" `
  -Model gpt-4o-mini `
  -Protocol paper `
  -Sources factconsolidation_sh_6k `
  -MaxQuestions 5 `
  -EmbedDevice auto `
  -Output runs\memoryagentbench-standard-pilot
```

`auto` usa CUDA quando o PyTorch detecta uma GPU, senão CPU. `cuda` exige CUDA e
falha se ela não estiver disponível. O reranker usa o mesmo dispositivo. Isso é
registrado no manifesto; não há mudança silenciosa para TF-IDF. O teste desta
integração encontrou `torch.cuda.is_available() == False` no Python informado.

## Rodar o benchmark completo

Todas as 3.671 perguntas, incluindo os avaliadores externos:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-memoryagentbench-openai.ps1 `
  -Python "C:\Users\rodri\anaconda3\python.exe" `
  -Model gpt-4o-mini `
  -Protocol paper `
  -Suite all `
  -EmbedDevice auto `
  -Judge `
  -Output runs\memoryagentbench-standard-paper
```

Para medir somente as tarefas principais da Tabela 3, substitua `-Suite all`
por `-Suite paper` e use outra pasta de saída.

Cada contexto constrói uma memória isolada, ingerindo os chunks na ordem
oficial; cada chamada de extração vê somente o chunk atual. A consolidação do
grafo ocorre depois da aquisição, como permitido para RAG estruturado. As
perguntas reutilizam essa memória. Perguntas/respostas anteriores não são
memorizadas. Gabaritos, keypoints, tipos anotados e decomposições não são
passados à recuperação nem ao leitor.

## Artigo versus configs publicadas

Não existe um único conjunto de parâmetros coincidente nas duas fontes:

| Parâmetro | `paper` — PDF, Tabelas 14/15 | `official` — YAML do commit fixado |
|---|---|---|
| Chunks de RULER, LongMemEval e FactConsolidation | 512 tokens | 4.096 tokens |
| Outros chunks | 4.096 tokens | 4.096 tokens |
| Saída de LongMemEval | 100 tokens | 50 tokens |
| Saída de DetectiveQA | 500 tokens | 2.000 tokens |
| RULER / EventQA / MCC / Recsys / Summary / FactConsolidation | 50 / 40 / 20 / 300 / 1.200 / 10 | mesmos limites |

O padrão é `paper`, para a comparação pedida com o artigo anexado. Para comparar
diretamente com uma execução do código atual dos autores, use `-Protocol official`
e uma pasta diferente. Ambos usam `top-k=10`, a temperatura 0,7 das configs RAG,
o tokenizador `o200k_base` do GPT-4o-mini e o algoritmo de agrupamento de
sentenças dos autores, sem overlap e sem truncar o corpus. O algoritmo oficial
permite uma sentença exceder o orçamento do chunk; esse comportamento foi
preservado. O campo `context_max_length` dos YAML não é usado para cortar os
dados: os próprios autores carregam o contexto completo da fonte escolhida.

A adequação do nosso método está explicitamente registrada: o leitor recebe os
40 fatos escolhidos, pacotes lógicos e trechos de origem do local-v2, em vez de
JSON conversacional do LoCoMo. A reflexão é interna, na mesma chamada de
resposta, adaptada ao formato final original de cada tarefa. Só essa resposta
é avaliada, com o teto oficial exato. Backends de reasoning que aumentariam
automaticamente esse teto são recusados.

Na tarefa contrafactual, a reflexão usa somente as premissas numeradas, sem a
orientação do LoCoMo para completar inferências usando conhecimento geral.

Nos conflitos, os fatos recebem proveniência do número de série real presente
no texto. O leitor recebe a regra oficial de preferir a versão mais recente em
cada passo de uma cadeia. Fatos antigos são preservados como contrapontos;
não há exclusão automática por predicado, consulta ao gabarito ou substituição
por fatos reais do mundo. As datas efetivamente escritas nos dados continuam
disponíveis ao mecanismo temporal. Os horários de ingestão dos wrappers foram
trocados por ordinais estáveis, explicitamente distintos das datas de evento.
O relatório registra quantos fatos entregues têm série identificada e quantos
não têm; a qualidade da extração e a recuperação da atualização continuam
sendo responsabilidades do método.

Desde a revisão de recência, o adaptador de conflitos usa
`-ConflictRecencyWeight 0.65` por padrão. É um prior multiplicativo sobre a
relevância, calculado entre versões com o mesmo sujeito e predicado: a versão
mais antiga recebe fator 0,35 e a mais nova fator 1. Fatos sem versões
concorrentes ou com série desconhecida recebem prioridade neutra. Isso evita
dar relevância a um fato desconexo somente porque seu número é alto. Nenhum
fato é apagado, e o mesmo predicado não é declarado automaticamente funcional.

Versões ausentes da extração também contam: a política encontra registros com
redação original idêntica, variando somente o span literal do objeto da tripla
citada. Sujeito, predicado, negação e outros qualificadores permanecem fixos.
Isso envelhece uma premissa antiga sem criar uma tripla nova ou extrair a
resposta. O leitor recebe etiquetas explícitas de versão mais recente e versão
antiga, vinculadas às séries originais, e continua verificando conflito/escopo.

Esse prior participa dos pontos de partida, expansão, seleção antes dos cortes
de grounding, feixe, reranking de planos e preenchimento dos 40 fatos. O score
de uma cadeia usa a menor recência entre seus passos, para que o último passo
atualizado não esconda uma premissa antiga. A busca lexical de trechos originais
também prioriza atualizações dentro de um grupo com relevância suficiente; até
quatro registros pertinentes aparecem primeiro, em ordem numérica decrescente,
dentro do orçamento de caracteres já usado pelo resgate de trechos adicionais.
Isso recupera atualizações que não viraram triplas. A reflexão compara versões
do mesmo fato antes de responder e trata a versão antiga como contraponto.

A política é aplicada somente ao pool numerado de FactConsolidation; os outros
datasets e o código compartilhado de LoCoMo não mudam. O peso e a versão da
política ficam no manifesto, e os diagnósticos por pergunta registram recência
e séries realmente entregues. `-ConflictRecencyWeight 0` desativa o prior da
busca para ablação (mantém a instrução de resolução de conflitos no leitor).
Resultados anteriores exigem uma pasta nova por causa da mudança de código.

### Reteste das mesmas cinco perguntas

O piloto final ficou em `runs/memoryagentbench-recency-source-pilot/`:
**5/5**, contra **2/5** no piloto anterior. O primeiro reteste intermediário
ficou em **3/5** e foi preservado em `runs/memoryagentbench-recency-pilot/`.
`comparison.md`/`comparison.json` conferem IDs, perguntas, referências, modelo,
extração, configurações e limites iguais. Os três erros anteriores passaram a
ser respondidos com Pesäpallo, The Fairly OddParents e India.

Continuam sendo cinco questões usadas para depuração, com temperatura 0,7 e
sem seed; não representam um resultado do benchmark completo. Cada execução
fez cinco chamadas de leitor. Os tokens de prompt passaram de 7.257 para 9.280
(+27,9%), por causa dos rótulos e das instruções explícitas de versões; os
tokens de saída continuaram em 19. A extração dos 473 fatos foi reutilizada.

## Métricas e avaliadores

| Tarefa | Métrica principal |
|---|---|
| RULER, EventQA, FactConsolidation SH/MH | substring exact match normalizado |
| Cinco classificações e DetectiveQA | exact match com o pós-processamento oficial |
| Movie Recommendation | Recall@5, catálogo `entity2id.json` e edit distance oficiais |
| LongMemEval | juiz GPT-4o, prompt original específico ao tipo de pergunta |
| Sumarização | fluência × F1 de precisão/recall, três julgamentos HELMET originais |

O parser de classificação não transforma `label: 43` em `43`. O scorer de filmes
preserva inclusive a ordem de empates baseada em `set`, e o script fixa
`PYTHONHASHSEED=0` para reproduzi-la. Não usamos ROUGE ou F1 lexical como
substituto do julgamento de LongMemEval/sumarização. F1 lexical secundário pode
aparecer em `results.jsonl`; ele não entra na coluna principal dessas tarefas.

LongMemEval usa `gpt-4o`, temperatura 0, teto 10, sem seed. Sumarização usa
`gpt-4o-2024-05-13`, temperatura 0,1, top-p 0,9, seed 42 e teto 4.096, como no
avaliador publicado. As três rubricas e seus exemplos são idênticos aos
publicados. A forma aninhada dos gabaritos de contextos com uma pergunta é
preservada como no `ConversationCreator`, inclusive no prompt do juiz.

Sem `-Judge`, as predições dessas duas tarefas ficam salvas com avaliação
**pendente**. Para julgar depois, sem repetir geração ou extração:

```powershell
& "C:\Users\rodri\anaconda3\python.exe" -X utf8 -u -m benchmarks.memoryagentbench judge `
  --output runs\memoryagentbench-standard-paper
```

Chamadas bloqueadas por filtro contam zero, com uma marca explícita; não são
retiradas do denominador. Falhas de infraestrutura interrompem a execução e
deixam a pergunta pendente. Um juiz inválido também fica pendente. O relatório
não publica um overall completo enquanto faltarem perguntas ou julgamentos.

## Retomada, custos e artefatos

Repita o comando inicial para retomar. O script usa `--resume`; perguntas já
salvas não são consultadas de novo. Chunks e extrações são cacheados. Retomar
um contexto parcialmente respondido reconsolida o índice local e reutiliza
os resultados de extração. O tempo dessa reconstrução é contabilizado.
O custo de memória apresentado é o observado naquela execução. Se a extração
já existia em cache, os tokens originalmente pagos em outra execução não são
cobrados novamente nem reconstruídos artificialmente no total. O índice
registra quantos chunks foram reutilizados; um total zero de tokens nessa
situação não significa que construir a memória inicialmente tenha sido grátis.
Você pode remover `-MaxQuestions` de um piloto para completar a mesma seleção.
Mudar fontes, protocolo, modelo, código, orçamento ou dispositivo exige outra
pasta: isso evita misturar configurações num resultado.

- `manifest.json`: hashes, seleção, parâmetros, método, modelo, dispositivo.
- `results.jsonl`: uma linha durável por pergunta, entrada do leitor, resposta,
  evidências, diagnósticos e uso de tokens.
- `index.jsonl`: custo de construção/reconstrução por contexto.
- `query-failures.jsonl` e `judge-failures.jsonl`: custos conhecidos de tentativas
  interrompidas, incluídos nos totais sem contar como perguntas concluídas.
- `judgments.jsonl`: julgamentos externos, prompts fixos e respostas brutas.
- `summary.json` e `report.md`: cobertura, resultados por fonte, médias por
  competência, overall e custos de memória/consulta/juiz separados.
- `official_exports/`: predições por fonte com o layout `data` dos autores,
  `answer`, `query`, `output` e `qa_pair_id`; nomes com `*` usam `_star` no Windows.

O bloqueio `run.lock` impede dois escritores na mesma saída. Uma interrupção
normal libera o bloqueio. Se o processo for encerrado à força, confira o PID
registrado no arquivo antes de remover um bloqueio que ficou abandonado.

Para atualizar somente o relatório, sem modelos ou rede:

```powershell
& "C:\Users\rodri\anaconda3\python.exe" -X utf8 -m benchmarks.memoryagentbench report `
  --output runs\memoryagentbench-standard-paper
```

As médias principais são primeiro calculadas por tarefa, depois por competência,
e por fim entre as quatro competências, como na Tabela 3. Uma média sobre todas
as perguntas daria peso desproporcional às tarefas maiores.

## Validação

Os testes verificam equivalência ao código oficial dos métricos e da
fragmentação, os limites exatos da API, ausência de gabarito nas entradas do
agente, ingestão sem contexto futuro, isolamento entre memórias, manutenção
dos números de série, retomada após interrupção, métricos do juiz e agregação
por tarefa. Para executá-los:

```powershell
& "C:\Users\rodri\anaconda3\python.exe" -m pytest tests/test_memoryagentbench.py -q
```

Uma execução com `--backend stub`/`--embed-backend tfidf` serve para depuração e
é marcada como tal no manifesto. Ela não representa um score publicável.

## Executar somente selective forgetting com Qwen no servidor

O lançador específico seleciona apenas `factconsolidation_sh_262k` e
`factconsolidation_mh_262k`, as duas colunas SF da tabela principal. São
100 perguntas por tarefa, protocolo paper e orçamento padrão de 40 fatos.
Utiliza o vLLM existente; não inicia outro modelo ou benchmark.

```bash
mkdir -p runs
EMBED_GPU=0 FACT_BUDGET=40 \
  nohup bash scripts/run-memoryagentbench-sf-qwen.sh > runs/standard-memoryagentbench-sf-qwen14b-facts40.log 2>&1 < /dev/null &
tail -f runs/standard-memoryagentbench-sf-qwen14b-facts40.log
```

`EMBED_GPU` controla somente embeddings e reranking. O Qwen permanece na GPU
do vLLM em `http://127.0.0.1:8095/v1`. `BENCH_PYTHON`, `CACHE_DIR`,
`OUTPUT_DIR` e `OPENAI_BASE_URL` permitem caminhos e endpoint explícitos.
O lançador retoma a mesma configuração e impede dois escritores na mesma
saída. Não o execute enquanto uma run dessa saída estiver ativa.
Os scores são substring exact match oficial; esta seleção não precisa dos
juízes GPT-4o e não gera overall das quatro competências.

### Fila depois do SF

`run-memoryagentbench-after-sf-qwen.sh` aguarda um SF concluído e avaliado
com 200 IDs únicos, orçamento 40, Qwen2.5-14B e protocolo paper. Só então
inicia as oito tarefas restantes da tabela principal: 124 contextos e
1.471 perguntas. A morte do processo SF não é condição de conclusão.

```bash
EMBED_GPU=0 nohup bash scripts/run-memoryagentbench-after-sf-qwen.sh > runs/standard-memoryagentbench-rest-qwen14b-facts40.log 2>&1 < /dev/null &
```

A saída é `runs/standard-memoryagentbench-rest-qwen14b/facts40`.
`SF_OUTPUT_DIR`, `OUTPUT_DIR`, `CACHE_DIR` e `BENCH_PYTHON` permitem
caminhos explícitos. O script usa retomada e bloqueio contra outra fila
para a mesma pasta. Não agenda os juízes oficiais de LongMemEval/sumarização.
As duas saídas precisam ser combinadas na análise para apresentar a tabela
completa; não há overall válido enquanto os julgamentos estiverem pendentes.

### Adaptação AR com fontes literais (`ar-source-v2`)

O perfil opcional `--adaptation ar-source-v2` é restrito à Accurate Retrieval.
Ele registra o conteúdo bruto dos chunks oficiais, sem o diálogo artificial de
ingestão. SH-QA, MH-QA e EventQA usam instruções documentais; LME(S*) conserva os
papéis da conversa e datas `Chat Time` observáveis, inclusive na continuação dos
chunks. A extração usa sentenças completas, IDs e offsets de origem, exige uma
citação literal por fato e registra rejeições de fonte e metadados.
Se o JSON da extração ultrapassar o limite de saída ou vier sem o esquema
esperado, a extração tenta grupos menores dos mesmos registros completos;
para um único registro, reduz o teto de itens. Há no máximo três níveis de
retentativa, registrados como `schema_split_retries`, sem aceitar JSON parcial.
O custo das chamadas continua contabilizado e falhas persistentes interrompem
a execução para diagnóstico.
Uma resposta curta e incompleta que esgota o orçamento, ou uma falha de esquema,
também pode usar a recuperação `source-record-reference-v1`: o modelo retorna
triplas e IDs, sem reescrever a citação. O código resolve cada ID no grupo de
entrada e usa seu registro literal como evidência. IDs inexistentes são rejeitados.
Essa chamada não usa `response_format=json_object`, mas continua exigindo JSON
válido e o mesmo teto de tokens. Seu custo fica em `index.openie.repair`, com
cache separado e política no manifesto. Os caches válidos da extração principal
continuam aproveitáveis; a resolução por ID confirma origem, não verdade semântica.

O executor continua local-v2 sem chamadas generativas. O perfil adiciona uma
preferência suave de alinhamento do predicado com os embeddings já disponíveis,
elimina joins sobre metadados de ingestão apenas nos documentos e fornece até
2.400 caracteres de citações originais por pergunta. Cobertura lexical zero
não é um veto universal; planos e seus scores continuam hipóteses.

Chunks, consultas, temperatura, limites de saída e métricas permanecem oficiais.
Os caches da extração incluem o novo prompt e a política de fontes. O manifesto
registra a adaptação e impede retomada misturando código ou configurações.
LoCoMo e o adaptador `standard` não são alterados.

```bash
BENCH_PYTHON="$PWD/.venv-bench/bin/python" EMBED_GPU=0 \
  nohup bash scripts/run-memoryagentbench-ar-adapted-qwen.sh \
  > runs/adapted-memoryagentbench-ar-qwen14b-facts40.log 2>&1 < /dev/null &
```

A ordem é **SH-QA → MH-QA → LME(S*) → EventQA**, com 12 contextos e
1.000 perguntas (100 + 100 + 300 + 500). A saída é
`runs/adapted-memoryagentbench-ar-qwen14b/facts40`. O Qwen existente na GPU 7
atende as chamadas; embeddings e reranking usam a GPU 0. O lançador utiliza
bloqueio e retomada, e não inicia SF, TTL ou LRU. LME(S*) exige o juiz oficial
para obter sua pontuação final; este lançador não agenda chamadas pagas.
