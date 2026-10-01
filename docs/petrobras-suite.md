# GPT-4.1-mini na Petrobras (WSL2 + Azure OpenAI)

O lançador baixa os datasets no ambiente e executa três etapas em sequência:

1. **LoCoMo:** dez conversas, 1.540 perguntas das categorias 1–4; perfil padrão `witnessrag-local`, busca local v2, reflexão conjunta com o leitor, MiniLM, 40 fatos. Corpus em blocos de 2.048 tokens, janelas de extração de 512, top-k 5 e resposta de 128 tokens. Sem replanejamento experimental.
2. **Selective Forgetting / Conflict Resolution:** FC-SH e FC-MH, 100 perguntas cada, protocolo/suite `paper`, adaptador padrão, recência 0,65 e 40 fatos. Igual à configuração SF do Qwen, trocando o modelo generativo.
3. **Accurate Retrieval:** SH-QA (100), MH-QA (100), LME(S*) (300) e EventQA (500), nessa ordem; `ar-source-v2`, extração documental, referências verificadas à fonte, papéis/datas reais em LME e até 2.400 caracteres de fontes.

Não executa TTL/LRU, não inicia vLLM e não chama juízes. O score oficial de LME fica pendente. Respeita as temperaturas, limites e métricas do protocolo MemoryAgentBench já implementado, inclusive temperatura 0,7 do leitor. Cada modelo conclui LoCoMo → SF → AR antes de começar o próximo modelo. As saídas são separadas por deployment/etapa, e os caches compartilhados têm identidade por modelo/provedor. O deployment Azure é usado na extração, reflexão e resposta; nenhum fato extraído pelo Qwen é importado.

O embedding padrão é **`embedding-3-small-global`**, correspondente ao Text Embedding 3 Small no hub informado pelo usuário. Para Large, passe `--embed-model embedding-3-large-global`. Isso muda a configuração de retrieval em relação à run Qwen com BGE-M3; registre essa diferença na comparação. O reranker continua sendo MiniLM local, por padrão em CPU; não há um deployment de reranker na tabela informada.

## Instalação no WSL2

Use o repositório no filesystem Linux (por exemplo `~/WitnessRAG`). Em um checkout já existente, use `git pull --ff-only`; para começar:

```bash
git clone https://github.com/RodrigoFlexa/WitnessRAG.git
cd WitnessRAG
python3 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-memoryagentbench.txt matplotlib
```

Use Python 3.10 ou superior. Se o ambiente já tem dependências/PyTorch configurados para CUDA, use esse Python com `BENCH_PYTHON=/caminho/para/python`. O padrão usa **embeddings remotos Azure e MiniLM em CPU**. A API generativa roda no Azure; nenhuma GPU para o GPT é necessária. Transformers/PyTorch ainda são necessários para o tokenizador Qwen que fixa a segmentação e para o reranker MiniLM. Não são baixados pesos generativos do Qwen.

## Acesso ao modelo

Mantenha seu `.env` local com `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_BASE_URL` **ou** `AZURE_OPENAI_ENDPOINT`, e a versão de API aceita pelo gateway. `AZURE_OPENAI_CA_BUNDLE` continua sendo respeitado pelo cliente Azure. Use caminhos Linux para certificados no WSL2. Não versionar `.env` ou certificados.

O modelo informado é o **nome exato do deployment**; o padrão do lançador é `gpt-4-1-mini-petrobras`. Não confundir com o identificador público `gpt-4.1-mini`. **`o4-mini-petrobras` é o4-mini, não GPT-4o-mini.** O usuário confirmou que GPT-4o-mini não está disponível: a execução atual seleciona somente GPT-4.1-mini.

GPT-4.1-mini e GPT-4o-mini usam o transporte chat não reasoning já implementado, com `temperature`, JSON quando necessário e limites por tarefa. Não precisam da configuração específica de o4-mini. O `.env` determina a versão da API aceita pelo gateway; o script não a substitui se já estiver definida. Endpoint/base URL também precisam aceitar os dois deployments; uma URL fixa para um único deployment exige a configuração correspondente.

Downloads exigem acesso a Hugging Face, ao snapshot oficial do LoCoMo em `raw.githubusercontent.com` e aos recursos NLTK/tokenizadores. O repositório não transporta datasets nem resultados. Se houver bloqueio corporativo nesses domínios, a preparação falha explicitamente e a próxima tarefa não começa. Configure proxy/certificados de download pelo ambiente, sem desabilitar a verificação TLS.

## Executar e retomar

Inspecionar os comandos sem baixar dados ou fazer chamadas:

```bash
bash scripts/run-petrobras-suite.sh --dry-run
```

Rodar primeiro GPT-4.1-mini:

```bash
mkdir -p runs/petrobras-suite
nohup bash scripts/run-petrobras-suite.sh \
  --model gpt-4-1-mini-petrobras \
  --output runs/petrobras-suite \
  > runs/petrobras-suite/queue.log 2>&1 < /dev/null &
echo $!
```

O lançador também aceita `--models ID1 ID2` para experimentos futuros explicitamente selecionados; cada deployment completa as três etapas antes de iniciar o próximo. O padrão não enfileira outro modelo. Não inicie outra fila enquanto a anterior estiver ativa. Trocar a chave API não muda a identidade, mas trocar endpoint/deployment/embedding/código impede misturar checkpoints.

Para MiniLM em GPU, acrescente `--embed-device cuda --embed-gpu 0`; os embeddings continuam no Azure. CUDA precisa estar disponível no PyTorch dentro do WSL2; uma solicitação explícita de CUDA falha se não estiver disponível, sem fallback silencioso para CPU. O dispositivo escolhido deve ser mantido nas retomadas. Para reproduzir o embedding BGE-M3 da run Qwen, use `--embed-backend st --embed-model BAAI/bge-m3`.

Repita **o mesmo comando** para retomar. O lançador revalida e pula etapas completas, continua as perguntas/conversas pendentes e reutiliza caches do mesmo modelo/provedor. Não avance manualmente nem altere código, deployment, configuração ou pastas no meio da run. A suite rejeita uma versão/configuração diferente; use outra pasta de saída para um experimento novo. Por padrão, cada tentativa LoCoMo tem prazo de 168 horas; `--locomo-hours` pode ampliar o prazo de uma retomada.

O arquivo `suite.lock` usa bloqueio do sistema operacional. Um `run.lock` abandonado do MemoryAgentBench só é removido após verificar que o PID não existe. Um worker ainda ativo impede escritores duplicados. A suite para na primeira falha; não fica repetindo erros determinísticos.

## Acompanhar

```bash
tail -f runs/petrobras-suite/queue.log
tail -f runs/petrobras-suite/gpt-4-1-mini-petrobras/locomo/benchmark.log
tail -f runs/petrobras-suite/gpt-4-1-mini-petrobras/sf.log
tail -f runs/petrobras-suite/gpt-4-1-mini-petrobras/ar.log
```

Os logs de uma etapa aparecem quando ela começa. Acurácia por tarefa de SF/AR:

```bash
.venv-bench/bin/python scripts/watch-memoryagentbench.py \
  --output runs/petrobras-suite/gpt-4-1-mini-petrobras/sf \
  --log runs/petrobras-suite/gpt-4-1-mini-petrobras/sf.log --interval 15

.venv-bench/bin/python scripts/watch-memoryagentbench.py \
  --output runs/petrobras-suite/gpt-4-1-mini-petrobras/ar \
  --log runs/petrobras-suite/gpt-4-1-mini-petrobras/ar.log --interval 15
```

Dentro da pasta de cada deployment, o estado fica em `suite.json`; o resultado LoCoMo em `locomo/locomo_agregado.json`; SF/AR em suas pastas `summary.json`, `report.md` e `official_exports/`. O lançador valida 1.540 perguntas em dez conversas do LoCoMo, 200 IDs únicos no SF e 1.000 no AR antes de considerar a respectiva etapa completa. Filtros do gateway permanecem contabilizados no protocolo; revise-os nos relatórios antes de comparar modelos. A primeira checagem de cada tentativa usa uma chamada chat curta e um embedding de teste.
