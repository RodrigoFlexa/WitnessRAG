# WitnessRAG padrão no servidor: Qwen2.5-14B, GPU 7

Os scripts usam uma única **A100 80 GB**, selecionada por `GPU=7` e
`CUDA_VISIBLE_DEVICES=7`. O Qwen em BF16, os embeddings BGE-M3 e o reranker
MiniLM rodam nessa GPU. O vLLM reserva até 80% da memória; o restante fica
disponível para os modelos de recuperação e demais processos. O executor de
junções e planos continua em CPU. Dentro de cada processo, a GPU física 7
aparece como `cuda:0`.

A solução é `witnessrag-local`: planos locais v2, memória de fatos datados,
executor relacional, reranking MiniLM e reflexão na mesma chamada do leitor.
O padrão é **40 fatos**; o comparativo executa **20 e 40**, sem trocar os
outros componentes. Não é a variante experimental com reflector separado.

## Atualizar e instalar

No servidor Linux, na raiz do checkout:

```bash
git pull --ff-only origin main
nvidia-smi
```

Para um servidor novo, instale `uv` conforme a
[documentação oficial](https://docs.astral.sh/uv/getting-started/installation/)
e prepare os dois ambientes:

```bash
GPU=7 bash scripts/setup-qwen-server.sh
```

O instalador usa Python 3.12 e seleção de wheels PyTorch pelo driver NVIDIA,
conforme a [instalação GPU do vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).
Os ambientes de inferência e benchmark são separados para evitar conflitos
entre dependências. As versões instaladas e a GPU ficam registradas em
`runs/server-setup/`. Se os ambientes já existem, o script os utiliza;
`VLLM_VERSION` permite escolher uma versão específica antes de instalar.
Não execute a instalação durante uma run; preserve os ambientes depois de
iniciar os experimentos.

## 1. Iniciar o servidor vLLM

Em uma sessão persistente do servidor:

```bash
tmux new -s witness-vllm
GPU=7 bash scripts/serve-qwen-vllm.sh
```

Espere o servidor ficar pronto. Para sair da sessão sem encerrar o processo,
pressione `Ctrl+b`, depois `d`. Para voltar: `tmux attach -t witness-vllm`.

O endpoint é `http://127.0.0.1:8095/v1`, modelo
`Qwen/Qwen2.5-14B-Instruct`, contexto de 32.768 tokens, BF16, até quatro
sequências ativas. `--generation-config vllm` evita herdar parâmetros de
geração do checkpoint que alterem o protocolo da tarefa; veja os
[argumentos oficiais do servidor](https://docs.vllm.ai/en/latest/cli/serve/).
`GPU`, `PORT`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION` e `MAX_NUM_SEQS`
podem ser ajustados antes de iniciar. Use a mesma porta nos benchmarks.
Se houver falta de memória, reduza a reserva ou a concorrência do servidor.

Se você fixar os pesos com `MODEL_REVISION=<commit HF>`, exporte a mesma
revisão nas duas sessões e mantenha-a nas retomadas. O valor é registrado
na identidade do benchmark e no cache. O servidor usa a revisão tanto dos
pesos quanto do tokenizer.

## 2. LoCoMo completo: 20 e 40 fatos

Em outra sessão persistente:

```bash
tmux new -s witness-locomo
GPU=7 bash scripts/run-standard-qwen-variants.sh locomo runs/standard-locomo-qwen14b
```

O script roda as dez conversas primeiro com 20 e depois com 40 fatos.
Cada variante cobre as 1.540 perguntas das categorias 1–4 usadas na avaliação
padrão; a categoria adversarial fica fora, como na run anterior.
A extração e os embeddings reutilizam o cache entre variantes. Os leitores
com contextos diferentes geram chamadas diferentes.

Resultados:

* `runs/standard-locomo-qwen14b/facts20/locomo_agregado.json`
* `runs/standard-locomo-qwen14b/facts40/locomo_agregado.json`

O log detalhado fica em `facts20/benchmark.log` e `facts40/benchmark.log`.
O limite padrão é 72 horas **por variante**; `HOURS` altera esse limite.
Falhas ou limite de tempo interrompem a sequência. Para retomar, repita o
mesmo comando; o piloto reconhece os checkpoints existentes.

Para rodar somente um orçamento:

```bash
GPU=7 FACT_BUDGET=20 bash scripts/run-standard-qwen.sh locomo runs/locomo-qwen14b-facts20
GPU=7 FACT_BUDGET=40 bash scripts/run-standard-qwen.sh locomo runs/locomo-qwen14b-facts40
```

## 3. LoCoMo experimental: checker de suficiência e um replanejamento

Antes ou depois da execução padrão do LoCoMo, você também pode avaliar a
[variante experimental com checker antes da leitura padrão](replanejamento-reflexao.md),
em outra sessão, executando:

```bash
GPU=7 REFLECTION_REPLAN=1 bash scripts/run-standard-qwen-variants.sh locomo
```

O checker recebe apenas query e fatos datados. Ele pode pedir uma nova busca;
depois, em ambos os caminhos, executa-se a reflexão/reader padrão.
Ela grava `runs/replan2-locomo-qwen14b/facts20` e `facts40`. Não reutilize
as pastas antigas `replan1`. Execute as
variantes sequencialmente para comparar os tempos na mesma GPU. Essa opção
está disponível somente para LoCoMo; a solução padrão permanece como controle.

## 4. MemoryAgentBench: depois do LoCoMo

Mantenha o mesmo vLLM ativo e inicie outra sessão:

```bash
tmux new -s witness-memoryagentbench
GPU=7 bash scripts/run-standard-qwen-variants.sh memoryagentbench runs/standard-memoryagentbench-qwen14b
```

Executa 20 e 40 fatos, sequencialmente, em todos os 146 contextos e 3.671
perguntas do release fixado. O modo `PROTOCOL=paper` segue os parâmetros
descritos no artigo. Cada tarefa mantém seu orçamento de saída, prompt,
métrica e configuração de geração; não se aplica o teto de 128 tokens do
LoCoMo a todas as tarefas. A recência adicional é restrita à tarefa de
resolução de conflitos, usando a ordem observável dos registros.

Os relatórios ficam em `facts20/report.md` e `facts40/report.md`. Para
executar apenas as dez tarefas da tabela principal do artigo, use
`SUITE=paper` e outra pasta de saída. A execução completa também produz
a média dessas tarefas, separada das tarefas adicionais.

**Os avaliadores oficiais de LongMemEval e sumarização usam GPT-4o.** O
Qwen é o modelo da nossa solução em todas as tarefas. As predições são
salvas, mas os scores dessas duas tarefas ficam pendentes até rodar os
juízes oficiais. Trocá-los por Qwen mudaria o protocolo de comparação.
Para completar a avaliação, forneça uma chave da API pública no ambiente
ou `.env` do servidor e execute, fora do ambiente de endpoint Qwen:

```bash
for budget in 20 40; do
  OPENAI_BASE_URL=https://api.openai.com/v1 \
    .venv-bench/bin/python -m benchmarks.memoryagentbench judge \
    --backend openai --cache runs/.cache/memoryagentbench-qwen14b \
    --output "runs/standard-memoryagentbench-qwen14b/facts$budget"
done
```

Esse passo faz chamadas pagas aos juízes. Detalhes dos dados fixados,
diferenças `paper`/`official`, métricas e aquisição sequencial em
[memoryagentbench.md](memoryagentbench.md).

## Conferência e retomada

Antes de cada variante, o script verifica se o modelo correto está servido
e se o ambiente de benchmark realmente enxerga CUDA. Registra a versão
do servidor e a GPU em `cache/server.json` dentro da pasta de saída.
Erros interrompem a execução, sem trocar o dispositivo automaticamente.
`WRAG_EMBED_STRICT_DEVICE=1` também impede que uma falha de carregamento do
BGE-M3 leve a uma execução em CPU.

As saídas e caches são separados entre LoCoMo e MemoryAgentBench. Dentro
de um benchmark, as duas variantes compartilham apenas os caches que
dependem dos mesmos modelos e entradas. Os manifestos de retomada rejeitam
mudanças de configuração, código ou dados. Preserve o checkout durante
as execuções; para uma nova configuração, use outra pasta.

Os testes locais validam protocolo, retomada e os comandos gerados pelos
scripts. A inicialização real do vLLM e a disponibilidade de memória precisam
ser verificadas na A100 do servidor.
