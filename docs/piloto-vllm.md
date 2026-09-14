# Piloto na A100 de 80 GB

Use a GPU física 5 com `Qwen/Qwen2.5-14B-Instruct` em BF16. É um teste exploratório da implementação e da comparação entre métodos; não estabelece superioridade científica. O modelo também pode errar a extração e a compilação das consultas.

O [modelo oficial](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) tem 14,7 bilhões de parâmetros: os pesos BF16 ocupam aproximadamente 29,4 GB, além de cache e memória de execução. A configuração inicial reserva 80% da GPU para vLLM e usa contexto de 16.384 tokens. O restante permite executar os embeddings no mesmo dispositivo. Isso precisa ser confirmado no servidor, especialmente se houver outros processos.

A entrada `qwen2.5:14b` do Ollama não é o identificador usado aqui. O launcher usa os pesos Hugging Face, baixados na primeira execução, ou um diretório local desses pesos passado em `--model`. Não reutiliza automaticamente os blobs do Ollama. O [suporte GGUF do vLLM](https://docs.vllm.ai/en/latest/features/quantization/gguf/) tem limitações; não é necessário para esta GPU.

## Instalação no servidor Linux

Na raiz deste repositório, use Python 3.11 ou 3.12. Dois ambientes separam as dependências de inferência das do benchmark:

```bash
python3 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install vllm
python3 -m venv .venv-bench
.venv-bench/bin/python -m pip install --upgrade pip
.venv-bench/bin/python -m pip install -r requirements-pilot.txt
```

A instalação de vLLM deve ser compatível com o driver e CUDA do servidor; consulte a [documentação oficial](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). A instalação inicial não está incluída no tempo do experimento. Downloads feitos pelo launcher estão incluídos.

## Executar

```bash
.venv-bench/bin/python -m wrag.pilot --gpu 5 \
  --vllm-python "$PWD/.venv-vllm/bin/python" \
  --hours 6.5 --output runs/qwen14b-a100-pilot
```

Para conferir a configuração antes de executar, acrescente `--dry-run`. O diretório de saída precisa ser novo ou vazio. Dentro dos processos, a GPU física escolhida passa a ser `cuda:0`. O launcher inicia um servidor local na porta 8085 e encerra apenas os processos que iniciou. `--port` permite escolher outra porta; `--existing-server` reutiliza um servidor já ativo sem encerrá-lo.

Parâmetros úteis: `--questions 50`, `--concurrency 4`, `--embed-device cpu`, `--model /caminho/pesos-hf` e `--model-revision COMMIT`. Não é necessário exportar manualmente as variáveis de modelo, endpoint ou GPU.

## Protocolo padrão

- 100 perguntas de 2WikiMultiHopQA, semente 42, corpus candidato das perguntas selecionadas e até 300 distratores aleatórios; limite de 1.500 passagens, sem descartar silenciosamente candidatos para caber.
- Na preparação verificada, isso resultou em 759 candidatos + 300 distratores = 1.059 passagens, de um corpus fonte com 6.119. É um corpus reduzido, não o benchmark completo.
- Métodos: dense, BM25, GraphRAG, HippoRAG, HippoRAG2, relational e WitnessRAG. Os comparadores são adaptações locais, não reproduções oficiais dos artigos.
- Mesmo Qwen para as tarefas LLM; embeddings `BAAI/bge-base-en-v1.5`. WitnessRAG mantém grounding semântico e aquisição habilitada. O comparador SQL usa correspondência exata. `--no-acquisition` permite uma ablação separada.
- Extração compartilhada registrada separadamente. Sua parcela é atribuída a cada método dependente no gráfico; não some essas barras para obter o custo real da rodada.
- Perguntas embaralhadas e métodos alternados por pergunta. Relatórios comparam apenas IDs concluídos por todos os métodos, com intervalos bootstrap pareados para as comparações.

O prazo de 6,5 horas inclui inicialização, indexação e avaliação; dois minutos são reservados para encerrar processos e gerar gráficos. Não é uma previsão de que 100 perguntas terminarão. Se atingir o prazo, preserva os resultados parciais. Se parar durante a indexação, pode não haver comparação de qualidade. A disponibilidade e a velocidade reais da GPU só serão conhecidas na execução. Uma amostra interrompida também pode ter viés associado ao tempo de processamento.

## Resultados

O diretório escolhido contém `pilot.json`, `status.json`, `vllm.log`, `benchmark.log`, dados selecionados e seus hashes. Registra também informações da GPU e versões de pacotes quando disponíveis.

Em `benchmark/ID/` ficam `report.md`, `report.json`, registros por pergunta e `figures/`. Os gráficos são produzidos em PNG, SVG e PDF:

1. EM/F1 do leitor, Recall@5 e recuperação de todas as passagens de apoio.
2. Latência e tokens LLM por pergunta.
3. EM estrutural e cobertura por testemunha completa.
4. Tempo de indexação compartilhada e própria.

Para regenerar os gráficos:

```bash
.venv-bench/bin/python -m wrag.eval.plots runs/qwen14b-a100-pilot/benchmark/ID
```

O launcher não oferece retomada automática da rodada interrompida. Os registros permanecem disponíveis para análise; uma nova execução usa outro diretório. O executor interno possui verificações de retomada, mas elas exigem a mesma configuração, código e dados.

## Validação realizada localmente

Testes cobrem isolamento de GPU no plano, parâmetros e cache do backend, comunicação HTTP compatível com OpenAI, seleção dos dados, pareamento dos gráficos e interrupção/retomada do executor. Um teste funcional executou os sete métodos com backend simulado e produziu os gráficos. Esses resultados simulados não medem a qualidade do Qwen. A execução real com CUDA/vLLM e seu tempo total ainda precisam ser verificados no servidor.
