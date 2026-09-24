# HotpotQA, RULER 128K e NarrativeQA no protocolo do GAM

Código: `wrag/gambench/` (`python -m wrag.gambench`). Script: `scripts/run-gam-bench.sh`.
Testes: `tests/test_gambench.py`.

## O que é medido

As mesmas colunas da Tabela 1(b) do GAM (arXiv:2511.18423):

| benchmark | recortes | métrica | amostras |
|---|---|---|---:|
| HotpotQA | 56K, 224K, 448K tokens | F1 | 128 cada |
| RULER 128K | Retri. (8 tarefas NIAH), MT (vt), AGG. (cwe, fwe), QA (qa_1, qa_2) | acurácia | 500 por tarefa |
| NarrativeQA | test, 300 perguntas | F1 | 300 |

## De onde vem cada parâmetro

O artigo é curto nos detalhes. O resto foi lido do código de avaliação publicado
pelos autores (github.com/VectorSpaceLab/general-agentic-memory, commit
`565db2cc`, pasta `research/`) e fixado em `wrag/gambench/protocol.py`.

| parâmetro | valor | fonte |
|---|---|---|
| HotpotQA | `BytedTsinghua-SIA/hotpotqa`: eval_400, eval_1600, eval_3200 | `scripts/eval_hotpotqa.sh` |
| RULER | `lighteval/RULER-131072-Qwen2.5-Instruct`, 13 tarefas, todas as amostras | `download_ruler.py`, `eval_ruler.sh` |
| separação do RULER | `split_input` (exemplo, instrução, contexto, pergunta) | `download_ruler.py` |
| NarrativeQA | `deepmind/narrativeqa` test; `random.seed(42)`, `shuffle`, 300 primeiras | `eval_narrativeqa.sh`, `narrativeqa_test.py` |
| páginas | 2.048 tokens do tokenizador do BGE-M3, sem sobreposição, cabeçalho `[Session i]` | `*_test.py`, `--max-tokens 2048 --embedding-model-path BAAI/bge-m3` |
| páginas recuperadas | 5 | artigo, Seção 3.1 |
| leitor | prompt literal, uma mensagem de usuário, sem system prompt, temperatura 0,3, 256 tokens | `*_test.py` (working generator) |
| F1 | `qa_f1_score` do LongBench, máximo sobre as respostas | `hotpotqa_test.py`, `narrativeqa_test.py` |
| acurácia RULER | correta só se todas as saídas aparecem (três regras de casamento) | `ruler_test.py` (`evaluate_answer`) |

Os arquivos são baixados por commit e conferidos por SHA-256. O `split_input`
portado foi comparado com o original nas 6.500 amostras do RULER: saídas idênticas.
A seleção do NarrativeQA reproduz os primeiros índices (418, 5545, 6257, 5098,
2465); a média de tokens das 300 perguntas dá 86K com o tokenizador do GPT-4o
(o artigo diz 87K).

As colunas do RULER são a média das tarefas do grupo. O artigo não lista a
composição; usamos a da RULER original. Os números publicados são consistentes
com 500 amostras por tarefa: toda coluna Retri. é múltiplo de 1/4.000
(arredondado), MT de 1/500, AGG. e QA de 1/1.000.

## O que é do método, não do protocolo

Fica de fora tudo o que é do GAM como método: o memorizador, os prompts de
pesquisa e reflexão, os system prompts por tarefa que o GAM dá ao memorizador
no RULER, e as dicas extras que ele dá ao pesquisador no `niah_multivalue`
("There are 4 different special magic numbers..."). O nosso método não recebe
nada disso.

No GAM, o contexto do leitor é o resumo da pesquisa. Aqui é o conjunto de até
5 páginas escolhidas pelo método, na ordem em que ele as entrega.

## Motores

| motor | o que escolhe as páginas | chamadas de LLM além do leitor |
|---|---|---|
| `rag` | BGE-M3 denso, top-5 (a linha RAG do artigo) | nenhuma |
| `hybrid` | BGE-M3 + BM25 (RRF): o BUSCAR do WitnessRAG | nenhuma |
| `witnessrag-lite` | BUSCAR + política de contexto sem LLM | nenhuma |
| `witnessrag` | o método completo (v3): REGISTRAR sobre todas as páginas, BUSCAR, PLANEJAR, PROVAR, VERIFICAR | extração (NER + OpenIE em janelas de 512 tokens), até 2 planos e 2 verificações por pergunta |

O `witnessrag` usa a configuração registrada do LoCoMo
(`scripts/run-witness-proof-locomo.sh`), sem o que só existe para conversa:
extração de diálogo, anotações de data e o leitor de evidências.

O motor nunca vê as respostas nem o nome da tarefa: a pergunta que ele recebe
tem a lista de respostas vazia e `qtype` vazio (há um teste para isso).

## Diferenças conscientes em relação ao código do GAM

1. **Seed.** O GAM não envia seed. Enviamos 42 para que duas execuções deem a
   mesma resposta; a temperatura continua 0,3. `--reader-seed none` reproduz o
   GAM literalmente.
2. **Tokenizador.** Se o tokenizador do BGE-M3 falha, o GAM passa, sem avisar,
   a paginar com o tiktoken do GPT-4o. Aqui a execução para.
3. **HotpotQA 448K.** O `download_data.sh` do GAM baixa `eval_6400`, mas o
   `eval_hotpotqa.sh` e o artigo usam `eval_3200` (448K). Usamos `eval_3200`.
4. **Denso do `rag`.** O `DenseRetriever` do GAM codifica páginas com
   `max_length=512` (padrão do FlagEmbedding), ou seja, vê só o início de cada
   página. O artigo não diz se a linha RAG usou esse retriever. Por padrão
   codificamos a página inteira; `EMBED_MAX_SEQ_LENGTH=512` imita o GAM.
5. **Modelo.** O artigo usa GPT-4o-mini e Qwen2.5-14B-Instruct. Rodamos agora
   com o GPT-4.1-mini da Petrobras. As linhas do artigo entram na tabela como
   referência; comparação direta só com o mesmo modelo (Qwen2.5-14B, depois).

## Falhas

Como no GAM: no HotpotQA e no NarrativeQA uma amostra sem resposta (erro, ou
bloqueio do filtro de conteúdo do Azure) sai da média do F1; no RULER conta como
incorreta. O relatório mostra também a média estrita (toda falha vale 0) e o
número de bloqueios e erros por recorte. Um erro é tentado de novo na retomada.

## Custo

`python -m wrag.gambench estimate --data-dir data/gam --price-in 0.40 --price-out 1.60`
(sem chamar LLM; RULER medido em 20 contextos por tarefa):

| recorte | páginas | leitor (qualquer motor) | extra do `witnessrag` |
|---|---:|---|---|
| HotpotQA 56K | 3.527 | 128 chamadas, 1,3 M tokens | 35 mil chamadas, 25 M + 8 M tokens |
| HotpotQA 224K | 13.961 | 128, 1,3 M | 140 mil, 95 M + 34 M |
| HotpotQA 448K | 27.848 | 128, 1,3 M | 278 mil, 188 M + 67 M |
| NarrativeQA | 9.002 (211 livros) | 300, 3,1 M | 90 mil, 63 M + 22 M |
| RULER (13 tarefas) | 411.350 | 6.500, 68 M | 4,1 milhões, 2.820 M + 990 M |

Com o preço de lista do GPT-4.1-mini (US$ 0,40 por milhão de tokens de entrada e
US$ 1,60 de saída): leitor de todos os benchmarks ≈ US$ 33 por motor;
`witnessrag` ≈ US$ 300 no HotpotQA, US$ 60 no NarrativeQA e US$ 2.700 no RULER.
Os tokens de resposta da extração são suposições (80 por NER, 400 por OpenIE).
Por isso o script não roda o `witnessrag` no RULER sem `WITNESS_ON_RULER=1`;
com o Qwen local o custo é de tempo, não de dinheiro.

Embeddings: ~466 mil páginas de 2.048 tokens no total. Na GPU são horas; na CPU,
dias. O script usa a GPU quando `nvidia-smi` funciona.

## Como rodar no servidor

```bash
cd ~/WitnessRAG
export BENCH_PYTHON="$HOME/WitnessRAG/venv/bin/python"
"$BENCH_PYTHON" -m pip install pyarrow        # só para preparar os dados

# 1. dados (baixa ~2,8 GB do Hugging Face e confere SHA-256)
"$BENCH_PYTHON" -m wrag.gambench prepare --data-dir data/gam

# 2. teste rápido: 5 amostras por recorte, os dois motores
END_IDX=5 GPU=0 bash scripts/run-gam-bench.sh runs/gam-smoke

# 3. rodada completa (retomável: rode de novo o mesmo comando)
GPU=0 bash scripts/run-gam-bench.sh

# tabela
"$BENCH_PYTHON" -m wrag.gambench report --output-root runs/gam-azure-gpt-4-1-mini-petrobras
```

Depois, com o Qwen2.5-14B num servidor com vLLM:

```bash
LLM=qwen PORT=8095 WITNESS_ON_RULER=1 GPU=1 bash scripts/run-gam-bench.sh
```

## Saídas

```
runs/gam-<modelo>/
  gam_table.md / gam_table.json          tabela no formato da Tabela 1(b)
  <motor>/<benchmark>/<recorte>/
    predictions.jsonl                     uma linha por amostra
    manifest.json                         o que define a rodada (conferido na retomada)
    report.md / report.json
```

Cada linha de `predictions.jsonl` tem a pergunta, as respostas, a resposta do
leitor, a métrica, as páginas escolhidas, os tokens, o custo do motor e o
diagnóstico do controlador de prova.
