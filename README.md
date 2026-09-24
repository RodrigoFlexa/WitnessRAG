# WITNESS-RAG

Protótipo de memória orientada à preservação de **testemunhas de consultas conjuntivas**. A implementação distingue provas sobre uma base formal de fatos de aproximações semânticas produzidas por LLM e embeddings.

A proposta original está em [docs/proposta.md](docs/proposta.md). Como o método funciona hoje, etapa por etapa, com o que cada uma garante e o que não garante, está em [docs/witness-atual.md](docs/witness-atual.md). O estado efetivo do código, as correções da revisão e as limitações estão em [docs/revisao-implementacao.md](docs/revisao-implementacao.md).

**Controlador de prova (24/09/2026):** o desenho v3 ([docs/paper/witnessrag-proposta-v3.tex](docs/paper/witnessrag-proposta-v3.tex), versão formal em [witnessrag-proposta-v2.tex](docs/paper/witnessrag-proposta-v2.tex)) está implementado como `--proof-controller`: Registrar (memória datada), Buscar, Planejar com evidências, Provar, Verificar e Responder, para toda pergunta e sem rótulo de categoria. Rodada no Azure: `scripts/run-witness-proof-locomo.sh`; relatório pareado: `scripts/proof-report.py`; replay offline sem LLM: `scripts/proof-offline-eval.py`.

**Estado anterior (23/09/2026):** a explicação didática completa, incluindo o controlador agnóstico à categoria (contrato de evidência + roteamento por pertença ao fragmento), as lentes de memória e o desenho experimental, está em [docs/witnessrag-metodo.md](docs/witnessrag-metodo.md). A seção de método para o artigo, em LaTeX, está em [docs/paper/witnessrag-method.tex](docs/paper/witnessrag-method.tex).

A [auditoria do piloto de 14/09/2026](docs/revisao-resultados-2026-09-14.md) revisa os resultados salvos e apresenta as ablações experimentais `--binding-aware-grounding` e `--verify-witnesses`, disponíveis na CLI e no piloto. Ambas são desligadas por padrão; ainda não têm ganho medido com modelo real.

## O que está implementado

- Compilação falível de perguntas em consultas conjuntivas positivas, com uma variável de resposta: single-hop, cadeias e interseções. Comparações, contagem e negação ficam fora da execução lógica.
- Junções dirigidas com atribuições consistentes de variáveis e proveniência por conjuntos de fatos. A identidade simbólica preserva pontuação: `C++`, `C#` e `C` são diferentes.
- Aterramento `semantic` por padrão: usa similaridade com limiar de relação e é aproximado, sem garantia lógica. `--grounding exact` ativa o controle simbólico.
- Seleção de contexto por testemunhas completas; uma prova que excede `top-k` não é cortada para parecer completa. O leitor recebe as passagens inteiras, sem o antigo corte oculto de 1.500 caracteres; isso pode aumentar o consumo de tokens.
- Seleção de fatos por orçamento com ILP, preservando o E dentro de cada testemunha e o OU entre alternativas da mesma demanda.
- Aquisição dirigida reversível por pergunta. A política atual é uma heurística de similaridade menos custo, não uma estimativa calibrada de valor da informação.
- Resposta estrutural e resposta do leitor avaliadas separadamente. Scores de suporte e risco são **não calibrados**.

## Opções experimentais

Todas desligadas por padrão: uma rodada sem elas reproduz o comportamento
anterior, e por isso as rodadas já medidas continuam comparáveis.

| opção (`cli run` e `wrag.pilot`) | efeito |
|---|---|
| `--answer-set` | a resposta é o **conjunto** de atribuições certas, com prova por item, em vez da testemunha mais barata; habilita `aggregation="set"` e `aggregation="count"`; o verificador julga cada membro, e o leitor enumera os itens sustentados pelas passagens |
| `--vocab-compile` | a compilação recebe as relações e entidades do grafo mais próximas da pergunta, para não inventar predicados que nenhum fato instancia |
| `--query-plans` | planejamento adaptativo: começa com até duas consultas, executa todas, compartilha fatos adquiridos, replaneja após lacunas ou rejeições e tenta outra consulta quando o verificador rejeita a primeira |
| `--max-query-plans K` | orçamento global de planos distintos por pergunta; padrão `5`; não aumenta o número de documentos entregues ao leitor |
| `--hybrid-fallback` | o fallback do WITNESS-RAG passa a ser fusão recíproca de postos entre denso e BM25 |
| `--dialogue-ie` | extração adaptada a diálogo: falante como sujeito, correferência dentro do bloco e escopo temporal no fato. Muda a base de fatos compartilhada e exige nova extração |
| `--no-relation-family-merge` | ablação que preserva cada flexão de relação separada no índice, sem refazer a extração |
| `--top-k N` | blocos entregues ao leitor |
| `--agnostic-router` | uma chamada de planejamento escreve um contrato de evidência a partir só do texto da pergunta; a rota (DIRECT ou COMPOSE) é função determinística do contrato. Não lê a categoria do benchmark |
| `--memory-lenses` | lentes escolhidas pelo contrato (recência/estabilidade, saliência, confiança corroborada) reordenam só a cauda do contexto; exige `--agnostic-router` |
| `--lenses L1,L2`, `--lens-max-swaps N` | ablações das lentes: quais podem ser ligadas e quantas posições podem trocar (padrão 1) |
| `--route-override compose\|direct` | ablação: planeja, mas força a rota para todas as perguntas |
| `--proof-controller` | desenho v3: memória datada (data do evento e importância de cada fato), um plano por pergunta escrito depois da primeira busca (consulta, período de referência, pesos de tempo e importância), prova no grafo com a mesma pontuação, verificação só quando a prova mudaria o contexto; até 2 trechos novos para prova confirmada de plano composto, 1 para plano simples; prefixo protegido. Exige `--evidence-reader` no LoCoMo |
| `--proof-cycles N` | ciclos buscar-planejar-provar-verificar (padrão 2) |
| `--no-proof-verify`, `--partial-evidence` | ablações do controlador de prova (sem verificação; com as sondas/lacuna do controlador seletivo na última vaga) |

A completude do conjunto de respostas **não é certificada**: ele contém o que a
memória prova, e nada limita o que ficou de fora por falha de extração, de
compilação ou de corte. `count` herda essa limitação.

No modo de diálogo, relações flexionadas recebem uma família morfológica comum
(`paint`, `painted`, `painting`) sem remover palavras nem preposições; por isso
`work at` e `work for` permanecem distintas. A forma original do fato continua
armazenada para auditoria. O compilador também repara a variável de resposta
quando existe uma única variável possível e registra o reparo.

O relatório traz a tabela **onde a pergunta parou**, com a taxa de disparo da
testemunha. Ela vem antes de qualquer leitura de F1: com disparo baixo, a tabela
principal mede o recuperador de reserva, não o executor. O relatório separa
prova encontrada de **contexto realmente alterado**, pois uma testemunha que
devolve o mesmo top-k do fallback não constitui uma intervenção na leitura.

## Métodos

Para executar somente WitnessRAG na primeira conversa do LoCoMo, com perguntas
single-hop e multi-hop, consulte [o piloto LoCoMo](docs/locomo-pilot.md).
Para comparar planejamento simples e múltiplos planos sobre a mesma conversa,
execute `scripts/run-locomo-planning-ablation.sh`. O script roda as duas formas
com a mesma configuração e só imprime a tabela pareada se ambas terminarem as
102 perguntas. `scripts/compare-planning.py --allow-partial` existe apenas para
diagnóstico de rodadas interrompidas.

| Identificador | Implementação local |
|---|---|
| `dense` | Similaridade de embeddings de passagens |
| `bm25` | BM25 lexical |
| `graphrag` | Adaptação local com comunidades e ranking de passagens |
| `hipporag` | Adaptação com entidades e difusão PPR |
| `hipporag2` | Adaptação com triplas, filtro e nós de passagem |
| `hybrid` | Fusão recíproca de postos entre denso e BM25 |
| `relational` | SQL exato, mesmo compilador/leitor, sem aquisição nem cortes de testemunhas |
| `witnessrag` | Busca de testemunhas, limites configuráveis e aquisição opcional |
| `witnessrag-annotated` | Diagnóstico com tradução heurística de anotações privilegiadas |
| `witnessrag-oracle` | Nome legado da condição anotada; não é um teto garantido |

Os comparadores com nomes de artigos são **adaptações**, com extração compartilhada. Seus números não reproduzem automaticamente os sistemas publicados. O controle SQL serve para separar os efeitos do executor dos efeitos da compilação, do leitor e da aquisição.

## Instalação e testes

PowerShell, em um ambiente Python apropriado:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Para uso apenas com Azure, há `requirements-azure.txt`. A suíte de testes usa backends determinísticos e não chama APIs pagas. PuLP está limitado à série anterior à versão 4, pois a interface CBC usada aqui tem deprecações anunciadas.

```powershell
python -m wrag.cli prepare-data
python -m wrag.cli selftest --methods dense,bm25,graphrag,hipporag,hipporag2,relational,witnessrag
```

`prepare-data` baixa os arquivos a partir do repositório HippoRAG e requer rede. `selftest` precisa dos dados locais e força LLM stub e TF-IDF; seus resultados verificam a execução, não a qualidade científica. Os testes de integração de `pytest` criam seus próprios dados e dispensam o download.

## Configuração dos provedores

Use `.env.example` como referência para configurar `.env`. Escolha explicitamente os provedores antes de uma rodada científica:

```ini
WRAG_LLM_BACKEND=azure
WRAG_EMBED_BACKEND=azure
WRAG_AZURE_DEPLOYMENT=<deployment-chat>
WRAG_AZURE_EMBED_DEPLOYMENT=<deployment-embedding>
AZURE_OPENAI_API_KEY=<chave>
AZURE_OPENAI_BASE_URL=<url-do-gateway>
```

Endpoint, versão da API e certificado corporativo dependem do ambiente; consulte `.env.example` e `wrag/config.py`. `python -m wrag.cli diag-azure` testa a conexão e faz chamadas reais. Embeddings locais são opcionais; `auto` pode cair para TF-IDF, condição sinalizada no relatório. Use `WRAG_LLM_BACKEND=stub` e `WRAG_EMBED_BACKEND=tfidf` para execuções offline.

## Experimentos

Novas execuções usam aterramento semântico por padrão. Use embeddings semânticos reais para avaliar essa capacidade; TF-IDF continua sendo um fallback para testes offline. O método `relational` permanece sempre exato.

Controle do executor, com a mesma extração e sem aquisição:

```powershell
python -m wrag.cli run --datasets musique --methods dense,bm25,hipporag2,relational,witnessrag -n 100 --grounding exact --exhaustive --no-acquisition --tag executor
```

`--exhaustive` requer `--grounding exact` e desativa os três cortes: candidatos por átomo, feixe e testemunhas finais. Pode exigir tempo e memória exponenciais no tamanho da consulta. O modo exato com limites só registra completude quando nenhum corte ocorreu. Não há completude garantida para linguagem natural.

Ablação puramente estrutural, sem preencher lacunas com recuperação densa:

```powershell
python -m wrag.cli run --datasets musique --methods relational,witnessrag -n 100 --grounding exact --exhaustive --no-acquisition --no-dense-fallback --tag estrutural
```

Seleção sob orçamento:

```powershell
python -m wrag.cli run --datasets musique --methods relational,witnessrag -n 100 --budget 0.5 --no-acquisition --train-questions caminho/treino.json --tag budget
```

O treino precisa ser um JSON separado no formato do dataset; sobreposição de IDs ou texto das perguntas é rejeitada. Na ausência de demandas suficientes, são sintetizadas demandas sobre o grafo, com essa origem registrada. Não são usadas as perguntas restantes da avaliação como treino.

O ILP é ótimo **sobre as testemunhas fornecidas**, somente quando o solver comprova otimalidade. Solução viável sob limite de tempo e fallback guloso são explicitamente distintos. A fração de orçamento controla uma máscara de fatos visíveis: não representa redução medida de RAM, pois o índice base permanece carregado.

## Protocolo e retomada

O padrão mantém o corpus completo disponível nos arquivos locais ao sortear perguntas. `--subset-corpus` é um piloto explicitamente reduzido. Os relatórios registram hashes de corpus, perguntas, código e configuração.

```powershell
python -m wrag.cli run --datasets musique --methods relational,witnessrag -n 100 --grounding exact --exhaustive --no-acquisition --resume-dir runs/ID
python -m wrag.cli report runs/ID
```

Para retomar, repita a configuração original. Configuração, código ou dados incompatíveis são rejeitados. Registros concluídos são preservados; `--fresh` os substitui. A reindexação da retomada aparece em `index_attempts.json`, sem apagar o custo inicial. Custos de etapas interrompidas antes do checkpoint podem estar incompletos.

`report.md` e `report.json` apresentam:

- Recall e all-recall por passagens, EM/F1 do leitor comum e diferenças com bootstrap pareado.
- A mesma interseção de IDs para todos os métodos; a união dos bloqueios por filtro é excluída dessa comparação. Essa é uma estimativa condicional, não disponibilidade sobre todas as solicitações.
- EM estrutural, existência de testemunha completa no contexto e cobertura lexical de triplas dirigidas. Correspondência lexical não é auditoria semântica do texto.
- Seletividade por score não calibrado, incluindo falhas sem prova e aceitando empates em bloco.
- Custos de extração compartilhada, indexação própria, seleção e consultas. Tokens lógicos e sem cache são distintos; tokens de embeddings e custos financeiros totais não são medidos.

## Piloto local com vLLM

Para testar os sete métodos na A100 de 80 GB com Qwen2.5-14B, seleção `--gpu 5`, prazo de 6,5 horas e gráficos automáticos, veja [o guia do piloto](docs/piloto-vllm.md). O launcher é `python -m wrag.pilot --gpu 5`; o guia inclui a instalação em ambientes separados e os limites da comparação.

## Limites científicos

A implementação não demonstra novidade científica nem superioridade sobre GraphRAG/HippoRAG. Permanecem necessários: esquema e ontologia controlados, fatos tipados com tempo e escopo, consultas ouro verificadas, calibração de erros, um modelo efetivo de aquisição por valor da informação e experimentos em múltiplas sementes e corpora. A completude formal do executor não elimina erros de extração, de identidade ou de compilação.
