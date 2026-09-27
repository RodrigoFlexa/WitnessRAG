# Multiplanos experimentais: execução e comparação

Implementação de 27/09/2026. Perfil `witnessrag-multiplan`, opção
`--multiplan-portfolio`; desligada por padrão. O controlador e os prompts antigos
continuam disponíveis. Esta versão implementa os mecanismos operacionais do
desenho em `multiplanos-teoria-e-validacao.md`; não demonstra ganho em LoCoMo.

## Rodar as duas versões em um comando

Na raiz do WitnessRAG, com o ambiente Python do projeto e a chave já configurada
no `.env`:

```powershell
python scripts/run-multiplan-comparison.py
```

Roda as dez conversas, todas as perguntas não adversariais, primeiro
`witnessrag-robust` e depois `witnessrag-multiplan`, usando gpt-4o-mini.
Detecta CUDA disponível; caso contrário, usa CPU. Memória em proposições,
reranker, datas, resumos, leitor, amostragem, orçamento de fatos e parâmetros
de busca ficam iguais. A referência é o robusto atual **com as correções da
revisão**, não uma reprodução do executável histórico anterior às correções.
As duas versões compartilham o cache de extração e embeddings.

Saídas: `runs/multiplan-comparison/witnessrag-robust`,
`runs/multiplan-comparison/witnessrag-multiplan` e relatório agregado
`runs/multiplan-comparison/compare.md`/`compare.json`.
O relatório calcula F1 oficial, BLEU-1 local, categorias, diferenças pareadas,
IC 95% por conversa e tokens. BLEU por categoria e diagnósticos agregados estão
no JSON. BLEU local ainda não tem equivalência confirmada com a tabela Zero-Mem.

Para começar com um piloto de oito perguntas da conv05:

```powershell
python scripts/run-multiplan-comparison.py --conversation 5 --questions 8 --output runs/multiplan-pilot
```

Para acrescentar o WitnessRAG sem as melhorias robustas e o robusto sem plano:

```powershell
python scripts/run-multiplan-comparison.py --include-controls --output runs/multiplan-four
```

São quatro rodadas; somente esse controle adicional altera reranker/datas no
WitnessRAG antigo. A comparação principal sempre isola o novo controlador
contra a mesma entrega robusta. Os controles ampliam custo e duração.

Outras opções: `--device cpu`, `--device cuda`, `--model`, `--locomo-file`,
`--cycles` (2), `--max-plans` (3), `--fact-budget` (40), `--concurrency` (8).
`--hours` é um prazo por variante. O custo de busca/chamadas não é equiparado
artificialmente: fica registrado, e a comparação não separa todos os efeitos
internos do novo controlador.

```powershell
python scripts/run-multiplan-comparison.py --dry-run
```

Esse comando valida as opções e mostra a execução sem baixar dados nem chamar
a API. Repetir o comando normal retoma rodadas parciais e preserva as completas.
O manifesto verifica código e configuração; quando eles mudarem, use outro
`--output`. O relatório recusa corpora diferentes, perguntas faltantes,
duplicadas ou com gabaritos diferentes. `--report-only` recalcula o relatório
sem executar rodadas incompletas.

## O que está implementado

- Contrato fixo extraído da pergunta: operação, tipo, ambiguidades e requisitos
  ancorados em trechos literais. O requisito `q0` sempre inclui a pergunta
  inteira, protegendo também condições omitidas pelo extrator. Replanejamentos
  não podem substituir esse contrato.
- Portfólio limitado de rotas diretas, cadeias, interseções, eventos e premissas.
  Todas as rotas aceitas pertencem à mesma interpretação. Ambiguidades são
  sinalizadas e não certificadas como uma resposta singular resolvida.
  Renomear variáveis não cria um plano distinto.
- Vocabulário, exemplos dirigidos das vizinhanças de entidades e modalidades
  realmente observadas no grafo. A visão é parcial; sua ausência não veta
  pontes fora da busca inicial. Predicados alternativos continuam dentro de
  átomos; conjunções não viram uniões por aproximação lexical.
- Junção existente condicionada a bindings e compartilhamento dos aterramentos
  de átomos entre planos. O cache é local à pergunta e distingue escopo,
  pontuação, limiares, modalidade de aterramento e horizonte da memória.
  Aterramentos condicionados a bindings ainda usam o cache interno de cada
  junção, não um cache global entre planos.
- Execução de todas as rotas novas do ciclo antes da parada. Listas acumulam
  membros aprovados de rotas compatíveis; contagens registram o número
  recuperado sem afirmar enumeração completa.
- Verificação por candidata e requisito, com citações cuja existência é
  conferida na fala de origem da própria testemunha. Uma candidata sem cobertura
  ou com citação inventada não vira prova. Composição entre fontes e paráfrases
  são permitidas; a decisão de implicação continua sendo falível e feita pelo LLM.
- Conflitos singulares provocam recuperação discriminativa e uma reconsideração
  conjunta das fontes. Persistindo o conflito, os pacotes são premissas para o
  leitor; não há votação nem escolha automática pelo maior escore.
- Replanejamento com lacuna, bindings, prefixo casado mas ainda não verificado,
  cortes e motivos de rejeição. A busca da ponte usa a entidade ligada da lacuna.
  O contrato permanece fixo, mesmo quando uma nova rota compacta é proposta.
- Reservas atômicas de fatos por testemunha antes do preenchimento/reranker.
  Uma prova admitida conserva todos os seus índices de origem, mesmo que cinco
  fatos venham da mesma fala. Pacotes que não cabem são registrados; seus
  fragmentos não recebem prioridade de prova. A entrega real é conferida.
- Operadores sobre intervalos explicitamente datados: primeiro/mais recente
  dentre eventos recuperados; antes/depois entre dois endpoints nomeados;
  duração como faixa de dias. Datas de sessão não substituem automaticamente
  datas de eventos e sobreposição fica indeterminada. Comparação geral e
  inferência recebem premissas; não são convertidas em prova dedutiva.

O modelo ainda pode produzir contratos errados, sinônimos inadequados ou
interpretações incompletas. Não há álgebra temporal universal, certificação de
completude global nem garantia de desempenho em qualquer pergunta. O desenho
de pesquisa inclui ainda experimentos de diversidade versus orçamento de busca
e auditoria humana; o comando pareado não substitui esses estudos.

## Validação feita

Testes locais cobrem contrato, congelamento, citações, junções, união de membros,
conflitos, contagem incompleta, cache, pacotes atômicos e intervalos temporais.

Foram feitas sete chamadas ao gpt-4o-mini com dados sintéticos, sem extração nem
leitor do benchmark: quatro para planejamento/verificação de uma cadeia e uma
lista, duas para contrastes trabalhar/liderar e possuir/tocar, uma para confirmar
uma correção de prompt. O primeiro contraste de instrumentos recusou também a
candidata correta porque o modelo citou a pergunta em `q0`. O código rejeitou
a citação; o prompt passou a distinguir explicitamente citação do requisito e
citação da fonte, com exemplo completo. A chamada seguinte aceitou somente o
instrumento tocado. A falha original foi preservada nos artefatos.

Resultados e drivers da checagem: `runs/multiplan-prompt-smoke/`. Esses contrastes
validam funcionamento básico e uma correção concreta; não medem F1/BLEU,
generalização ou falso positivo de prova em diálogos reais. Não foi iniciada a
comparação completa de dez conversas.
