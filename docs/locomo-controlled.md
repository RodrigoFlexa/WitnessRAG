# Execução controlada do Witness

No servidor Linux, depois de atualizar o código:

```bash
cd /home/rodrigo.flexa/WitnessRAG
GPU=4 bash scripts/run-locomo-controlled.sh runs/locomo-controlled-01
```

O mesmo comando retoma a execução. Não use a pasta de uma rodada antiga.
Os caches históricos não são lidos nem modificados. Guarde as rodadas anteriores.

O padrão usa `.venv-bench/bin/python` para o benchmark e
`.venv-vllm/bin/python` para o servidor. `BENCH_PYTHON`, `VLLM_PYTHON`,
`GPU`, `PORT` (início da faixa), `HOURS` (18 por tentativa), `MAX_RESUMES` (8),
`MODEL_REVISION` e `LOCOMO_FILE` podem ser definidos no ambiente.
`LOCOMO_CONVERSATION=0` executa somente a primeira conversa, em uma saída separada.
Não execute dois processos na mesma pasta. O launcher obtém um bloqueio exclusivo,
procura uma porta livre e encerra somente o servidor que o piloto iniciou.

## O que roda

1. Uma extração e um grafo com vetores por conversa, salvos com checksum.
2. Auditoria em duas perguntas de cada categoria por conversa, escolhidas pelo hash
   do qid. Controle e soft-v2 são reproduzidos com respostas LLM gravadas. Mudança
   em requisição, plano, scores ou diagnóstico interrompe a execução. Uma repetição
   nova sem cache mede variabilidade; uma repetição com cache mede custo quente.
3. Duas recuperações por pergunta: evidence e soft-v2. A memória é a mesma.
   Um cache novo, com identidade estável independente da porta, compartilha respostas
   para requisições idênticas. Não se usa cache de experimento anterior.
4. Dois leitores por recuperação: comum e pistas. As passagens, textos, ordem e
   pistas são congelados antes da leitura. As leituras são novas, sem cache LLM.
5. Relatório pareado para as quatro células, com F1 oficial, single/multi, AR@5,
   bootstrap por conversa, chamadas, tokens, latências e hits de cache.

O soft-v2 rejeita placeholders, listas malformadas e combinações inconsistentes
de tier/missing. Faz no máximo uma tentativa de reparo; se falhar, fica unavailable.
Nenhum requisito inválido é simplesmente apagado para promover o plano a full.

## Saídas para trazer de volta

- `comparison.md`: tabela principal.
- `comparison.json`: métricas e custos detalhados.
- `cells/`: respostas das quatro células em JSONL.
- `controlled/`: memória, auditoria, recuperações e chamadas completas por pergunta.
- `pilot/`: logs e relatórios de indexação do executor existente.
- `controlled-manifest.json`: configuração, versões e hash do código.

Preferencialmente traga a pasta `runs/locomo-controlled-01` completa. `cache/` é
dispensável para ler os resultados, mas deve ser preservado no servidor para retomar.
`memory.pkl` contém objetos Python locais: só carregue checkpoints de confiança.

Regerar o relatório sem GPU:

```bash
python -m wrag.controlled runs/locomo-controlled-01 --report
```

Inspecionar o comando sem iniciar servidor nem criar rodada:

```bash
GPU=4 bash scripts/run-locomo-controlled.sh runs/locomo-controlled-01 --dry-run
```

## Interpretação e limites

A tabela padrão do piloto mostra apenas evidence/common. A comparação oficial
deste experimento é `comparison.md`, gerada somente com todas as células completas.
O custo da recuperação pertence ao par de leitores: não o some duas vezes.
Custos frios/quentes são estimados na amostra de auditoria; a execução principal
tem cache compartilhado e informa os hits efetivos. Replay em memória é uma
verificação distinta de medir cache em disco. Embeddings permanecem cacheados;
"frio" aqui significa inferência LLM sem cache, não inicialização fria da GPU.

A auditoria é por conversa antes da avaliação dessa conversa. O teste de replay
usa a mesma instância do processo; a memória congelada também protege retomadas,
mas não estabelece determinismo universal entre versões, máquinas ou kernels.
As repetições novas quantificam uma amostra da variabilidade, não a média de várias
execuções completas. Estas conversas já orientaram o projeto: é estudo de
desenvolvimento, não validação em dados inéditos.

Código, configuração e versões diferentes exigem outra pasta. Uma interrupção
antes do checkpoint pode repetir chamadas; custos dessas chamadas interrompidas
não entram no relatório final. Checkpoints concluídos de recuperação e de cada
leitor são reutilizados, mesmo que o piloto reinicie a conversa.

Esta rodada prepara a decisão entre evidence e soft-v2. Trechos literais e coleta
dirigida entre sessões serão o próximo incremento, depois de analisar os resultados.
