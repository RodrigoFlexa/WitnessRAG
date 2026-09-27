# Atualização do manuscrito e da big picture

Revisão local em 27/09/2026, sobre o código `78e6674` e a árvore de trabalho.
Escopo principal: representar a implementação efetiva e reconstruir a Figura 1.
O perfil completo é definido por `scripts/proof-profiles.sh`, opção `witnessrag`,
equivalente a `proof-v4-memory`. Não confundir esse perfil com os valores isolados
da classe de configuração, onde várias extensões são desligadas por padrão.

## Figura

- Dois estágios: **Memory Construction** e **Memory Retrieval**.
- A recuperação inclui busca híbrida, plano, executor, verificador, seleção de
  proposições e leitor. Context assembly não é um terceiro estágio.
- Sem exemplos ou diálogos fictícios, conforme a orientação final do autor.
- Layout, grafo, cores, textos e fluxos próprios. A imagem do GAM serviu apenas
  como referência para a separação entre construção e recuperação.
- `figures/overview-diagram.tex`: fonte vetorial editável.
- `figures/overview-standalone.tex`: documento de compilação da figura isolada.
- `figures/overview-standalone.pdf`: exportação vetorial usada pelo artigo,
  preservando as fontes e o posicionamento independentemente da classe AAMAS.
- `figures/overview.tex`: inclusão, legenda e descrição acessível.

Para alterar a figura, edite o diagrama, compile `overview-standalone.tex` a
partir de `figures/`, e depois compile `main.tex`. No Overleaf, selecione
temporariamente o documento standalone como principal para regenerar o PDF,
ou gere-o localmente e substitua a exportação. O PDF exportado acompanha o projeto.

## Correções sustentadas pelo código

| Tema | Implementação consultada | Correção no texto |
|---|---|---|
| Representação atual | `wrag/ie.py`, `_facts_from_memories`; `scripts/proof-profiles.sh` | Proposição autocontida + tripla + fala de origem + tempo + tipo de evento. |
| Data e proveniência | `wrag/witness/dated_memory.py`, `_best_turn`, `_register` | Fala declarada quando válida; fallback por sobreposição; data de sessão como proxy; tratamento específico de planos futuros. |
| Importância | `dated_memory.py`, `arousal`, `DatedMemory.__init__` | Heurística lexical determinística; média por trecho normalizada; não é estimador aprendido de utilidade. |
| Score dos trechos | `wrag/witness/scoring.py`, `passage_scores`, `WeightLevels` | RRF normalizada; normal/none com pesos adicionais zero; strong 0,3 temporal e 0,1 importância. |
| Escala temporal | `scoring.py`, `proximity_scale` | Piso de sete dias também para as referências pontuais. |
| Entrada do planner | `wrag/methods/witnessrag.py`, `_evidence_facts`, `_retrieve_proof` | Seleção limitada de fatos datados e vocabulário; não todos os textos integrais. |
| Plano e tipos | `wrag/witness/plan.py`, `plan_from_data`; `_prove_items` | Slots temporais opcionais, tipos como sinal de ranking/verificação, comparação delegada ao leitor. |
| Matching | `wrag/witness/search.py`, `_match_fact`, `_extend` | Relações, argumentos e verbalização; consistência de bindings; aproximação semântica separada da semântica exata. |
| Pontuação da testemunha | `search.py`, `_State.cost`, `_extend`, `_to_witnesses` | Produto dos escores, acumulado em log; custo com penalidade pelo número de passagens; não média geométrica. |
| Aceitação | `WitnessRAGRetriever._prove`, `_prove_items` | Produto do matching e confiança; regras diferentes para valor único e conjunto; cortes não invalidam automaticamente membros de conjuntos. |
| Verificação | `_retrieve_proof`; `wrag/witness/confirm.py` | Condicional e por item; testemunha já presente pode ser aceita sem chamada; não é garantia formal. |
| Entrega atual | `_fact_context`; `wrag/eval/runner.py`, `read(..., facts_mode=...)` | Leitor recebe fatos e resumos; `pids` dos logs não representam seu contexto textual integral. |
| Prioridade e orçamento | `_fact_context` | Bônus aditivos 2/1, relevância pergunta/átomos, até 40 fatos e 4 por fala; nenhuma garantia incondicional de preservar testemunha inteira. |
| Resumos e custo | `_chunk_summary` | Geração sem pergunta, sob demanda e cache local; chamadas adicionais na primeira utilização. |
| Premissas abdutivas | `_retrieve_proof`, bloco `if cfg.fact_delivery` | Blocos intermediários são substituídos pela entrega por fatos; não apresentar a via abdutiva como entrada adicional do leitor no perfil completo. |

## Escopo editorial

Atualizados: resumo, introdução, formulação da tarefa, método e afirmações sobre
o próprio WitnessRAG em background/related work. Eliminada a equação duplicada
com o mesmo rótulo `eq:objective`. As fontes bibliográficas preexistentes e as
comparações específicas com trabalhos externos não receberam uma auditoria
bibliográfica completa nesta tarefa.

Não foram inventados resultados nem transformados pilotos de desenvolvimento
em evidência de superioridade geral. Experimental Setup e Results continuam
pendentes de preenchimento com execuções e protocolos validados. O resumo
nesta revisão descreve o método sem reivindicar ganhos numéricos. O texto
continua sendo um manuscrito em desenvolvimento, não uma submissão final.

Nenhum algoritmo Python foi alterado. A revisão documenta limitações existentes,
incluindo a diferença entre uma testemunha aceita e evidência efetivamente
entregue. O backup anterior às alterações está em `tmp/pdfs/overleaf-before-update.zip`
na raiz do repositório, fora deste projeto Overleaf.

## Orientação final e revisão de escrita

Após a solicitação expressa do autor, a geração e o redesenho da figura foram
interrompidos. Os arquivos gráficos permanecem no último estado de trabalho;
a figura não é apresentada como aprovada ou finalizada. A partir dessa
orientação, a revisão se concentrou no texto.

A passagem editorial final:
- reorganizou a introdução em problema, lacuna delimitada, fundamento,
  proposta e contribuições metodológicas;
- removeu generalizações universais e alegações de superioridade não
  demonstradas na comparação com a literatura;
- substituiu o tom de auditoria de implementação por uma descrição científica
  do método, mantendo a precisão das equações e das condições de aceitação;
- distinguiu interpretação da pergunta, execução, verificação e entrega;
- acrescentou discussão e conclusão sem inventar resultados empíricos;
- preservou as seções de avaliação ainda incompletas, para preenchimento
  posterior com evidência validada.

Verificação do comportamento usado como base da descrição: 49 testes existentes
em test_proof_controller.py, test_v4.py e test_temporal_context.py passaram.
Isso verifica mecanismos locais; não constitui avaliação científica de desempenho.

Validação editorial final: compilação LaTeX concluída; sem rótulos duplicados, referências cruzadas ausentes ou chaves bibliográficas ausentes. O artigo compilado tem oito páginas no estado atual, incluindo referências; a avaliação experimental ainda está incompleta.
