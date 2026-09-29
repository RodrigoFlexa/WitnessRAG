# Memória com consulta-alvo e reflector separado

Experimento pareado de 10, 20 e 40 fatos na primeira conversa. Reutiliza a extração e **a execução v2 já salva** em `runs/local-plans-v2-gpt4omini-all`. Não recalcula embeddings, planos, junções ou reranking. Como a nova consulta-alvo é apenas uma hipótese para o reflector, ela não altera essa recuperação: compartilhar a execução é um controle do experimento.

## Fluxo

1. Uma chamada `memory.target` por pergunta produz a consulta conjuntiva semântica, a operação e restrições. Não recebe fatos, categorias ou respostas ouro. A consulta fica registrada como hipótese, sem executar seus predicados no grafo.
2. A execução v2 existente fornece os mesmos 40 fatos e fontes para todas as variantes. O carregador verifica perguntas, corpus, índices, IDs dos fatos, cache de extração, falas de origem e presença dos fatos no contexto salvo. Cache ausente ou incompatível causa erro; nunca inicia nova extração.
3. Uma chamada `memory.reflect` por variante recebe pergunta, consulta-alvo e até 10/20/40 fatos. Pacotes de junções selecionadas têm prioridade e entram completos. Pacotes que não cabem são registrados e excluídos do preenchimento, para evitar entregar uma cadeia cortada como pacote completo. O limite pode resultar em menos fatos quando necessário.
4. O respondedor recebe o contexto original completo, idêntico entre as variantes, mais as conclusões de seu reflector. A reflexão conjunta antiga do reader fica desligada. O número 10/20/40 limita **o reflector**, não o respondedor.

O corpo padrão é a frase factual já extraída (`statement`), mantendo qualificadores como os intervalos 25/5 do exemplo de Pomodoro. `-Body triple` seleciona a tripla pura para outro experimento, em uma pasta diferente. Datas de sessão ficam agrupadas; datas de evento só são apresentadas como tais quando resolvidas a partir de uma expressão explícita. Modalidade e proveniência continuam registradas.

O reflector retorna conclusões, pontes de inferência, conflitos e lacunas. A validação rejeita IDs ausentes, inferências sem ponte declarada e conflitos com menos de duas premissas. **Isso valida estrutura e referências, não verdade ou suficiência semântica.** As inferências não são gravadas no grafo.

Quando o modelo declara uma cabeça não ligada (`answer_var: x`) mas usa uma única variável nos átomos (por exemplo `?time`), a cabeça é normalizada para essa variável e o ajuste é registrado em `normalizations`. Um termo que coincide exatamente com a variável declarada também recebe o prefixo ausente (`x` → `?x`), e o texto literal de exemplo `optional ?time` é removido do campo opcional de data. Nenhum predicado é alterado e nenhuma chamada extra é feita. Com mais de uma variável candidata, é necessário o reparo descrito abaixo; o código não escolhe uma delas por aproximação.

Consultas `yesno` são booleanas: seus átomos expressam as condições e `answer_var` é normalizado para `null`, pois não há entidade/data projetada. Variáveis presentes são existenciais. Esse caso se limita à consulta consultiva; não modifica a interface do executor v2.

Para uma cabeça não ligada com várias variáveis, a chamada de reparo recebe somente a pergunta e a consulta, escolhe entre as variáveis existentes e retorna apenas `answer_var`. O código conserva os átomos e as restrições originais. Uma escolha fora da lista é rejeitada; não há seleção heurística nem consulta às respostas ouro.

## Rodar as três variantes

Na raiz do projeto:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-memory-variants-openai.ps1 `
  -Python "C:\Users\rodri\anaconda3\python.exe" `
  -Model gpt-4o-mini `
  -Output runs\memory-reflector-conv00
```

O comando usa as 152 perguntas da primeira conversa e os três tetos. A chave é lida do mesmo `.env` usado pela run anterior. Um smoke com quatro perguntas equilibradas usa `-Questions 4 -Output runs\memory-reflector-conv00-smoke`. A opção `-DryRun` valida os arquivos e os cortes **sem chamar a API e sem criar resultados de benchmark**.

Para terminar na memória, adicione `-MemoryOnly` e use uma pasta própria. Para um modelo servido localmente, passe seu nome em `-Model` e o endereço em `-BaseUrl`, por exemplo `http://127.0.0.1:8085/v1` para um servidor já configurado.

Tetos iniciais de saída da memória: consulta-alvo 384 e reflector 1.024 tokens, respeitados exatamente pelo backend para modelos sem reasoning. Para o respondedor, 128 é o orçamento solicitado ao reader existente; seu backend mantém o piso de configuração (2.048 nesta execução), portanto não representa um teto físico de 128. O relatório contabiliza os tokens efetivamente retornados. Uma resposta truncada ou com estrutura/referências inválidas da memória permite uma única chamada de reparo de saída, registrada em `memory.target.repair` ou `memory.reflect.repair`, com seu custo e a saída recusada. Consulta ainda inválida, truncamento persistente ou envelope JSON malformado interrompem a execução, conservando as variantes concluídas. Bloqueios de conteúdo não recebem esse reparo. Uma mudança de configuração exige nova pasta.

Se o reflector ainda contiver entradas inválidas após esse reparo, mas o envelope e as listas estiverem íntegros, cada conclusão/conflito é revalidado separadamente. Entradas sem premissas, com IDs ausentes ou com tipo/ponte inválidos são descartadas, sem inventar referências; as demais são preservadas. `discarded_entries` registra texto, posição e motivo, e `generation_repairs` conserva também a saída do reparo. O reader é informado de que houve interpretações omitidas. A contagem dos descartes aparece no comparativo. Uma reflexão sem conclusões válidas pode terminar com listas vazias e lacunas explícitas.

## Comparativo, custo e retomada

- `comparison.md` / `comparison.json`: F1/BLEU, diferenças pareadas, categorias, fatos entregues, tokens de entrada/saída da memória e tokens do respondedor. Compara apenas perguntas concluídas nas três variantes e informa se a experiência está completa.
- `facts-10/`, `facts-20/`, `facts-40/`: resultados por pergunta, `predictions.jsonl` e resumo por variante. Registram a consulta, o pacote com proveniência e a reflexão completa.
- `shared/`: consulta-alvo gerada uma vez por pergunta.
- `attempts/`: uso de chamadas de todas as tentativas, inclusive interrompidas.
- `manifest.json`: hashes das fontes e do código, prompts, modelo e configuração. Retomada incompatível é recusada.

A execução completa, sem reparos, faz **1.064 chamadas de tarefa**: 152 consultas-alvo, 456 reflexões e 456 respostas. Com `-MemoryOnly`, são 608. Reparos acrescentam no máximo uma chamada por geração de consulta/reflexão em cada tentativa. O cache local pode eliminar requisições ao provedor. Tokens lógicos de cada variante incluem sua consulta-alvo; para gasto físico, não somar essa consulta três vezes. O relatório `execution_attempt_usage` distingue chamadas do cache local das demais. Retentativas de transporte pertencem ao backend existente.

O script retoma automaticamente a mesma pasta, pulando perguntas/variantes já salvas. Cada resultado é gravado atomicamente antes de atualizar o comparativo. Um lock permite apenas um escritor por pasta. Alterar modelo, perguntas, corpo dos fatos, prompts ou código exige outra pasta para manter o experimento coerente.

Exceção para correções de código: uma tentativa que ainda não tenha consulta compartilhada nem resultado por variante pode atualizar apenas seu hash de código na retomada. Configuração, fontes e prompts precisam permanecer idênticos. O manifesto anterior fica em `manifest-history/`, e os custos das chamadas já feitas ficam preservados em `attempts/`.

Quando o usuário autoriza corrigir bugs durante uma execução já iniciada, `-AllowCodeUpdate` permite continuar também com resultados parciais. Apenas o código pode mudar; modelo, fontes, perguntas, prompts e orçamentos precisam permanecer idênticos. Resultados concluídos ficam intactos, versões anteriores são arquivadas e resultados novos registram `implementation_code_hash`.

O F1 da run original aparece como referência nas mesmas perguntas, mas ela tinha reflexão conjunta no reader. O controle principal são as três variantes entre si. Como o respondedor ainda recebe todo o contexto original, seu F1 mede o efeito do reflector compacto sobre esse leitor; não demonstra, sozinho, suficiência da memória compacta. Construção inicial e tempo histórico do executor são reportados separadamente do custo novo de inferência.
