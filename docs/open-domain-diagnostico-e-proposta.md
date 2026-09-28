# Open-domain: diagnóstico das runs e proposta de melhoria

Auditoria local em 27/09/2026. Não foram feitas chamadas ao modelo, alterações nos prompts de produção ou alterações nas predições originais. Os números foram recalculados das respostas salvas. Este documento apresenta evidência observada, interpretação e experimentos propostos; não apresenta ganhos de uma implementação nova.

O principal problema observado é a passagem de **pistas pessoais recuperadas para uma conclusão que exige conhecimento de mundo**. Há também erros de formato/tipo de resposta e um defeito comprovado na entrega das premissas. A primeira intervenção recomendada é preservar o robust e o cascade, corrigir essa entrega e testar um leitor que execute explicitamente a inferência. Os logs não justificam aumentar indiscriminadamente o número de planos.

## 1. Recortes, modelo e métricas

Os metadados das runs robust usam `Qwen/Qwen2.5-14B-Instruct`; os exemplos pareados inspecionados do cascade usam o mesmo deployment. O identificador de backend é `openai`, por ser uma interface compatível; isso não significa que o leitor é GPT-4o-mini.

Reproduzi a tabela enviada usando as primeiras 1.117 perguntas do cascade, em ordem de pasta de conversa e linha do JSONL, pareadas por `qid`. Esse prefixo reproduz também os quatro tamanhos de categoria. Os arquivos locais já contêm 1.235 perguntas pareadas, incluindo 77 open-domain. O robust tem as dez conversas completas, 1.540 perguntas e 96 open-domain. Não se deve comparar as 96 perguntas do robust às 77 do cascade como se fossem a mesma amostra.

| Recorte | n geral | n open-domain | F1 geral robust | F1 geral cascade | F1 open robust | F1 open cascade |
|---|---:|---:|---:|---:|---:|---:|
| Tabela enviada, reproduzida | 1.117 | 69 | 53,84 | 54,04 | 16,91 | 14,12 |
| Snapshot local mais recente | 1.235 | 77 | 54,08 | 53,89 | 19,19 | 16,31 |
| Robust completo, sem comparação pareada ao cascade | 1.540 | 96 | — | — | 24,49 | — |

Na tabela original, BLEU-1 geral é 47,08 versus 47,32; EM geral é 29,27 versus 29,90. A inversão de ordem geral no snapshot mais recente reforça que ainda não há um vencedor estável em qualidade. O bootstrap pareado por conversa dá IC95% de [-0,67; +1,32] para o delta geral de +0,21 do cascade no prefixo original; para open-domain, delta -2,80 e IC95% [-8,05; +3,52]. Com poucas conversas, esses intervalos são exploratórios.

O cascade conserva aproximadamente o F1 e reduz os tokens de inferência registrados: 9.643 para 6.747 por pergunta no prefixo original, aproximadamente 30%. Isso exclui a construção da memória e não é uma medição de latência ou custo monetário. Em open-domain, o leitor recebe quase o mesmo volume: 1.896 versus 1.891 tokens. A queda não se explica simplesmente por receber menos tokens.

O F1 segue a implementação local do avaliador oficial. O BLEU-1 local usa unigramas, stemming/normalização e penalidade de brevidade; sua comparabilidade com a tabela do artigo de referência requer conferir o código daquele protocolo. Não há razão para assumir automaticamente que todo BLEU-1 publicado usa esse mesmo pré-processamento.

## 2. O que os logs efetivamente mostram

O artigo do LoCoMo define open-domain como integrar informação fornecida por um falante com conhecimento externo, incluindo senso comum e fatos de mundo. Portanto, um fato recuperado pode ser apenas uma premissa da resposta, e não a própria resposta. [Artigo original, seção 4.1](https://arxiv.org/html/2402.17753v1#S4.SS1).

| Pergunta / caso | Pista presente no contexto efetivo | Resposta robust / cascade | Ouro | Diagnóstico |
|---|---|---|---|---|
| `conv-47:qa16`, identificar jogo de cartas | Cartas de cores/números; jogar carta de mesma cor ou número; comprar cartas | Descrição do jogo / mesma descrição | UNO | Recuperação das pistas funcionou; faltou identificação por conhecimento de mundo |
| `conv-47:qa30`, país dos bilhetes em julho | James comprou bilhetes para Toronto; viagem prevista para julho | Toronto / Toronto | Canada | Resposta tem tipo errado; falta a ponte cidade → país. Ambos passaram pelo plano |
| `conv-43:qa27`, técnica para preparar exames | Tim estuda 25 minutos e faz pausa de 5 | Pomodoro Technique / breaks | Pomodoro technique | As duas entregas contêm a pista decisiva; DIRECT perde o reconhecimento do conceito neste caso |
| `conv-43:qa3`, escolher autor entre Lewis e Greene | Preferência de Tim por fantasia, Harry Potter e GoT | likely yes / likely yes | C. S. Lewis | Pergunta de escolha respondida como binária; não é falta de uma preferência recuperada |
| `conv-43:qa66`, livro de Star Wars | Preferências literárias de Tim | The Hobbit / fantasy books | Star Wars: Jedi Apprentice… | No robust há prova no contexto, mas a resposta viola a condição Star Wars; cascade não nomeia um livro |

Referências dos exemplos nas runs originais:

- Robust: `runs/multiplan-comparison/witnessrag-robust/conversations/conv06/benchmark/20260927-222739-620843-conv06/locomo/witnessrag.jsonl`, linhas 44 (UNO) e 20 (Toronto).
- Cascade: `runs/multiplan-comparison/witnessrag-cascade/conversations/conv06/benchmark/20260927-235541-368195-conv06/locomo/witnessrag.jsonl`, mesmas linhas.
- Robust: `runs/multiplan-comparison/witnessrag-robust/conversations/conv04/benchmark/20260927-212854-682187-conv04/locomo/witnessrag.jsonl`, linhas 78 (Pomodoro) e 57 (escolha de autor).
- Cascade: `runs/multiplan-comparison/witnessrag-cascade/conversations/conv04/benchmark/20260927-232109-720970-conv04/locomo/witnessrag.jsonl`, mesmas linhas.
- Todos os casos pareados, incluindo o livro de Star Wars, estão em `runs/open-domain-audit/cases.json`, com caminho e linha de cada origem.

No recorte de 69 perguntas, o robust tem 46 respostas com F1 zero e 16 respostas `insufficient information`; o cascade tem 51 e 18, respectivamente. As abstinências estão incluídas no total de zeros, não são uma parcela adicional.

No robust, 34/69 encerram com `respostas_demais` e 18/69 têm prova no contexto. Destas 18, **13 ainda têm F1 zero**. No cascade, as oito perguntas open-domain com prova no contexto têm F1 zero. Isso evidencia que uma prova recuperada não equivale à validação da operação inferencial ou da resposta final. Não prova que todas essas respostas são semanticamente erradas: há também desencontros lexicais com o ouro.

Minha interpretação é que o controlador frequentemente tenta provar uma relação final como `live in`, `use` ou `enjoy`. Em open-domain essa relação pode não existir literalmente no grafo: o que existe são pistas para inferi-la. Alternativas de relação ajudam a localizar premissas, mas não executam sozinhas a conversão Toronto → Canada, regras do jogo → nome do jogo, ou preferência por fantasia → escolha entre autores.

### Defeito comprovado: premissas preparadas e descartadas

Em `wrag/methods/witnessrag.py`, a etapa `abductive_premises` registra `diagnosticos['premissas']` e acrescenta um bloco com falas originais a `extras` (aproximadamente linhas 2563–2568). Depois, `fact_delivery` substitui a lista inteira:

```python
extras = [{"title": "Facts from the memory", "text": facts_text}]
```

Esse reset, aproximadamente na linha 2587, descarta o bloco anterior. No prefixo original, **19 perguntas open-domain do robust registram seleção de premissas, mas zero recebem o bloco de premissas no contexto final registrado**. No snapshot mais recente, são 20/77; no robust completo, 25/96. Alguns fatos selecionados podem reaparecer por outras rotas na entrega por fatos; o defeito comprovado é o descarte do bloco especializado, não a ausência de toda premissa dessas perguntas.

Há também substituição de um eventual bloco `plan_to_reader`, mas essa opção está desligada nas runs inspecionadas e não explica seus resultados.

### O leitor permite inferência, mas a instrução fica ambígua

O `QA_FACTS` herda `QA_EVIDENCE_TEMPLATE`: ele já permite conhecimento comum e diz para não abster quando a inferência não é citada literalmente. Ao mesmo tempo, pede resposta curta, proíbe inventar fatos ausentes e apresenta os dados como fatos da memória. Minha hipótese é que o Qwen frequentemente aplica uma leitura predominantemente extrativa. Essa hipótese é compatível com UNO, Toronto e Pomodoro, mas só um replay controlado do leitor pode medir sua contribuição causal.

A normalização em `wrag/eval/reader.py` também considera perguntas iniciadas por `Would`, `Could`, etc. elegíveis para reduzir respostas iniciadas por yes/no. Ela não distingue escolha entre alternativas. No caso Lewis/Greene, o log registra `likely yes`; não há informação suficiente para atribuir a origem dessa resposta exclusivamente ao normalizador. O problema observável é o contrato de resposta binária aplicado a uma tarefa de escolha.

### DIRECT é busca simples; inferência não é necessariamente simples

O roteador atual considera opiniões, razões e questões sobre o que é provável como DIRECT. A decisão olha somente a pergunta e está baseada na necessidade de compor fatos da memória. Isso é útil para economizar busca, mas não descreve a necessidade de aplicar conhecimento de mundo: uma única pista pode exigir uma inferência conceitual.

No prefixo original, a decomposição pareada é:

| Rota escolhida pelo cascade | n | F1 robust nessas mesmas perguntas | F1 cascade | Delta |
|---|---:|---:|---:|---:|
| DIRECT | 44 | 20,99 | 15,29 | -5,70 |
| PLAN | 25 | 9,74 | 12,06 | +2,32 |

A perda se concentra no subconjunto DIRECT. Isso não significa que encaminhar todas essas perguntas para PLAN resolveria o problema, nem que as perguntas PLAN são intrinsecamente piores. Os subconjuntos têm dificuldades diferentes. Os contextos finais também não são idênticos entre as variantes, e o hash de código difere em algumas conversas; a associação com a rota não é um isolamento causal do efeito do roteador.

Na ablação completa, com as mesmas 96 open-domain entre suas cinco variantes, o F1 é: full **20,09**, no-plan **20,31**, no-proof **20,25**, no-verify **21,85**, no-temporal-score **19,74**. Esses valores pertencem a outra comparação; não se deve calcular seus deltas contra o robust 16,91 do prefixo parcial. A ablação sugere que simplesmente retirar ou ampliar planos/provas não ataca o gargalo principal. O resultado no-verify, isoladamente, não justifica remover a verificação.

### Métricas e perguntas subdeterminadas

F1 e BLEU-1 penalizam palavras diferentes: `progressive`/`left` versus `Liberal` podem ter pontuação zero, mesmo quando a interpretação se aproxima. Recomendações também podem admitir várias respostas plausíveis, enquanto o ouro contém uma escolha específica. Isso demanda auditoria semântica complementar, mas não explica erros claros de país, opção ou assunto do livro.

Não recomendo acrescentar justificativa a todas as respostas. O avaliador oficial da categoria 3 usa apenas o trecho do ouro antes de `;`; explicações indiscriminadas podem piorar precisão lexical. Essa regra foi verificada no [avaliador original](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py#L186-L192). A necessidade de responder com razão deve vir da pergunta, não do rótulo da categoria.

O caso `conv-44:qa43` (estado de Audrey/Andrew, ouro Minnesota) é menos seguro que Toronto. O corpus contém a pista `Fox Hollow`, ausente dos fatos finais inspecionados; não encontrei Minnesota/Voyageurs explicitamente nessa busca textual. O nome é potencialmente ambíguo e não demonstra sozinho o estado. Portanto, não o considero uma correção garantida nem proponho memorizar uma associação específica para satisfazer esse ouro.

## 3. Proposta: separar busca de premissas e inferência da resposta

Preservar o plano robusto e o roteamento atual como baseline. A extensão deve ser opcional e inicialmente concentrada na entrega/leitura. Uma evolução coerente com a teoria de planos é distinguir **plano de evidência** de **operação de conclusão**: o primeiro busca fatos verificáveis da conversa; a segunda aplica uma ponte de conhecimento e produz o tipo solicitado. Uma conclusão provável não deve ser armazenada como um fato comprovado da conversa.

### A. Corrigir a entrega das premissas dentro do orçamento

Fazer a seleção abductiva alimentar `_fact_context`, em vez de criar um bloco que será apagado. Reservar inicialmente até oito fontes dentro do limite existente de fatos/tokens, substituindo itens de menor prioridade. Preservar falante, negação, modalizadores e identificadores de origem. Quando a paráfrase apaga uma pista essencial, substituir parte do orçamento por uma citação curta da fala original. Não basta concatenar todo o bloco: isso confundiria o ganho de seleção com o ganho de contexto maior.

Registrar IDs selecionados e efetivamente entregues. Usar esses IDs para medir cobertura do contexto que o leitor realmente recebe; os `pids` de trechos preservados apenas para métricas não garantem que todos os seus fatos tenham sido entregues no modo `fact_delivery`.

### B. Introduzir uma operação de resposta independente de PLAN/DIRECT

A representação pode distinguir busca `DIRECT`/`PLAN` e leitura `EXTRACT`/`INFER`, com tipo de saída e opções explícitas quando presentes. O detector usa somente pergunta e contexto recuperado, nunca `categoria_locomo`, ouro ou evidência anotada.

O leitor INFER executa, em uma chamada curta, as operações:

1. identificar o tipo pedido e as opções/qualificadores obrigatórios;
2. selecionar as premissas pessoais relevantes;
3. aplicar conhecimento comum à ponte necessária;
4. escolher uma resposta compatível com tipo, opções e condições.

Registrar separadamente premissas da memória e ponte de conhecimento de mundo. A saída ao avaliador continua sendo apenas a resposta curta. Para escolha entre autores, retornar um autor; para país, um país; para uma técnica, seu nome. Preferências inferidas devem permanecer probabilísticas. Quando não houver pista suficiente, conservar a abstinência.

Começar com o conhecimento paramétrico do mesmo Qwen, sem mudar o modelo ou adicionar busca web. Não codificar tabelas de respostas do LoCoMo e não usar os exemplos desta auditoria como demonstrações no prompt; usar exemplos sintéticos de habilidades equivalentes. Se depois houver necessidade de uma base externa, sua fonte, versão e custo precisam ser congelados e reportados.

### C. Adaptar planos para pedir premissas quando a conclusão não está no grafo

Somente depois de validar A/B: quando a resposta não é explicitamente sustentada pelo grafo, permitir um pequeno plano de busca de pistas e manter a operação de inferência como etapa distinta. As leituras alternativas devem procurar famílias diferentes de evidência relevante, sem pressupor a resposta: preferências e contraexemplos para escolhas; destino e data para país; regras/características para identificar objetos.

Inicialmente limitar a duas hipóteses/leituras de evidência e no máximo uma expansão adicional, acionada por pista ausente, contradição ou tipo incompatível. Não retomar o controlador multiplano amplo que perdeu para o robust sem uma ablação que justifique isso. A falta de prova literal da conclusão não deve automaticamente disparar vários replanejamentos se já há premissas suficientes para uma inferência.

### D. Validar formato sem exigir que toda conclusão esteja citada

Uma verificação barata deve distinguir pergunta binária de escolha, nome de conceito de descrição e país de cidade. Deve também rejeitar violações explícitas, como um livro que não pertence ao universo solicitado. Não exigir citação literal da resposta inferida, pois isso reintroduziria o problema original. Se houver revisão por modelo, ela precisa conhecer a separação entre premissas pessoais e ponte de mundo e entrar na contagem de tokens.

## 4. Experimento pequeno antes de uma rodada completa

Primeiro, um replay com o contexto final congelado de **24 perguntas de desenvolvimento**, selecionadas antes dos resultados novos: geografia, identificação de conceitos, escolhas e inferências de perfil, além de controles com evidência insuficiente. Os casos revisados nesta auditoria são desenvolvimento; não podem ser apresentados como teste independente.

O baseline já está salvo. Comparar nele o leitor atual e o leitor INFER: aproximadamente 24 novas chamadas, usando o mesmo deployment. Esse teste separa a capacidade do leitor de novos efeitos de recuperação e permite descobrir se o conhecimento paramétrico necessário existe.

Se houver sinal consistente, testar separadamente a correção da entrega de premissas e depois a combinação. Congelar memória/indexação, versão de código, orçamento de fatos/tokens, temperatura, limites de saída e conjunto de perguntas. O baseline também deve ser reexecutado nessa etapa para medir estabilidade. Incluir um pequeno conjunto de controle single-hop/temporal, para verificar se o novo formato prejudica casos que já funcionam.

Registrar além de F1/BLEU-1/EM: pistas efetivamente entregues, resposta com tipo correto, contradições, abstinências e tokens. Uma revisão semântica manual de uma amostra deve separar erro de conhecimento de desencontro lexical; ela complementa, não substitui, as métricas oficiais.

Só promover a opção após ganho repetido no replay e nas conversas reservadas, sem regressão material nos controles. Depois rodar as dez conversas, com todas as opções congeladas, e reportar comparação pareada e intervalos por conversa. Como várias dessas conversas já serviram para formular hipóteses, declarar o uso de desenvolvimento e reservar dados adicionais para uma alegação forte de generalização. Nenhuma promessa de alcançar a tabela de estado da arte é sustentada pelos logs atuais.

## 5. Reproduzir esta auditoria

Na raiz do repositório, sem acesso ao endpoint do modelo:

```powershell
python scripts/audit-open-domain.py --limit-paired 1117 --output runs/open-domain-audit/snapshot-1117
python scripts/audit-open-domain.py
```

Cada saída contém `summary.json`, `cases.json` (perguntas, respostas, diagnósticos e referências aos logs) e `questions.tsv`. O script valida a identidade das perguntas e do ouro e recalcula as métricas; o limite reproduz o prefixo parcial. O bootstrap e a contagem de tokens reutilizam `scripts/paired-report.py`.

Confiança alta: descarte das premissas, incompatibilidades de tipo/operação nos exemplos e existência de pistas já entregues. Confiança moderada: predominância de uma leitura extrativa e insuficiência do critério atual de DIRECT para inferências. Ganho da correção e do leitor proposto: ainda precisa ser medido.
