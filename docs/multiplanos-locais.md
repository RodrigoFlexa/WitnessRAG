# Múltiplos planos locais — versão experimental

Esta versão substitui geração/replanejamento/verificação/roteamento por LLM
por construção local de consultas e reranking de consulta + testemunha. O perfil
é `witnessrag-local`; o cascade e as quatro células da ablação de reflexão não
mudam. Não há resumo gerado nem reflexão de resumo. A única chamada generativa
em tempo de consulta é a do reader com reflexão conjunta existente.

## Versão atual: v2

O perfil `witnessrag-local` agora seleciona **v2**. A v1 foi preservada para
comparações; a opção `--local-plan-version v1` ativa o comportamento anterior.
O piloto com memória congelada usa `--plan-version v1` ou `v2` e grava a versão
no manifesto. Use pastas diferentes: uma run não pode mudar de versão ao retomar.

A v2 mantém múltiplas origens simultaneamente: entidades literais, objetos e
sujeitos de fatos relevantes. A presença de um nome não desativa as outras
origens. O perfil tem até 12 pontos iniciais, feixe 32, até 4.000 execuções por
pergunta, profundidade 3 e 96 planos enviados ao reranker. Esses são limites
iniciais de engenharia, sem calibração no teste do benchmark.

O orçamento de geração é independente da largura do reranking. Até 20% é
reservado às sementes e até 30% às interseções, que são avaliadas antes dos
caminhos. O restante é dividido pelas profundidades. O escalonamento alterna
origens; a fronteira conserva diferentes constantes e formas de consulta. Uma
expansão pode mover a resposta para outro nó ou manter a resposta enquanto
acrescenta um participante/qualificador. Execuções repetidas são memorizadas.

Um contrato local identifica operação, cabeça da resposta, papel sujeito/objeto,
participantes, conjunção explícita, escopo possessivo e unidade de contagem.
Ele permanece uma hipótese de gramática inglesa limitada. Os campos e as
checagens aparecem nos logs e no contexto do reader, junto à pergunta original.
Em uma conjunção, cada participante deve ocupar o papel solicitado no predicado
solicitado. `paint(A,x) AND love(B,x)` não satisfaz "ambos pintaram x".

O ranking passou a priorizar projeções compatíveis e conjunções completas. Para
`both/each`, a identidade ampla de um cluster não basta para afirmar um valor
específico compartilhado: exige-se suporte lexical entre os valores ligados.
Aliases semanticamente equivalentes sem esse suporte continuam disponíveis nas
falas, mas podem perder um plano elegível. Isso é uma restrição conservadora,
não um solucionador completo de correferência ou um verificador semântico.

O score combina verificações estruturais, relevância local e a curva temporal;
a penalidade fixa por quantidade de átomos foi removida. O score não é
probabilidade e pode ultrapassar 1. O reranker continua lendo programa, fatos e
falas originais, sem treino novo nem acesso a respostas ouro.
Antes do corte de testemunhas são filtrados os bindings incompatíveis; o
reranker recebe testemunhas por alternância entre planos, até `3 * candidates`
pares (288 no perfil). Um valor específico que aparece depois de dois bindings
genéricos deixa de ser descartado apenas pela ordem do arquivo.

Conjuntos admitem pacotes com múltiplas testemunhas completas. Contagens separam
menções por fonte e unem predicados duplicados quando entidade e data explícita
do evento coincidem; datas amplas ou de sessão não certificam identidade de
eventos. Os grupos são hipóteses de ocorrência, com completude pendente. Planos
futuros não testemunham perguntas que explicitamente pedem eventos ocorridos.

Além dos fatos e falas das testemunhas, uma busca BM25 sobre as falas originais
reserva suporte por participante e procura candidatos a antecedentes de
referências descritivas. Retorna texto literal com proveniência, em orçamento
separado de até `excerpt_max_chars`; não cria arestas nem fatos inferidos.
Antecedentes candidatos não são anunciados como referências resolvidas.
As falas também recebem a curva de proximidade temporal como prioridade suave,
sem transformar data de sessão em data de ocorrência. O ponto/janela resolvido
é apresentado ao reader. Um timestamp explícito fora da restrição não pode ser
a projeção de resposta temporal, mesmo quando o estado do fato é incerto.

O prompt compartilhado do reader e as quatro células da ablação de reflexão
permanecem iguais. Nesta versão o contexto contém instruções derivadas do
contrato para conferir papéis, combinar fontes, listar membros e contar
ocorrências. Existe uma única chamada `qa`, com a reflexão conjunta já existente.

## Construção e execução da v1 preservada

1. Identificar entidades presentes literalmente na pergunta. Na ausência delas,
   propor até três sujeitos dos fatos recuperados por embedding.
2. Construir átomos a partir dos predicados incidentes às entidades. A busca pode
   começar pelo objeto, mas nunca inverte semanticamente sujeito e objeto.
3. Executar cada consulta usando os groundings indexados por predicado e entidade
   e o `WitnessSearcher.join` existente. Igualdade das variáveis intermediárias
   usa a identidade de entidade já adotada pelo grafo.
4. Expandir os bindings encontrados em caminhos dirigidos, até três átomos por
   padrão; construir interseções de padrões com uma resposta em comum.
5. Manter um feixe diversificado por sequência de relações. Memorizar execuções
   repetidas; limitar execuções a `4 * candidates * depth`. O feixe é aproximado
   e os cortes são registrados.
6. Reordenar até 48 planos, com até duas testemunhas por plano, usando o mesmo
   cross-encoder local configurado em `fact_rerank`. O par contém o programa,
   os fatos e excertos literais. Manter até três pacotes com evidência adicional,
   sem deixar que testemunhas com os mesmos fatos ocupem todas as vagas.
7. Entregar os fatos de cada pacote integralmente dentro do orçamento de 40
   fatos. Completar por relevância e adicionar falas originais curtas, sem
   resumo. Uma cadeia que não cabe não é registrada como entregue integralmente.

O inventário de operações inclui valor, conjunto, contagem, data, duração,
primeiro/último, comparação e premissas. Conjuntos são enumerados pelo executor;
contagem/completude, comparação e inferência continuam responsabilidades do
reader. Tipos e requisitos da pergunta são hipóteses: este parser leve não
produz um contrato semântico completo nem estabelece que toda restrição foi
compreendida. A pergunta original acompanha o contexto.

## Curva temporal

Mantém-se `exp(-distância / escala)` e as escalas de `MemoryScorer`: janela
citada, presente da memória, início ou ponto de evento. Pesos normal/strong vêm
da configuração existente, sem importância. A curva reordena as testemunhas,
mas não torna um evento fora de uma restrição calendárica admissível.

A filtragem explícita `within/before/after` é conservadora: atua no primeiro
átomo ancorado, somente quando há expressão temporal de evento. A data de sessão
e o começo de um estado contínuo não são tratados como intervalo de vigência
completo. Átomos de ponte não recebem automaticamente o filtro do evento.
O escopo temporal completo continua pendente de checagem pelo reader.

Âncoras relativas a eventos são propostas por correspondência lexical em fatos
da entidade, com data explícita. Se houver mais de um intervalo candidato, a
âncora permanece não resolvida; não se escolhe silenciosamente o mais recente.
Primeiro/último e contagem não são anunciados como globalmente completos.

## Limites científicos

O reranker atual foi treinado para relevância textual, não para reconhecer
equivalência pergunta–programa. Sua pontuação não é probabilidade de correção.
Junção executada não demonstra fidelidade semântica à pergunta nem verdade da
extração. Os diagnósticos marcam os resultados como candidatos não verificados.
Treinar um pontuador de planos com negativos de papel, modalidade e escopo é uma
evolução possível; esta implementação não inventa que esse treinamento ocorreu.

Zero tokens de planejamento significa zero chamadas generativas dessas etapas.
Embeddings, cross-encoder e busca continuam consumindo CPU/GPU. Construção do
grafo permanece uma despesa anterior, a ser amortizada e reportada separadamente.

## Executar no servidor

Com Qwen2.5-14B já disponível na porta 8095, na raiz do repositório:

```bash
PORT=8095 GPU=1 LOCOMO_CONVERSATION=0 \
  bash scripts/run-local-plans-qwen.sh runs/local-plans-qwen-conv00
```

`0` é a primeira conversa. Para as dez, use `LOCOMO_CONVERSATION=all`.
A execução usa BGE-M3 e BGE reranker em GPU por padrão; mantenha `BENCH_PYTHON`
apontando ao ambiente do benchmark. O script existente retoma a mesma pasta.
Não inicie simultaneamente dois processos sobre a mesma pasta de saída.

Para comparar v1 e v2 em um comando, com os mesmos modelos e cache de extração:

```bash
PORT=8095 GPU=1 LOCOMO_CONVERSATION=0 \
  bash scripts/run-local-plans-comparison-qwen.sh runs/local-plans-comparison-conv00
```

As duas variantes são multiplanos locais. O script grava `v1/`, `v2/` e
`comparison.md`, com F1/BLEU pareados e tokens lógicos incluindo cache. Para
controle estrito da memória extraída, forneça o checkpoint compatível via
`WRAG_FROZEN_MEMORY_ROOT`, conforme o protocolo já existente do repositório.
Essa comparação é separada da ablação de reflexão de quatro células.

Pode-se usar o perfil também no script PowerShell existente:

```powershell
./scripts/run-witness-openai-locomo.ps1 -Method proof -Profile witnessrag-local `
  -Conversation 0 -Output runs/local-plans-openai-conv00
```

## Piloto com memória congelada e sem nova extração

`run-local-plans.py` aceita checkpoints confiáveis criados pelo repositório,
verifica checksum e identidade literal dos trechos e registra o modelo da
extração separadamente do reader. Os metadados de proveniência podem ter sido
enriquecidos entre versões, mas ids, títulos e textos precisam coincidir.

```powershell
python scripts/run-local-plans.py `
  --data-root runs/cascade-reflection-ablation/cascade `
  --memory-root runs/locomo-controlled-01 --conversation 0 `
  --retrieval-only --questions 12 `
  --reranker cross-encoder/ms-marco-MiniLM-L6-v2 `
  --output runs/local-plans-retrieval-smoke
```

Para respostas reais, troque `--retrieval-only` por
`--backend openai --model gpt-4o-mini`; `--questions 0` avalia a conversa inteira.
Para Qwen num endpoint existente, use `--backend openai`
`--model Qwen/Qwen2.5-14B-Instruct --base-url http://127.0.0.1:8095/v1`.
Use `--device cuda` e o reranker BGE para reproduzir o perfil do servidor.
Uma retomada exige `--resume` e manifesto idêntico, incluindo o código.

O piloto grava `predictions.jsonl`, `manifest.json`, `summary.json` e `report.md`.
Os logs preservam planos selecionados, oito candidatos do topo, fatos originais,
pontuação semântica/temporal, cortes e custo do reader. Uma guarda em execução
recusa qualquer chamada generativa diferente de `qa`.

## Validação local concluída da v1

O piloto completo está em `runs/local-plans-conv00-final/`: 152 perguntas da
primeira conversa (`conv00`), reader GPT-4o-mini, embeddings BGE-M3 e reranker
MiniLM em CPU. A memória congelada contém 948 fatos extraídos anteriormente por
Qwen2.5-14B. Não houve extração nova nem construção de resumo neste piloto.

| Categoria | n | F1 | BLEU-1 |
|---|---:|---:|---:|
| single-hop | 70 | 54,65 | 48,63 |
| multi-hop | 32 | 39,80 | 30,92 |
| temporal | 37 | 68,92 | 65,57 |
| open-domain | 13 | 35,96 | 30,58 |
| total | 152 | 53,40 | 47,48 |

A média foi de 2.169 tokens lógicos do reader por pergunta, incluindo chamadas
servidas pelo cache. Planejamento e resumo tiveram **zero chamadas e zero
tokens**, independentemente do cache. Foram construídos em média 452 programas,
reordenados 48 planos por pergunta e entregues três pacotes com fatos distintos.
A auditoria não encontrou perguntas duplicadas nem cadeias selecionadas com
fatos ausentes do contexto. A recuperação levou em média 6,77 s em CPU.

Esses números demonstram funcionamento e custo de consulta; não estabelecem
superioridade sobre o cascade. O modelo do reader e o reranker do piloto diferem
do perfil Qwen/BGE do servidor. Multi-hop continua um ponto a investigar numa
comparação pareada com a mesma extração, os mesmos modelos e as mesmas perguntas.
Não houve ajuste do prompt nem dos pesos a partir das respostas ouro deste piloto.

Os resultados brutos foram preservados. A auditoria em `audit.json` documenta
dois problemas de metadados corrigidos depois da rodada: o classificador antigo
registrava `motivo_parada="sem plano"` apesar dos planos locais presentes, e o
manifesto do piloto mantinha alguns campos globais padrão do harness. Os ids,
categorias, respostas e métricas das 152 linhas são de LoCoMo. Essas correções
não alteram a recuperação, os prompts, as respostas ou os valores acima.

## Validação da v2 final

Depois dos testes sem API, foram avaliadas as **32 perguntas multi-hop** da
primeira conversa. A comparação pareada reutiliza os resultados da v1, a mesma
memória congelada, reader GPT-4o-mini, embeddings BGE-M3 e reranker MiniLM em CPU.
As saídas finais estão em `runs/local-plans-v2-conv00-multihop-final/`.

| Versão | F1 | BLEU-1 | Tokens lógicos/pergunta |
|---|---:|---:|---:|
| v1 | 39,80 | 30,92 | 2.207 |
| v2 | 53,17 | 38,67 | 2.410 |
| diferença | +13,38 pp | +7,75 pp | +202 |

O F1 aumentou em seis perguntas, diminuiu em uma e empatou em 25. Houve zero
chamadas generativas de planejamento ou resumo. Em média foram construídos
1.387 programas, executadas 1.395 consultas e reordenados 198 pares
plano–testemunha. Construção: 0,61 s/pergunta em CPU; reranking: 3,89 s, com cache
local de pares. O tempo de recuperação total médio foi 4,61 s e não representa
uma medição de CPU fria nem de GPU. Tokens do reader incluem cache de LLM.

No exemplo das duas participantes que pintaram o mesmo tema, a primeira
consulta selecionada foi `paint(Caroline,x) AND paint(Melanie,x)` com binding
`sunset`. O reader respondeu `sunset` (F1 1), enquanto a v1 respondeu `feelings`
(F1 0). O livro recomendado passou da descrição genérica para `Becoming Nicole`.
Nem todas as falhas foram resolvidas: a data do passeio após a viagem continua
sem resposta adequada. A v2 deixou de projetar uma data explícita incompatível,
mas isso não produz evidência ausente nem certifica referências ambíguas.

A verificação final de 12 perguntas equilibradas (três por categoria) está em
`runs/local-plans-v2-conv00-final-check/`: single-hop 75,24→71,43; temporal
59,52→50,00; open-domain 50,26→51,85; multi-hop 55,56→55,56. A amostra é pequena
e **não demonstra melhora geral**. A queda temporal inclui uma resposta que
manteve a expressão relativa sem escrever o ano normalizado; a queda single-hop
inclui paráfrase e um item adicional. São limitações reais para F1/BLEU.

**452 testes passaram**, incluindo direção das relações, interseção de papéis,
projeção da resposta, aliases genéricos, escopo de filhos, conjuntos, ocorrências,
datas, associação plano–testemunha, ausência de acesso a metadados ouro e uma
única chamada `qa`. Uma rodada de diagnóstico foi interrompida ao encontrar um
bug de associação de filas; está marcada `invalid_aborted` e não entra nos
resultados acima. O comparador recusa essa pasta.

Esta é uma conversa de desenvolvimento. O ganho mede o pacote v2 (busca,
checagens e entrega ao reader), sem isolar a contribuição de cada componente.
O script do servidor compara v1/v2 com Qwen/BGE; a GPU não foi validada neste
ambiente. A comparação nas dez conversas é necessária antes de recomendar
substituição do método ou apresentar resultados de artigo.

## Protótipo lite: investigação adiada

Também existe a opção experimental `lite`, explicitamente separada da v1/v2.
Nenhum dos comandos do servidor acima a seleciona. Ela usa BM25 sobre fatos e
fontes literais, sem embedding da pergunta nem cross-encoder, e inspeciona os
átomos, papéis, proveniência e entrega integral das testemunhas por código.
O refletor registra suporte estrutural e pendências; nunca certifica suficiência
semântica nem completude de conjuntos. O respondedor permanece uma única chamada
`qa`, com inferência interna e saída curta no campo `answer`.

No piloto offline da primeira conversa, as 152 recuperações levaram em média
0,22 s em CPU, sem chamadas de inferência neural. A memória congelada já havia
sido construída: esse custo anterior não é zero. Nas mesmas 32 multi-hop da v2,
a cobertura média de falas anotadas caiu de 47,66% para 35,16%; essa é uma métrica
de recuperação, não F1 de resposta. Em oito respostas com GPT-4o-mini, o F1 caiu
de 73,21 para 65,71 frente à v2 nas mesmas perguntas. Não há evidência para
substituir a v2 por esse protótipo; sua investigação foi adiada.

Após a inclusão do protótipo, a suíte completa passou com **461 testes**.
Os scripts `audit-local-lite.py` e `run-local-plans.py --plan-version lite`
permitem reproduzir os diagnósticos usando uma memória congelada confiável.
Artefatos brutos de execução permanecem locais; os resultados resumidos estão
documentados aqui.
