# Multiplanos como núcleo do WitnessRAG

Proposta de evolução teórica e protocolo de validação — 27/09/2026.
**Este documento descreve o desenho de pesquisa. Uma primeira implementação
experimental está em [multiplanos-experimental.md](multiplanos-experimental.md);
não há ainda melhoria de F1 demonstrada para ela.** As correções da revisão estão em
[revisao-plano-robusto.md](revisao-plano-robusto.md).

## Tese que vale perseguir

Uma pergunta pode exigir a mesma informação por diferentes representações da
memória. Um único plano pode falhar porque escolheu o predicado, a ancoragem de
entidade, a decomposição ou o caminho errado. Um planejador competente deve
produzir um **conjunto pequeno de planos complementares, fiel à pergunta e
executável sobre a memória disponível**, e usar os resultados da execução para
decidir se já há evidência suficiente ou se uma revisão localizada é necessária.

O objetivo principal continua sendo melhorar o desempenho de QA no LoCoMo.
F1, BLEU, orçamento e latência são os resultados finais. Reranking e apresentação
de datas ficam controlados nos experimentos que avaliam a contribuição dos
multiplanos, para que o ganho não seja atribuído ao componente errado.

“Adaptável a qualquer tipo de pergunta” deve significar reconhecer a operação
necessária e construir uma estratégia adequada, inclusive quando a resposta
depende de inferência ou falta informação. Não significa transformar toda
pergunta em uma consulta conjuntiva ou garantir resposta correta para qualquer
entrada sobre uma memória incompleta.

## O que o sistema atual faz e o que ainda falta

A implementação atual possui um plano principal, sinônimos por átomo e leituras
alternativas tentadas quando a principal não produz uma prova utilizável, ou
quando ela é recusada. Isso é uma boa base de recuperação diante de falhas de
representação. Ainda não é um conjunto de estratégias coordenadas para cobrir
as exigências da pergunta.

Auditoria das predições robustas da conv05:

- 195 chamadas de planejamento, incluindo replanejamentos.
- 128 leituras alternativas registradas ao longo dos ciclos.
- 91/128 mantêm exatamente a mesma sequência de sujeitos, objetos e posições
  de tempo da principal; 37 mudam essa estrutura. Isso mede apenas a estrutura
  literal dos argumentos: não prova que as 91 sejam redundantes, pois tipos,
  operadores e relações também podem mudar.
- Só 3 perguntas encerram com uma leitura alternativa aceita, incluindo uma
  cuja verificação foi dispensada pelo código anterior.
- 21/30 perguntas multi-hop terminam com plano de um átomo. A categoria do
  benchmark, portanto, não identifica diretamente composição no executor.

Há três limitações concretas. Os exemplos às vezes ensinam a relaxar a pergunta
para obter uma prova; a busca para assim que a principal basta, mesmo quando
outro plano poderia trazer membros adicionais de uma lista; e o período
temporal é sobretudo um sinal de ranking, não uma álgebra temporal completa.

## Separar quatro objetos

### 1. Requisitos da pergunta

Extrair um contrato C(q), comum aos planos de uma mesma interpretação:

- entidade e papel de cada participante;
- variável e tipo de resposta;
- operação: localizar, enumerar, contar, selecionar extremo, comparar, calcular
  tempo, reunir premissas para inferência;
- restrições de tempo, local, companhia, origem e modalidade;
- exigência de cobertura e condições que a resposta precisa satisfazer.

Cada restrição deve apontar para as palavras da pergunta que a justificam.
Resoluções vindas da memória precisam apontar para sua evidência. Isso não torna
o contrato automaticamente correto: sua fidelidade também deve ser avaliada.

O contrato é fixado durante a comparação de planos equivalentes. A dificuldade
de encontrar testemunhas não autoriza retirar uma restrição. Se a pergunta é
realmente ambígua, manter contratos alternativos identificados; não misturar
respostas de interpretações incompatíveis como se fossem a mesma consulta.

### 2. Variação lexical dentro de um plano

Alternativas como `reside in` e `live in` podem ajudar a casar predicados.
Devem preservar papéis, modalidade e significado relevante. `own` não substitui
`play`; `work at` não substitui `lead`; `move to` não identifica necessariamente
a residência atual. Relações próximas mas não equivalentes podem ser pistas
de recuperação, sem serem automaticamente provas da restrição original.

Esse nível deve ser resolvido por alternativas no átomo, sem consumir uma
leitura inteira apenas para repetir uma paráfrase.

### 3. Planos lógicos alternativos

São representações distintas da mesma necessidade informacional: relação direta,
cadeia por entidade intermediária, interseção ou ligação através de um evento.

Exemplo: “Em qual cidade fica o laboratório dirigido por Omar?”

- Plano A: `lead(Omar, ?lab) ∧ located_in(?lab, ?city)`.
- Plano B: identificar o laboratório numa proposição que diz “Omar dirige X” e
  usar a localização de X, com a resolução nominal ligada à fala de origem.
- Uma proposição direta “Omar dirige um laboratório em Recife” pode sustentar
  uma rota compacta, desde que a evidência cubra **liderança e localização do
  laboratório**, não apenas a cidade onde Omar trabalha ou mora.

Não exigir que a extração produza todos esses predicados. A geração deve ler
uma visão compacta do esquema e das vizinhanças relevantes do grafo: predicados,
direção, tipos observados e exemplos de fatos, sem acesso a respostas ouro.
Uma evidência inicial insuficiente também não deve proibir o planejador de
procurar uma ponte fora dessa vizinhança inicial.

### 4. Estratégias de execução

O mesmo plano lógico admite começar por uma entidade mais seletiva, propagar
bindings pela cadeia ou intersectar conjuntos. Essa escolha deve usar os
índices e as cardinalidades observadas, não exigir que o LLM invente detalhes
de execução. O executor atual já oferece junções e aterramento condicionado a
bindings; a evolução deve aproveitar isso e compartilhar subconsultas entre
planos, em vez de refazer toda busca para cada leitura.

Uma inversão da ordem de busca não autoriza inverter semanticamente sujeito e
objeto. Traversal reverso e predicado inverso são operações diferentes.

## Estratégia e parada por operação

| Necessidade | Uso dos multiplanos | Condição de parada ou limite |
|---|---|---|
| Single-hop | Desambiguar entidade/predicado e localizar suporte direto | Uma resposta sustentada, sem conflito relevante detectado e com todas as restrições cobertas |
| Cadeia multi-hop | Procurar pontes por rotas alternativas, mantendo bindings | Testemunha conectada que cubra a pergunta; joins parciais não bastam |
| Lista/interseção | Combinar membros verificados obtidos por planos compatíveis | Saturação sob orçamento; declarar que isso não demonstra completude global |
| Quando | Localizar evento e preservar expressão original, data de sessão e granularidade | Suporte para o evento correto e a precisão pedida |
| Antes/depois, duração, primeiro/último | Recuperar os eventos comparáveis e executar a operação temporal adequada | Relação temporal demonstrável; intervalos sobrepostos podem deixar a ordem indeterminada |
| Contagem | Enumerar entidades/eventos distintos e verificar o escopo | Distinguir contagem comprovadamente completa de contagem dos itens recuperados |
| Hipótese/open-domain | Planos de suporte e de possível contradição | Síntese baseada em premissas, identificada como inferência; não como prova dedutiva do grafo |
| Informação ausente | Diagnosticar a lacuna e tentar outra representação | Orçamento esgotado ou ausência de nova evidência; reconhecer insuficiência |

Não há necessidade de fornecer o rótulo LoCoMo ao sistema. A operação deve ser
inferida da própria pergunta. Algumas perguntas combinam operações: conjunto
mais janela temporal, cadeia mais comparação, entidade mais mudança de estado.
Logo, a representação deve compor operadores, não simplesmente escolher uma
das quatro categorias do benchmark.

## Seleção, verificação e adaptação

**A unidade de seleção deve ser evidência para requisitos, não apenas o plano
com maior suporte bruto.** Uma consulta ampla e curta pode casar muito bem e
continuar respondendo à pergunta errada. Primeiro verificar elegibilidade:
quais requisitos cada testemunha satisfaz, quais dependem da fala original e
quais permanecem sem suporte. Só então usar relevância, novidade e custo.

Para uma resposta singular, planos concordantes podem corroborar a evidência;
planos conflitantes pedem resolução, não votação cega. Testemunhas que derivam
da mesma fala não são evidências independentes. Para uma lista, planos
equivalentes podem contribuir com membros distintos verificados. Não unir
respostas de hipóteses semânticas incompatíveis.

O retorno do executor ao planejador deve identificar o defeito:

- relação sem aterramento → revisar lexicalização ou representação;
- junção interrompida → preservar o prefixo sustentado e procurar a ponte;
- resposta fora do tipo → reparar essa condição sem trocar o objeto pedido;
- período incompatível → localizar o evento correto ou rever sua resolução;
- resposta ambígua → recuperar evidência discriminativa;
- conjunto incompleto sob o orçamento → procurar membros novos;
- resposta recusada → reutilizar o motivo explícito da recusa.

A revisão deve modificar somente a parte que falhou quando possível. O número
de planos/ciclos é um orçamento, não uma obrigação de produzir três leituras
mesmo quando há uma interpretação evidente. Para listas, a primeira prova
utilizável não é automaticamente uma boa regra de parada.

O pacote entregue ao leitor deve preservar testemunhas compostas como unidades
quando isso couber no orçamento. Um reranker de fatos individuais pode remover
a ponte menos parecida com a pergunta e destruir a utilidade do plano. O registro
de fontes acrescentado nesta revisão permite medir essa perda; trocar a política
de orçamento exige um experimento específico.

## O que pode ser afirmado formalmente

Se Q1...Qm preservam o mesmo contrato e cada testemunha é correta **em relação à
memória formalizada**, a união de suas respostas corretas continua obedecendo
a esse contrato. Antes da seleção por orçamento, adicionar um plano mantendo
os anteriores não reduz o conjunto de candidatos recuperados.

Essas propriedades são condicionais. Não provam que o LLM extraiu corretamente
o contrato, que a memória representa a realidade, que a similaridade denota
implicação, que o conjunto está completo, ou que o F1 do leitor aumentará.
Com orçamento finito e leitor probabilístico, nem sequer a monotonicidade da
qualidade final decorre da união. O texto científico deve distinguir essas
garantias formais das hipóteses empíricas.

Uma formulação do objetivo do controlador é maximizar a cobertura verificada
dos requisitos e, para enumerações, a cobertura de respostas sustentadas, sob
limites de tokens e execução. A fidelidade ao contrato é condição de
elegibilidade, não um peso que um escore alto de similaridade pode compensar.

## Como demonstrar competência do planejador

O experimento final deve manter memória, reranker, apresentação temporal e leitor
fixos, comparando:

1. Entrega robusta sem plano.
2. Um plano forte com o mesmo orçamento de busca.
3. Várias amostras do mesmo planejador, controlando o número de chamadas.
4. Multiplanos atuais.
5. Multiplanos orientados por contrato e falhas do executor.

Controle importante: dar ao plano único candidatos/beam equivalentes, para não
atribuir a “diversidade de planos” um ganho explicado apenas por mais busca.

Além de F1/BLEU/custo, medir separadamente:

- fidelidade do contrato e das leituras à pergunta;
- executabilidade e correção de bindings/tipos;
- diversidade útil: evidência válida exclusiva acrescentada por cada plano;
- recall de suporte condicionado à informação existir na memória;
- testemunha encontrada versus testemunha efetivamente entregue;
- conflitos, falsos positivos de prova e falhas corrigidas por replanejamento.

O diagnóstico pode usar gabarito **depois da execução**; o método não pode usá-lo
para decidir planos, rotas, parada ou seleção. Casos sintéticos contrastivos
servem para testar preservação semântica: possuir/tocar, trabalhar/liderar,
planejar/realizar, ir sozinho/ir acompanhado, sessão/evento, primeiro/recente.
Eles não substituem QA real e não bastam para anunciar generalização.

A primeira mudança de prompt a testar deve ajustar instruções, esquema e
**todos os exemplos** simultaneamente para refletir essa representação. O
histórico do próprio projeto mostra que acrescentar campos opcionais sem
exemplos consistentes não é um teste adequado do conceito. O novo prompt deve
permanecer experimental até passar pelos contrastes e pelo piloto pareado.

## Relação com trabalhos anteriores

Planejar recuperação, adaptar esforço à pergunta e navegar grafos já são linhas
existentes: [PlanRAG](https://arxiv.org/abs/2406.12430),
[Plan*RAG](https://arxiv.org/abs/2410.20753),
[Adaptive-RAG](https://arxiv.org/abs/2403.14403) e
[Think-on-Graph](https://proceedings.iclr.cc/paper_files/paper/2024/hash/10a6bdcabbd5a3d36b760daa295f63c1-Abstract-Conference.html).
Portanto, “gerar vários planos” isoladamente não deve ser apresentado como
novidade estabelecida.

O diferencial a investigar é sua combinação precisa para memória conversacional
imperfeita: contratos explícitos, planos logicamente compatíveis, testemunhas com
tempo e proveniência, compartilhamento de subconsultas e revisão orientada por
lacunas verificáveis. Isso é uma hipótese de contribuição; esta revisão não é
uma busca exaustiva de anterioridade nem demonstra novidade por si só.
