# WITNESS-RAG — proposta teórica

> Transcrição organizada da discussão que originou o método, para que a teoria
> fique versionada junto do código. As referências a arquivos e seções do
> repositório foram acrescentadas na coluna da direita de cada bloco.

**Ideia central.** Construir e manter a memória para preservar demonstrações das
respostas às perguntas relevantes; recuperar essas demonstrações por algoritmos
que respeitem a estrutura lógica da pergunta.

Uma **testemunha** é um conjunto de fatos suficiente para demonstrar uma
resposta.

---

## 1. Semântica antes de topologia

Considere um corpus $D$ e uma interpretação de referência $G^\dagger$, contendo
os fatos que o corpus sustenta sob um esquema previamente definido. O símbolo
$\dagger$ não significa que conhecemos perfeitamente essa interpretação: ela é o
alvo semântico, anotável em pequenos conjuntos de avaliação e aproximado pelo
extrator no sistema real.

Relações tipadas, por exemplo:

$$\mathrm{trabalhaEm} : \mathit{Pessoa} \times \mathit{Instituição}, \qquad
\mathrm{localizadaEm} : \mathit{Instituição} \times \mathit{Cidade}.$$

Cada ocorrência de um fato carrega também

$$f = (\text{predicado}, \text{argumentos}, \text{fonte}, \text{trecho}, \text{tempo}, \text{status}),$$

o que permite distinguir "Ana trabalha na Atlas" de "Ana trabalhou na Atlas em
2018", ou de "um documento afirma, sem confirmação, que Ana trabalha na Atlas".

Para relações com vários participantes (uma transferência com remetente,
destinatário, objeto e data), usar relações de maior aridade ou nós de evento.
Decompor tudo em pares independentes cria combinações que o texto nunca afirmou.

> **Não existe topologia capaz de corrigir uma semântica que já foi perdida na
> extração.**

Também é necessário fixar a representação antes de contar hops: uma relação
representada por nó de evento pode exigir duas arestas físicas embora corresponda
a uma única afirmação semântica.

*No código:* `wrag/ie.py` (`Fact` com `pid` como fonte/trecho e `confidence` como
status). A tipagem é **aprendida**, não fixada: relações são canonizadas por
embedding em `wrag/graph.py`.

---

## 2. Single-hop e multi-hop como classes de consulta

Consultas conjuntivas positivas:

$$q(\bar x) = \exists \bar y \ \bigwedge_{i=1}^{m} R_i(\bar z_i).$$

Base de exemplo:

| fato | afirmação |
|---|---|
| $e_1$ | Ana trabalha na Atlas |
| $e_2$ | Atlas fica em Recife |
| $e_3$ | Ana pesquisa Óptica |
| $e_4$ | Bruno trabalha na Atlas |
| $e_5$ | Bruno pesquisa Óptica |

- Single-hop: $q_1(x) = \mathrm{trabalhaEm}(\text{Ana}, x)$
- Cadeia: $q_2(c) = \exists i\ \mathrm{trabalhaEm}(\text{Ana}, i) \wedge \mathrm{localizadaEm}(i, c)$
- Interseção: $q_3(p) = \mathrm{trabalhaEm}(p, \text{Atlas}) \wedge \mathrm{pesquisa}(p, \text{Óptica})$

A última exige que a **mesma** pessoa satisfaça as duas condições. Recuperar um
funcionário da Atlas e outra pessoa que pesquisa Óptica não basta. Esse é o
motivo para não identificar multi-hop com "encontrar um caminho": algumas
perguntas envolvem caminhos, outras envolvem ramificações, interseções e
restrições compartilhadas.

**Fora do escopo inicial:** negação por ausência, contagens, comparações
numéricas e perguntas globais ("quais são os principais temas?"). Essas extensões
exigem semânticas e garantias adicionais.

*No código:* `wrag/witness/query.py`. `ConjunctiveQuery.shape()` classifica em
`single-hop` / `chain` / `intersection`, e o relatório separa os resultados por
essa classe. `aggregation` registra quando a pergunta pede comparação — a busca
ainda traz os fatos a comparar, mas o sistema não reivindica garantia sobre a
comparação.

---

## 3. Grafo ótimo, definido de forma verificável

Seja $Q$ a classe de consultas de interesse. Dois grafos são equivalentes para
essa classe quando

$$G_1 \equiv_Q G_2 \iff \forall q \in Q,\ \mathrm{Ans}(q, G_1) = \mathrm{Ans}(q, G_2).$$

A memória ideal de menor custo é

$$G^* \in \arg\min_{G \in \mathcal{G}_{adm}} C(G) \quad \text{sujeito a} \quad G \equiv_Q G^\dagger,$$

onde $\mathcal{G}_{adm}$ define as representações permitidas e $C(G)$ contabiliza
armazenamento, incluindo identificadores e evidências. Isso responde "ótimo em
que sentido?": **menor custo, preservando exatamente as respostas de uma classe
especificada**.

**Limite importante.** Se permitirmos todas as perguntas atômicas sobre fatos
específicos ("Ana trabalha na Atlas?"), nenhum fato distinto pode ser apagado de
uma representação explícita sem alterar alguma resposta. A demonstração é
imediata: se dois grafos diferem no fato $R(a,b)$, a consulta $R(a,b)$ os
distingue. Portanto, redução sem perdas precisa explorar restrições nas
perguntas, redundâncias demonstráveis, ou codificação comprimida com
decodificação. **Não se pode prometer um grafo esparso que preserve
arbitrariamente toda a informação interrogável do texto.**

---

## 4. Testemunhas mínimas

Fixado o esquema e as identidades das entidades, seja $F$ a base de referência e
$S \subseteq F$ a memória conservada. Para uma resposta $a$ à consulta $q$:

$$\mathcal{W}_{q,a} = \{ W \subseteq F : W \models q(a) \text{ e } W \text{ é mínimo por inclusão} \}.$$

No exemplo: "Recife" para $q_2$ tem testemunha $\{e_1, e_2\}$; "Ana" para $q_3$
tem $\{e_1, e_3\}$; "Bruno" para $q_3$ tem $\{e_4, e_5\}$.

"Mínimo por inclusão" significa que nenhum fato pode ser removido sem perder a
suficiência. Não significa que seja a testemunha de menor custo.

**Propriedade fundamental:**

$$\boxed{\ a \in \mathrm{Ans}(q, S) \iff \exists W \in \mathcal{W}_{q,a} : W \subseteq S\ }$$

Vale para bases finitas e consultas conjuntivas positivas, na semântica fixada.
A prova tem duas direções: se uma testemunha está em $S$, seus fatos demonstram a
resposta; se a resposta existe em $S$, uma atribuição das variáveis fornece fatos
suficientes, e removendo redundâncias chega-se a uma testemunha mínima.

Esse resultado transforma **preservar respostas** em **preservar conjuntos
completos de evidências**. Há antecedente direto na teoria de bancos de dados
(Hu e Sintos, *Finding Smallest Witnesses for Conjunctive Queries*), então esta
parte é fundamento conhecido, não reivindicação de novidade.

*No código:* `wrag/witness/search.py`, função `_minimal_only`.

---

## 5. Memória sob orçamento

Demandas $t = (q, a)$ com pesos $w_t \ge 0$:

$$U(S) = \sum_t w_t \, \mathbf{1}[\exists W \in \mathcal{W}_t : W \subseteq S], \qquad
\max_{S \subseteq F} U(S) \ \text{ s.a. } \sum_{e \in S} c_e \le B.$$

**A utilidade não é submodular.** Para uma demanda que exige $\{e_1, e_2\}$, seja
$f(S) = \mathbf{1}[\{e_1,e_2\} \subseteq S]$:

$$f(\{e_2\}) - f(\emptyset) = 0, \qquad f(\{e_1,e_2\}) - f(\{e_1\}) = 1.$$

O ganho **aumentou**: há complementaridade. Esse contraexemplo explica por que
"guardar as arestas mais relevantes individualmente" pode eliminar justamente as
combinações necessárias ao multi-hop, e impede invocar garantias clássicas de
cobertura gulosa.

**Formulação inteira.** Variáveis binárias: $x_e$ (conservar o fato $e$),
$y_{t,W}$ (conservar integralmente a testemunha $W$), $z_t$ (preservar a resposta
da demanda $t$).

$$
\begin{aligned}
y_{t,W} &\le x_e && \forall e \in W \\
y_{t,W} &\ge \textstyle\sum_{e \in W} x_e - |W| + 1 \\
z_t &\ge y_{t,W} && \forall W \in \mathcal{W}_t \\
z_t &\le \textstyle\sum_{W \in \mathcal{W}_t} y_{t,W} \\
\textstyle\sum_e c_e x_e &\le B
\end{aligned}
\qquad \max \sum_t w_t z_t
$$

As restrições codificam exatamente o "E" dentro de uma testemunha e o "OU" entre
testemunhas alternativas. Se todas as testemunhas pertinentes forem enumeradas e
o solucionador terminar com certificado de otimalidade, temos ótimo global do
problema finito. Enumerando só algumas, a garantia se restringe às alternativas
enumeradas.

**Verificação na instância pequena.** Com as três demandas (empregador de Ana,
peso 1; cidade da instituição de Ana, peso 3; alguém na Atlas que pesquisa
Óptica, peso 4) e orçamento de três fatos, $\{e_1, e_2, e_3\}$ é a única solução
ótima, com valor 8.

*No código:* `wrag/witness/budget.py`; verificado em
`tests/test_witness.py::test_budget_toy_instance` e
`::test_submodularity_counterexample`.

---

## 6. Recuperação com garantia

Para consultas em cadeia, o algoritmo mantém simultaneamente a entidade atual e a
posição na consulta. Para $q_2$:

$$(\text{Ana}, 0) \longrightarrow (\text{Atlas}, 1) \longrightarrow (\text{Recife}, 2).$$

A primeira transição aceita somente `trabalhaEm`; a segunda, somente
`localizadaEm`. Uma relação `moraEm` não satisfaz a segunda etapa, mesmo chegando
à mesma cidade. Formalmente, o produto entre o grafo de dados e um autômato da
consulta, $G \times A_q$ — técnica estabelecida para *regular path queries*
(Casel e Schmid).

Para cadeia ancorada com $k$ relações, execução por camadas tem limite
$O(k(|V| + |E|))$, além do custo de materializar a saída. Para consultas com
interseção, junções relacionais: todas as ocorrências de uma variável precisam
receber o mesmo valor, e a largura de árvore da consulta passa a importar tanto
quanto o número de relações.

**A garantia:** dada uma consulta corretamente compilada, uma memória fixa e
execução exaustiva do algoritmo apropriado, toda resposta retornada satisfaz a
consulta na memória, e toda resposta existente na memória é encontrada.

Ela **não** afirma que o extrator capturou tudo que o texto dizia, e deixa de
valer como completude quando se introduz corte por top-k, busca aproximada ou
poda não demonstrada.

*No código:* `wrag/witness/search.py`. A junção por feixe reproduz a caminhada
por camadas para cadeias e a junção relacional para interseções. **O feixe é
onde a completude é abandonada**: com `beam_width` infinito a busca é exaustiva;
com feixe finito, a garantia vale sobre o que o feixe reteve. O relatório grava
`beam_width` e `feixe_exaustivo` por pergunta.

---

## 7. Proveniência

Anotando cada fato com um símbolo $X_e$, uma resposta sustentada por duas
evidências conjuntas recebe um produto; derivações alternativas somam:

$$P_{q,a} = X_1 X_2 + X_3 X_4.$$

O produto representa fatos exigidos conjuntamente; a soma, derivações
alternativas. A construção se apoia em semianéis de proveniência (Green,
Karvounarakis e Tannen) e fornece estrutura para explicar respostas, reconhecer
evidências compartilhadas e atualizar resultados quando fontes são removidas.

**Sobre grupos livres.** Relações usuais não têm inversas no sentido de grupos.
Se Ana e Bruno trabalham na Atlas, percorrer
Ana $\xrightarrow{\text{trabalhaEm}}$ Atlas $\xrightarrow{\text{inversa}}$ Bruno
não retorna a Ana: a relação inversa recupera funcionários, não desfaz a
operação. Impor $rr^{-1} = 1$ criaria equivalências incorretas. Uma abstração
mais apropriada é uma **categoria livre de caminhos tipados**, interpretada em
relações: composição quando os tipos se encaixam, sem exigir inversibilidade.

**Sobre ribbon graphs.** Acrescentam ordem cíclica às incidências em cada
vértice. Para memória factual genérica, falta justificar qual informação
semântica essa ordem representa; só valeria a pena num domínio com estrutura
cíclica relevante e mensurável. Ordem temporal normalmente pede relações
temporais explícitas.

*No código:* `wrag/witness/provenance.py`. A direção invertida é aceita com
**penalidade multiplicativa**, não como inversa de grupo — OpenIE inverte sujeito
e objeto com frequência, e recusar a direção invertida descartaria testemunhas
corretas por acidente de extração.

---

## 8. Incerteza

Três coisas a separar:

1. O texto realmente sustenta o fato extraído?
2. O fato sustentado pelo texto é verdadeiro no mundo?
3. A consulta formal representa o que o usuário perguntou?

Para um primeiro artigo, o alvo da extração é a primeira: **fidelidade ao
corpus**. Veracidade externa é outra tarefa.

Com mundos possíveis $\omega$ e $F_\omega$ o conjunto válido naquele mundo:

$$p_t(S) = \Pr_{\omega \mid D}[\exists W \in \mathcal{W}_t : W \subseteq S \cap F_\omega],
\qquad U_{\text{incerto}}(S) = \sum_t w_t \, p_t(S).$$

**Não se pode simplesmente multiplicar confianças de arestas sem justificar
independência.** Fatos extraídos do mesmo trecho, erros de identidade
compartilhados e documentos copiados produzem dependências. Tampouco somar
probabilidades de provas que compartilham evidências.

O **limite da união dispensa independência**:

$$\Pr(\text{resposta inválida}) \le \delta_{\text{interpretação}} + \sum_{e \in W} \delta_e + \delta_{\text{verbalização}}.$$

Isso exige limites válidos para o procedimento considerado. Confiança declarada
pelo LLM não basta; testemunhas escolhidas adaptativamente exigem avaliação que
considere a seleção. O limite trata **invalidade**, não omissão de respostas
adicionais.

*No código:* `wrag/witness/provenance.py` devolve `score` (noisy-or, para
ordenar, sem pretensão de calibração) e `risk` (limite da união). Só o segundo é
apresentado como limite, e o README registra que a testemunha de menor limite é
escolhida depois de ver os dados.

---

## 9. Extração obedecendo ao objetivo da memória

Em vez de extrair sempre os fatos mais salientes, o sistema prioriza verificações
capazes de **completar ou corrigir testemunhas importantes**. Se muitas perguntas
dependem de saber onde ficam instituições, e a memória tem muitas relações
`trabalhaEm` mas poucas `localizadaEm`, o sistema direciona uma nova leitura aos
trechos que provavelmente contêm essas informações.

Valor esperado da informação de uma ação $a$ (reler um trecho, verificar uma
identidade):

$$\mathrm{VOI}(a) = \mathbb{E}_y\Big[\max_{S : C(S) \le B} U(S \mid D, y, a)\Big]
- \max_{S : C(S) \le B} U(S \mid D) - \lambda\,\mathrm{custo}(a).$$

Com um modelo probabilístico correto, isso define a melhor decisão **de um
passo** entre as ações consideradas. Uma sequência globalmente ótima é problema
mais difícil.

> **Hipótese específica do WITNESS-RAG:** usar a utilidade de testemunhas
> completas para orientar conjuntamente a seleção da memória e a aquisição de
> novas evidências melhora a recuperação multi-hop por unidade de custo.

*No código:* `wrag/methods/witnessrag.py::_plan_acquisition`. O primeiro termo é
aproximado pela similaridade densa entre a sonda (âncora + relação faltante) e a
passagem. É a parte mais frágil da implementação e a de maior espaço para
melhoria.

---

## 10. Atualização

As demonstrações registradas permitem identificar quais respostas dependem de um
fato alterado. Remover uma evidência elimina as demonstrações que dependem dela,
preservando alternativas ainda válidas. A reotimização pode penalizar mudança
excessiva:

$$\max_{S : C(S) \le B} \big[ U_t(S) - \kappa\, C_{\text{mudança}}(S, S_{t-1}) \big].$$

*Não implementado neste protótipo.* Exige protocolo de corpus evolutivo, que é um
experimento diferente.

---

## 11. Antecedentes e o que pode ser reivindicado

| componente | antecedente relevante |
|---|---|
| raciocínio guiado por formas lógicas | KAG |
| converter perguntas em padrões de grafo | SimGRAG |
| resumir grafos conforme interesses que mudam | APEX² |
| encontrar sub-bases que preservam respostas | Smallest Witness Problem |

"LLM + grafo + consulta lógica" não é reivindicação suficiente. A contribuição
candidata está na **otimização orientada por testemunhas, com dependências de
incerteza, usada também para decidir o que extrair e revisar**. A busca de
antecedentes identifica trabalhos próximos; não estabelece que a combinação seja
inédita.

---

## 12. Desenho experimental

Quatro condições, para separar a qualidade da teoria da qualidade do LLM:

| condição | o que permite medir |
|---|---|
| grafo e consultas corretos por construção | recuperação e otimização sem erros de linguagem |
| grafo extraído; consultas corretas | perdas causadas pela extração |
| grafo correto; consultas produzidas pelo LLM | perdas causadas pela interpretação |
| ambos produzidos pelo LLM | desempenho do sistema completo |

Comparações: recuperação textual, HippoRAG, HippoRAG 2, métodos próximos
(KAG/SimGRAG) e, sobretudo, **um executor relacional convencional sobre os mesmos
fatos extraídos** — essa última revela se a contribuição supera uma aplicação
direta de técnicas existentes.

Medir: respostas corretas, recuperação de testemunhas completas, custo total de
extração e consulta, latência, armazenamento, comportamento após atualizações, e
risco versus proporção de perguntas respondidas. Orçamento de evidências,
extrator e modelo de resposta controlados.

**As perguntas de avaliação devem ficar separadas das usadas para construir a
memória.** Otimizar para as próprias perguntas de teste apenas demonstra
memorização do conjunto de avaliação.

### Critérios de refutação

A proposta perde força se:

- a vantagem desaparecer diante do executor relacional usando os mesmos fatos;
- a economia de consulta for anulada pelo custo de extração;
- a preservação de testemunhas não generalizar para novas composições de
  perguntas.

---

## Referências citadas na discussão

- Hu e Sintos. *Finding Smallest Witnesses for Conjunctive Queries.*
- Casel e Schmid. *Fine-Grained Complexity of Regular Path Queries.*
- Green, Karvounarakis e Tannen. *Provenance Semirings.*
- Mulase e Penkava. Definição de ribbon graph.
- Gutiérrez et al. *HippoRAG* (2024) e *HippoRAG 2 / From RAG to Memory* (2025).
- Han et al. *Retrieval-Augmented Generation with Graphs* (survey).
- Edge et al. *GraphRAG* (2024).
