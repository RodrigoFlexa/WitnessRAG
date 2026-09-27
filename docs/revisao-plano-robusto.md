# Revisão técnica e científica do WitnessRAG robusto

Data: 27/09/2026. Escopo: implementação local, anexos, nove variantes da conv05,
ablação anterior de dez conversas e protocolo da tabela Zero-Mem. Nenhuma chamada
à API foi feita nesta revisão. As instruções contidas nas conversas anexadas
foram tratadas como histórico, não como ordens para iniciar experimentos.

## Parecer

**Preservar os prompts atuais como referência experimental e corrigir as falhas
de execução e medição. A evolução central deve ser a geração e execução de
multiplanos**, conforme a prioridade explicitada pelo autor. Detalhei essa
evolução em [multiplanos-teoria-e-validacao.md](multiplanos-teoria-e-validacao.md).
O reranker, o preenchimento pela pergunta
e a apresentação temporal têm um sinal de desenvolvimento útil. Entretanto, os
dados ainda não sustentam que o planejamento seja o principal responsável pelo
ganho, nem que o sistema supere a tabela de referência no LoCoMo completo.

Minha avaliação editorial seria **revisão substancial**, caso a alegação central
fosse superioridade em qualidade e eficiência graças a provas no grafo. Existe
uma hipótese de pesquisa defensável: planos executáveis como mecanismo de
seleção de evidências, com proveniência e verificação textual sobre memória
imperfeita. Essa hipótese precisa de uma avaliação que isole sua contribuição.

## Resultados que consegui reproduzir

Recalculei as métricas a partir das respostas, sem aceitar cegamente os escores
armazenados. Na conv05, todas as nove variantes têm os mesmos 123 identificadores,
perguntas, categorias e respostas de referência. Os F1 armazenados coincidem com
os recalculados. Todos os números desta seção são de **predições anteriores às
correções desta revisão**, não uma avaliação do código corrigido.

| Configuração | Perguntas | F1 oficial | BLEU-1 local | Tokens leitor | Tokens totais de consulta |
|---|---:|---:|---:|---:|---:|
| WitnessRAG anterior, conv05 | 123 | 51,50 | 45,98 | 1.816 | 8.150 |
| Plano robusto isolado | 123 | 53,65 | 47,93 | 1.818 | 9.268 |
| Reranker isolado | 123 | 58,68 | 53,22 | 1.879 | 8.213 |
| Robusto completo | 123 | 62,96 | 57,42 | 2.016 | 9.465 |
| Mesma entrega robusta sem plano | 123 | 61,63 | 56,47 | 2.006 | 2.006 |
| Busca híbrida por trechos | 123 | 52,93 | 47,76 | 10.263 | 10.263 |

Robusto contra anterior: **+11,46 F1**, IC 95% por perguntas [6,62; 16,64].
Robusto contra robusto sem plano: **+1,32 F1**, IC [−1,66; 4,45], e **+0,95
BLEU-1 local**, IC [−1,92; 3,65]. São 12 perguntas melhores e 6 piores em F1.
Esses intervalos descrevem esta conversa de desenvolvimento; não quantificam
generalização para conversas inéditas nem corrigem a seleção entre variantes.

O sistema completo usa aproximadamente **4,7 vezes** os tokens de consulta do
controle robusto sem plano. Contra os trechos, a redução é aproximadamente 80%
**no leitor**, mas somente 8% na consulta inteira. Nenhum desses totais inclui a
construção da memória; latência do reranker também deve ser medida separadamente.
Não é correto transformar a economia de contexto do leitor em economia igual
de tokens ou de tempo do sistema inteiro.

Também reavaliei a ablação anterior, com bootstrap pareado de **conversas**:

| Configuração anterior | n | F1 | Δ contra completo [IC 95%] |
|---|---:|---:|---|
| Completo | 1.540 | 55,35 | — |
| Sem plano | 1.540 | 54,86 | −0,50 [−1,39; 0,41] |
| Sem prova | 1.540 | 54,31 | −1,04 [−1,69; −0,33] |
| Sem verificação | 1.540 | 55,00 | −0,35 [−0,78; 0,06] |

O contraste com “sem prova” favorece o controlador completo. Contudo, essa
ablação também retira verificação e replanejamento; não identifica sozinha o
efeito causal da junção. A comparação com “sem plano” continua inconclusiva.
A variante antiga “sem nota temporal” tem apenas 146 registros e não deve ser
colocada como se fosse uma ablação completa pareada das 1.540 perguntas.

Relatórios reproduzidos: `runs/review-20260927/conv05/`, `plan-effect/` e
`full-ablation/`. Pequenas diferenças nos limites dos ICs em relação aos antigos
relatórios decorrem de usar a mesma semente por comparação, independentemente
da ordem das variantes; as médias não mudaram.

## Achados e correções implementadas

### 1. Verificação dispensada indevidamente na entrega por fatos — alta prioridade

Em `WitnessRAGRetriever._retrieve_proof`, uma prova não era verificada quando
seus trechos já estavam recuperados. Isso é compatível com a entrega dos mesmos
trechos inteiros, mas não com `_fact_context`, onde seus fatos passam a ocupar
lugares prioritários e podem substituir outros fatos.

Nos registros robustos, **9/123 perguntas** terminaram em
`prova_ja_no_contexto`, incluindo uma leitura alternativa. Corrigi a decisão:
com entrega por fatos, uma prova utilizável passa pelo verificador quando ele
está habilitado. A ablação explícita sem verificação continua funcionando.
Isso pode aumentar o custo e alterar respostas; não alego ganho de F1 sem medir.

### 2. Resposta recusada voltava como candidata prioritária — alta prioridade

Mesmo depois de o verificador recusar uma candidata, seus fatos ainda podiam
receber a prioridade intermediária do plano na seleção final. Um teste com
rejeição explícita reproduziu o problema: dois fatos continuavam nesse nível.
Agora a recusa é mapeada da lista de testemunhas verificadas para a lista de
candidatas, e essa candidata perde o privilégio. Os fatos continuam elegíveis
pela relevância comum: recusar uma resposta não torna toda sua fonte inútil.

### 3. Deduplicação apagava tempo ou modalidade — prioridade média

Proposições com `statement` eram deduplicadas apenas pelo texto, ao contrário
das triplas, que já consideravam a data. Agora a chave inclui data resolvida e
modalidade (`kind`), preservando ocorrências distintas e a diferença entre
plano e evento. Duplicatas equivalentes continuam colapsadas.

Testes sintéticos demonstram a perda anterior. Na memória local da conv05,
encontrei duas colisões entre descrições iguais de imagens em sessões distintas;
isso demonstra que a condição existe nos dados, **não que ela explique erros
temporais de QA ou prometa ganho mensurável**. Exemplos salvos em
`runs/review-20260927/temporal-collisions.json`.

### 4. Prioridades eram bônus numéricos — prioridade média

Somar 2 à prova e 1 ao plano não garante a ordem prova → plano → preenchimento
quando a similaridade pode ser negativa. O reranker também tinha um empate
possível nos extremos 0/1. Troquei essa soma por ordenação lexicográfica:
primeiro o nível, depois a relevância. Os testes reproduzem ambos os casos.

Isso garante precedência na ordenação, não completude de toda prova entregue:
o orçamento total e o limite de quatro fatos por fala continuam vigentes.
Uma política de entrega de testemunhas inteiras merece uma ablação própria.

### 5. Relatórios não eram reproduzíveis a partir dos arquivos exportados

`paired-report.py` só procurava árvores `**/locomo/*.jsonl`, enquanto os arquivos
da conv05 estão diretamente em `runs/conv05-robust/*.jsonl`. IDs repetidos eram
sobrescritos e conjuntos distintos de perguntas viravam uma interseção silenciosa.

O script agora aceita exportações, recusa duplicatas, exige pareamento exato,
confere perguntas/gabaritos/categorias e recalcula as métricas. Inclui BLEU-1
local, intervalos por conversa quando possível e `--output` para preservar
relatórios históricos. Também corrigi a impressão Unicode no Windows.

O contador “prova por outra leitura” tinha um erro adicional: contava qualquer
plano final alternativo, mesmo recusado ou sem prova. O valor correto do robusto
é **3 aceitas: 2 verificadas e 1 dispensada**, não 7. A contagem 7 correspondia a
3 aceitas, 3 casos de respostas demais e 1 prova recusada.

### 6. Proveniência da entrega e perfil de controle

Adicionei índices, IDs dos fatos, trechos e falas efetivamente selecionados em
`fatos_entregues`. Isso não acrescenta texto ao leitor, mas permite auditar a
entrega real. O `recall@k` existente continua sendo de **trechos recuperados**:
no modo fatos, ele não mede o que chegou ao leitor. Os registros antigos não
permitem reconstruir essa correspondência com a mesma precisão.

Registrei `robust-no-plan` como perfil nos scripts Windows e Bash, com a mesma
extração e entrega robusta e `--ablation no-plan`. Antes era apenas o nome de uma
combinação descrita no texto, não uma opção reconhecida pelos scripts.

## Questões conceituais que não alterei sem experimento

**Mais leituras não equivalem a melhores provas.** Os exemplos do prompt incluem
`play → own`, `lead → work at`, `work as → work at` e a retirada da esposa de um
sujeito coletivo. Possuir um instrumento não implica tocá-lo; trabalhar num
laboratório não implica liderá-lo; ir sozinho não satisfaz necessariamente uma
pergunta sobre uma viagem do casal. `live in` e `move to` também não são
equivalentes sem condições temporais. O verificador textual é uma proteção
falível, não uma demonstração de equivalência entre essas leituras.

**A formalização deve explicitar essa limitação.** Disjunções de predicados
podem ser escritas como UCQs. Entretanto, similaridade de embeddings, resolução
de entidades e leituras propostas por LLM não dão garantia de consequência
lógica em linguagem natural. A testemunha é relativa à memória extraída e à
interpretação escolhida. Uma prova de item também não demonstra a completude
do conjunto de respostas. Convém explicitar essas condições no artigo.

**Multi-hop do benchmark não é sinônimo de junção de vários átomos.** Na conv05,
21/30 planos finais de perguntas multi-hop tinham um átomo; cinco eram
interseções, três desconectados e um inválido. Recuperar uma lista espalhada
por sessões pode melhorar essa categoria sem executar uma cadeia de entidades.
Nas 12 perguntas com plano final conectado, o Δ exploratório robusto contra
sem plano foi −7,01 F1 [−25,35; 4,65]. Esse grupo é pequeno e selecionado por uma
saída do modelo: não é estimativa causal nem evidência de que composição não
funciona. É motivo para investigar, em vez de afirmar que o ganho multi-hop
já valida composição.

**Datas estão parcialmente no ranqueamento e no leitor.** Um período temporal
altera prioridades, mas não constitui por si só um filtro lógico rígido de
intervalos. “Primeiro emprego”, relações antes/depois e eventos planejados
exigem cuidado adicional. Usaria no texto “tempo do evento e da enunciação”; a
expressão “bitemporal” precisa ser definida, sem sugerir um banco temporal com
todas as garantias de validade e tempo de transação.

**Recuperação não tem garantia automática de nunca piorar o híbrido.** Adicionar
ou priorizar evidências pode deslocar informação útil e mudar a resposta do
leitor. Essa propriedade precisaria de uma definição e demonstração próprias.

## Comparação com a tabela anexada

A tabela é a Tabela 1 de [Zero-Mem](https://arxiv.org/html/2607.29377v1).
O artigo declara leitor final comum, orçamento equivalente e cinco itens
primários de recuperação. Os alvos globais com GPT-4o-mini são 59,15 F1 e 52,96
BLEU-1; a comparação por categoria também deve preservar o protocolo.

O [repositório oficial](https://github.com/Zero-Mem/Zero-mem) consultado contém
apenas a promessa de liberar código após revisão. Não consegui confirmar seu
tokenizador, normalização e agregação exatos do BLEU. O
[avaliador original do LoCoMo](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py)
define o F1 usado aqui; não define este BLEU-1 local.

Assim, **62,96/57,42 na conv05 não prova superar 59,15/52,96 no conjunto inteiro**.
Além da amostra diferente, o BLEU local aplica normalização, stemming e
penalidade de brevidade à resposta inteira. Em multi-hop, o F1 oficial faz
melhor casamento de itens separados por vírgula e não penaliza itens extras da
mesma maneira. Precisão de itens e respostas incorretas adicionais merecem uma
auditoria separada; não se deve otimizar listagem excessiva para explorar isso.

## Próxima contribuição que eu testaria

Minha prioridade seria **leituras alternativas que preservem as restrições da
pergunta**. A alternativa pode mudar vocabulário ou representar uma relação
direta como cadeia, mas deve manter participantes, modalidade, período e tipo
da resposta. Resoluções como “Kai e a esposa → Kai” precisam de suporte para a
participação da esposa na evidência. Leituras úteis apenas para encontrar
premissas podem alimentar a recuperação, sem receber o status de prova da
pergunta completa.

Isso ataca uma falha visível dos exemplos atuais e conecta a contribuição ao
plano. Não implementei um novo esquema ou novos exemplos porque validar esse
comportamento exige um teste controlado com o planejador real. Não acrescentaria
comunidades, fusão agressiva de entidades ou novos pesos antes disso.

Sequência experimental recomendada, ainda não executada:

1. Congelar esta correção e rodar um piloto pareado de 12 perguntas previamente
   escolhidas de conv01/conv05, distribuídas entre single-hop, temporal e
   multi-hop, verificando também perdas. Manter as perguntas fixas antes de
   executar; não selecionar só casos que favorecem a proposta.
2. Nesse piloto, comparar o código anterior e o corrigido com os mesmos prompts
   e memória. Só depois experimentar os exemplos que preservam restrições.
   Registrar número real de chamadas e tokens; 12 perguntas não significam
   somente 12 chamadas quando há planejamento, verificação e leitura.
3. Se a execução estiver correta, comparar robusto e robusto sem plano nas dez
   conversas, com mesmo leitor, memória, reranker e limite de entrega. Publicar
   F1 e BLEU com definição explícita, custos de construção/consulta separados,
   intervalos por conversa e resultados por categoria e conversa.
4. Para isolar o plano, incluir depois robusto sem alternativas/leituras,
   mantendo toda a entrega robusta; e execução sem verificação com o mesmo
   orçamento de ciclos. O atual `wr-plan` sobre a entrega antiga não isola
   a contribuição marginal dos planos robustos dentro do sistema completo.

As conversas 01/05 já são desenvolvimento. As demais também tiveram resultados
globais examinados em rodadas anteriores: não chamá-las de teste completamente
intocado. Congelar as decisões prospectivamente, declarar o histórico de ajuste
e usar outro conjunto para uma afirmação forte de generalização. Um benchmark
com cadeias explícitas seria validação complementar, não substituto do LoCoMo.

## Validação e reprodução

- Antes: 328 testes existentes passaram.
- Regressões novas: 14 casos; oito falhas reproduzidas antes das respectivas
  correções, além de controles de comportamento e de pareamento estatístico.
- Depois: **342 testes passaram**, sem chamadas a servidores de modelos.
- Perfis Bash e PowerShell conferidos sem executar o experimento; sintaxe válida.
- Prompts e nove arquivos de predições históricos conferidos por SHA-256 e
  preservados. `git diff --check` sem erros de espaços.
- Nenhum ganho novo de F1 é atribuído às correções. Os testes demonstram as
  propriedades de execução corrigidas; não substituem uma avaliação de QA.

Comandos que reproduzem a análise, sem API:

```text
python -m pytest -q
python scripts/audit-robust-results.py runs/conv05-robust --output runs/review-20260927
python scripts/paired-report.py runs/conv05-robust witnessrag wr-plan wr-fill wr-time wr-rerank wr-no-rerank witnessrag-robust robust-no-plan hybrid-chunks --output runs/review-20260927/conv05
python scripts/paired-report.py runs/conv05-robust robust-no-plan witnessrag-robust --output runs/review-20260927/plan-effect
python scripts/paired-report.py runs/ablation-openai-c2048-k5 full no-plan no-proof no-verify --output runs/review-20260927/full-ablation
```

As modificações anteriores do usuário foram preservadas. As cópias dos quatro
arquivos alterados que já continham trabalho anterior estão em
`runs/review-20260927/before/`; o manifesto de entrada e os diagnósticos também
estão nessa pasta de revisão. Não foi criado commit.
