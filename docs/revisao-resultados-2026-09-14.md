# WitnessRAG: auditoria dos resultados e revisão proposta

Revisão de 14/09/2026. Análise dos JSONL e manifestos locais, inspeção de código,
consulta ao artigo do HippoRAG2 e testes offline. Nenhuma nova execução do Qwen
no servidor foi realizada. Os recursos de busca/verificação acrescentados nesta
revisão são experimentais: não há ganho de benchmark medido para eles.

## O que os resultados realmente mostram

Os resultados do relatório correspondem à **witness-v2**, não à witness-v3.
A v3 registra a propagação parcial rejeitada e piora para F1 43,26 / R@5 62,75.
O código atual já voltou ao fallback denso.

| Método/rodada | F1 | R@5 | Todas as evidências @5 |
|---|---:|---:|---:|
| Witness inicial | 42,86 | 61,75 | 40,00 |
| Dense inicial | 44,60 | 75,25 | 47,00 |
| Witness v2 | 45,76 | 76,25 | 49,00 |
| HippoRAG inicial | 57,41 | 92,75 | 82,00 |
| HippoRAG2 inicial | 59,93 | 94,00 | 89,00 |

São 100 perguntas sobre **1.059 passagens em corpus reduzido**, com adaptações
locais dos métodos. Não equivalem ao benchmark oficial completo. A v2 tem três
ganhos de F1 e nenhuma perda frente ao dense, mas o ganho médio é só 1,17 ponto.
Bootstrap pareado descritivo (10.000 reamostragens, semente 42) produz intervalo
de 95% [0,00; 2,67]. Contra HippoRAG2: −14,17 pontos, [−22,83; −6,00]. Esses
intervalos não corrigem a seleção de variantes após repetidas consultas ao teste.

Os hashes de corpus e perguntas coincidem, mas **a extração não é a mesma**:
11.988 fatos / 4.541 relações na inicial, 11.950 / 4.567 na v2. Os clusters da v2
são 8.945, a partir de 9.959 entidades; os números 10.055 → 9.001 do relato não
descrevem seu manifesto. A identidade de provedor também mudou; sua chave inclui
URL e revisão do modelo, portanto isso não prova que os pesos mudaram, mas impede
assumir execução idêntica. A v2 registra 2.118 chamadas de indexação sem cache.
Não se pode atribuir causalmente toda a diferença a fallback e canonicalização.

O comparador anterior verificava deployment, embedder e três parâmetros, mas não
os hashes de dados. Foi corrigido e agora recusa essa diferença de identidade de
provedor. A auditoria descritiva registra a ressalva em vez de escondê-la.

## Três conclusões do relato precisam ser enfraquecidas

**“O mecanismo é superior quando dispara” não está demonstrado.** Nas nove
composicionais com testemunha, R@5 é 94,44 contra 77,78, porém **F1 é 33,33 contra
37,04**. O grupo foi selecionado pelo próprio sucesso de fechamento do Witness,
não por um critério independente. Das nove respostas estruturais, quatro têm EM
correto. No conjunto das 16 perguntas com testemunha, são quatro EM corretos.
EM penaliza aliases legítimos; esses 25% não são uma auditoria semântica de
precisão das provas. Da mesma forma, 4/45 = 8,9% é taxa de acerto estrutural sobre
todas as composicionais, não precisão condicional. A cobertura lexical de triplas
também não mede equivalência semântica.

**“A única raiz é o vocabulário de relações” é hipótese, não identificação causal.**
A diversidade de relações é compatível com OpenIE e prejudica um executor que
precisa casar predicados. Não prova que normalizar relações resolva identidade,
direção, consulta incorreta ou seleção prematura de candidatos. Não encontrei
o cache de OpenIE do servidor nesta cópia; o percentual de relações singleton
não foi recalculado a partir dos fatos.

**“Falta apenas recuperar melhor” também é incompleto.** Em 49 perguntas a v2
recupera todas as passagens ouro; 18 ainda têm F1 zero. O leitor pode falhar por
distração, compreensão, conteúdo contraditório ou inadequação da anotação.
É necessário inspecionar esses casos antes de classificar todos como erro do
leitor. Dois exemplos têm resposta estrutural exatamente correta e F1 final zero:
local de detenção do intérprete de “B Boy” e causa da morte do diretor de
“Destination: Dewsbury”. Não substituir automaticamente o leitor pela resposta
estrutural: as demais testemunhas mostram por que isso seria perigoso.

## Gargalos concretos

1. **Top-k antes da ligação.** Para `R(A, ?y) ∧ S(?y, ?x)`, o código ranqueia
   globalmente fatos para `S` antes de descobrir `?y`. Um fato verdadeiro
   `S(B,C)` pode desaparecer entre relações genéricas de milhares de outras
   entidades. A junção não consegue recuperar o que o pool já excluiu. Este é
   um problema independente do vocabulário e foi reproduzido em teste.
2. **Similaridade aceita equivalências indevidas.** `expected_type` é registrado,
   mas não restringe o executor. Na pergunta sobre o local de nascimento do
   diretor de “The Ape (2005 Film)”, a resposta estrutural é `April 19, 1978`.
   Na pergunta sobre a avó materna de Eleanor of Brittany, é `King Henry II of
   Castile`. Fechamento de junção aproximada não é demonstração textual.
3. **Identidade ainda não é segura.** Contenção de tokens mais cosseno continua
   sendo heurística. A regra permite candidatar `Henry`/`Henry VIII`; remove
   `jr`/`sr` e trata conjuntos de tokens sem ordem. Além disso, união transitiva
   pode conectar extremos que nunca foram verificados entre si. Não chamar isso
   de identidade certificada. Auditar clusters e manter aliases como candidatos
   até haver evidência contextual, especialmente para parentes e dinastias.
4. **Escopo limitado.** 26 comparações e 22 bridge-comparisons caem no denso por
   agregação não suportada: 48% da amostra. Nas bridge-comparisons, todas as
   evidências @5 são 0% contra 81,82% do HippoRAG2. O baixo impacto atual em F1
   desse grupo não elimina a fragilidade de evidência. Nas 45 composicionais,
   36 não fecham testemunha e têm R@5 65,28 contra 94,44.
5. **Aquisição prematura e estreita.** A política segue apenas uma lacuna do
   melhor estado parcial, que pode conter uma entidade errada. Releitura dirigida
   pelo predicado errado reforça esse erro; normalização não elimina esse ciclo.

O HippoRAG2 usa OpenIE para orientar recuperação, integrando passagens ao grafo
e filtrando triplas por relevância; não precisa provar equivalência de cada
predicado de uma consulta formal. Isso explica por que o mesmo tipo de extração
pode servir melhor à difusão que à junção estrita. Essa distinção vem do
[artigo original](https://arxiv.org/html/2502.14802v1), seções 3.2–3.5; não é
evidência de que uma variante nova nossa já o superaria.

## Revisão da teoria

Preservar a definição de testemunha para consultas conjuntivas **exatas sobre
uma base fixa**. Separar explicitamente três objetos:

- testemunha simbólica: satisfaz a consulta compilada no grafo;
- evidência textual: sustenta os fatos, as identidades e a interpretação da pergunta;
- contexto entregue: cabe no orçamento e permite ao leitor responder.

Uma implicação válida no primeiro nível não estabelece os outros dois. O grafo
deve propor candidatos; o texto deve justificar sua promoção. A contribuição
mais defensável é **selecionar conjuntos suficientes de evidência e adquirir os
elos faltantes sob orçamento**, com controle empírico de erros e custos. Apenas
adicionar junções ou um verificador LLM não estabelece novidade científica.

Para uma consulta q, ligação parcial θ e átomo i, gerar candidatos em
`F_i(θ) = {f: argumentos de f são compatíveis com θ}` e aplicar o top-k dentro
desse conjunto. A compatibilidade usa a identidade já ligada, sem reabrir uma
busca vetorial pela entidade intermediária. Relações ainda precisam de validação.

Reformular o objetivo empírico como ganho de utilidade de resposta/evidência
menos custo de busca, aquisição e verificação, sob orçamento de contexto.
Calibrar separadamente o risco de substituir o ranking de referência, usando
desenvolvimento disjunto. Aprovação do LLM e produto de cossenos não são
probabilidades. Se forem desejadas garantias, é preciso controlar os erros de
compilação, extração, identidade e verificação também após seleção adaptativa.

Canonicalização deve ser uma representação adicional reversível: predicado
canônico, direção, argumentos, tipos, qualificadores e trecho fonte, preservando
a tripla original. Consulta e fatos precisam usar o mesmo contrato. Não fundir
`born in` com `born on`, nem colapsar `mother`, `father` e `parent` como sinônimos.
Relações genéricas podem servir para gerar candidatos, com a condição específica
verificada depois. Vocabulário e exemplos devem vir de treino/corpus, não das
100 perguntas que já orientaram ajustes.

## Implementado nesta revisão

- `--binding-aware-grounding`: busca semântica visita a fronteira ligada e
  consulta as adjacências antes do corte local. Respeita direção, ligações,
  restrições repetidas e máscara de memória. Modo exato e candidatos fornecidos
  explicitamente mantêm a semântica anterior. `n_candidatos_por_atomo` continua
  descrevendo o pool inicial; não deve ser confundido com a expansão local.
- `--verify-witnesses`: antes de promover passagens, avalia até cinco testemunhas
  que cabem no contexto, usando pergunta, consulta, atribuições, fatos e textos.
  Exige aprovação explícita e citação literal válida para cada átomo. Rejeição,
  bloqueio e saída inválida mantêm o fallback configurado. Registra custo no
  estágio `witness.verify` e decisões em `diagnosticos.verificacao`. Não envia
  respostas ou evidências ouro. Verifica testemunhas completas, não todos os
  aterramentos candidatos; portanto não resolve sozinha a falta de cobertura.
- Comparação: checagem de hashes, identidade/configuração do leitor, duplicatas
  de qid e interseção vazia. Ainda não é auditoria completa de equivalência de
  versões do prompt ou de fatos extraídos entre rodadas.
- Auditoria reproduzível em `scripts/audit-witness.py` e resultados derivados em
  `docs/audit-witness.json`, incluindo estratos, exemplos e diferenças pareadas.

As duas opções novas são **desligadas por padrão**, permitindo medir cada efeito.
O extrator compartilhado e as regras de identidade existentes não foram alterados.
Checar citações impede fontes inventadas, mas não prova que a citação implique
o átomo: essa decisão continua falível e precisa de auditoria humana em amostra.
Falsos positivos que não chegam ao top-5 de testemunhas não são reparados.

## Próximo experimento no servidor

Primeiro congelar uma extração e o endpoint/revisão, e rodar controles no mesmo
ambiente atual. O `--cache-dir` apenas aponta para uma pasta; trocar URL/revisão
troca a chave e pode reextrair tudo. Confirmar o hit no log. Se o cache histórico
não estiver disponível sob a mesma identidade, contabilizar nova extração e
reexecutar os comparadores sobre ela.

Exemplo de matriz em bash no servidor (mesma configuração e GPU do relato):

```bash
common=(--gpu 3 --vllm-python "$PWD/.venv-vllm/bin/python" --port 8087
        --cache-dir "$PWD/runs/qwen14b-pilot/cache" --hours 2)

.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --methods dense,hipporag2,witnessrag --output runs/revision-control

.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --methods witnessrag --binding-aware-grounding --output runs/revision-bound

.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --methods witnessrag --verify-witnesses --output runs/revision-verify

.venv-bench/bin/python -m wrag.pilot "${common[@]}" \
  --methods witnessrag --binding-aware-grounding --verify-witnesses \
  --output runs/revision-both

.venv-bench/bin/python scripts/compare-runs.py runs/revision-both runs/revision-control
```

Usar essa amostra somente para diagnóstico. Congelar decisões antes de avaliar
perguntas novas e corpus maior; repetir em outro dataset. Reportar F1, R@5,
evidência completa, aceitação do verificador, precisão semântica auditada,
chamadas/tokens e latência. Rodar a ablação sem aquisição de forma pareada também.
O verificador pode aumentar precisão e reduzir recall; a busca condicionada pode
ampliar cobertura e ampliar falsos positivos. A combinação precisa de medição.

Na etapa seguinte: (1) esquema tipado reversível; (2) aquisição guiada por
fronteiras verificadas, com múltiplas ligações candidatas e orçamento; (3) planos
de evidência para comparação, deixando agregação explícita; (4) diagnóstico do
leitor com passagens ouro. Para o objetivo estritamente competitivo, avaliar
HippoRAG2 como gerador/fallback com seleção de evidência Witness por cima, rotulado
como **híbrido** e comparado ao HippoRAG2 sozinho sob custo equivalente. Ganho de
engenharia dessa composição não demonstraria superioridade do executor Witness
isolado. Evitar novas varreduras de limiar sobre o mesmo teste.

## Validação local

Testes de regressão cobrem perda de candidato no top-k global, máscara de fatos,
modo exato, candidatos externos, citações ausentes/inventadas, tipos recusados,
booleans inválidos, bloqueio, orçamento de verificações e preservação do fallback.
A suíte completa também exercita integração, junção versus SQL e ILP.
Resultado: **76 testes passaram**, sem testes ignorados; 143 avisos de deprecação
do PuLP. As opções novas também foram conferidas nas interfaces CLI e piloto.
PuLP e rank-bm25 foram instalados em `.cache/revision-deps` para esses testes,
sem substituir dependências globais. Nenhum teste chama um LLM real.
