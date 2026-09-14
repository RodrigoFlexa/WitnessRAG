# Revisão científica e técnica da implementação

> Atualização de 11/09/2026: por decisão experimental, `semantic` passou a ser o padrão. O modo `exact` permanece disponível explicitamente e é sempre usado pelo comparador SQL. As referências ao padrão exato abaixo descrevem o estado ao término da revisão original.

Data: 10/09/2026. Escopo: `wrag/`, benchmark, testes e documentação. A proposta conceitual original foi preservada em `proposta.md`.

## Parecer

A implementação tinha os componentes centrais da proposta: consultas conjuntivas, testemunhas, seleção por orçamento, proveniência e aquisição dirigida. Entretanto, misturava garantias formais com aproximações semânticas e continha falhas de avaliação que impediam interpretar os números como evidência científica do método. As correções aproximam o código da proposta; ainda não constituem demonstração de novidade ou superioridade experimental.

A contribuição precisa ser formulada como **preservação e aquisição de evidência suficiente sob orçamento**, com consultas e hipóteses explícitas. Junção relacional e testemunhas mínimas, isoladamente, não estabelecem novidade. Um executor SQL é um controle indispensável: se a contribuição alegada for apenas responder consultas conjuntivas exatas, bancos relacionais já realizam essa operação.

## Contrato matemático que o código pode sustentar

Considere uma base finita F de fatos binários dirigidos, com símbolos canônicos fixos, e uma consulta positiva q(x) = ∃y ⋀ᵢ Rᵢ(uᵢ,vᵢ). Uma atribuição às variáveis deve satisfazer simultaneamente todos os átomos. Para uma resposta a, seja W(q,a) a família de subconjuntos mínimos por inclusão de F que satisfazem q(a).

Para S ⊆ F:

    a ∈ Ans(q,S) ⇔ existe W ∈ W(q,a) com W ⊆ S.

Isso decorre da monotonicidade da consulta positiva e da finitude da base. O código agora respeita direção, predicado e identidade de argumentos no modo `exact`. Não transforma similaridade em igualdade. Cada testemunha devolvida satisfaz a consulta formal; a enumeração é completa quando nenhum corte de candidatos, feixe ou saída ocorreu. `--exhaustive` desativa os três limites. A enumeração pode ser exponencial.

A afirmação é relativa aos símbolos e fatos fornecidos. Ela não demonstra que a extração corresponde ao texto, que duas menções se referem à mesma pessoa ou que a consulta formal preserva o significado da pergunta. `expected_type` ainda não impõe tipagem; ontologia, temporalidade, negação e agregações exigem outra camada de modelagem.

Para demandas t=(q,a), pesos wₜ ≥ 0 e custos cₑ ≥ 0:

    U(S) = Σₜ wₜ · 1[existe W ∈ Wₜ com W ⊆ S]
    maximizar U(S), sujeito a Σₑ∈S cₑ ≤ B.

O ILP implementa a conjunção dentro de uma testemunha e a disjunção entre testemunhas da mesma demanda. `optimal=True` exige certificado do solver, além de validação da solução. A otimalidade refere-se às testemunhas enumeradas, não a todas as perguntas possíveis ou ao grafo semanticamente ótimo extraído do texto. A utilidade não é submodular em geral: para uma demanda que exige {e₁,e₂}, acrescentar e₂ tem ganho zero no vazio e ganho um após guardar e₁.

## Problemas encontrados e correções

| Área | Problema anterior | Resultado da correção |
|---|---|---|
| Aterramento | Constante semelhante podia compensar relação errada; inversão automática tornava relações aproximadamente simétricas | Modo exato dirigido por padrão; modo semântico explicitamente aproximado, com limiar de relação e sem inversão automática |
| Identidade | Similaridade e normalização agressiva podiam fundir entidades distintas | Fusão vetorial desativada por padrão; símbolos preservam pontuação e acentos |
| Junções | Deduplicar apenas por conjunto de fatos eliminava atribuições e respostas diferentes | Estado identificado por fatos e atribuições; fatos reutilizados contam uma vez na testemunha |
| Completude | Cortes de candidatos e de testemunhas não invalidavam a marca de busca exaustiva | Todos os cortes registrados; zero desativa cada limite |
| Controle independente | Faltava um executor relacional de referência | Novo `relational`, com junções SQLite parametrizadas, sem aquisição nem feixe |
| Contexto do leitor | Ranking podia cortar a prova para caber em k; havia ainda corte oculto de 1.500 caracteres por passagem | Seleção por conjuntos completos de passagens e envio dos textos integrais; aumento potencial de tokens |
| Proveniência | Noisy-or com desconto dependia da ordem e dava aparência probabilística | Score da melhor testemunha, invariável à duplicação de provas; riscos explicitamente não calibrados |
| Métrica de testemunha | A comparação ignorava predicado e direção | Cobertura lexical de triplas completas; antiga medida relaxada separada; nenhuma é validação semântica |
| Resposta estrutural | Diagnósticos podiam medir uma prova ausente do contexto e ser confundidos com resposta do leitor | EM estrutural e presença de testemunha no contexto registradas separadamente |
| Aquisição | Escritas podiam alterar índices do grafo compartilhado; repetição gerava duplicatas | Memória local reversível, cópia dos índices mutáveis, deduplicação e reativação temporária de fatos podados quando relidos |
| Demandas | Perguntas restantes da avaliação eram usadas como treino; provas alternativas viravam demandas independentes | Treino externo disjunto ou síntese explícita; alternativas da mesma demanda agrupadas por OU |
| Solver | Estado de solução sob limite de tempo podia ser interpretado como ótimo | Validação de integralidade, orçamento e estado de solução; viável e fallback distintos de ótimo |
| Corpus e cache | Amostragem reduzia o universo implicitamente; identidade/cache dependiam de prefixos ou só IDs | Corpus completo por padrão; hashes do conteúdo completo e parâmetros; TF-IDF ajustado apenas ao corpus |
| Comparação | Resultados parciais podiam ter denominadores diferentes ou omitir um método | Mesma interseção de IDs; ausências e duplicatas explícitas; bootstrap das diferenças pareadas |
| Seletividade | Empates podiam ser quebrados arbitrariamente; falhas sem prova ficavam fora | Empates em bloco e falhas incluídas; avaliação sobre acerto estrutural |
| Custos e retomada | Custos de indexação misturados; retomada não reutilizava corretamente a rodada; execução fresh podia duplicar linhas | Custos por etapa e pergunta; configuração/código/dados verificados; custos iniciais preservados e reindexações registradas |
| HippoRAG2 local | Filtro vazio recolocava candidatos rejeitados | Resultado vazio aciona recuperação densa; pesos negativos não entram como sementes |
| Anotações | “Oracle” era apresentado como teto garantido e podia cair silenciosamente para LLM | Condição `annotated` explicitamente privilegiada e heurística; indisponibilidade registrada |

## Validação realizada

Antes das alterações, a suíte existente teve 14 testes aprovados e uma falha. Essa falha envolvia o cache de um embedder de teste cujo vocabulário muda entre instâncias; o teste agora usa codificação direta, sem cache incompatível.

**Resultado final: 55 testes aprovados**, com 143 avisos de deprecação; `compileall` também concluído sem erros.

A suíte ampliada inclui contraprovas para relação errada, direção invertida, identidade de entidades, interseção inconsistente, átomos repetidos, limites de busca, integridade das provas no contexto, passagens longas, idempotência, isolamento da aquisição e reaquisição após poda. Compara também resultados de junções WITNESS com SQL em bases pequenas geradas por semente e confere o ILP contra enumeração de todas as memórias de uma instância pequena, para cada orçamento.

Testes de integração exercitam os sete métodos padrão, relatório, retomada, preservação de custos, configuração incompatível, execução fresh, demandas sintéticas e rejeição de sobreposição entre treino e avaliação. Nenhum desses testes chama LLMs pagos.

Uma execução adicional pela CLI foi concluída com os sete métodos e a condição anotada, usando o `sample` público do HippoRAG (1 pergunta, 3 passagens), LLM stub e TF-IDF. O artefato está em [report.md](../runs/20260910-230140-529391-review-offline/report.md). É um teste funcional, sem poder para comparar métodos. Nesse ambiente, comunidades usaram o fallback NetworkX porque igraph não estava instalado. Fontes e hashes do sample estão registrados na pasta externa `review-data/sources.json` da revisão.

PuLP e rank-bm25 foram instalados em uma pasta de dependências isolada, sem substituir o ambiente global. As deprecações do PuLP são avisos de compatibilidade futura, não falhas dos testes. Os provedores Azure/OpenAI e os modelos de embeddings reais não foram exercitados nesta revisão.

## O que continua em aberto

1. **O grafo ótimo extraído do texto.** Hoje o ILP escolhe fatos de F; não escolhe conjuntamente esquema, entidades, fatos, verificações e extração. A distribuição de demandas e os custos definem o ótimo. Sem especificá-los, não existe um grafo ótimo universal.
2. **Compilação e esquema.** Relações lexicalmente diferentes podem significar a mesma coisa, e nomes iguais podem denotar pessoas diferentes. É preciso um esquema controlado com normalização verificável e resolução de identidade; liberar similaridade não resolve logicamente esse problema.
3. **Incerteza.** Confiança fixa de extração e similaridade não são probabilidades. Mesmo um limite da união requer limites de erro válidos para os eventos envolvidos e tratamento da seleção adaptativa. A representação atual guarda passagem de origem, mas não oferece verificação de spans, dependências de fontes ou calibração conjunta.
4. **Valor da informação.** A aquisição atual ordena similaridade menos custo. Não calcula a esperança da utilidade após observar evidência, não mantém posterior e não otimiza uma política de múltiplos passos. Os nomes históricos `voi` e `expected_gain` não devem ser interpretados literalmente como estimativas calibradas.
5. **Armazenamento real.** O orçamento é uma máscara sobre fatos; entidades, vetores, textos e índice base continuam disponíveis. Experimentos sobre bytes ou RAM precisam materializar índices compactos e contabilizar também os mecanismos de aquisição.
6. **Protocolo científico completo.** Faltam condições com consultas formalmente verificadas e fatos auditados, múltiplas sementes e comparação com implementações oficiais. O diagnóstico anotado não substitui as quatro condições controladas propostas originalmente.
7. **Reprodutibilidade e custo.** Ainda é necessário fixar revisões externas, versões de modelos e dependências em uma rodada científica. O custo financeiro não está completo: embeddings não têm tokens contabilizados e etapas interrompidas antes do checkpoint podem perder registro parcial.

## Experimento decisivo recomendado

Primeiro, usar bases formais conhecidas e consultas ouro, sem aquisição: WITNESS exaustivo e SQL devem concordar em respostas e testemunhas mínimas. Introduzir cortes e medir perda de completude, tempo e memória por tamanho e estrutura da consulta.

Depois, em consultas de teste disjuntas, comparar seleção por testemunhas, poda por fatos individuais, seleção aleatória e memória completa em uma grade de orçamentos. Manter a mesma família de demandas de treino, custos e executor. Reportar utilidade de teste, respostas completas, custo e distância ao ótimo apenas onde um ótimo foi realmente certificado.

Por fim, introduzir separadamente erros de extração, identidade e compilação. Para testar aquisição, comparar a política dirigida com nenhuma aquisição, releitura aleatória e releitura por similaridade com o mesmo orçamento de chamadas. Somente após esse controle a política pode sustentar uma alegação de ganho por preservação de provas ou por valor da informação.

Grupos livres e ribbon graphs não são necessários para o fragmento implementado: junções e hipergrafos de testemunhas já representam cadeias e interseções. Uma estrutura algébrica adicional só se justifica se expressar uma propriedade observável que esse modelo não captura e gerar uma previsão testável.

## Preservação dos arquivos originais

Não foi detectado um repositório Git neste diretório ou nos ancestrais examinados. Antes das alterações, código, testes e documentos foram copiados para [witnessrag-before-review.zip](C:/Users/rodri/.codex/visualizations/2026/09/10/01a08cc3-ee8e-7cf0-83d8-d14465318798/witnessrag-before-review.zip). A proposta em `docs/proposta.md` permaneceu intacta.
