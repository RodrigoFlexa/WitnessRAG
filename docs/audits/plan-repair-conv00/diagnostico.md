# WitnessRAG: diagnóstico causal e piloto refutável na conv00

> **Atualização após o piloto:** a recomendação H1 abaixo foi refutada e está retirada. O leitor citado obteve F1 multi-hop 0,243936 (controle 0,398074) e single-hop 0,577784 (controle 0,637681), com maior latência. Não executar as repetições/ampliação sugeridas abaixo para promover esta versão. Ver `reader-failure/postmortem.md` e os resultados reproduzidos por `scripts/audit_reader_cited.py`. O restante deste documento preserva a hipótese anterior para rastreabilidade.

Auditoria local de 21/09/2026. Unidade: pergunta, ligada por `qid`. Conv00 do launcher é `conv-26` nos identificadores do dataset. Foram comparadas as mesmas 102 perguntas, 70 single-hop e 32 multi-hop oficiais. Os rótulos oficiais são usados para avaliação, não para acionar rotas. Há perguntas multi-hop com uma única passagem ouro: bloco de evidência não é hop lógico.

**Recomendação:** manter o controlador reduzido e testar primeiro a substituição de sua chamada de leitura por uma enumeração citada de itens, em recuperação congelada. Não aumentar replanejamentos nem ligar `proof_reader`. Corrigir a instrumentação do verificador antes de tentar obter mais provas. Só depois testar uma troca seletiva de passagem baseada em membro útil, separadamente do leitor.

O código de produção não foi alterado nesta auditoria. Foram adicionados um auditor reproduzível, um leitor experimental isolado, quatro testes e os resultados da auditoria. Nenhuma inferência Qwen foi executada aqui; não existe ganho novo medido.

## Evidência e reprodução local

- `questions.csv`: uma linha para cada uma das 102 perguntas, respostas, F1, delta pareado, apoios ausentes, localização desses apoios na fronteira/aquisição/candidatos, fechamento/cobertura dos planos, verificação, custo e identidade de contexto.
- `summary.json`: contagens reproduzidas, comparação do controle, hashes de entrada e intervalos bootstrap.
- `audit-output.txt`: saída completa do auditor, incluindo as 32 linhas multi-hop.
- `scripts/audit-plan-repair-local.py`: recalcula os arquivos acima sem inferência ou alteração dos resultados originais.
- `wrag/eval/reader_pilot.py`: piloto experimental executável com exatamente os contextos da versão reduzida; modo `--preflight-only` não chama LLM.

Fontes primárias locais:

1. `runs/locomo-plan-repair-02/benchmark/20260920-234016-169721-qwen-pilot/locomo/witnessrag.jsonl`.
2. `runs/locomo-plan-repair-01/conversations/conv00/benchmark/20260920-192726-557869-conv00/locomo/witnessrag.jsonl`.
3. `runs/locomo-controlled-01/controlled/158a0d2a68689a0db8839076/{memory.json,audit.json,answers/,retrieval/}`.
4. Corpus textual em `runs/locomo-plan-repair-02/data/` e configurações `run.json`/`pilot.json` das duas execuções.

## O que os números permitem concluir

| Medida | Primeira versão | Reduzida |
|---|---:|---:|
| F1 oficial single-hop | 0,614305 | 0,637681 |
| F1 oficial multi-hop | 0,396329 | 0,398074 |
| F1 multi-hop antes da guarda da primeira versão | 0,427579 | — |
| Chamadas lógicas LLM/pergunta | aproximadamente 15,2 | 7,205882 |
| Tempo recuperação + leitura/pergunta | aproximadamente 43,7 s | 16,938647 s |

Na reduzida: recuperação média 15,20694 s e leitura 1,73171 s. São latências por pergunta registradas pelo harness, não throughput do servidor nem custo monetário. As 735 chamadas se dividem em 102 compilações, 102 replanejamentos, 218 avaliações de obrigações, 92 aquisições, 119 verificações e 102 leituras. Há só 4 hits de cache LLM; não explicam a redução de custo. Máximo observado: 9 chamadas/pergunta. Isso não é um teto teórico: retries e obrigações podem excedê-lo em outras perguntas.

O multi-hop tem 6 ganhos, 6 perdas, 20 empates. O saldo é apenas +0,001744. `qa40` contribui sozinho +1,0 de soma de F1, ou +0,03125 na média. Nos outros 31 casos, o saldo é −0,94418 de soma de F1. Portanto, o resultado **não demonstra que cortar o controlador preservou sua qualidade intrínseca**: o ganho da retirada da guarda encobre outras perdas de contexto/leitura. Também não demonstra que o corte causou essas perdas, pois várias mudanças ocorreram juntas.

## Separação das etapas

As contagens de sintomas abaixo se sobrepõem. Não somar colunas nem interpretar o rótulo terminal do controlador como uma atribuição causal exclusiva.

| Etapa/sintoma na reduzida | 102 perguntas | 32 multi-hop | Leitura causal |
|---|---:|---:|---|
| Apoios anotados completos no top 5 | 88 | 18 | Presença dos blocos; não garante suficiência textual |
| Algum plano fecha junção | 89 | 31 | Falta de fechamento não é o principal gargalo multi-hop |
| Algum plano fechado com cobertura aprovada | 33 | 19 | Avaliação de obrigações elimina muitos fechamentos |
| Algum plano tem átomo sem candidato | 26 | 2 | Há problema de aterramento, concentrado em single-hop |
| Algum plano executável não fecha apesar de candidatos em todos os átomos | 8 | 1 | Incompatibilidade de bindings/cortes; não ausência simples de fatos |
| Verificações de testemunhas | 119 | 50 | Unidade: testemunha, não pergunta |
| Testemunhas aceitas | 17 | 12 | 13 perguntas com alguma aceitação; 8 multi-hop |
| Prova rotulada `full` | 11 | 8 | Nem sempre significa conjunto completo |
| Prova provisória | 2 | 0 | Contexto, não prova lógica completa |
| Sem prova aceita | 89 | 24 | Isso não impede responder pelo fallback |
| Contexto alterado pelo Witness relativamente ao fallback | 1 | 1 | Quase todo F1 é do fallback + leitor |

Há 220 planos finais: 168 fechados, dos quais 42 cobertos e 126 não cobertos; 2 não executáveis. No multi-hop, 49 dos 54 planos fecham, 27 desses sem cobertura. Há 19 registros de corte de candidatos e 31 de testemunhas em planos multi-hop, nenhum corte de feixe registrado ali. Em single-hop há um registro de corte de feixe. Não é correto culpar o beam por todo `binding_conflict_or_beam_cut`.

Os rótulos terminais são: `closed_coverage_rejected` 61/102 (15 multi), `no_verified_witness` 12 (8 multi), `zero_atom_candidates` 12 (nenhum multi), `binding_conflict_or_beam_cut` 3 (nenhum multi), `invalid_or_unsupported_plan` 1 (multi), sucesso 13 (8 multi). A precedência no `if/elif` de `_retrieve_active` esconde causas secundárias: `qa66`, por exemplo, tem conflito numa junção, mas termina como cobertura rejeitada devido a outro plano fechado.

**Recuperação e seleção.** Os 14 multi-hop sem todos os apoios têm F1 0,325614; os 18 completos, 0,454431. Em 10 dos 14, pelo menos um apoio ausente já está na fronteira; em 7, todos os apoios ausentes estão nela. Em 6, um apoio ausente aparece nos candidatos registrados da junção. Nenhum dos apoios ausentes no top 5 aparece nas aquisições registradas. Isso favorece testar seleção de contexto antes de pagar mais OpenIE. Não significa que toda passagem ouro seja suficiente ou única: `qa43` tem texto alternativo sobre arte abstrata em D17:13, e `qa37` responde parcialmente sem todos os blocos anotados.

**Compilação e junção.** `qa40` tem plano `count` sem átomos, erro `variavel_de_resposta_ausente`; o fallback ainda responde 2. `qa66` tenta `hike(Melanie, ?y)` e `do with family(?y, ?x)`: 8 e 23 candidatos, mas bindings incompatíveis; a alternativa fecha uma estrutura desconectada sem ligar a atividade à caminhada. A rejeição de cobertura dessa alternativa é correta. `qa48` chega a verificar “okay” como tipo de cerâmica; `qa61` verifica “beach” e “painting” como artistas musicais. Nesses casos, quantidade de candidatos não equivale a qualidade e gastar o limite nos dois primeiros custa membros úteis.

**Obrigações.** `REPAIR_PLAN_TEMPLATE` já manda contar `source_conditions` como requisitos preservados. Em `qa91`, a condição “necklace belongs to Caroline” está presente, mas o checker declara faltar “Caroline's necklace”. Isso é descumprimento semântico do contrato pelo LLM, não campo ausente no código. O retry de `assess_plan_repair` só detecta esquema inconsistente: uma justificativa semanticamente errada, mas bem formada, passa. Em `qa47`, a condição “Caroline has a negative experience” não liga explicitamente o suporte ao evento; aqui há também subespecificação real. Não promover todas as condições textuais automaticamente a cobertura/prova.

**Verificação.** Das 102 rejeições de testemunha: 34 `not_explicit`, 25 `invalid_evidence`, 22 `wrong_answer_type`, 13 `wrong_identity`, 6 `incomplete_answer`, 2 `missing_atom`. No multi-hop: respectivamente 15, 7, 11, 3, 1, 1. São classificações do verificador, não auditoria humana de 102 erros. A categoria `invalid_evidence` mistura citação inexata, pid/índice errado, cobertura de átomos/condições e filtro lexical; os logs não distinguem qual ramo falhou.

**Leitura.** `proof_reader=false`: o leitor não recebeu o candidato “Melanie” do verificador em `qa47`; ele produziu a mesma resposta errada independentemente a partir dos cinco blocos. Não atribuir esse caso a contaminação por dicas de prova. `qa78` é ainda mais claro: “new shoes” foi aceito, sua passagem foi selecionada, mas o leitor respondeu só “figurines”. Corrigir verificação não resolve esse erro final.

## Contratos e decisões de código corrigíveis

1. **`full` mistura validade de membro e completude do conjunto.** `verify_witnesses` diz explicitamente que um membro não precisa provar exaustividade. `_retrieve_active` aceita qualquer lista não vazia e registra `classe_prova="full"`. Cinco das oito provas multi-hop `full` nem contêm todos os apoios anotados: qa24, qa60, qa15, qa51, qa70. Isso não prova falsidade dos membros; prova que o rótulo não sustenta a conclusão “resposta completa”. Separar `member_validity`, `requirement_coverage` e `enumeration_completeness=unknown/explicit`, sem afirmar completude por duas testemunhas.

2. **A pré-seleção descarta candidatos antes de saber se satisfazem o tipo/condição.** `cover_answers(fitting, limit)` limita a dois antes da verificação, e `result.witnesses = accepted` remove os demais candidatos da seleção de contexto. `nao_avaliadas` recebe a lista já cortada, portanto pode ser zero mesmo com dezenas de candidatos não verificados. Em qa61, dois slots são gastos em praia e pintura. Guardar contagens antes/depois do corte e separar pool de propostas, membros verificados e candidatos úteis para leitura. Não rotular candidatos não verificados como provas.

3. **O veto lexical a condições é incompatível com diálogo.** `_condition_quote_plausible` exige duas palavras exatas quando a condição tem pelo menos três palavras relevantes. A condição `instruments Melanie play` não casa com “playing my violin”; primeira pessoa, flexão e tipo implícito bastam para reprovar. Em qa60, “violin for self-care” tem uma citação presente em D2:5, motivo favorável do LLM e rejeição `invalid_evidence`; o clarinete é aceito e o violino fica fora do contexto. O mesmo padrão aparece em qa91. O log não preserva o JSON bruto de `condition_evidence`, então não permite provar qual subteste causou cada rejeição histórica. O falso negativo do predicado local é reproduzível, mas a distribuição exata de causas exige instrumentação. Substituir o veto por diagnóstico, mantendo origem literal, cobertura de IDs de condições e julgamento semântico do verificador; preservar speaker/turno e permitir frases vizinhas para resolver “I/my”. Testar essa alteração separadamente, pois pode aumentar falsos positivos.

4. **`not_explicit` pode proibir o raciocínio que o benchmark exige.** qa71 precisa ligar “Becoming Nicole”, recomendado em D7:11, a “that book you recommended” em D17:10. O plano `read(Melanie, ?x)` não representa a ligação da recomendação e o candidato “book” não identifica o título. Relaxar apenas o verificador não cria essa ligação. Distinguir prova de cada premissa de inferência composicional; não exigir que a conclusão inteira apareça numa frase.

5. **Seleção cega da quinta passagem foi uma mudança real entre execuções.** Não reintroduzi-la indiscriminadamente. Na nova execução, mesmo 157 fatos novos de aquisição não produzem uma troca útil de contexto na maioria das perguntas. O controlador tenta recuperar/validar provas enquanto o leitor continua vendo o fallback original em 101/102 casos.

6. **Contagem de filhos é um contrato diferente de contagem de menções.** qa75 enumera `kids` e `son`; o primeiro não é um indivíduo distinto, o segundo não permite contar todos os filhos. `member_before_count` já corrige o tipo `number` no verificador atual. Não voltar a rejeitar todo membro por não ser número, nem converter duas evidências/menções em total 2. O código já mantém limite inferior para contagens; preservar isso. A guarda numérica tem um ramo de validação de contagem inalcançável após o retorno antecipado atual; limpar o código morto é manutenção, não hipótese de ganho.

7. **O detector antigo de conjuntos não cobre “Who supports …”.** Ligar de novo apenas `answer_guard` não corrige qa47: `looks_like_answer_set` reconhece padrões `what/which`, e a guarda não recebe a agregação do plano. Além disso, a guarda exige itens literais separados por vírgulas, incompatíveis com várias paráfrases válidas. O piloto novo usa a agregação prevista pelo compilador apenas para escolher a rota, sem passar os candidatos ao leitor.

8. **Telemetria insuficiente.** `_trim` elimina flags e parte das citações por decisão. `verification` precisa persistir `local_failure_reason`, resposta bruta, `plan_id`, `witness_id`, quantos candidatos foram excluídos antes do verificador e os spans aceitos. O diagnóstico deve permitir múltiplas causas por pergunta. Também salvar fallback top 20, consultas/vetores de sondas, vocabulário inicial/replanejado, scores antes/depois da seleção e hash dos textos efetivamente enviados.

## Por que as passagens mudaram

Os hashes de corpus e perguntas são idênticos. Ambos os `pilot.json` registram a mesma fonte congelada e digest de manifestos `bdf2516e…`. O pickle local da conv00 foi verificado contra seu SHA-256: `0cd33dbf…dc69`. Ambos usam `st-BAAI_bge-m3`.

Os hashes do código de `run.json` foram reproduzidos usando os arquivos `wrag/*.py` do Git: `635b840` corresponde à primeira execução (`eda75d32…3300`); `e239880`/`2398615` correspondem à reduzida (`0e1e24c4…05b0`). O diff mostra a condição `and not cfg.plan_repair` acrescentada à troca da última passagem por uma da fronteira.

O mesmo diff mostra outra mudança de contrato: a primeira versão podia escolher plano sem condições textuais sem verificação quando `verify_witnesses=false`; a reduzida verifica sempre que `plan_repair=true`. Logo, nem as contagens de provas das duas versões são diretamente equivalentes sem distinguir o caminho de aceitação.

Há 93 conjuntos diferentes, apenas 9 idênticos; no multi-hop, 30 diferentes e 2 idênticos. As primeiras quatro posições são idênticas em 94/102. Na primeira execução, as 85 perguntas sem prova alteraram o contexto. **Reaplicar a regra antiga usando a fronteira/aquisições antigas e o top 5 novo reproduz exatamente os 85 contextos antigos.** Esses 85 casos não precisam de uma hipótese de índice ou cache diferente para serem explicados. Os oito conjuntos diferentes restantes pertencem a perguntas com prova `full` na primeira versão, afetadas pela seleção/verificação; não atribuir cada diferença residual a uma única causa sem replay das chamadas antigas.

Só 47 fronteiras completas são iguais. Isso é compatível com planos/sondas diferentes: planos iniciais múltiplos vêm de geração LLM, vocabulário replanejado depende de feedback/aquisição e o orçamento mudou. A memória congelada não congela as chamadas de consulta. A aquisição escreve em `MemoryView`, com rollback ao final por padrão; não altera os vetores do fallback denso. Não há evidência local de contaminação cumulativa entre perguntas.

Os caches LLM são diferentes e a chave normal inclui a URL/porta do endpoint (8097 versus 8098), além de modelo/prompt. Revisão do modelo está vazia nos manifestos. Os caches de embeddings/LLM destas duas execuções não estão disponíveis na cópia local; não é possível comparar seus bytes. Não usar “temperatura zero” como prova de identidade: no controle, 3/8 repetições frescas da amostra conv00 diferiram no estado de recuperação, embora o replay das chamadas gravadas fosse idêntico. O cache de embeddings por modelo também não fixa revisão de pesos. O próximo ensaio deve registrar revisões efetivas, não só nomes.

## Evidência que refuta alternativas promissoras

No controle, com os mesmos contextos por par de leitores:

| Recuperação/leitor | F1 single | F1 multi |
|---|---:|---:|
| evidence/common | 0,661029 | 0,403639 |
| evidence/proof | 0,632621 | 0,338535 |
| soft-v2/common | 0,635026 | 0,419888 |
| soft-v2/proof | 0,610756 | 0,362293 |

As hashes de recuperação e listas de pids foram conferidas em todos os pares. `proof_reader` causa delta multi −0,065104 na recuperação evidence (3 ganhos/4 perdas/25 empates) e −0,057595 na soft-v2 (5/10/17). Isso não refuta todo leitor estruturado, mas refuta a recomendação de simplesmente ligar as dicas atuais. O soft-v2/common tem resultado exploratório melhor que 0,3981, porém muda recuperação; não é ensaio isolado da correção proposta.

## Inventário das 32 perguntas multi-hop

`A` indica todos os blocos anotados no top 5; `I`, algum ausente. O diagnóstico principal é uma hipótese auditável apoiada pelo texto/trace, não uma causa única. CSV contém os flags simultâneos.

| qa | A/I | F1 novo | Diagnóstico e possibilidade de reparo |
|---|---|---:|---|
| 3 | A | 1,000 | Adoção correta; proteção contra regressão |
| 4 | A | 0,667 | “transgender” perde “woman”; escopo/tipo de resposta |
| 7 | A | 0,000 | “single parent” futuro em D2:14 exige inferência; não forçar com prova literal |
| 11 | A | 1,000 | Sweden correto sem prova aceita |
| 13 | A | 0,667 | Omite público trans da carreira; não necessariamente pergunta enumerativa |
| 15 | I | 0,750 | Falta swimming; prova de membros não é coleção completa |
| 18 | A | 0,667 | Lê mountains/forest, omite beach presente; candidato do leitor |
| 19 | I | 0,000 | Atividades viram gostos dos filhos; mistura recuperação, identidade e relação |
| 23 | A | 0,500 | Charlotte's Web recuperável; Nothing is Impossible ausente no texto/caption; limite de modalidade |
| 24 | I | 0,500 | Falta bloco da pottery; prova aceita nature/family, seleção muda contexto |
| 32 | I | 0,533 | Falta school speech; lista ampla com itens relacionados exige auditoria semântica |
| 34 | A | 0,286 | Mentoring verbose; adoption research não é evento de ajuda a crianças; school speech omitido |
| 37 | I | 0,286 | Sunset recuperável por evidência alternativa; detalhe excedente reduz F1 |
| 38 | I | 0,333 | Faltam blocos de atividades familiares; não corrigível só encurtando a resposta |
| 39 | I | 0,582 | Verbos/qualificadores e itens adicionais; suporte alternativo possível |
| 40 | A | 1,000 | Plano inválido; leitor 2 correto, guarda antiga estragava; manter rota |
| 43 | I | 0,000 | Pergunta estilo, resposta lista meios/objetos; D17:13 contém abstract em apoio alternativo |
| 47 | A | 0,000 | Melanie substitui grupos de suporte; vincular D3:11–13 a D12:1 |
| 48 | A | 0,286 | Tipos bowls/cup descritos com decoração; principal alvo de concisão sem perda de sentido |
| 51 | I | 0,074 | Legendas longas e atribuição de autoria; leitor + evidência ausente |
| 52 | A | 1,000 | Pets corretos apesar de verificador rejeitar identidade; proteger |
| 55 | I | 0,000 | Interseção de autoria; responde horses, falta apoio anotado; requer duas premissas |
| 56 | A | 0,400 | Omite transgender symbol, inclui objetos afins; modalidade/atribuição exigem inspeção |
| 60 | I | 0,667 | Violino útil rejeitado por invalid_evidence; clarinete apenas; alvo da seleção posterior |
| 61 | I | 0,500 | Candidatos errados gastam verificação; falta Matt Patterson |
| 65 | A | 0,208 | Mudanças corporais/perda de amigos diluídas em descrição ampla; possível reparo de escopo |
| 66 | I | 0,000 | Junção errada + bloco útil nos candidatos não selecionado |
| 70 | I | 0,333 | Poetry reading ausente; confunde LGBTQ geral com transgender-specific |
| 71 | A | 0,000 | Resolver recomendação de livro entre D7:11 e D17:10; leitor atual abstém |
| 75 | A | 0,000 | “2 younger kids” não é total; parentesco e imagem de três crianças, ambiguidade textual |
| 76 | A | 0,000 | “yesterday” em sessão de 20/10 → 19/10; exige referência à caminhada, não enumeração |
| 78 | A | 0,500 | Shoes verificado e entregue, mas leitor omite; alvo forte de enumeração |

## Menor mudança e orçamento

**Hipótese H1:** ao ler o mesmo top 5, coletar itens com suas citações antes de renderizar respostas curtas recupera membros omitidos e o nível de abstração pedido, sem usar dicas dos candidatos do grafo. Isso deve ajudar sobretudo qa47, qa48, qa78 e qa18; qa34/qa65 são alvos secundários mais arriscados. Não resolve a ausência do violino em qa60, nem o título ausente de qa23. O piloto deixa contagens na rota original. Datas qa76 e scalar qa71 ficam fora se o detector não classificar o plano como conjunto; não prometer corrigi-las nesta alteração mínima.

Implementação experimental entregue: `eligible` usa `looks_like_answer_set` OU algum plano compilado com agregação `set`, excluindo `how many`. Ativa 50/102: 24 multi-hop e 26 single-hop. Nenhum ouro vai ao prompt ou à decisão. O prompt pede relação correta, sujeito, qualificadores, evidência de todos os blocos e resposta no nível de tipo/entidade solicitado. Sem nomes específicos destes exemplos. Cada item traz uma ou mais citações literais; validação local verifica origem por item. Itens com citação inválida são descartados individualmente, sem descartar os outros; saída inválida/truncada abstém, sem usar resposta ouro, resposta antiga ou segunda chamada. O JSON bruto permanece salvo.

Limitação deliberada: citar texto verdadeiro não prova que o item o segue semanticamente. Esta checagem não deve ser anunciada como “respostas sem alucinação”; a auditoria de precisão abaixo é obrigatória.

**Orçamento H1:** uma chamada de leitura por pergunta por sistema, nenhuma chamada adicional ao controlador, cinco passagens originais completas e na mesma ordem. Leitor atual: máximo 512 tokens; candidato: máximo 900. Custo em tokens pode subir apesar de chamadas constantes. Em implantação sobre estas perguntas: 7,2059 chamadas/pergunta históricas; nenhuma nova aquisição ou verificação. Piloto pareado: 102 chamadas comuns + 50 alternativas = 152 chamadas novas por repetição; nas 52 rotas idênticas, reutiliza-se a mesma leitura comum. Essa reutilização é explícita nos registros. Em três repetições: 456 chamadas, e ainda 32 unidades multi-hop, não 96.

Critério operacional proposto: média total estimada ≤20 s/pergunta, média de chamadas ≤7,3 e teto de leitura 900 tokens; informar também p50/p95 e tokens reais. Latência de ponta a ponta precisa de confirmação após a promoção, pois somar recuperação histórica a leitura nova é apenas uma estimativa. Para a futura rodada do controlador, impor teto duro de 9 chamadas incluindo retries, reservando uma para QA; ao esgotar, usar fallback. Essa mudança de orçamento deve ser ablada separadamente de H1.

**Hipótese H2, somente depois:** quando há indício observável de cobertura incompleta, aproveitar um membro/condição novo já localizado fora do top 5, sem aumentar o orçamento. `all_recall@5` serve só para estratificar resultados; é proibido usá-lo como gate em produção. O gate pode usar condições não satisfeitas no texto e candidatos com provenance válida. Mantém quatro passagens base, admite no máximo uma troca da quinta, e exige que a troca cubra requisito/membro novo sem remover a única evidência de outro requisito já identificado. Sem candidato elegível, preservar contexto. Para um novo verificador por lote, reutilizar um dos dois slots existentes; não acrescentar chamadas. O lote julga membros, não exaustividade, e precisa de controle de contexto/tokens. Testar primeiro qa60/qa61/qa24 como regressões qualitativas, mas medir as 102 perguntas.

Este gate é uma hipótese, não conhecimento de completude. Pode falhar em qa38/qa51, com dois apoios faltantes e necessidade de mais de uma troca; em qa43 por abstração; em qa55 por autoria; e em perguntas cujo suporte não está no pool. A fronteira oferece todos os apoios faltantes só em 7/14 casos. Não prometa resolver 14 com uma substituição.

## Experimento que pode refutar H1

1. Fixar o corpus/ordem/102 qids e os cinco textos + pids + ordem da reduzida. O módulo valida hashes de corpus/perguntas e salva snapshots completos com hash; não instancia embedder nem retriever.
2. Reexecutar leitor comum e candidato no mesmo servidor/modelo/revisão, temperatura zero e seed 42, alternando AB/BA. Cache LLM desligado. Repetições medem instabilidade de serving. Os 50 casos elegíveis recebem um leitor por braço; os demais têm uma chamada idêntica compartilhada. Nunca selecionar o melhor resultado entre repetições.
3. Métrica principal: `question_score` oficial, categoria 1 e 4. Calcular delta por qid, média multi, vitórias/perdas/empates e bootstrap pareado por pergunta, agrupando repetições antes de reamostrar. A diferença entre baseline reexecutada e histórica é controle negativo de serving, não efeito do candidato.
4. Proteção: single-hop ≥ baseline comum contemporânea; margem tolerada no piloto ≤0,01, mas qualquer regressão deve ser investigada. Reportar estratos A/I, elegíveis/não elegíveis e respostas vazias/truncadas. Não escolher gate usando o ouro.
5. Custo: separar recuperação histórica, novas chamadas comuns, novas chamadas alternativas e rotas compartilhadas. Registrar tokens e wall time por braço; não somar a recuperação duas vezes nem chamar 152/102 de custo de implantação. Instalar/servir/indexar não entra na latência de QA.
6. Auditoria semântica cega de todos os itens emitidos nos 32 multi-hop e nas 26 perguntas single-hop elegíveis, em ambos os braços: correto com apoio, contradito, sem apoio, incerto, duplicado. Inspecionar também os itens rejeitados pela checagem local. Relatar fração de itens sem apoio, precisão por item e proporção de perguntas com qualquer item sem apoio. O F1 oficial de categoria 1 faz média do melhor match por item ouro e NÃO penaliza itens extras; ganho por lista inflada não valida H1. Ajudar a auditar com citações não substitui revisão de sujeito/relação/tempo. Se houver dois avaliadores, usar avaliações independentes e adjudicação; guardar divergências.
7. Refutação prática: não promover se o delta multi ≤0, se single perde mais de 0,01, se ganho depende de itens sem apoio, se truncamentos crescem ou se custo excede o orçamento. Não retocar o prompt repetidamente nos mesmos 32 casos e chamar a última rodada de confirmação.

## Ordem de implementação

**0 — Já entregue:** auditor, CSV/JSON, piloto isolado e quatro testes. `pytest tests/test_reader_pilot.py -q`: 4 passaram. Preflight confirmou 102 perguntas, 50 elegíveis, hash de snapshots `8324b7e03f52cc0f96d3ee11a2a7608435f6a0d4fac357418817590fb05e67c1`. Nenhuma execução no servidor foi feita.

**1 — Executar H1 sem editar o controlador.** Módulo experimental contém pseudocódigo executável: `eligible → uma chamada JSON → cited_answer por item → renderização por vírgula → F1 oficial`. Fazer auditoria cega dos itens antes de decidir promoção.

**2 — Se H1 passar, integrar atrás de flag explícita.** `wrag/config.py:QAConfig` ganha `cited_set_reader=False`; `wrag/pilot.py:parser/_run_config/_validate_resume` registra a flag e o limite de tokens; `wrag/eval/runner.py:_answer_standard` passa somente a decisão de rota derivada dos planos, jamais ouro; `wrag/eval/reader.py:read` chama o leitor citado em vez de QA comum quando aplicável; `wrag/prompts.py` recebe o template versionado. O mesmo leitor deve valer para todos os métodos em qualquer futura comparação entre métodos. Atualizar o docstring que hoje presume leitor idêntico sem descrever ablações.

**3 — Instrumentar e corrigir contratos, uma alteração por ablação.** Em `wrag/witness/verification.py:verify_witnesses`, registrar causa local exata e JSON bruto antes de alterar critérios. Em `wrag/eval/runner.py:_trim`, manter IDs e motivos em sidecar integral. Em `wrag/witness/research.py:PlanAssessment/assess_plan_repair`, representar requisitos presentes versus pendentes de verificação; não fundir isso com verdade da fonte. Em `wrag/methods/witnessrag.py:_retrieve_active`, separar validade de membro, cobertura e completude; registrar tamanhos originais antes de `cover_answers`.

Pseudocódigo para evitar descarte silencioso:

```python
proposed = result.witnesses
to_check = select_typed_candidates(proposed, reserved_verification_budget)
accepted, decisions = verify_members(to_check)  # preserva condições + citações
diagnostics.n_unchecked = len(proposed) - len(to_check)
proof.members = accepted
proof.enumeration_complete = explicit_completion_certificate_or_unknown()
context_candidates = accepted + useful_uncertified_sources(proposed, decisions)
# Só accepted é prova; fonte útil não vira resposta certificada.
```

O seletor por tipo é nova hipótese: nem embeddings nem o `expected_type` atual “other” bastam. Implementar metadados/condições tipadas e testar candidatos errados antes de gastar o slot. Não adicionar um classificador LLM por candidato.

**4 — H2 e orçamento rígido.** `select_evidence` em `wrag/witness/research.py` recebe requisitos e candidatos com provenance; `_retrieve_active` permite no máximo uma troca justificada mesmo sem prova completa; ledger de chamadas em torno de compile/obligations/acquire/verify reserva QA e inclui retries. Novo resultado de recuperação é congelado para comparar H2 com H1 no mesmo leitor. Não usar script default “all”.

Testes adicionais necessários antes das etapas 2–4: referências pronominais com speaker; citação literal válida com `play/playing` e nome ausente; condição semanticamente errada apesar de palavras em comum; membro rejeitado sem apagar os outros; `count` sem exaustividade não vira total; `set` com um membro não vira completo; n_unchecked antes do corte; custo inclui retries e reserva QA; máximo de cinco passagens; troca que apagaria evidência exclusiva é rejeitada; replay com mudança de um byte/ordem de pid falha; pergunta sem gold no prompt; datas relativas e captions não identificam autoria automaticamente. Usar nomes/fatos sintéticos nos testes de contrato, e os qids da conv00 apenas como fixtures de regressão diagnóstica.

## Comandos exatos no servidor

Do diretório `/home/rodrigo.flexa/WitnessRAG`, com os novos arquivos sincronizados. Usar saída nova; o módulo recusa sobrescrever um diretório existente. O piloto não precisa de embeddings, OpenIE ou reindexação.

Se não houver Qwen servindo, iniciar num terminal, com GPU/porta livres. Fixar `--revision` e `--tokenizer-revision` ao hash de pesos usado na execução reduzida quando esse hash for recuperado do servidor; os manifestos locais não o fornecem. Sem isso, declarar explicitamente que a revisão histórica não foi reproduzida; o par contemporâneo continua controlado.

```bash
cd /home/rodrigo.flexa/WitnessRAG
CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv-vllm/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen2.5-14B-Instruct \
  --served-model-name Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8098 --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.8 \
  --max-num-seqs 8 --tensor-parallel-size 1 --generation-config vllm
```

Com o servidor pronto, em outro terminal:

```bash
cd /home/rodrigo.flexa/WitnessRAG
export PYTHONHASHSEED=42 WRAG_SEED=42 WRAG_LLM_BACKEND=vllm
export OPENAI_MODEL=Qwen/Qwen2.5-14B-Instruct
export OPENAI_BASE_URL=http://127.0.0.1:8098/v1 OPENAI_API_KEY=local-pilot
export WRAG_LLM_CACHE=0 WRAG_AZURE_MAX_RETRIES=0
.venv-bench/bin/python -m wrag.eval.reader_pilot \
  --baseline runs/locomo-plan-repair-02/benchmark/20260920-234016-169721-qwen-pilot/locomo/witnessrag.jsonl \
  --data-dir runs/locomo-plan-repair-02/data \
  --output runs/locomo-reader-cited-01 --repeats 1
```

Executar primeiro com `--preflight-only` se quiser validar o snapshot sem inferência. Para medir repetibilidade depois da primeira inspeção, usar o MESMO prompt/flags com `--output runs/locomo-reader-cited-repeat-01 --repeats 3`. Não há resume: saídas parciais permanecem para diagnóstico e exigem outra saída. O módulo não inicia/encerra servidor existente. Após validar H1, a rodada de integração deve ser conv00 explicitamente: o script atual `scripts/run-locomo-plan-repair.sh` tem default `--locomo-conversation all`; passar `--locomo-conversation 0` ao final. Não executar esse script como substituto do ensaio congelado.

## Meta 0,45 e quando ampliar

Faltam aproximadamente `32 × (0,45 − 0,3980737434) = 1,66164` pontos somados de F1. Isso é aproximadamente dois erros completos corrigidos sem perdas, mas não há garantia de que sejam recuperáveis. Corrigir só qa40 novamente não ajuda: já vale 1.

Se só os 18 com apoio completo melhorarem, sua média precisa subir de 0,454431 para 0,546745, mantendo os outros 14. Se só os 14 incompletos melhorarem, precisam subir de 0,325614 para 0,444303. Nenhum desses números é previsão; são requisitos aritméticos.

Um cenário concreto de H1: qa48 de 0,285714 para 1 (+0,714286), qa47 de 0 para 0,888889 com “mentors, family, friends”, qa78 de 0,5 para 1 (+0,5). Total +2,103175, média final 0,463798. Não foram executadas essas respostas; são contrafactuais para mostrar que poucos casos plausíveis bastam, e também como uma ou duas perdas destroem o ganho. A meta não justifica acrescentar palavras sem apoio ou copiar o ouro.

Bootstrap de 20.000 reamostragens por pergunta, seed 42: F1 reduzido multi com intervalo percentil 95% aproximadamente [0,2840; 0,5147]; delta reduzida−primeira versão [−0,0914; +0,1023]. São intervalos condicionais exploratórios desta conversa; as perguntas compartilham memória e não são conversas independentes. Não dão um intervalo válido de generalização para dez conversas. Repetir Qwen não aumenta o número de perguntas independentes.

Para ampliar, exigir ganho pareado estável nas repetições, multi ≥0,45 na conv00, proteção single, custo no orçamento e ausência de aumento de itens sem apoio. Um intervalo pareado que ainda cruza zero implica sinal preliminar; não vender superioridade estabelecida. Depois congelar implementação/prompt e avaliar primeiro um pequeno conjunto de conversas não usado para ajustes. Como já existem resultados do controle de dez conversas, verificar histórico de desenvolvimento: se todas já orientaram decisões, chamá-las de validação interna, não holdout inédito. Só então ampliar ao conjunto completo e usar intervalos por conversa. Se H1 falhar, publicar a falha e testar H2 isoladamente; não escalar o controlador caro para tentar alcançar 0,45 na mesma amostra.
