# Próxima hipótese após a falha do leitor citado

Esta nota substitui a recomendação H1 de `diagnostico.md`. É um desenho experimental, não um ganho medido.

## Qual foi a melhor condição?

Depende do denominador. Na conv00, `frontier` da ablação de provas chegou a F1 oficial multi-hop **0,4941**, single-hop **0,6258**, com **5,20 chamadas LLM/pergunta**. Esse é o maior F1 multi-hop registrado nos JSONL completos da conv00 que conferi. Nas dez conversas, o mesmo braço ficou em **0,3998** multi e **0,6228** single. Uma seleção pela maior marca da conv00 seria frágil.

O melhor equilíbrio observado na conv00 foi `soft-only`: **0,4690** multi, **0,6533** single, **6,20 chamadas/pergunta** e **7,30 s/pergunta** no registro daquela rodada. Nas dez conversas, `soft-only` teve o maior multi-hop entre os agregados completos examinados: **0,4242** em 282 perguntas, com **0,6603** single-hop em 841. O braço `evidence` teve **0,4136** multi e **0,6607** single: delta pareado `soft-only` +0,0107, 54 ganhos/45 perdas/183 empates. No controle mais rigoroso, `soft-v2/common` versus `evidence/common` deu apenas +0,0029 multi e −0,0099 single. Assim, a tendência positiva de `soft-only` é interessante, mas o tamanho do efeito não está estabelecido. As rodadas antigas não compartilham todas a memória, cache, código e seleção de contexto da reduzida.

O resultado `locomo-todas` de 0,4981 multi cobre **três** conversas e não compete como melhor resultado nas dez. A versão reduzida de plan-repair teve **0,3981** multi e **0,6377** single na conv00, com **7,21 chamadas** e **16,94 s**; a primeira versão, 0,3963/0,6143, 15,2 chamadas e 43,7 s. A nova leitura citada caiu para 0,2439/0,5778 na recuperação congelada. O leitor `focused` da ablação de leitura, que reduziu o contexto para trechos, perdeu −0,0711 multi nas dez conversas. `proof_reader` piorou nas duas recuperações controladas. Os resultados favorecem manter o leitor comum e as cinco passagens integrais.

## Mecanismo que merece teste

Usar o grafo para **propor novas passagens**, mas decidir a troca por cobertura de turnos de diálogo verificáveis, não por um rótulo de “prova completa”. O `soft-only` localiza fontes relevantes sem exigir uma prova rígida e mantém o leitor comum. A diferença entre F1 0,4242 e a meta 0,45 nas dez conversas é **0,0258**, ou aproximadamente **7,26 pontos somados de F1 em 282 perguntas**. Esta é a lacuna a testar; não se deve afirmar que uma troca de passagem já a resolve.

Há suporte parcial para essa direção no experimento `gap-01`, com a mesma recuperação congelada por braço e o mesmo leitor comum: trocar **uma** passagem de cauda por evidência lexical dirigida elevou multi de 0,4056 para **0,4136** (delta +0,0081; 24 ganhos/21 perdas). A variante com verificação LLM chegou a **0,4148** (+0,0093; 21 ganhos/8 perdas), mas gastou **357 chamadas adicionais** no experimento e seu intervalo por conversa ainda inclui zero. Trocar duas passagens piorou para 0,3993. Na conv00 do `gap-01`, uma troca lexical fez 3 ganhos e nenhuma perda, mas não passou de 0,4356 naquele baseline. O gate do experimento usou `tipo == multi-hop` do benchmark; isso não está disponível numa pergunta real e deve ser removido antes de qualquer promoção.

### Proposta precisa, sem LLM adicional

Indexar offline as falas de cada bloco com `pid`, sessão, turno, speaker e texto. Legendas de imagem ficam em campo distinto e não são prova de autoria por si só. Na consulta, usar o top 5 de `soft-only`, mais candidatos da busca híbrida já disponível. Para cada exigência da pergunta ou condição pendente do plano, ranquear falas por ator, verbo/relação, objeto e tempo. Uma fala candidata precisa acrescentar uma exigência ou membro **novo** em relação às falas das cinco passagens atuais. Manter as quatro primeiras e considerar, no máximo, substituir a quinta. Recusar a troca se a quinta tiver a única fala encontrada para outra exigência. Retornar as cinco passagens completas ao leitor comum, na ordem determinada antes de olhar a resposta. Restringir inicialmente a perguntas enumerativas reconhecidas por sintaxe/estrutura do plano; deixar perguntas de contagem e datas na rota atual.

Esboço:

```python
base = soft_only_retrieve(question, top_k=5)
if not answer_blind_enumerative_gate(question, plan):
    return base
facets = question_facets(question, plan.conditions)  # sujeito, operação, qualificadores
present = match_dialogue_turns(facets, base)          # com speaker e sessão
candidate = best_new_turn(facets, present, candidate_pool, exclude=base)
if candidate and candidate.novelty > fixed_threshold \
        and not sole_support_for_other_facet(base[4], present):
    return base[:4] + [candidate.pid]
return base
```

O novo gate não usa categoria `multi-hop`, passagens ou respostas ouro. O limiar deve ser fixado antes de avaliar as perguntas-alvo; se calibrado com rótulos do LoCoMo, separar conversas de treino e teste. A experiência `gap-01` mostra que agressividade de duas trocas pode destruir evidência; por isso o limite de uma passagem não é arbitrário. `qa60` (violino), `qa61` (Matt Patterson) e `qa24` (pottery) são casos diagnósticos de evidência ausente; `qa48` mostra que mesmo cinco passagens completas podem exigir outra forma de resposta, fora do escopo da primeira alteração.

### Segunda rota, somente depois

Nas perguntas com evidência suficiente, preservar a resposta do leitor comum e testar correções **determinísticas e restritas ao operador** em replay, em vez de um novo prompt que reescreve tudo. Exemplo de alvo: `qa48`, que pede *types of pottery* e recebe descrições de objetos. Qualquer regra que reduza `bowl with ...` a `bowl` deve ser testada em **todas** as perguntas do mesmo padrão, com exemplos negativos onde o complemento distingue o tipo; caso contrário, seria ajuste à resposta ouro. Não aplicar essa regra a nomes de livros, locais ou pessoas. Casos como `qa47` requerem vínculo semântico entre falas e provavelmente uma hipótese separada; não afirmo que uma regra lexical os resolverá. Uma nova chamada LLM para revisar respostas fica fora da primeira ablação, pois a guarda numérica e o leitor citado já produziram regressões.

## Ensaio refutável e orçamento

Primeira etapa: reproduzir `soft-only` contra `evidence/common` **na memória congelada da conv00**, mesmo Qwen na GPU 1, mesmo leitor comum, sem `proof_reader` nem `plan_repair`. Registrar hashes de memória, pids, textos, chamadas, tokens e latência. A comparação histórica 0,4690 versus 0,3981 não é pareada e não basta. Se a vantagem não aparecer na reprodução, usar `evidence/common` como base; não ajustar a hipótese ao melhor run antigo.

Segunda etapa: congelar a recuperação dessa base e comparar top 5 original com a troca única, lendo ambos com o mesmo prompt comum. Executar todas as 102 perguntas; para a métrica principal, 32 multi-hop. `single-hop` é proteção. Não usar rótulo de tipo no gate. Não rodar as dez conversas antes de observar direção positiva e auditoria por pergunta na conv00. Se a troca só recupere mais apoios anotados mas F1 não subir, ela falhou como solução de resposta.

Orçamento: nenhuma chamada LLM adicional por pergunta na variante de troca; máximo cinco passagens integrais. A indexação por turno é feita uma vez por conversa. A média de chamadas deve ficar na da base reproduzida, idealmente ≤6,5 e certamente abaixo das 7,21 da reduzida. Comparar latência de recuperação e leitura separadamente; evitar inferir custo pela rodada histórica aquecida. Uma eventual segunda rota de correção de resposta só será considerada com gate verificável e orçamento novo explícito.

Critério para continuar: ganho pareado positivo multi na conv00 sem regressão single superior a 0,01, sem aumento de respostas amplas sem apoio e com explicação por pergunta dos ganhos/perdas. Mesmo que passe, não declarar 0,45 validado em dez conversas. Ao congelar código e limiar, testar conversas adicionais por bloco e calcular intervalo por conversa. Os runs anteriores das dez conversas já serviram ao desenvolvimento; não são holdout novo.
