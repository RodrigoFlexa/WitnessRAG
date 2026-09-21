# Leitor citado: hipótese refutada

Fonte: `runs/locomo-reader-cited-01/{manifest.json,pairs.jsonl,summary.json}`. Auditoria offline: `python -X utf8 scripts/audit_reader_cited.py`. Nenhuma chamada nova ao modelo.

## Resultado

| Avaliação | Single-hop | Multi-hop |
|---|---:|---:|
| Leitor comum | 0,637681 | 0,398074 |
| Leitor citado final | 0,577784 | 0,243936 |
| Respostas brutas do citado, ignorando citações* | 0,623807 | 0,393204 |

*Contrafactual diagnóstico: só JSON parseável, saídas truncadas continuam sem recuperação. Não valida as respostas, não corrige omissões e pode premiar itens sem apoio. Não é resultado de uma solução alternativa pronta.

As 102 hashes de contexto foram verificadas. O F1 do controle contemporâneo coincide com o histórico em cada uma das 102 perguntas. Não há motivo para atribuir a queda observada a recuperação diferente ou drift de F1 do controle.

Multi-hop: 2 ganhos, 13 perdas, 17 empates. Nove perdas têm rejeição local de citação, uma tem truncamento e três decorrem de resposta gerada pior sem rejeição. Essas categorias indicam o caminho observado, não causalidade exclusiva: retirar a rejeição não necessariamente recupera a resposta do controle. O intervalo pareado multi informado pelo piloto, [−0,2633; −0,0476], está inteiramente abaixo de zero, condicionado à conv00.

## Falhas do projeto proposto

1. **O contrato dependia de aderência não demonstrada do Qwen.** Das 50 chamadas elegíveis, 45 emitiram um único objeto `items`; 14 desses continham vírgulas no campo answer, agrupando possíveis membros. O validador rejeita o objeto inteiro se qualquer citação falha. A promessa de rejeição por membro não se realizou quando o modelo agrupou membros num objeto. Os testes sintéticos cobriam JSON ideal, mas não demonstravam aderência real.
2. **Copiar identificadores longos e texto literal virou requisito de resposta.** Nos JSONs parseáveis há 68 citações exatas no pid declarado, 35 exatas em outra das cinco passagens e 10 que não aparecem literalmente em nenhuma delas. São 113 citações, não 113 perguntas. O problema não é só ausência de evidência. Em qa98, a resposta reproduz a resposta comum, mas o pid aponta para o bloco errado. Em qa78, “Those figurines…” difere do original “These figurines…” e o item é descartado. Ao todo, 22 objetos foram rejeitados por proveniência.
3. **Citações longas aumentaram custo e truncamento.** Três respostas terminaram por comprimento: qa18, qa149 e outra pergunta sem perda de F1. Houve 21 abstenções finais nas 50 chamadas alternativas. Em qa18, a saída repete evidências e mistura Grand Canyon com camping, estourando 900 tokens. Aumentar o teto não corrige relação errada nem justifica o custo.
4. **O próprio conteúdo também piorou.** qa15 passa de lista com F1 0,75 para apenas “painting landscapes and still life” com 0,10; qa52 omite Bailey e passa de nomes separados por vírgula para “Luna and Oliver”; qa3 muda adoption agencies para counseling/mental health antes mesmo do veto local. Exigir evidência não garantiu enumeração, escopo ou completude.
5. **O gate foi amplo demais.** Ativou 26 single-hop e produziu 12 perdas contra 2 ganhos nesse grupo. Usar qualquer plano com `aggregation=set` não foi proteção adequada. Não há autorização científica para consertar isso roteando pela categoria ouro.

## Os casos usados para motivar a hipótese

- qa48 funcionou: bowls/cups, F1 1,0 contra 0,2857. Esse ganho isolado é real e não salva a proposta.
- qa47 gerou “Melanie, friends, family, mentors” dentro de um único item, rejeitado por proveniência. Continua incluindo Melanie; F1 bruto alto não demonstraria ausência de membros sem apoio.
- qa78 continuou omitindo shoes e ainda perdeu figurines no veto textual.
- qa18 truncou; não virou o ganho esperado.

O cenário anteriormente apresentado de atingir 0,4638 pressupunha corrigir qa47/qa48/qa78 sem perdas. Só qa48 melhorou. A projeção aritmética não foi evidência de que a alteração implementada produziria tais respostas.

## Decisão

Retirar H1, manter o leitor comum e não ampliar esta versão. O leitor comum da produção não foi modificado. Retirar o validador recuperaria parte da queda, mas os scores brutos ainda ficam abaixo do controle em ambos os grupos; portanto, não recomendar “desligar o filtro” como solução. Também não recomendar aumentar tokens ou repetir a mesma versão na GPU.

O erro foi escolher uma mudança que adicionou geração de citações, nova serialização, nova forma de resposta, gate e veto obrigatório ao mesmo tempo, tratando um ganho plausível de concisão como justificativa para esse conjunto. A próxima decisão deve partir de ablações menores e evidência já salva; o resultado atual não sustenta promessa de 0,45 nem uma nova rodada cara.
