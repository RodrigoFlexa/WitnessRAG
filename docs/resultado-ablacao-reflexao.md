# Auditoria da ablação de reflexão — 28/09/2026

As quatro runs de `runs/cascade-reflection-ablation` estão completas: 1.540 perguntas nas dez conversas, sendo 841 single-hop, 282 multi-hop, 321 temporais e 96 open-domain. Recalculei os resultados pelas respostas salvas, conferi os contextos efetivamente entregues e reproduzi uma correção de espaços sem chamar modelos. O código de produção, os prompts e as runs originais foram preservados.

A minha proposta de reader reflexivo, nesta implementação, piorou o resultado. Há uma perda de formato bastante importante e também regressões reais de tipo, unidade e seleção de evidência. As memórias inferenciais têm sinais promissores, mas esta execução não isolou seu efeito causal. Não há evidência aqui para reformar o cascade ou ampliar os planos.

## Resultados reproduzidos

| Variante | F1 geral | BLEU-1 local | single-hop | multi-hop | temporal | open-domain |
|---|---:|---:|---:|---:|---:|---:|
| cascade | 53,69 | 46,99 | 58,94 | 47,78 | 54,34 | 22,82 |
| summary-reflection | 54,43 | 47,88 | 59,98 | 46,56 | 55,59 | 25,03 |
| reader-reflection | 51,41 | 43,50 | 54,45 | 48,11 | 55,21 | 21,80 |
| both | 51,39 | 43,53 | 54,52 | 46,99 | 55,72 | 22,38 |

Contra cascade, summary-reflection ganha 0,74 ponto de F1, com IC95% [-0,30; +1,85]. O ganho observado de open-domain é +2,21, com IC95% [-1,24; +6,15]. São diferenças exploratórias, pequenas e sem intervalo separado de zero. Os intervalos usam bootstrap pareado de conversas, 4.000 amostras, semente 0; há apenas dez conversas.

O reader perde 2,28 pontos de F1 geral contra cascade, IC95% [-3,26; -0,89]. Sua adição aos resumos inferenciais custa 3,04 pontos contra summary-reflection, IC95% [-4,40; -1,48]. Os ganhos de multi-hop (+0,33 e +0,43 nos respectivos pares) e temporal não demonstram uma melhoria consistente: seus intervalos também cruzam zero.

BLEU-1 aqui é a implementação local com normalização e stemming. Sua equivalência com o protocolo da tabela de Zero-Mem não foi estabelecida. Não usar esses números diretamente para declarar vitória ou derrota contra aquela tabela.

## O que não ficou controlado

As quatro células têm as mesmas perguntas e corpus bruto. Entretanto:

| Par | Perguntas com fatos diferentes | Seleções de resumos diferentes | Rotas diferentes | Texto dos resumos diferente |
|---|---:|---:|---:|---:|
| cascade → reader-reflection | 1 | 0 | 0 | 0 |
| summary-reflection → both | 0 | 0 | 0 | 0 |
| cascade → summary-reflection | 1.540 | 973 | 4 | 1.540 |
| reader-reflection → both | 1.540 | 973 | 4 | 1.540 |

O texto dos resumos precisa mudar ao adicionar as inferências. Já os fatos, a seleção de resumos e as rotas deveriam permanecer iguais para isolar esse fator. O controle existente registrou **5.035 divergências de campos**, não 5.035 perguntas, e recusou gerar `reflection-ablation.md`. O `compare.md` é uma comparação descritiva pareada; ele não substitui o controle da ablação.

O manifesto divide as variantes em duas portas: cascade/reader em 8095, summary/both em 8096. Todas as dez conversas de cada grupo têm a mesma identidade de provedor dentro do grupo, mas identidades diferentes entre os grupos:

- 8095: `16053480c5fdbe370ddbac659574f8c7a1d930627783825b77e40dbb4ee329c4`;
- 8096: `ff608582c4e217798880939845dd07d8a99fd846e1a0ef7ee16c837ab70c23c1`.

`OpenAICompat.cache_identity()` inclui a URL do servidor no modo comum. A chave da extração em `wrag/ie.py` inclui essa identidade. Usar a mesma pasta de cache, portanto, não garantiu a mesma memória extraída. A hipótese de reconstrução independente é consistente com os manifestos e com a mudança observada dos fatos. Os logs não demonstram que os dois servidores tenham modelos de qualidade diferente, nem que a porta em si altere a capacidade do modelo.

O único caso divergente no primeiro par é `locomo:conv-47:qa45`. Excluí-lo deixa 1.539 contextos idênticos e a queda do reader praticamente igual: -2,26 pontos, IC95% [-3,24; -0,88]. Logo a conclusão sobre o reader não depende dessa exceção. O segundo par está completamente controlado para os quatro campos conferidos.

## Perda de formato: listas sem espaços

O reader reflexivo produziu 142 respostas com vírgula imediatamente seguida de letra, contra 28 no cascade. Exemplos:

| Pergunta / ID | Cascade | Reader reflexivo | Diagnóstico |
|---|---|---|---|
| Nomes das cobras de Jolene — conv-48:qa16 | `Susie, Seraphim` | `Susie,Seraphim` | Mesmo conteúdo; F1 passa de 100% para 0% |
| Jogos do torneio de John — conv-47:qa141 | `Fortnite, Apex Legends, Overwatch` | `Fortnite,Apex Legends,Overwatch` | Mesmo conteúdo; F1 passa de 100% para 0% |

O [avaliador oficial fixado no projeto](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py) remove vírgulas, sem inserir espaços. Nas categorias single-hop, temporal e open-domain, os dois nomes podem se fundir num único token. Em multi-hop, o F1 primeiro separa itens por vírgulas; por isso essa perda não aparece da mesma forma no F1 dessa categoria. O BLEU-1 local avalia a resposta inteira e sofre também em multi-hop.

Fiz um replay somente de apresentação: substituir vírgula seguida de letra por vírgula e espaço. A regra é igual para as quatro variantes, não acessa categoria, pergunta ou ouro e preserva separadores numéricos como `1,000`. O avaliador permanece intacto.

| Variante | F1 original | F1 com espaços | BLEU-1 com espaços | single-hop com espaços |
|---|---:|---:|---:|---:|
| cascade | 53,69 | 53,99 | 47,57 | 59,48 |
| summary-reflection | 54,43 | 54,52 | 48,29 | 60,14 |
| reader-reflection | 51,41 | 53,07 | 46,20 | 57,40 |
| both | 51,39 | 52,90 | 46,04 | 57,22 |

São resultados contrafactuais de pós-processamento das respostas existentes, **não uma nova execução nem demonstração de um reader melhor**. A regra aumenta F1 em 56 respostas do reader e diminui em duas; em both, aumenta em 49 e diminui em quatro. Algumas respostas ouro também contêm separadores sem espaço. A correção não garante melhora em toda pergunta.

Aplicando o mesmo tratamento às duas pontas, a perda geral reader/cascade cai para -0,92, IC95% [-1,90; +0,49], mas permanece em single-hop: -2,08, IC95% [-3,36; -0,62]. No par both/summary, a perda geral permanece -1,62, IC95% [-2,57; -0,22], e em single-hop -2,92, IC95% [-3,64; -1,95]. Não é correto concluir que tudo se resolve com espaços.

Um detalhe revelador: o BLEU-1 multi-hop do reader inicialmente perde 4,63 pontos contra cascade; após formatar ambas as variantes, passa a +0,50. O pequeno ganho original de F1 multi-hop e a grande perda de BLEU eram, em parte, um efeito dos diferentes tratamentos de listas.

## Regressões que continuam existindo

| ID | O que a pergunta solicita | Cascade | Reader reflexivo |
|---|---|---|---|
| conv-43:qa81 | Duração do surfe | `five years` | `5` |
| conv-43:qa100 | Duração da convivência com os colegas do basquete | `four years` | `4` |
| conv-44:qa76 | Duração do passeio dos cães | `about an hour` | `1` |
| conv-43:qa148 | Duração do piano até dezembro | `about four months` | `since August 2023` |
| conv-48:qa98 | Quando recebeu o primeiro console | `when she was 10` | `2010` |
| conv-41:qa41 | Praia ou montanhas | `beach` | `likely yes` |
| conv-48:qa40 | Praia ou montanhas | `the beach` | `yes` |
| conv-42:qa93 | Jogo do segundo torneio vencido | `Street Fighter` | `Counter-Strike: Global Offensive` |

Há 30 perguntas contendo “how long”. O cascade não responde a nenhuma com apenas um número; reader e both fazem isso em dez perguntas cada. Duração não é contagem, início não é duração e idade não autoriza inventar um ano de calendário. A perda das unidades aparece na resposta antes da avaliação; não é causada pelo canonicalizador de contagens, que atua somente em “how many”.

O caso do segundo torneio mostra também uma seleção inadequada de evidência: o contexto contém Street Fighter e CS:GO, mas a leitura deve associar o jogo ao evento pedido, não à preferência habitual do participante. No caso da praia, `answer_kind=boolean` recebe `schema_valid=true`: validar um enum não verifica a adequação semântica da resposta à pergunta.

As respostas `insufficient information` diminuem de 99 no cascade para 70 no reader; não há respostas vazias. As respostas do reader têm menos palavras por separação em espaços (4,12 versus 4,84), embora essa contagem também seja afetada pela ausência de espaços nas listas. Assim, os logs não sustentam a explicação de que o reader piorou porque ficou mais abstencionista ou mais prolixo.

Há 109 tipos fora do enum no reader e 56 em both. Nem todo tipo fora do enum é uma resposta semanticamente errada: `duration` pode até ser uma classificação razoável, mas não é aceito pelo contrato implementado. O problema principal é que o campo não governa uma verificação efetiva de tipo e unidade.

Esta versão pede reflexão interna na mesma chamada de resposta; não registra uma revisão independente nem comprova que uma checagem ocorreu. No artigo, descrevê-la como leitura com instrução conjunta de inferência e reflexão, sem equipará-la automaticamente a um verificador de respostas.

## O que deu certo e por que open-domain ainda está baixo

- **Pomodoro**, conv-43:qa27: somente both retorna `Pomodoro Technique`. As outras versões repetem “25 minutes on, then 5 minutes off” ou uma descrição do estudo. A premissa 25/5 já aparece nos fatos; as memórias inferenciais selecionadas não nomeiam Pomodoro. Esse caso é compatível com uma interação útil, mas não prova que uma memória de reconhecimento foi criada.
- **Preferência entre veículos**, conv-50:qa39: cascade abstém; summary, reader e both respondem `Dodge Charger`. O contexto contém o interesse por carros clássicos. É um exemplo do tipo de inferência que a proposta deveria favorecer.
- **Toronto → país**, conv-47:qa30: todas as variantes retornam `Toronto`, embora a pergunta solicite um país. A premissa relevante está no contexto; falha a passagem ao nível de descrição pedido.
- **Jogo de cartas**, conv-47:qa16: todas as versões repetem a descrição em vez de reconhecer UNO. O contexto inclui cartas coloridas/numeradas e a regra de igualar cor ou número. Há premissas úteis, mas o reader não conclui com o nome do conceito.

As inferências dos resumos chegam a todas as perguntas: em média 18,09 memórias por consulta. Foram observados 129 chunks selecionados distintos, com 395 memórias aceitas e 121 rejeitadas pelo validador. Esses são totais dos chunks efetivamente vistos nas respostas, não necessariamente de todo o cache.

Nos casos examinados, várias memórias são generalizações amplas sobre motivação, apoio, criatividade e hobbies. Elas podem ajudar escolhas baseadas em preferências, mas não substituem reconhecimento de conceitos ou mudança de nível geográfico. Há ainda duas limitações arquiteturais: gerar no máximo quatro inferências para um chunk grande pode omitir uma ponte útil; selecionar seis chunks traz todas as suas inferências, inclusive as que não ajudam a pergunta.

Além disso, cada memória adiciona citações originais. Um eventual ganho pode vir da inferência, de fatos que a extração perdeu, ou da repetição de evidência. A média de tokens do reader sobe de 1.913 para 3.576 (+87%) com summary-reflection; o total por consulta sobe de 6.736 para 8.358 (+24%), sem incluir construção da memória. Não atribuir exclusivamente à abstração semântica um ganho obtido com mais evidência e contexto.

## Proposta de próxima revisão

1. **Manter cascade e planos robustos.** Nesta rodada, a alteração que degradou o resultado está no reader e na entrega. Não promover reader-reflection ou both como padrão. Manter summary-reflection como experimento promissor, sem afirmar que seu ganho foi isolado.
2. **Separar apresentação de raciocínio.** Corrigir a serialização de listas de forma comum aos leitores. O ideal é serializar itens delimitados com `, `, preservando separadores de números e identificadores; a regra de replay usada aqui é diagnóstica e não substitui um contrato de saída completo. Não mudar o avaliador ou ajustar por categoria do benchmark.
3. **Revisar minimamente o reader.** Preservar o prompt de evidências que já funcionava; testar uma extensão curta que exija a menor resposta completa. Contagem pode ser número; duração precisa de unidade; alternativas precisam da opção; país precisa do país. Se o fato explícito já responde, não reinterpretá-lo. Se não responde, aplicar a ponte necessária e conferir todos os qualificadores. Usar exemplos sintéticos fora do LoCoMo, sem copiar Pomodoro/Toronto/UNO do benchmark.
4. **Isolar o efeito do schema.** O atual enum mistura duração/data em `time` e deixa `value` muito amplo; o modelo também ignora parte do contrato. Comparar a instrução reflexiva com o JSON simples original antes de obrigar novos campos. Uma saída estruturada só é útil se a interpretação e o renderizador preservarem unidades, listas e opções. Não exigir uma segunda chamada de reflexão por padrão antes de demonstrar benefício.
5. **Selecionar as inferências individualmente.** Reaproveitar a recuperação e o reranker para escolher poucas memórias cuja conclusão/ponte atende à pergunta, com suas premissas. Não entregar automaticamente todas as interpretações de cada chunk. Expandir a cobertura de reconhecimento/classificação só depois de um teste controlado; não começar acrescentando regras sobre os casos ouro.
6. **Congelar as entradas experimentais.** Salvar a extração/grafo e um snapshot de recuperação por pergunta; aplicar os leitores ao mesmo contexto, sem refazer planos. O projeto já tem mecanismos de memória congelada e replay em `wrag/eval/controlled.py`, mas o launcher desta ablação precisa ser integrado a eles; exportar uma variável não garante, por si só, o controle completo. O namespace explícito `WRAG_EXPERIMENT_CACHE_ID` elimina a dependência da porta, mas não substitui snapshots e verificações.
7. **Medir o valor das inferências além das citações.** Com o mesmo contexto congelado, comparar resumo atual, resumo acrescido somente das citações e resumo acrescido de citações + inferências. Parear os dois últimos no orçamento de tokens, sem truncar evidência decisiva. Isso separa parte do ganho de contexto do ganho de interpretação.

Antes de repetir tudo, usar 30–40 perguntas por variante numa amostra de desenvolvimento, distribuída por operações: lista, duração/idade, escolha, classificação/reconhecimento, evidência explícita e ausência de suporte. Reutilizar os contextos salvos: apenas chamadas de QA, sem reconstrução de memória ou planos. Os casos desta auditoria servem para diagnóstico; confirmar a revisão em perguntas não usadas no ajuste e congelar o prompt antes da rodada completa. Como estas conversas já orientaram mudanças, registrar seu papel de desenvolvimento no artigo.

## Artefatos e reprodução

```text
python scripts/audit-reflection-runs.py runs/cascade-reflection-ablation --output runs/reflection-audit
```

O comando é inteiramente local, não chama API ou servidor. Saídas:

- `runs/reflection-audit/audit.json`: métricas originais e replay, intervalos, contrastes do reader, checagem de contextos, tipos/unidades, identidades dos manifestos e SHA256 dos 40 arquivos de predições;
- `runs/reflection-audit/cases.json`: 49 casos selecionados para inspeção, respostas/contextos das quatro células e caminho/linha do registro original;
- `runs/reflection-audit/questions.tsv`: todas as 1.540 perguntas, respostas e F1 original/replay;
- `scripts/audit-reflection-runs.py`: análise reproduzível.

Foram gastos zero tokens de API nesta auditoria. As propostas acima ainda não foram implementadas ou avaliadas como novas variantes.
