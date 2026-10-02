# Requisitos de prova e reflexão por pertencimento (v6)

## Diagnóstico que motivou o desenho

Na ablação v2 (10 conversas, Qwen2.5-14B, 40 fatos), reflexão e referência
temporal não mexeram no F1. A decomposição da perda mostrou:

| origem da perda | pts de F1 |
|---|---|
| evidência ausente ou incompleta | 12,6 |
| resposta errada com evidência presente | 9,2 |
| resposta parcial | 8,4 |
| tempo (forma diferente do ouro: artefato da métrica) | 2,9 |
| tempo errado / outros | 4,4 |
| abstenção / contagem | 3,3 |

- O sinal "prova parcial" do executor não separava perguntas com e sem
  evidência (79,7% vs 81,4%): os programas locais nascem da memória.
- A resposta amarrada pela testemunha não serve de segunda opinião (certa em
  17 de 409 erros do leitor).
- Reflexão com o mesmo modelo e o mesmo contexto não traz informação nova.

Na conversa de desenvolvimento (conv00), 16 das 25 perguntas com evidência
incompleta pedem um CONJUNTO; um membro no contexto basta para o requisito
parecer atendido. Os membros que faltam EXISTEM na memória (só 3 de 34 falas
faltantes não viraram fato), mas o cross-encoder não reconhece pertencimento
("family" ~ "kids", "activities" ~ "swimming").

## Desenho

**Planejador** (`plan.requirements`, só a pergunta e a data presente, em
cache): 1 a 4 requisitos, cada um um fato a encontrar; uma cadeia vira um
requisito por elo; cada requisito declara `all` (precisa de TODAS as
instâncias: lista, contagem, padrão) e a janela de tempo copiada da pergunta.

**Executor**: suporte de um requisito = relevância do cross-encoder entre ele e
os fatos entregues, respeitando a janela (tempo declarado pelo fato; data de
sessão nunca contradiz).

**Reflexão** (`reflect.members`): para um requisito `all`, os fatos das
pessoas nomeadas, de falas ainda não entregues (um por fala), são mostrados a
um juiz de pertencimento (LLM), que diz quais são instâncias. Até 8 membros
entram no lugar do preenchimento menos relevante; fatos de prova nunca saem e o
orçamento não muda. Para requisitos de um fato, a busca na memória inteira
acrescenta o melhor fato quando nenhum fato entregue o sustenta.

O leitor é o da v2, com o mesmo prompt; o texto do juiz nunca entra no leitor,
só os fatos e suas falas originais.

## Protocolo

Decisões só na conv00 (desenvolvimento). Medições na conv00:

| etapa | resultado |
|---|---|
| suporte por requisito (melhor fato) | não separa evidência incompleta; 0 falas-ouro alcançadas |
| enumeração pelo cross-encoder | 1 de 12 falas faltantes |
| juiz de pertencimento (LLM) | 3–4 de 21 falas faltantes (a anotação do LoCoMo é incompleta: precisão medida contra ela subestima) |
| **completo, conv00, 40 fatos** | **58,47 vs 57,38 (v2)**; multi-hop 54,62 vs 50,38; demais categorias iguais |

Teste: conv07 (ablação completa, com trava) e conv04 (só o completo), nenhuma
usada nas decisões deste desenho.

## Ablação

| variante | remove |
|---|---|
| `full` | nada |
| `no-witness` | busca planejada por testemunhas (a reflexão continua) |
| `no-time-reference` | janelas dos requisitos; pesos de tempo da gramática |
| `no-time-model` | tempos resolvidos (implica a anterior) |
| `no-reflection` | completar requisitos (os requisitos são medidos; leitor igual) |

```bash
python scripts/run-locomo-ablation-conv.py --conversation 7 --fact-budget 40 \
  --output runs/locomo-v6-conv07 --cache runs/.cache/locomo-ablation-qwen14b \
  --baseline runs/locomo-ablation-qwen14b/facts40 --gpu 0 --port 8096
```

Código: `wrag/witness/requirements.py` (flags `--requirements`,
`--requirement-max-members`); testes em `tests/test_requirements.py`.
