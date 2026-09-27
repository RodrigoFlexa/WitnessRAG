# Cascata: o plano só quando a pergunta precisa (27/09/2026)

Opção `--plan-router llm` (desligada por padrão); perfil `witnessrag-cascade`
= `witnessrag-robust` + roteador.

## Ideia

Na conv05 (gpt-4o-mini), o robusto sem plano fez 74,5 em single-hop contra
73,4 com plano; o plano rendeu em multi-hop (+4,3) e temporal (+4,2), ao custo
de ~7,5 mil tokens por pergunta. A cascata gasta o plano só onde ele rende:

1. **Roteador.** Uma chamada curta (até 16 tokens de saída) lê **só o texto da
   pergunta** e responde `{"route": "PLAN"}` ou `{"route": "DIRECT"}`.
2. **DIRECT:** exatamente o caminho sem plano: os 40 fatos mais relevantes para
   a pergunta (reranker, datas ancoradas, resumos). Nenhuma chamada de plano ou
   verificação.
3. **PLAN:** o controlador de prova completo do `witnessrag-robust`.

O roteador escolhe PLAN quando a pergunta pede tempo (quando, quanto tempo,
ordem, primeira/última vez), vários itens ou contagem espalhados pelas
conversas, ou ligar fatos (a resposta é achada através de outro fato). Os
critérios descrevem o que a busca precisa fazer; o prompt não cita categorias
do benchmark e não vê a memória, a resposta nem anotações. Qualquer falha do
roteador (chamada bloqueada, saída ilegível) vai para PLAN: o roteador só pode
economizar, nunca pular o plano por acidente.

Na simulação com regras fixas de palavras na conv05, a cascata ficou em 62,88
contra 62,96 do robusto, com 53% menos tokens. O roteador por LLM substitui
aquelas regras.

## Rodar no servidor (Qwen)

```bash
bash scripts/serve-qwen-vllm.sh                 # outro terminal, se não estiver de pé
bash scripts/run-cascade-qwen.sh                # 10 conversas, 3 rodadas + relatórios
LOCOMO_CONVERSATION=5 bash scripts/run-cascade-qwen.sh runs/qwen-cascade-conv05   # teste
```

Roda `witnessrag-robust`, `robust-no-plan` e `witnessrag-cascade` com a mesma
memória, leitor e cache; a cascata por último, porque seus planos e leituras
saem do cache das outras duas e só o roteador é chamada nova. No fim:

- `compare.md`: F1 pareado, tokens e planos por pergunta das três;
- `cascade.md`: para as perguntas mandadas a PLAN e a DIRECT, o F1 das três
  rodadas nesses mesmos subconjuntos. O roteador é bom se, nas PLAN, o robusto
  supera o sem plano, e nas DIRECT, o sem plano empata ou supera.

## Código

| peça | onde |
|---|---|
| roteador (prompt, parser, padrão PLAN) | `wrag/witness/router.py` |
| desvio para o caminho sem plano | `WitnessRAGRetriever._retrieve_proof` (`plan_router`) |
| opção e perfil | `wrag/pilot.py` (`--plan-router`), `scripts/proof-profiles.sh` |
| rodadas e relatórios | `scripts/run-cascade-qwen.sh`, `scripts/cascade-report.py` |
| testes | `tests/test_plan_router.py` |
