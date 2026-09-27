# Ablação do WitnessRAG

Cada variante tira um componente do método completo. Memória (extração em
proposições), leitor e orçamento de leitura são os mesmos em todas.

| variante | perfil | opção | o que tira |
|---|---|---|---|
| completo | `witnessrag` | | nada |
| sem plano | `abl-no-plan` | `--ablation no-plan` | Planejador, Executor e Refletor: nenhuma chamada de planejamento ou verificação; os fatos são escolhidos só pela similaridade com a pergunta |
| sem prova | `abl-no-proof` | `--ablation no-proof` | a execução do plano no grafo (junção, prova, verificação, replanejamento); os átomos do plano só orientam a similaridade |
| sem verificação | `abl-no-verify` | `--ablation no-verify` | a confirmação da prova nas falas de origem |
| sem nota temporal | `abl-no-temporal-score` | `--ablation no-temporal-score` | pesos de tempo e importância da nota (fica só similaridade); as datas continuam na memória e no leitor |

`witnessrag` = proof-v4 + `--ie-style memory --fact-delivery facts+summary`
(o antigo `proof-v4-memory`).

## Como rodar

Windows, gpt-4o-mini (chave no `.env`):

    powershell -ExecutionPolicy Bypass -File scripts\run-ablation.ps1

Linux / servidor, gpt-4o-mini:

    LLM=openai bash scripts/run-ablation.sh

Servidor, Qwen2.5-14B (com `scripts/serve-qwen-vllm.sh` de pé):

    LLM=qwen EMBED_DEVICE=cuda GPU=1 bash scripts/run-ablation.sh

Padrão: trechos de 2048 tokens e k = 5 (protocolo do GAM e do Zero-Mem); outro
orçamento com `-ChunkTokens`/`-TopK` ou `CHUNK_TOKENS`/`TOP_K`. O completo roda primeiro; depois as ablações, cada uma em
`runs/ablation-<llm>-c<chunk>-k<k>/<variante>`. Repetir o comando retoma.
No fim, `scripts/ablation-report.py` escreve `ablation.md` e `ablation.json`
na mesma pasta: F1 oficial por categoria, Δ para o completo com IC 95% por
bootstrap de conversas, e tokens por pergunta (leitor e total, sem a
construção da memória).
