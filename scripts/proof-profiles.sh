#!/usr/bin/env bash
# Perfis do controlador de prova, compartilhados pelos scripts Azure, Qwen e
# OpenAI. `proof_profile_flags PERFIL` preenche o array PROOF_FLAGS.
#
#   proof            desenho v3 registrado
#   proof-no-verify  ablação v3: prova utilizável muda o contexto sem verificação
#   proof-partial    ablação v3: sondas/lacuna na última vaga
#   proof-1cycle     ablação v3: um único plano
#   proof-v4         desenho v4 completo: variáveis tipadas, prova de conjunto
#                    item a item, entrega mista (trocas v3 até k_W e falas de
#                    origem para o resto) e premissas abdutivas
#                    (docs/witnessrag-v4.md)
#   proof-v4-memory  método B (docs/memoria-v2.md): proof-v4 + memória de
#                    proposições (--ie-style memory) e leitor com fatos +
#                    resumos (--fact-delivery facts+summary)
#   witnessrag       o método completo (igual a proof-v4-memory)
#   abl-no-plan | abl-no-proof | abl-no-verify | abl-no-temporal-score
#                    ablações de um componente (docs/ablacao.md)
#   witnessrag-robust  WitnessRAG + plano robusto (relações alternativas,
#                    leituras alternativas) + fatos pela pergunta + reranker +
#                    datas bitemporais (docs/plano-robusto.md)
#   wr-no-rerank     witnessrag-robust sem o reranker (sem custo de CPU)
#   wr-plan | wr-alt | wr-readings | wr-fill | wr-rerank | wr-time
#                    uma parte do witnessrag-robust de cada vez
#   v4-typed         só tipos + conjuntos item a item, entrega por trechos (pages)
#   v4-excerpts      só a entrega como falas de origem, sem trocar trechos
#   v4-mixed         só a entrega mista
#   v4-abductive     só as premissas abdutivas
#   v4-no-types      v4 sem os tipos (isola o átomo unário)
# A memória (OpenIE, clusters, famílias de relação) é a mesma em todos.
proof_profile_flags() {
  local profile=$1
  PROOF_FLAGS=(--binding-aware-grounding --vocab-compile --hybrid-fallback
    --dialogue-ie --gap-context-rescue --proof-controller)
  case "$profile" in
    proof) ;;
    proof-no-verify) PROOF_FLAGS+=(--no-proof-verify) ;;
    proof-partial) PROOF_FLAGS+=(--partial-evidence) ;;
    proof-1cycle) PROOF_FLAGS+=(--proof-cycles 1) ;;
    proof-v4) PROOF_FLAGS+=(--typed-variables --item-set-proofs --witness-delivery mixed
      --abductive-premises) ;;
    proof-v4-memory) PROOF_FLAGS+=(--typed-variables --item-set-proofs --witness-delivery mixed
      --abductive-premises --ie-style memory --fact-delivery facts+summary) ;;
    witnessrag|abl-no-plan|abl-no-proof|abl-no-verify|abl-no-temporal-score)
      # WitnessRAG completo (= proof-v4-memory) e as ablações de um componente
      # (docs/ablacao.md). Todas usam a mesma memória e o mesmo leitor.
      PROOF_FLAGS+=(--typed-variables --item-set-proofs --witness-delivery mixed
        --abductive-premises --ie-style memory --fact-delivery facts+summary)
      # "if" and not "[[ ]] &&": under "set -e" a false test as the last
      # command made the function fail and the caller exit silently.
      if [[ "$profile" == abl-* ]]; then PROOF_FLAGS+=(--ablation "${profile#abl-}"); fi
      ;;
    witnessrag-robust|witnessrag-multiplan|robust-no-plan|wr-no-rerank|wr-plan|wr-alt|wr-readings|wr-fill|wr-rerank|wr-time)
      # Plano robusto e seleção de fatos (docs/plano-robusto.md), sobre o
      # WitnessRAG completo. witnessrag-robust liga tudo; wr-* liga uma parte
      # (wr-plan = relações alternativas + leituras alternativas).
      PROOF_FLAGS+=(--typed-variables --item-set-proofs --witness-delivery mixed
        --abductive-premises --ie-style memory --fact-delivery facts+summary)
      case "$profile" in
        witnessrag-multiplan) PROOF_FLAGS+=(--relation-alternatives --plan-readings 2
          --fact-fill question --fact-rerank --fact-time both --multiplan-portfolio
          --portfolio-max-plans 3) ;;
        witnessrag-robust) PROOF_FLAGS+=(--relation-alternatives --plan-readings 2
          --fact-fill question --fact-rerank --fact-time both) ;;
        robust-no-plan) PROOF_FLAGS+=(--ablation no-plan --fact-fill question
          --fact-rerank --fact-time both) ;;
        wr-no-rerank) PROOF_FLAGS+=(--relation-alternatives --plan-readings 2
          --fact-fill question --fact-time both) ;;
        wr-plan) PROOF_FLAGS+=(--relation-alternatives --plan-readings 2) ;;
        wr-alt) PROOF_FLAGS+=(--relation-alternatives) ;;
        wr-readings) PROOF_FLAGS+=(--plan-readings 2) ;;
        wr-fill) PROOF_FLAGS+=(--fact-fill question) ;;
        wr-rerank) PROOF_FLAGS+=(--fact-rerank) ;;
        wr-time) PROOF_FLAGS+=(--fact-time both) ;;
      esac
      ;;
    v4-typed) PROOF_FLAGS+=(--typed-variables --item-set-proofs) ;;
    v4-excerpts) PROOF_FLAGS+=(--witness-delivery excerpts) ;;
    v4-abductive) PROOF_FLAGS+=(--abductive-premises) ;;
    v4-no-types) PROOF_FLAGS+=(--item-set-proofs --witness-delivery mixed
      --abductive-premises) ;;
    v4-mixed) PROOF_FLAGS+=(--witness-delivery mixed) ;;
    *)
      echo "PROFILE desconhecido: $profile (veja scripts/proof-profiles.sh)" >&2
      return 2
      ;;
  esac
}
