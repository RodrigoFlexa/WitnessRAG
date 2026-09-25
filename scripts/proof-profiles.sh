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
