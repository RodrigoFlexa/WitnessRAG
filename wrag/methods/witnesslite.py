"""Scalable Witness context policy without corpus-wide OpenIE."""
from __future__ import annotations

from wrag.methods.base import RetrievalResult, Retriever
from wrag.methods.dense import HybridRetriever
from wrag.witness.context_selection import select_complement


class WitnessLiteRetriever(Retriever):
    """Hybrid top-pool plus the same answer-blind, one-tail context policy.

    This deployment mode is intended for 56k--448k passage indexes and 128k
    isolated contexts, where Qwen OpenIE over every passage is not a meaningful
    cost comparison.  It makes zero retrieval-time LLM calls.
    """
    name = "witnessrag-lite"

    def __init__(self, ctx):
        super().__init__(ctx)
        self.hybrid = HybridRetriever(ctx, ctx.run.witness.hybrid_rrf_k)

    def index(self):
        self.hybrid.index()
        self.indexed = True

    def _retrieve(self, question, k):
        pool_k = max(k, self.ctx.run.witness.candidate_pool_k)
        pids, scores = self.hybrid.search(question.question, pool_k)
        base = pids[:k]
        cfg = self.ctx.run.witness
        selection = select_complement(
            self.corpus, question.question, base, {}, k,
            temporal=cfg.temporal_memory, complementary=cfg.complementary_context,
            candidate_pids=pids[k:])
        final = selection.pids
        score_by_pid = dict(zip(pids, scores))
        return RetrievalResult(
            pids=final, scores=[score_by_pid.get(pid, 0.0) for pid in final],
            diagnostics={"modo": "witnessrag-lite", "pool": pool_k,
                         "chamadas_llm_recuperacao": 0,
                         "selecao_contexto": {
                             "alterou": selection.changed, "adicionada": selection.added,
                             "removida": selection.removed, "motivo": selection.reason,
                             "score": round(selection.score, 4)}})

    def index_report(self):
        return {"modo": "hibrido_mais_selecao_complementar",
                "openie": False, "grafo": False,
                "candidate_pool_k": self.ctx.run.witness.candidate_pool_k}
