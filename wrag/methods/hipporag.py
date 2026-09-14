"""
HippoRAG (Gutiérrez et al., 2024), reimplementado sobre a extração compartilhada.

O caminho online do artigo, na ordem:

1. NER sobre a pergunta produz C_q = {c_1..c_n}.
2. Cada c_i é ligado ao nó de frase mais próximo por embedding: R_q = {r_1..r_n},
   com r_i = argmax_j cos(M(c_i), M(e_j)).
3. O vetor de reset dá massa igual a cada nó de R_q, multiplicada pela
   especificidade s_i = |P_i|^{-1} — o sinal IDF-like que o artigo defende como
   calculável localmente em cada nó.
4. PPR com damping 0.5 sobre nós de frase, arestas de relação e de sinonímia.
5. p = Pᵀ·n′ agrega a probabilidade dos nós nas passagens, e p ordena o ranking.

O que NÃO está aqui, de propósito: nós de passagem e filtragem de triplas. São as
duas coisas que o HippoRAG 2 acrescenta, e mantê-las fora é o que faz a
comparação entre as duas versões medir a contribuição delas.
"""

from __future__ import annotations

import numpy as np

from wrag import config as C
from wrag import prompts
from wrag.data import Question
from wrag.embed import cosine_topk
from wrag.graph import build_graph, passage_scores_from_nodes, personalized_pagerank
from wrag.llm import GenParams
from wrag.llm.filters import LEDGER
from wrag.methods.base import IndexContext, RetrievalResult, Retriever, pad_with_dense
from wrag.methods.dense import DenseRetriever
from wrag.util import get_logger

log = get_logger("wrag.methods.hipporag")


class HippoRAGRetriever(Retriever):
    name = "hipporag"
    uses_graph = True
    with_passage_nodes = False

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx)
        self._dense = DenseRetriever(ctx)
        self._specificity: np.ndarray | None = None

    def index(self) -> None:
        if self.ctx.kg is None:
            raise RuntimeError("o grafo compartilhado precisa ser construído antes dos métodos")
        self._dense.index()
        self._specificity = self.kg.node_specificity()
        self.indexed = True

    # -- ligação pergunta -> nós -------------------------------------------

    def query_entities(self, question: Question) -> tuple[list[str], bool]:
        result = self.ctx.llm.chat(
            prompts.QUERY_NER_TEMPLATE.format(question=question.question),
            system=prompts.QUERY_NER_SYSTEM,
            params=GenParams(temperature=0.0, max_tokens=400, json_mode=True),
            stage="retrieve.ner",
        )
        if result.filtered:
            LEDGER.add("query", self.ctx.dataset, self.name, question.qid, "NER da pergunta bloqueado")
            return [], True
        data = result.json() or {}
        return [str(e).strip() for e in (data.get("named_entities") or []) if str(e).strip()], False

    def seed_nodes(self, mentions: list[str]) -> dict[int, float]:
        """Nó de frase mais próximo de cada menção, com score = similaridade."""
        kg = self.kg
        if not mentions or kg.n_entities == 0:
            return {}
        vectors = self.ctx.embedder.encode(mentions)
        seeds: dict[int, float] = {}
        for vector in vectors:
            idx, scores = cosine_topk(vector, kg.entity_vectors, 1)
            if idx.size == 0:
                continue
            eid, score = int(idx[0]), float(scores[0])
            seeds[eid] = max(seeds.get(eid, 0.0), score)
        return seeds

    # -- recuperação --------------------------------------------------------

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        kg = self.kg
        dense_pids, dense_scores = self._dense.search(question.question, max(k, 20))

        mentions, blocked = self.query_entities(question)
        if blocked:
            # Sem NER não há semente. Cair para o denso mantém a pergunta na
            # tabela em vez de transformá-la num zero artificial, e a bandeira
            # `filtered` deixa o relatório decidir se ela deve contar.
            return RetrievalResult(pids=dense_pids[:k], scores=dense_scores[:k], filtered=True,
                                   diagnostics={"fallback": "denso (NER bloqueado)"})

        seeds = self.seed_nodes(mentions)
        if not seeds:
            return RetrievalResult(pids=dense_pids[:k], scores=dense_scores[:k],
                                   diagnostics={"fallback": "denso (nenhum nó ligado)",
                                                "mencoes": mentions})

        # O grafo compartilhado inclui nós de passagem (o HippoRAG 2 precisa
        # deles). Aqui a caminhada roda só no bloco de frases: incluir nós de
        # passagem já seria metade da contribuição do HippoRAG 2, e a comparação
        # entre as duas versões deixaria de medir o que promete medir.
        adjacency = kg.adjacency[: kg.n_entities, : kg.n_entities]
        reset = np.zeros(kg.n_entities, dtype=np.float64)
        specificity = self._specificity if self._specificity is not None else kg.node_specificity()
        for eid, score in seeds.items():
            # massa igual por nó de consulta (o artigo), modulada pela
            # especificidade; o score de ligação entra como desempate suave
            reset[eid] += float(specificity[eid]) * (0.5 + 0.5 * score)

        cfg = self.ctx.run.graph
        node_probs = personalized_pagerank(adjacency, reset, damping=cfg.ppr_damping,
                                           tol=cfg.ppr_tol, max_iter=cfg.ppr_max_iter)
        scores = passage_scores_from_nodes(kg, node_probs)

        order = np.argsort(-scores)[: max(k, 20)]
        pids = [self.corpus.passages[int(i)].pid for i in order if scores[int(i)] > 0]
        top_scores = [float(scores[int(i)]) for i in order if scores[int(i)] > 0]
        pids, top_scores = pad_with_dense(pids, top_scores, dense_pids, dense_scores, k)

        return RetrievalResult(
            pids=pids, scores=top_scores,
            diagnostics={"mencoes": mentions, "nos_semente": len(seeds)},
        )

    def index_report(self) -> dict:
        return {"grafo": self.kg.stats()}


def ensure_graph(ctx: IndexContext, with_passage_nodes: bool = True) -> None:
    """Constrói extração e grafo uma vez por contexto, para todos os métodos."""
    if ctx.extraction is None:
        from wrag.ie import extract_corpus

        ctx.extraction = extract_corpus(ctx.corpus, ctx.llm, ctx.run.ie)
    if ctx.kg is None:
        ctx.kg = build_graph(ctx.corpus, ctx.extraction, ctx.embedder, ctx.run.graph,
                             with_passage_nodes=with_passage_nodes)
