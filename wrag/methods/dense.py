"""
RAG clássico: recuperação por similaridade de embedding, e BM25 como variante.

É a linha de base que todo o resto precisa bater, e também o *fallback* dos
outros quatro métodos — por isso mora aqui a função que todos chamam.
"""

from __future__ import annotations

import numpy as np

from wrag.data import Question
from wrag.embed import cosine_topk
from wrag.methods.base import IndexContext, RetrievalResult, Retriever
from wrag.util import get_logger, normalize

log = get_logger("wrag.methods.dense")


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx)
        self._matrix: np.ndarray | None = None

    def index(self) -> None:
        if self.ctx.kg is not None and self.ctx.kg.passage_vectors.size:
            # reaproveita os vetores já calculados pelo grafo
            self._matrix = self.ctx.kg.passage_vectors
        else:
            self._matrix = self.ctx.embedder.encode(self.corpus.texts(), desc="embed(passagens)")
        self.indexed = True

    def search(self, text: str, k: int) -> tuple[list[str], list[float]]:
        assert self._matrix is not None
        query = self.ctx.embedder.encode([text])[0]
        idx, scores = cosine_topk(query, self._matrix, k)
        pids = [self.corpus.passages[int(i)].pid for i in idx]
        return pids, [float(s) for s in scores]

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        pids, scores = self.search(question.question, k)
        return RetrievalResult(pids=pids, scores=scores, diagnostics={"backend": self.ctx.embedder.name})

    def index_report(self) -> dict:
        return {"vetores_passagem": 0 if self._matrix is None else int(self._matrix.shape[0])}


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx)
        self._bm25 = None

    def index(self) -> None:
        from rank_bm25 import BM25Okapi

        tokens = [normalize(p.full).split() for p in self.corpus.passages]
        self._bm25 = BM25Okapi(tokens)
        self.indexed = True

    def search(self, text: str, k: int) -> tuple[list[str], list[float]]:
        assert self._bm25 is not None
        scores = np.asarray(self._bm25.get_scores(normalize(text).split()), dtype=np.float32)
        k = int(min(k, scores.shape[0]))
        idx = np.argsort(-scores)[:k]
        return [self.corpus.passages[int(i)].pid for i in idx], [float(scores[int(i)]) for i in idx]

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        pids, scores = self.search(question.question, k)
        return RetrievalResult(pids=pids, scores=scores)


class HybridRetriever(Retriever):
    """Fusão recíproca de postos entre o denso e o BM25.

    Existe para o fallback dos métodos com grafo em corpora conversacionais: a
    resposta costuma depender de um nome próprio raro ("Becoming Nicole") que o
    vetor de um bloco longo de diálogo dilui, e que o léxico acha de imediato.
    RRF não precisa de calibração entre as duas escalas de score, que é o motivo
    de ser preferível a uma soma ponderada aqui.
    """

    name = "hybrid"

    def __init__(self, ctx: IndexContext, rrf_k: int = 60) -> None:
        super().__init__(ctx)
        self._dense = DenseRetriever(ctx)
        self._bm25 = BM25Retriever(ctx)
        self.rrf_k = rrf_k

    def index(self) -> None:
        self._dense.index()
        self._bm25.index()
        self.indexed = True

    def search(self, text: str, k: int) -> tuple[list[str], list[float]]:
        depth = max(k, 20)
        fused: dict[str, float] = {}
        for retriever in (self._dense, self._bm25):
            pids, _scores = retriever.search(text, depth)
            for rank, pid in enumerate(pids):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        order = sorted(fused, key=lambda pid: (-fused[pid], pid))[:k]
        return order, [fused[pid] for pid in order]

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        pids, scores = self.search(question.question, k)
        return RetrievalResult(pids=pids, scores=scores,
                               diagnostics={"fusao": "rrf", "rrf_k": self.rrf_k})

    def index_report(self) -> dict:
        return {"fusao": "rrf", "rrf_k": self.rrf_k}
