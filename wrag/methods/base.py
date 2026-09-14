"""
Contrato dos métodos de recuperação e o contexto que todos compartilham.

`IndexContext` existe para tornar impossível o erro mais fácil de cometer neste
benchmark: deixar dois métodos verem bases de fatos diferentes. Corpus, LLM,
embedder, extração e grafo são construídos uma vez por dataset e injetados nos
cinco métodos. O que cada método pode variar é o que faz com isso.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from wrag import config as C
from wrag.data import Corpus, Question
from wrag.embed import Embedder
from wrag.graph import KnowledgeGraph
from wrag.ie import ExtractionResult
from wrag.llm import LLM
from wrag.util import get_logger

log = get_logger("wrag.methods")


@dataclass
class IndexContext:
    corpus: Corpus
    llm: LLM
    embedder: Embedder
    run: C.RunConfig
    extraction: ExtractionResult | None = None
    kg: KnowledgeGraph | None = None

    @property
    def dataset(self) -> str:
        return self.corpus.name


@dataclass
class RetrievalResult:
    """Saída de um método para uma pergunta.

    `diagnostics` é onde cada método deposita o que só ele tem: triplas filtradas
    no HippoRAG 2, comunidades no GraphRAG, a testemunha e o certificado de risco
    no WITNESS-RAG. O relatório lê esse campo para as métricas específicas, e ele
    é gravado por pergunta para permitir análise de erro depois.
    """

    pids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    filtered: bool = False
    latency_s: float = 0.0

    def top(self, k: int) -> list[str]:
        return self.pids[:k]


class Retriever(ABC):
    """Interface dos cinco sistemas. `index()` é offline; `retrieve()` é online."""

    name: str = "abstract"
    uses_graph: bool = False

    def __init__(self, ctx: IndexContext) -> None:
        self.ctx = ctx
        self.indexed = False

    @property
    def corpus(self) -> Corpus:
        return self.ctx.corpus

    @property
    def kg(self) -> KnowledgeGraph:
        if self.ctx.kg is None:
            raise RuntimeError(f"{self.name}: o grafo não foi construído para este contexto")
        return self.ctx.kg

    def index(self) -> None:
        self.indexed = True

    @abstractmethod
    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        ...

    def retrieve(self, question: Question, k: int | None = None) -> RetrievalResult:
        k = k or self.ctx.run.top_k
        started = time.perf_counter()
        result = self._retrieve(question, k)
        result.latency_s = time.perf_counter() - started
        return result

    def index_report(self) -> dict[str, Any]:
        """O que este método construiu offline. Entra no relatório de custo."""
        return {}


def pad_with_dense(pids: list[str], scores: list[float], dense_pids: list[str],
                   dense_scores: list[float], k: int) -> tuple[list[str], list[float]]:
    """Completa um ranking curto com recuperação densa.

    Todo método com grafo precisa disso: quando a busca estrutural devolve menos
    de k passagens, deixar o resto vazio faria o recall@5 medir "o grafo achou
    pouco" em vez de "o grafo achou errado". Completar com o denso é o que o
    HippoRAG 2 faz e é o que torna os cinco sistemas comparáveis no mesmo k.
    """
    have = set(pids)
    out_pids, out_scores = list(pids), list(scores)
    floor = min(scores) if scores else 0.0
    for pid, score in zip(dense_pids, dense_scores):
        if len(out_pids) >= k:
            break
        if pid in have:
            continue
        out_pids.append(pid)
        # abaixo do piso do ranking estrutural, para não reordenar o que veio dele
        out_scores.append(min(floor, 0.0) - 1.0 + float(score) * 1e-3)
        have.add(pid)
    return out_pids[:k], out_scores[:k]
