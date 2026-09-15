"""Registro dos métodos e construção do contexto compartilhado."""

from __future__ import annotations

from typing import Callable
import time

from wrag import config as C
from wrag.data import Corpus
from wrag.embed import Embedder, TfidfEmbedder, get_embedder
from wrag.llm import LLM, get_llm
from wrag.llm.base import usage_delta
from wrag.methods.base import IndexContext, RetrievalResult, Retriever
from wrag.methods.dense import BM25Retriever, DenseRetriever, HybridRetriever
from wrag.methods.graphrag import GraphRAGRetriever
from wrag.methods.hipporag import HippoRAGRetriever, ensure_graph
from wrag.methods.hipporag2 import HippoRAG2Retriever
from wrag.methods.witnessrag import WitnessRAGOracleRetriever, WitnessRAGRetriever, WitnessRAGAnnotatedRetriever
from wrag.methods.relational import RelationalRetriever
from wrag.util import get_logger

log = get_logger("wrag.methods")

REGISTRY: dict[str, Callable[[IndexContext], Retriever]] = {
    "dense": DenseRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "graphrag": GraphRAGRetriever,
    "hipporag": HippoRAGRetriever,
    "hipporag2": HippoRAG2Retriever,
    "witnessrag": WitnessRAGRetriever,
    "witnessrag-oracle": WitnessRAGOracleRetriever,
    "witnessrag-annotated": WitnessRAGAnnotatedRetriever,
    "relational": RelationalRetriever,
}

GRAPH_METHODS = {name for name, cls in REGISTRY.items() if cls.uses_graph}

__all__ = ["REGISTRY", "IndexContext", "RetrievalResult", "Retriever", "build_context",
           "build_methods", "ensure_graph"]


def build_context(corpus: Corpus, run: C.RunConfig, llm: LLM | None = None,
                  embedder: Embedder | None = None) -> IndexContext:
    llm = llm or get_llm()
    embedder = embedder or get_embedder()
    # O fallback TF-IDF precisa ver o corpus antes de vetorizar qualquer coisa;
    # os provedores reais ignoram isto.
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit(corpus.texts())
    return IndexContext(corpus=corpus, llm=llm, embedder=embedder, run=run)


def build_methods(ctx: IndexContext, names: list[str]) -> dict[str, Retriever]:
    """Instancia e indexa os métodos pedidos, construindo o grafo uma vez só."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"métodos desconhecidos: {unknown}. Disponíveis: {sorted(REGISTRY)}")

    if any(n in GRAPH_METHODS for n in names):
        # Nós de passagem entram sempre: o HippoRAG 2 precisa deles, e o
        # HippoRAG 1 fatia o bloco de frases para não usá-los.
        before = ctx.llm.usage.snapshot()
        started = time.perf_counter()
        ensure_graph(ctx, with_passage_nodes=True)
        ctx.shared_index_cost = {"seconds": time.perf_counter() - started,
                                 "usage": usage_delta(ctx.llm.usage.snapshot(), before)}

    methods: dict[str, Retriever] = {}
    for name in names:
        log.info("indexando método %s", name)
        retriever = REGISTRY[name](ctx)
        before = ctx.llm.usage.snapshot()
        started = time.perf_counter()
        retriever.index()
        retriever.index_cost = {"seconds": time.perf_counter() - started,
                               "usage": usage_delta(ctx.llm.usage.snapshot(), before)}
        methods[name] = retriever
    return methods
