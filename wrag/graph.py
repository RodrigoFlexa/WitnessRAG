"""
Índice compartilhado com entidades, passagens, relações e arestas de similaridade.

Os comparadores de difusão usam PPR; WITNESS-RAG executa junções dirigidas.
A identidade padrão preserva símbolos canônicos, sem fundir entidades por
similaridade. Fusão vetorial é uma opção experimental e não prova identidade.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from wrag import config as C
from wrag.data import Corpus
from wrag.embed import Embedder
from wrag.ie import ExtractionResult, Fact
from wrag.util import get_logger, canonical_symbol as normalize

log = get_logger("wrag.graph")

MERGE_MARGIN = 0.10  # limiar de fusão = limiar de aresta + esta margem

# Palavras que não discriminam identidade e por isso não contam na contenção.
_IDENTITY_STOPWORDS = frozenset(
    "a an the of de da do del la le les el los las in on at for and or to from "
    "s dr mr mrs ms sir jr sr".split()
)


@lru_cache(maxsize=4096)
def relation_signature(surface: str) -> str:
    """Assinatura morfológica conservadora para variantes do mesmo predicado.

    Mantém todas as palavras e preposições, portanto ``work at`` e ``work for``
    continuam distintas. Só reduz flexões como ``painted``/``painting``/``paint``.
    A extração ainda guarda a forma original para auditoria.
    """
    value = normalize(surface)
    if not value:
        return ""
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return " ".join(stemmer.stem(token) for token in value.split())
    except ImportError:  # o núcleo continua utilizável na instalação mínima
        return value


def _identity_tokens(surface: str) -> frozenset[str]:
    return frozenset(t for t in normalize(surface).split() if t and t not in _IDENTITY_STOPWORDS)


def _is_identity_variant(left: str, right: str) -> bool:
    """Identidade por contenção lexical, confirmada depois pela semântica.

    Similaridade de cosseno sozinha não prova identidade: 'the first half of X' e
    'the second half of X' ficam em 0.95 e não são a mesma coisa. Exigir que um
    conjunto de tokens contenha o outro descarta esse caso — 'first' e 'second'
    são tokens discriminantes que nenhum dos dois lados absorve — e ainda aceita
    'Juan Courten' ⊂ 'Juan de Courten' e 'New York' ⊂ 'New York City'.

    Conservador de propósito: perde sinônimos sem sobreposição ('TV'/'television').
    Numa proposta que emite proveniência, unir duas entidades distintas fabrica
    testemunha falsa, que custa mais caro do que deixar uma cadeia em aberto.
    """
    a, b = _identity_tokens(left), _identity_tokens(right)
    if not a or not b:
        return False
    if not (a <= b or b <= a):
        return False
    # Uma única palavra a mais pode inverter o referente ("Universidade X" vs
    # "Universidade X Campus Y"); o limite mantém a fusão perto da abreviação.
    return abs(len(a) - len(b)) <= max(1, min(len(a), len(b)) - 1)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class KnowledgeGraph:
    corpus: Corpus
    facts: list[Fact]

    # nós de frase
    entities: list[str] = field(default_factory=list)              # forma de superfície representativa
    entity_of: dict[str, int] = field(default_factory=dict)        # normalize(surface) -> eid
    entity_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    cluster_of: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    # relações canônicas
    relations: list[str] = field(default_factory=list)
    relation_of: dict[str, int] = field(default_factory=dict)
    relation_signature_of: dict[str, int] = field(default_factory=dict)
    relation_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    # passagens
    passage_row: dict[str, int] = field(default_factory=dict)
    passage_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    # índices derivados
    entity_passages: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    facts_by_subject: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    facts_by_object: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    facts_by_passage: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    fact_vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    # matrizes
    adjacency: sp.csr_matrix | None = None          # nós de frase + (opcional) de passagem
    node_passage: sp.csr_matrix | None = None       # |E| x |P|, contagem de aparições
    n_passage_nodes: int = 0
    synonym_edges: int = 0

    # -- construção ---------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def n_passages(self) -> int:
        return len(self.corpus.passages)

    def node_specificity(self) -> np.ndarray:
        """s_i = |P_i|^{-1}: o sinal IDF-like do HippoRAG, calculável localmente
        em cada nó (é o argumento de plausibilidade neurobiológica do artigo)."""
        counts = np.array([max(1, len(self.entity_passages.get(i, ()))) for i in range(self.n_entities)],
                          dtype=np.float32)
        return 1.0 / counts

    def entity_id(self, surface: str) -> int | None:
        return self.entity_of.get(normalize(surface))

    def cluster(self, eid: int) -> int:
        return int(self.cluster_of[eid]) if 0 <= eid < len(self.cluster_of) else -1

    def stats(self) -> dict[str, Any]:
        return {
            "n_nos_frase": self.n_entities,
            "n_clusters_canonicos": int(len(set(self.cluster_of.tolist()))) if len(self.cluster_of) else 0,
            "n_fatos": len(self.facts),
            "n_relacoes_canonicas": len(self.relations),
            "n_arestas_sinonimia": self.synonym_edges,
            "n_nos_passagem": self.n_passage_nodes,
        }


def build_graph(
    corpus: Corpus,
    extraction: ExtractionResult,
    embedder: Embedder,
    cfg: C.GraphConfig | None = None,
    with_passage_nodes: bool = True,
) -> KnowledgeGraph:
    cfg = cfg or C.GraphConfig()
    facts = list(extraction.facts)
    kg = KnowledgeGraph(corpus=corpus, facts=facts)

    # -- nós de frase
    surfaces: dict[str, str] = {}
    for f in facts:
        for surface in (f.subject, f.object):
            key = normalize(surface)
            if key and key not in surfaces:
                surfaces[key] = surface.strip()
    kg.entities = [surfaces[k] for k in surfaces]
    kg.entity_of = {k: i for i, k in enumerate(surfaces)}

    # -- relações canônicas (usadas pelo aterramento do WITNESS-RAG)
    rel_surfaces: dict[str, str] = {}
    rel_counts: Counter[str] = Counter()
    rel_groups: dict[str, list[str]] = {}
    for f in facts:
        key = normalize(f.relation)
        signature = relation_signature(f.relation) if cfg.merge_relation_inflections else key
        if not key or not signature:
            continue
        rel_counts[key] += 1
        if key not in rel_surfaces:
            rel_surfaces[key] = f.relation.strip()
        group = rel_groups.setdefault(signature, [])
        if key not in group:
            group.append(key)
    signatures = list(rel_groups)
    representative_keys = [
        max(rel_groups[sig], key=lambda key: (rel_counts[key], -len(key)))
        for sig in signatures
    ]
    kg.relations = [rel_surfaces[key] for key in representative_keys]
    kg.relation_signature_of = {signature: i for i, signature in enumerate(signatures)}
    kg.relation_of = {
        key: kg.relation_signature_of[signature]
        for signature, keys in rel_groups.items() for key in keys
    }

    for i, f in enumerate(facts):
        f.subj_id = kg.entity_of.get(normalize(f.subject), -1)
        f.obj_id = kg.entity_of.get(normalize(f.object), -1)
        f.rel_id = kg.relation_of.get(normalize(f.relation), -1)
        if f.subj_id >= 0:
            kg.facts_by_subject[f.subj_id].append(i)
            kg.entity_passages[f.subj_id].add(f.pid)
        if f.obj_id >= 0:
            kg.facts_by_object[f.obj_id].append(i)
            kg.entity_passages[f.obj_id].add(f.pid)
        kg.facts_by_passage[f.pid].append(i)

    # -- vetores
    log.info("vetorizando %d nós, %d relações, %d fatos, %d passagens",
             kg.n_entities, len(kg.relations), len(facts), kg.n_passages)
    kg.entity_vectors = embedder.encode(kg.entities, desc="embed(entidades)")
    kg.relation_vectors = embedder.encode(kg.relations, desc="embed(relações)")
    kg.fact_vectors = embedder.encode([f.verbalize() for f in facts], desc="embed(fatos)")
    kg.passage_vectors = embedder.encode(corpus.texts(), desc="embed(passagens)")
    kg.passage_row = {p.pid: i for i, p in enumerate(corpus.passages)}

    # -- sinonímia: um cálculo, dois usos (arestas e fusão)
    pairs = _similar_pairs(kg.entity_vectors, cfg.synonym_threshold, cfg.synonym_max_neighbors)
    kg.synonym_edges = len(pairs)
    uf = UnionFind(kg.n_entities)
    merge_threshold = min(0.99, cfg.synonym_threshold + MERGE_MARGIN)
    for i, j, sim in pairs:
        if cfg.merge_similar_entities and sim >= merge_threshold:
            uf.union(i, j)
        elif (cfg.merge_identity_variants and sim >= cfg.identity_merge_threshold
              and _is_identity_variant(kg.entities[i], kg.entities[j])):
            uf.union(i, j)
    kg.cluster_of = np.array([uf.find(i) for i in range(kg.n_entities)], dtype=np.int32)

    # -- matriz nó x passagem (o P do HippoRAG)
    kg.node_passage = _node_passage_matrix(kg)

    # -- adjacência
    kg.adjacency = _adjacency(kg, pairs, with_passage_nodes=with_passage_nodes)
    kg.n_passage_nodes = kg.n_passages if with_passage_nodes else 0

    log.info("grafo construído: %s", kg.stats())
    return kg


def _similar_pairs(vectors: np.ndarray, threshold: float, max_neighbors: int) -> list[tuple[int, int, float]]:
    """Pares acima do limiar, em blocos para não materializar |N|x|N|."""
    n = vectors.shape[0]
    if n < 2:
        return []
    pairs: list[tuple[int, int, float]] = []
    block = max(1, min(2048, 40_000_000 // max(1, n)))
    for start in range(0, n, block):
        stop = min(n, start + block)
        sims = vectors[start:stop] @ vectors.T
        for local, row in enumerate(sims):
            i = start + local
            row[i] = -1.0
            k = int(min(max_neighbors, n - 1))
            if k <= 0:
                continue
            idx = np.argpartition(-row, k - 1)[:k]
            for j in idx:
                j = int(j)
                if j <= i:
                    continue
                sim = float(row[j])
                if sim >= threshold:
                    pairs.append((i, j, sim))
    return pairs


def _node_passage_matrix(kg: KnowledgeGraph) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for f in kg.facts:
        col = kg.passage_row.get(f.pid)
        if col is None:
            continue
        for eid in (f.subj_id, f.obj_id):
            if eid >= 0:
                counts[(eid, col)] += 1
    for (r, c), v in counts.items():
        rows.append(r)
        cols.append(c)
        vals.append(float(v))
    return sp.csr_matrix((vals, (rows, cols)), shape=(max(1, kg.n_entities), max(1, kg.n_passages)))


def _adjacency(kg: KnowledgeGraph, synonym_pairs: Sequence[tuple[int, int, float]],
               with_passage_nodes: bool) -> sp.csr_matrix:
    """Adjacência simétrica sobre nós de frase e, opcionalmente, de passagem.

    Arestas de relação são pesadas pelo número de triplas que ligam o par: dois
    nós citados juntos em cinco passagens são mais associados que dois citados
    juntos uma vez, e é essa associatividade que o PPR propaga.
    """
    n_entities = kg.n_entities
    total = n_entities + (kg.n_passages if with_passage_nodes else 0)
    weights: dict[tuple[int, int], float] = defaultdict(float)

    for f in kg.facts:
        if f.subj_id >= 0 and f.obj_id >= 0 and f.subj_id != f.obj_id:
            key = (min(f.subj_id, f.obj_id), max(f.subj_id, f.obj_id))
            weights[key] += 1.0

    for i, j, sim in synonym_pairs:
        weights[(min(i, j), max(i, j))] += float(sim)

    if with_passage_nodes:
        # aresta de contexto "contains": passagem -> cada frase dela
        for f in kg.facts:
            col = kg.passage_row.get(f.pid)
            if col is None:
                continue
            node = n_entities + col
            for eid in (f.subj_id, f.obj_id):
                if eid >= 0:
                    weights[(min(eid, node), max(eid, node))] += 1.0

    if not weights:
        return sp.csr_matrix((max(1, total), max(1, total)))

    rows, cols, vals = [], [], []
    for (i, j), w in weights.items():
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    return sp.csr_matrix((vals, (rows, cols)), shape=(total, total))


def personalized_pagerank(
    adjacency: sp.csr_matrix,
    reset: np.ndarray,
    damping: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> np.ndarray:
    """PPR por iteração de potência.

    x = (1-d)·reset + d·Wᵀx, com W a adjacência normalizada por linha. `damping`
    é a probabilidade de CONTINUAR a caminhada; 0.5 é o valor do HippoRAG.
    Nós sem saída devolvem sua massa ao vetor de reset, e não ao vazio — sem
    isso a distribuição vaza e o ranking entre nós de grau baixo fica arbitrário.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    total = float(reset.sum())
    if total <= 0:
        return np.zeros(n, dtype=np.float32)
    reset = (reset / total).astype(np.float64)

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    dangling = degrees == 0
    inv = np.zeros_like(degrees)
    inv[~dangling] = 1.0 / degrees[~dangling]
    walk = sp.diags(inv) @ adjacency  # linhas normalizadas

    x = reset.copy()
    for _ in range(max_iter):
        leaked = damping * float(x[dangling].sum())
        nxt = (1.0 - damping) * reset + damping * (walk.T @ x) + leaked * reset
        s = nxt.sum()
        if s > 0:
            nxt /= s
        if np.abs(nxt - x).sum() < tol:
            x = nxt
            break
        x = nxt
    return x.astype(np.float32)


def passage_scores_from_nodes(kg: KnowledgeGraph, node_probs: np.ndarray) -> np.ndarray:
    """p = Pᵀ·n′: agrega probabilidade dos nós de frase nas passagens (HippoRAG)."""
    assert kg.node_passage is not None
    entity_probs = node_probs[: kg.n_entities]
    return np.asarray(kg.node_passage.T @ entity_probs).ravel()


def detect_communities(kg: KnowledgeGraph, resolution: float = 1.0,
                       min_size: int = 3) -> list[list[int]]:
    """Comunidades de nós de frase, para o GraphRAG.

    Usa Leiden via python-igraph quando disponível (é o algoritmo do artigo da
    Microsoft) e cai para o greedy modularity do networkx quando não. A queda é
    registrada no log porque muda a partição e, portanto, os relatórios.
    """
    assert kg.adjacency is not None
    sub = kg.adjacency[: kg.n_entities, : kg.n_entities].tocoo()
    edges = [(int(i), int(j), float(w)) for i, j, w in zip(sub.row, sub.col, sub.data) if i < j]
    if not edges:
        return []
    try:
        import igraph as ig

        g = ig.Graph(n=kg.n_entities, edges=[(i, j) for i, j, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]
        partition = g.community_leiden(objective_function="modularity", weights="weight",
                                       resolution=resolution, n_iterations=3)
        groups = [list(c) for c in partition]
    except ImportError:
        import networkx as nx

        log.warning("python-igraph indisponível; usando greedy modularity do networkx "
                    "(a partição difere da do artigo do GraphRAG)")
        g = nx.Graph()
        g.add_nodes_from(range(kg.n_entities))
        g.add_weighted_edges_from(edges)
        groups = [sorted(c) for c in nx.community.greedy_modularity_communities(g, weight="weight")]

    return [c for c in groups if len(c) >= min_size]
