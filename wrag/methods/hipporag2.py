"""
HippoRAG 2 (Gutiérrez et al., 2025), reimplementado sobre a mesma extração.

As quatro mudanças em relação ao HippoRAG 1, todas presentes aqui:

1. **Integração densa-esparsa.** Cada passagem vira um nó, ligado por aresta de
   contexto ("contains") a toda frase extraída dela. O ranking final lê a
   probabilidade dos NÓS DE PASSAGEM, não a soma agregada dos nós de frase.
2. **Contextualização mais profunda (query-to-triple).** A pergunta inteira é
   casada contra triplas, não entidades da pergunta contra nós. O artigo mede
   +12.5% de Recall@5 sobre NER-to-node com essa troca.
3. **Recognition memory.** Um LLM filtra as triplas recuperadas, T' ⊆ T, e só as
   sobreviventes viram sementes de frase.
4. **Pesos de reset separados.** Nós de frase recebem massa proporcional ao score
   de ranqueamento; nós de passagem recebem massa proporcional à similaridade
   densa, multiplicada por um fator de peso (0.05 no artigo, §6.2).

Se nenhuma tripla sobrevive ao filtro, o artigo manda usar o ranking denso
direto. Esse caminho é frequente e está implementado.
"""

from __future__ import annotations

import numpy as np

from wrag import prompts
from wrag.data import Question
from wrag.embed import cosine_topk
from wrag.graph import personalized_pagerank
from wrag.llm import GenParams
from wrag.llm.filters import LEDGER
from wrag.methods.base import IndexContext, RetrievalResult, Retriever
from wrag.methods.dense import DenseRetriever
from wrag.util import get_logger, normalize

log = get_logger("wrag.methods.hipporag2")


class HippoRAG2Retriever(Retriever):
    name = "hipporag2"
    uses_graph = True

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx)
        self._dense = DenseRetriever(ctx)

    def index(self) -> None:
        if self.ctx.kg is None:
            raise RuntimeError("o grafo compartilhado precisa ser construído antes dos métodos")
        if self.kg.n_passage_nodes == 0:
            raise RuntimeError("HippoRAG 2 exige nós de passagem no grafo compartilhado")
        self._dense.index()
        self.indexed = True

    # -- query to triple ----------------------------------------------------

    def rank_triples(self, question: str, n: int) -> tuple[list[int], list[float]]:
        kg = self.kg
        if kg.fact_vectors.size == 0:
            return [], []
        query = self.ctx.embedder.encode([question])[0]
        idx, scores = cosine_topk(query, kg.fact_vectors, n)
        return [int(i) for i in idx], [float(s) for s in scores]

    def recognition_memory(self, question: Question, fact_ids: list[int],
                           max_kept: int) -> tuple[list[int], bool]:
        """Filtro por LLM. Devolve os índices mantidos, na ordem em que vieram.

        O casamento de volta é por tripla normalizada, e não por índice devolvido
        pelo modelo: pedir índices convida o modelo a inventar um, e um índice
        inventado viraria semente de um fato que ninguém recuperou.
        """
        kg = self.kg
        if not fact_ids:
            return [], False
        if not self.ctx.run.hippo2.use_recognition_memory:
            return fact_ids[:max_kept], False

        triples = [kg.facts[i].triple for i in fact_ids]
        result = self.ctx.llm.chat(
            prompts.TRIPLE_FILTER_TEMPLATE.format(
                question=question.question, triples=prompts.format_triples(triples), max_kept=max_kept),
            system=prompts.TRIPLE_FILTER_SYSTEM,
            params=GenParams(temperature=0.0, max_tokens=800, json_mode=True),
            stage="retrieve.filter",
        )
        if result.filtered:
            LEDGER.add("query", self.ctx.dataset, self.name, question.qid, "filtro de triplas bloqueado")
            return fact_ids[:max_kept], True

        data = result.json()
        data = data if isinstance(data, dict) else {}
        kept_raw = data.get("fact") or data.get("triples") or []
        wanted = set()
        for item in kept_raw:
            if isinstance(item, dict):
                item = [item.get("subject"), item.get("relation"), item.get("object")]
            if isinstance(item, (list, tuple)) and len(item) == 3:
                wanted.add(tuple(normalize(str(x)) for x in item))
        kept = [i for i in fact_ids
                if tuple(normalize(x) for x in kg.facts[i].triple) in wanted]
        return kept[:max_kept], False

    # -- recuperação --------------------------------------------------------

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        kg = self.kg
        cfg = self.ctx.run.hippo2
        dense_pids, dense_scores = self._dense.search(question.question, max(cfg.n_passages_seed, k))

        fact_ids, fact_scores = self.rank_triples(question.question, cfg.n_triples_retrieved)
        kept, blocked = self.recognition_memory(question, fact_ids, cfg.n_triples_kept)
        score_of = dict(zip(fact_ids, fact_scores))

        if not kept:
            # "Se nenhuma tripla estiver disponível, recupera diretamente as
            # passagens mais bem ranqueadas pelo modelo de embedding." (§3.5)
            return RetrievalResult(pids=dense_pids[:k], scores=dense_scores[:k], filtered=blocked,
                                   diagnostics={"fallback": "denso (sem triplas)", "n_triplas": 0})

        # -- sementes de frase: score médio das triplas de que a frase saiu
        phrase_scores: dict[int, list[float]] = {}
        for fid in kept:
            fact = kg.facts[fid]
            for eid in (fact.subj_id, fact.obj_id):
                if eid >= 0:
                    phrase_scores.setdefault(eid, []).append(score_of.get(fid, 0.0))

        total_nodes = kg.adjacency.shape[0]
        reset = np.zeros(total_nodes, dtype=np.float64)
        for eid, scores in phrase_scores.items():
            reset[eid] += max(0.0, float(np.mean(scores)))

        # -- sementes de passagem: todas, com massa ∝ similaridade densa,
        #    multiplicada pelo fator de peso (0.05 no artigo)
        phrase_mass = float(reset.sum()) or 1.0
        passage_mass = 0.0
        for pid, score in zip(dense_pids, dense_scores):
            row = kg.passage_row.get(pid)
            if row is None:
                continue
            passage_mass += max(0.0, float(score))
        if passage_mass > 0:
            scale = cfg.passage_node_weight * phrase_mass / passage_mass
            for pid, score in zip(dense_pids, dense_scores):
                row = kg.passage_row.get(pid)
                if row is not None:
                    reset[kg.n_entities + row] += max(0.0, float(score)) * scale

        graph_cfg = self.ctx.run.graph
        node_probs = personalized_pagerank(kg.adjacency, reset, damping=graph_cfg.ppr_damping,
                                           tol=graph_cfg.ppr_tol, max_iter=graph_cfg.ppr_max_iter)

        # -- ranking pelos NÓS DE PASSAGEM (a diferença que a §3.2 introduz)
        passage_probs = node_probs[kg.n_entities: kg.n_entities + kg.n_passages]
        order = np.argsort(-passage_probs)[:k]
        pids = [self.corpus.passages[int(i)].pid for i in order]
        scores = [float(passage_probs[int(i)]) for i in order]

        return RetrievalResult(
            pids=pids, scores=scores, filtered=blocked,
            diagnostics={
                "n_triplas_recuperadas": len(fact_ids),
                "n_triplas_mantidas": len(kept),
                "triplas": [list(kg.facts[i].triple) for i in kept],
            },
        )

    def index_report(self) -> dict:
        return {"grafo": self.kg.stats()}
