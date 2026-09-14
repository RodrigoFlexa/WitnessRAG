"""
Comparador local inspirado em GraphRAG, com ranking de passagens adaptado.

Usa extração compartilhada, comunidades e vizinhança de entidades. Não é uma
reprodução fiel do GraphRAG publicado e não implementa sua busca global.
A comparação avalia este pipeline local sob o protocolo comum do repositório.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from wrag import prompts
from wrag.data import Question
from wrag.embed import cosine_topk
from wrag.graph import detect_communities
from wrag.llm import GenParams
from wrag.llm.filters import LEDGER
from wrag.methods.base import IndexContext, RetrievalResult, Retriever, pad_with_dense
from wrag.methods.dense import DenseRetriever
from wrag.util import get_logger, truncate

log = get_logger("wrag.methods.graphrag")


class GraphRAGRetriever(Retriever):
    name = "graphrag"
    uses_graph = True

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx)
        self._dense = DenseRetriever(ctx)
        self.communities: list[list[int]] = []
        self.community_of: dict[int, int] = {}
        self.reports: dict[int, dict] = {}
        self._report_vectors: np.ndarray | None = None
        self._report_ids: list[int] = []
        self._blocked_reports = 0

    # -- indexação ----------------------------------------------------------

    def index(self) -> None:
        if self.ctx.kg is None:
            raise RuntimeError("o grafo compartilhado precisa ser construído antes dos métodos")
        self._dense.index()
        cfg = self.ctx.run.graphrag
        self.communities = detect_communities(self.kg, min_size=cfg.min_community_size)
        for cid, members in enumerate(self.communities):
            for eid in members:
                self.community_of[eid] = cid
        log.info("GraphRAG: %d comunidades (>= %d nós)", len(self.communities), cfg.min_community_size)

        if cfg.build_community_reports and self.communities:
            self._build_reports()
        self.indexed = True

    def _build_reports(self) -> None:
        kg = self.kg
        cfg = self.ctx.run.graphrag
        # Comunidades maiores primeiro: com orçamento limitado, resumir as
        # grandes cobre mais do corpus por chamada.
        order = sorted(range(len(self.communities)), key=lambda c: -len(self.communities[c]))
        order = order[: cfg.max_communities_reported]

        batch, ids = [], []
        for cid in order:
            members = self.communities[cid]
            member_set = set(members)
            entities = [kg.entities[e] for e in members[:40]]
            rels = []
            for i, fact in enumerate(kg.facts):
                if fact.subj_id in member_set and fact.obj_id in member_set:
                    rels.append(f"- {fact.subject} | {fact.relation} | {fact.object}")
                if len(rels) >= 60:
                    break
            batch.append(prompts.COMMUNITY_TEMPLATE.format(
                entities=", ".join(entities), relationships="\n".join(rels) or "(sem relações internas)"))
            ids.append(cid)

        results = self.ctx.llm.chat_many(
            batch, system=prompts.COMMUNITY_SYSTEM,
            params=GenParams(temperature=0.0, max_tokens=600, json_mode=True),
            stage="graphrag.community", desc="relatórios de comunidade")

        texts, kept_ids = [], []
        for cid, result in zip(ids, results):
            if result.filtered:
                self._blocked_reports += 1
                LEDGER.add("index", self.ctx.dataset, self.name, f"community-{cid}",
                           "relatório de comunidade bloqueado")
                continue
            data = result.json()
            data = data if isinstance(data, dict) else {}
            summary = str(data.get("summary") or "").strip()
            if not summary:
                continue
            self.reports[cid] = {"title": str(data.get("title") or f"comunidade {cid}"),
                                 "summary": summary}
            texts.append(f"{self.reports[cid]['title']}. {summary}")
            kept_ids.append(cid)

        if texts:
            self._report_vectors = self.ctx.embedder.encode(texts, desc="embed(relatórios)")
            self._report_ids = kept_ids
        log.info("GraphRAG: %d relatórios de comunidade (%d bloqueados)",
                 len(self.reports), self._blocked_reports)

    # -- busca local --------------------------------------------------------

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        kg = self.kg
        cfg = self.ctx.run.graphrag
        dense_pids, dense_scores = self._dense.search(question.question, max(k, 20))

        if kg.n_entities == 0:
            return RetrievalResult(pids=dense_pids[:k], scores=dense_scores[:k],
                                   diagnostics={"fallback": "denso (grafo vazio)"})

        # 1. entidades de entrada: casamento direto pergunta -> nós
        query_vector = self.ctx.embedder.encode([question.question])[0]
        idx, sims = cosine_topk(query_vector, kg.entity_vectors, 10)
        seeds = {int(i): float(s) for i, s in zip(idx, sims) if s > 0}

        # 2. vizinhança: entidades ligadas às sementes por algum fato
        neighborhood: dict[int, float] = dict(seeds)
        for eid, weight in list(seeds.items()):
            for fid in kg.facts_by_subject.get(eid, [])[:60] + kg.facts_by_object.get(eid, [])[:60]:
                fact = kg.facts[fid]
                for other in (fact.subj_id, fact.obj_id):
                    if other >= 0 and other != eid:
                        neighborhood[other] = max(neighborhood.get(other, 0.0), weight * 0.5)

        # 3. unidades de texto: evidência acumulada das entidades que as citam
        unit_scores: dict[str, float] = defaultdict(float)
        for eid, weight in neighborhood.items():
            fids = (kg.facts_by_subject.get(eid, []) + kg.facts_by_object.get(eid, []))
            for fid in fids[: cfg.text_units_per_entity]:
                unit_scores[kg.facts[fid].pid] += weight

        # 4. relatórios de comunidade: reforçam as passagens das comunidades
        #    que a pergunta ativa. É o que distingue o local search de uma
        #    expansão de vizinhança pura.
        community_hits: list[str] = []
        if self._report_vectors is not None and self._report_vectors.size:
            ridx, rsims = cosine_topk(query_vector, self._report_vectors, 3)
            for pos, sim in zip(ridx, rsims):
                cid = self._report_ids[int(pos)]
                community_hits.append(self.reports[cid]["title"])
                members = set(self.communities[cid])
                for fid, fact in enumerate(kg.facts):
                    if fact.subj_id in members or fact.obj_id in members:
                        unit_scores[fact.pid] += 0.25 * float(sim)

        ranked = sorted(unit_scores.items(), key=lambda kv: -kv[1])[: max(k, 20)]
        pids = [pid for pid, _ in ranked]
        scores = [float(s) for _, s in ranked]
        pids, scores = pad_with_dense(pids, scores, dense_pids, dense_scores, k)

        return RetrievalResult(
            pids=pids, scores=scores,
            diagnostics={"n_sementes": len(seeds), "n_vizinhanca": len(neighborhood),
                         "comunidades": community_hits},
        )

    def index_report(self) -> dict:
        return {"n_comunidades": len(self.communities), "n_relatorios": len(self.reports),
                "relatorios_bloqueados": self._blocked_reports}
