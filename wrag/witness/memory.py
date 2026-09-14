"""
Visão de memória com escrita reversível.

A aquisição adaptativa cria fatos NOVOS em tempo de consulta. Escrevê-los direto
no grafo compartilhado causaria dois problemas, um científico e um metodológico:

* **Vazamento entre métodos.** O grafo é compartilhado de propósito, para que os
  cinco sistemas vejam a mesma base de fatos. Um fato que o WITNESS-RAG extraiu
  ao responder a pergunta 7 apareceria no índice do HippoRAG na pergunta 8, e a
  comparação deixaria de ser entre métodos.
* **Dependência da ordem.** Com memória que cresce, o resultado da pergunta 300
  depende de quais 299 vieram antes. Isso é uma propriedade interessante — é
  aprendizado contínuo — mas é um experimento diferente, com outro protocolo.

Então a aquisição escreve numa `MemoryView`, que registra tudo que acrescentou e
sabe desfazer (`reset()`). Por padrão o retriever reseta entre perguntas. Com
`acquisition_persist=True` ele não reseta, e aí o run vira explicitamente um
experimento de memória acumulativa — que o relatório marca.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from wrag.embed import Embedder, cosine_topk
from wrag.graph import KnowledgeGraph
from wrag.ie import Fact
from wrag.util import get_logger, canonical_symbol as normalize

log = get_logger("wrag.witness.memory")



class MemoryView:
    """Espelha a interface do KnowledgeGraph usada pela busca, com extensão
    reversível. Os vetores grandes do grafo base são compartilhados por
    referência até a primeira escrita."""

    def __init__(self, base: KnowledgeGraph) -> None:
        self.base = base
        self.corpus = base.corpus

        self.facts: list[Fact] = list(base.facts)
        self.entities: list[str] = list(base.entities)
        self.entity_of: dict[str, int] = dict(base.entity_of)
        self.relations: list[str] = list(base.relations)
        self.relation_of: dict[str, int] = dict(base.relation_of)

        self.entity_vectors: np.ndarray = base.entity_vectors
        self.relation_vectors: np.ndarray = base.relation_vectors
        self.fact_vectors: np.ndarray = base.fact_vectors
        self.passage_vectors: np.ndarray = base.passage_vectors
        self.passage_row = base.passage_row

        self._cluster_of: list[int] = base.cluster_of.tolist()

        # baselines para o rollback
        self._n_facts0 = len(self.facts)
        self._n_entities0 = len(self.entities)
        self._n_relations0 = len(self.relations)
        self._added_entity_keys: list[str] = []
        self._added_relation_keys: list[str] = []
        self._journal: list[tuple[dict, Any, int]] = []

        self.facts_by_subject = {eid: list(ids) for eid, ids in base.facts_by_subject.items()}
        self.facts_by_object = {eid: list(ids) for eid, ids in base.facts_by_object.items()}

    # -- interface lida pela busca -----------------------------------------

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    def entity_id(self, surface: str) -> int | None:
        return self.entity_of.get(normalize(surface))

    def cluster(self, eid: int) -> int:
        return self._cluster_of[eid] if 0 <= eid < len(self._cluster_of) else -1

    # -- escrita -----------------------------------------------------------

    def add_facts(self, facts: list[Fact], embedder: Embedder) -> list[int]:
        """Acrescenta fatos, resolvendo entidades e relações contra o que existe.

        Identidade usa símbolos canônicos; similaridade vetorial não funde
        entidades. Repetir a mesma tripla na mesma passagem é idempotente.
        """
        if not facts:
            return []

        known = {(normalize(f.subject), normalize(f.relation), normalize(f.object), f.pid)
                 for f in self.facts}
        fresh = []
        for fact in facts:
            key = (normalize(fact.subject), normalize(fact.relation), normalize(fact.object), fact.pid)
            if key not in known:
                known.add(key)
                fresh.append(replace(fact))
        facts = fresh
        if not facts:
            return []

        new_surfaces: list[str] = []
        for fact in facts:
            for surface in (fact.subject, fact.object):
                key = normalize(surface)
                if key and key not in self.entity_of and all(normalize(s) != key for s in new_surfaces):
                    new_surfaces.append(surface)

        new_relations = []
        for fact in facts:
            key = normalize(fact.relation)
            if key and key not in self.relation_of and all(normalize(r) != key for r in new_relations):
                new_relations.append(fact.relation)

        if new_surfaces:
            vectors = embedder.encode(new_surfaces)
            for surface, vector in zip(new_surfaces, vectors):
                eid = len(self.entities)
                self.entities.append(surface)
                key = normalize(surface)
                self.entity_of[key] = eid
                self._added_entity_keys.append(key)
                cluster = eid
                # Aquisição não converte proximidade vetorial em identidade.
                self._cluster_of.append(cluster)
            self.entity_vectors = (np.vstack([self.entity_vectors, vectors])
                                   if self.entity_vectors.size else vectors)

        if new_relations:
            vectors = embedder.encode(new_relations)
            for relation in new_relations:
                self.relation_of[normalize(relation)] = len(self.relations)
                self.relations.append(relation)
                self._added_relation_keys.append(normalize(relation))
            self.relation_vectors = (np.vstack([self.relation_vectors, vectors])
                                     if self.relation_vectors.size else vectors)

        fact_vectors = embedder.encode([f.verbalize() for f in facts])
        self.fact_vectors = (np.vstack([self.fact_vectors, fact_vectors])
                             if self.fact_vectors.size else fact_vectors)

        added: list[int] = []
        for fact in facts:
            fact.subj_id = self.entity_of.get(normalize(fact.subject), -1)
            fact.obj_id = self.entity_of.get(normalize(fact.object), -1)
            fact.rel_id = self.relation_of.get(normalize(fact.relation), -1)
            index = len(self.facts)
            self.facts.append(fact)
            added.append(index)
            for eid, table in ((fact.subj_id, self.facts_by_subject),
                               (fact.obj_id, self.facts_by_object)):
                if eid >= 0:
                    bucket = table.setdefault(eid, [])
                    self._journal.append((table, eid, len(bucket)))
                    bucket.append(index)
        return added

    def reset(self) -> None:
        """Desfaz tudo que foi acrescentado desde a construção."""
        if (len(self.facts) == self._n_facts0 and len(self.entities) == self._n_entities0
                and len(self.relations) == self._n_relations0 and not self._journal):
            return
        for table, key, length in reversed(self._journal):
            bucket = table.get(key)
            if bucket is not None:
                del bucket[length:]
                if not bucket:
                    table.pop(key, None)
        self._journal.clear()

        del self.facts[self._n_facts0:]
        del self.entities[self._n_entities0:]
        del self.relations[self._n_relations0:]
        del self._cluster_of[self._n_entities0:]
        for key in self._added_entity_keys:
            self.entity_of.pop(key, None)
        for key in self._added_relation_keys:
            self.relation_of.pop(key, None)
        self._added_entity_keys.clear()
        self._added_relation_keys.clear()

        self.entity_vectors = self.base.entity_vectors
        self.relation_vectors = self.base.relation_vectors
        self.fact_vectors = self.base.fact_vectors

    def stats(self) -> dict[str, Any]:
        return {"n_fatos": len(self.facts), "n_fatos_base": self._n_facts0,
                "n_entidades": len(self.entities)}
