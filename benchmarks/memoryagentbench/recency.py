"""Ordinal recency for the numbered knowledge pool, without deleting evidence.

This is a task adaptation of local-v2, not a new extraction or an answer rule.
Only validated source serials are used. Preference is within a subject/relation
family; an unrelated high serial does not make a fact an answer. Historical
facts stay in memory and all scope/conflict checks remain reader obligations.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.util import canonical_symbol
from wrag.witness.local_contract import SourceIndex
from wrag.witness.local_plans_v2 import MultiOriginPlanner, retrieve_local_v2
from wrag.witness.query import is_var
from wrag.witness.search import Grounding


class SerialPolicy:
    def __init__(self, corpus, facts, weight: float = .65):
        if not 0 <= weight <= 1:
            raise ValueError("Conflict recency weight must be between 0 and 1")
        self.weight = weight
        self.serials = []
        self.families = defaultdict(list)
        self.records = {}
        self.source_versions = {}
        self.newest_source_for_serial = {}
        for passage in corpus.passages:
            self.records[passage.pid] = {
                int(m[1]): m[2].strip() for m in re.finditer(
                    r"\[MAB:(\d+)\] Fact: (.*?)(?=\n\[MAB:|$)", passage.text, re.S)}
        for index, fact in enumerate(facts):
            records = self.records.get(fact.pid, {})
            turn = re.fullmatch(r"MAB:(\d+)", fact.turn_id)
            number = int(turn[1]) if turn and int(turn[1]) in records else None
            if number is None and fact.statement:
                candidates = [n for n, body in records.items()
                              if fact.statement.strip().casefold() in body.casefold()]
                if len(candidates) == 1:
                    number = candidates[0]
            self.serials.append(number)
            family = (canonical_symbol(fact.subject), canonical_symbol(fact.relation))
            self.families[family].append(index)
        self._index_source_versions(facts)
        # Only competing versions get a non-neutral priority. No predicate is
        # declared functional, and no old proposition is silently discarded.
        self.freshness = np.full(len(facts), .5, dtype=np.float32)
        self.newest = {}
        for indices in self.families.values():
            numbers = sorted({n for i in indices for n in
                              self.source_versions.get(i, [self.serials[i]]) if n is not None})
            if len(numbers) < 2:
                continue
            ranks = {number: pos / (len(numbers) - 1) for pos, number in enumerate(numbers)}
            for index in indices:
                if self.serials[index] is not None:
                    self.freshness[index] = ranks[self.serials[index]]
                    self.newest[index] = numbers[-1]

    @staticmethod
    def _body(text):
        return " ".join(re.sub(r"^\d+\.\s*", "", text).split()).casefold()

    def _index_source_versions(self, facts):
        # Find exactly repeated source wording with only the cited object's
        # literal span replaced. All subject/predicate/qualifier wording stays
        # fixed. This recognizes an update even if OpenIE omitted its triple;
        # it never extracts an answer or consults a reference label.
        templates = defaultdict(lambda: defaultdict(list))
        bodies = {n: self._body(body) for records in self.records.values() for n, body in records.items()}
        for index, fact in enumerate(facts):
            number = self.serials[index]
            body = bodies.get(number, "")
            obj = " ".join(fact.object.split()).casefold()
            matches = list(re.finditer(r"(?<!\w)" + re.escape(obj) + r"(?!\w)", body)) if obj else []
            if len(matches) != 1:
                continue
            match = matches[0]
            prefix, suffix = body[:match.start()], body[match.end():]
            subject = " ".join(fact.subject.split()).casefold()
            if not subject or subject not in prefix + suffix:
                continue
            templates[prefix][suffix].append(index)
        matched = defaultdict(set)
        # Prefix dictionary lookups keep this near linear in source length,
        # instead of comparing every extracted fact with every source record.
        for number, body in bodies.items():
            ends = [0] + [m.end() for m in re.finditer(r"\s+", body)]
            for end in ends:
                for suffix, indices in templates.get(body[:end], {}).items():
                    if body.endswith(suffix) and len(body) > end + len(suffix):
                        for index in indices:
                            matched[index].add(number)
        for index, numbers in matched.items():
            self.source_versions[index] = sorted(numbers)
            if len(numbers) > 1:
                for number in numbers:
                    self.newest_source_for_serial[number] = max(
                        self.newest_source_for_serial.get(number, number), max(numbers))

    def version_label(self, number):
        newest = self.newest_source_for_serial.get(number)
        if newest is None:
            return f"[source serial {number}]"
        if number < newest:
            return f"[source serial {number}; OLDER VERSION; newer matching record {newest}]"
        return f"[source serial {number}; LATEST MATCHING VERSION]"

    def blend(self, relevance, indices):
        # Multiplicative prior: a very recent unrelated fact cannot acquire
        # relevance solely from its serial, unlike an additive global bonus.
        return np.asarray(relevance) * ((1 - self.weight) +
                                       self.weight * self.freshness[list(indices)])

    def witness_score(self, indices):
        # A recent last hop must not conceal a stale first/intermediate hop.
        return float(min((self.freshness[i] for i in indices), default=.5))

    def describe(self):
        return {"version": "ordinal-subject-relation-v2", "weight": self.weight,
                "clock": "validated_source_serial", "old_facts_deleted": False,
                "known_serials": sum(n is not None for n in self.serials),
                "unknown_serials": sum(n is None for n in self.serials),
                "competing_version_facts": len(self.newest),
                "source_version_templates": len(self.source_versions),
                "versions_found_only_in_source": len({n for numbers in self.source_versions.values()
                                                       for n in numbers} - set(self.serials)),
                "chain_recency": "minimum_across_all_hops"}


class SerialPlanner(MultiOriginPlanner):
    def __init__(self, retriever, question):
        super().__init__(retriever, question)
        self.contract.time_weight = retriever.serial_policy.weight
        self.contract.pending.append("same_subject_relation_versions_require_scope_and_conflict_check")

    def similarities(self):
        relevance = super().similarities()
        # This ordering is used for starts, seed proposals and every expansion,
        # before execution/beam limits; recency is not merely a final tie-break.
        self.raw_similarity = relevance
        return self.r.serial_policy.blend(np.clip(relevance, 0, 1), range(len(relevance)))

    def temporal_score(self, candidate, witness):
        return self.r.serial_policy.witness_score(witness.facts)

    def score_cheap(self, candidate):
        super().score_cheap(candidate)
        policy = self.r.serial_policy
        freshness = max((policy.witness_score(w.facts) for w in candidate.witnesses), default=.5)
        candidate.score *= (1 - policy.weight) + policy.weight * freshness

    def rerank(self, candidates):
        ranked, scorer = super().rerank(candidates)
        results = []
        weight = self.r.serial_policy.weight
        for _, candidate, witness, semantic, freshness in ranked:
            checks = self.checks(candidate, witness)
            grade = int(checks['projection_compatible']) + int(checks['required_conjunction'])
            structural = .5 * checks['owner_coverage'] + .5 * checks['predicate_coverage']
            relevance = .6 * semantic + .4 * structural
            score = grade + relevance * ((1 - weight) + weight * freshness)
            results.append((score, candidate, witness, semantic, freshness))
        results.sort(key=lambda row: (-row[0], row[1].signature(), row[2].facts))
        return results, scorer

    def execute(self, query):
        signature = tuple((a.subject, a.relation, a.object, a.time) for a in query.atoms)
        if signature in self.memo:
            return self.memo[signature]
        groundings = []
        policy = self.r.serial_policy
        for pos, atom in enumerate(query.atoms):
            found = []
            for index in self.grounding_ids(atom):
                fact = self.memory.facts[index]
                if any(not is_var(term) and self.node(eid) not in
                       self.entity_nodes.get(canonical_symbol(term), set())
                       for term, eid in ((atom.subject, fact.subj_id), (atom.object, fact.obj_id))):
                    continue
                if not self.temporal_ok(index, pos == 0):
                    continue
                # Source order is independent of event dates, for EVERY atom.
                found.append(Grounding(index, (1 - policy.weight) +
                                       policy.weight * float(policy.freshness[index]), match=1.0))
            found.sort(key=lambda g: (-g.score, -float(self.raw_similarity[g.fact_index]), g.fact_index))
            limit = self.cfg.candidates_per_atom
            if limit and len(found) > limit:
                self.truncations.append("groundings")
                found = found[:limit]
            groundings.append(found)
        result = self.searcher.join(query, groundings)
        self.executions += 1
        self.truncations.extend(result.truncations)
        self.memo[signature] = result.witnesses
        return result.witnesses


class SerialSourceIndex(SourceIndex):
    """Prioritize recent original records even when extraction missed them."""
    def __init__(self, dated, policy):
        super().__init__(dated)
        self.policy = policy

    def rank(self, terms_, owner="", before=None, temporal=None):
        ranked = super().rank(terms_, owner, before, temporal=None)
        if not ranked or not self.policy.weight:
            return ranked
        top = ranked[0][0]
        # Preserve relevance: do not promote recent peripheral source turns.
        eligible = [(score, index) for score, index in ranked if score >= .5 * top]
        serials = {}
        for _, index in eligible:
            pid, _, turn = self.turns[index]
            match = re.fullmatch(r"MAB:(\d+)", turn.turn_id)
            if match and int(match[1]) in self.policy.records.get(pid, {}):
                serials[index] = int(match[1])
        values = sorted(set(serials.values()))
        if len(values) < 2:
            return ranked
        ranks = {n: pos / (len(values) - 1) for pos, n in enumerate(values)}
        w = self.policy.weight
        promoted = [((1 - w) * score / top + w * ranks.get(serials.get(index), .5), index)
                    for score, index in eligible]
        promoted.sort(key=lambda row: (-row[0], row[1]))
        admitted = {index for _, index in eligible}
        # Tail remains available, after the relevance-gated candidate group.
        return promoted + [(score / top * (1 - w), index) for score, index in ranked if index not in admitted]


class SerialWitnessRetriever(WitnessRAGRetriever):
    def index(self):
        super().index()
        self.serial_policy = SerialPolicy(self.corpus, self.memory.facts, self.conflict_recency_weight)
        self._local_source_index = SerialSourceIndex(self.dated, self.serial_policy)
        self._local_source_index_identity = (id(self.dated), len(self.memory.facts))

    def _retrieve_proof(self, question, k, pool_pids, pool_scores):
        result = retrieve_local_v2(self, question, k, pool_pids, pool_scores, planner_class=SerialPlanner)
        result.diagnostics["ordinal_recency"] = self.serial_policy.describe()
        return result

    def _rerank_facts(self, question, order, score, priority):
        from wrag.witness.rerank import get_reranker
        pool = order[:self.ctx.run.witness.fact_rerank_pool]
        texts = [(self.memory.facts[i].verbalize() + " " + self.dated.excerpt(i, window=0))[:700]
                 for i in pool]
        relevance = get_reranker(self.ctx.run.witness.fact_rerank).score(question.question, texts)
        blended = self.serial_policy.blend(relevance, pool)
        ranked = sorted(range(len(pool)), key=lambda j: (-priority.get(pool[j], 0), -blended[j], j))
        return [pool[j] for j in ranked] + order[len(pool):], len(pool)
