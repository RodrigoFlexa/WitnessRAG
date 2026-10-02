"""Bounded, bottom-up program search over the existing conversational graph.

No LLM, new facts, summaries, gold labels or annotated decompositions. Programs
are directed conjunctive queries, executed by WitnessSearcher with indexed
groundings. The graph establishes a join, NOT fidelity to the question. A local
cross-encoder ranks program+witness pairs; the reader checks the original text.
This initial semantic scorer is a pretrained relevance model, not a trained
semantic parser. Its ranking and the bounded search remain approximate.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace

import numpy as np

from wrag.methods.base import RetrievalResult
from wrag.util import canonical_symbol
from wrag.witness.plan import ProofPlan
from wrag.witness.query import Atom, ConjunctiveQuery, is_var, looks_like_answer_set
from wrag.witness.scoring import Period, proximity
from wrag.witness.search import Grounding, Witness
from wrag.witness.timeline import Interval, parse_anchor, parse_dates

_STOP = set("a an the is are was were do does did has have had who what which where when how why in on at to of for and or with from by that this it his her their as would could should about been before after during first last most current currently latest recently many long".split())


def terms(text):
    return set(re.findall(r"[^\W\d_]+", text.casefold())) - _STOP


@dataclass
class Contract:
    question: str
    operation: str
    expected_type: str
    period: Period
    temporal_side: str = ""
    reference_fact: int | None = None
    pending: list[str] = field(default_factory=list)
    time_weight: float = 0.0

    def to_dict(self):
        return {"operation": self.operation, "expected_type": self.expected_type,
                "period": self.period.to_dict(), "temporal_side": self.temporal_side,
                "reference_fact": self.reference_fact, "pending": self.pending,
                "time_weight": self.time_weight,
                "semantic_status": "hypothesis_checked_by_reader"}


def contract(question: str, dated) -> Contract:
    """Small grammatical inventory; unresolved clauses stay in the original q.

    No benchmark category is inspected. A calendar restriction is conservative
    and applied only to the first (anchored event) atom; other temporal scopes
    remain reader obligations. Session timestamps never become event constraints.
    """
    low = question.casefold()
    operation = ("count" if re.search(r"\bhow many\b", low) else
                 "duration" if re.search(r"\bhow long\b|\bhow much time\b", low) else
                 "date" if re.search(r"^when\b|\bwhat (?:date|year|month)\b", low) else
                 "first" if re.search(r"\bfirst\b|\bearliest\b", low) else
                 "last" if re.search(r"\blatest\b|\bmost recent\b|\blast (?:job|event|time)\b", low) else
                 "compare" if re.search(r"\bmore than\b|\bless than\b|\bcompare\b", low) else
                 "premises" if re.search(r"^why\b|^would\b|^could\b|^should\b|\blikely\b", low) else
                 "set" if looks_like_answer_set(question) else "value")
    expected = ("date" if operation == "date" else "number" if operation in {"count", "duration"} else
                "place" if re.search(r"^where\b|\b(?:city|country|place|town)\b", low) else
                "person" if re.search(r"^who\b", low) else "other")
    if not getattr(dated, "model_time", True):
        # The reader still sees the unmodified question and literal sources;
        # the retriever no longer compiles temporal operators or anchors.
        if operation in {"date", "duration", "first", "last"}:
            operation = "value"
        return Contract(question, operation, expected, Period("none", None, ""))
    now = dated.last
    period = Period("now", Interval(now, now) if now else None, "now")
    # Strip directional words so parse_anchor does not interpret an unqualified
    # 'before DATE' as the special expression 'the week before DATE'.
    cleaned = re.sub(r"\b(?:before|after|prior to|since)\b", " ", question, flags=re.I)
    anchor = parse_anchor(cleaned, now=now)
    side = ""
    if anchor:
        dates = parse_dates(cleaned)
        if len(dates) >= 2 and re.search(r"\bbetween\b|\bfrom\b", low):
            anchor = Interval(min(dates), max(dates), question)
        period = Period("window", anchor, anchor.text or cleaned)
        side = ("before" if re.search(r"\bbefore\b|\bprior to\b", low) else
                "after" if re.search(r"\bafter\b|\bsince\b", low) else "within")
    elif operation == "first" and dated.first:
        period = Period("start", Interval(dated.first, dated.first), "start")
    elif re.search(r"\bcurrently\b|\bnow\b|\brecently\b|\blatest\b|\bmost recent\b", low):
        side = "recent"
    pending = []
    if re.search(r"\bbefore\b|\bafter\b|\bsince\b|\bprior to\b", low) and not anchor:
        pending.append("event_relative_anchor_unresolved")
    if operation in {"set", "count", "first", "last"}:
        pending.append("retrieval_does_not_prove_global_completeness")
    if operation in {"duration", "compare", "premises"}:
        pending.append("reader_must_apply_operation_to_supported_premises")
    weight = .3 if period.kind != "now" or side == "recent" else 0.0
    return Contract(question, operation, expected, period, side, pending=pending, time_weight=weight)


@dataclass
class Candidate:
    query: ConjunctiveQuery
    witnesses: list[Witness]
    score: float = 0.0
    semantic: float = 0.0
    coverage: float = 0.0
    temporal: float = 0.0

    def signature(self):
        return tuple((a.subject, a.relation, a.object, a.time) for a in self.query.atoms)


class LocalPlanner:
    def __init__(self, retriever, question):
        self.r = retriever
        self.memory, self.dated = retriever.memory, retriever.dated
        self.cfg = retriever.ctx.run.witness
        self.text = question.question   # deliberately never retain the Question/gold
        hint = getattr(retriever, "_reflection_search_hint", "")
        self.search_text = self.text + ("\nSearch focus: " + hint if hint else "")
        self.contract = contract(self.text, self.dated)
        self.tokens = terms(self.text)
        self.sim = self.similarities()
        # The join implementation is shared, but its score/cache state is private.
        self.searcher = copy.copy(retriever.searcher)
        self.searcher.cfg = replace(self.cfg, answer_set=True, max_witnesses=min(8, self.cfg.max_witnesses or 8))
        self.searcher.portfolio_cache = None
        self.searcher.clear_scoring()
        self.by_relation, self.incident, self.entity_nodes = {}, {}, {}
        self.available = set()
        for i, f in enumerate(self.memory.facts):
            if not self.searcher._permitted(i):
                continue
            self.available.add(i)
            self.by_relation.setdefault(canonical_symbol(f.relation), []).append(i)
            for eid, surface in ((f.subj_id, f.subject), (f.obj_id, f.object)):
                node = self.node(eid)
                self.entity_nodes.setdefault(canonical_symbol(surface), set()).add(node)
                self.incident.setdefault(node, []).append(i)
        self.anchors = self.find_anchors()
        self.executions = 0
        self.rejected_temporal = 0
        self.memo = {}
        self.truncations = []
        self.generation_budget = self.cfg.local_plan_candidates * self.cfg.local_plan_depth * 4
        if self.cfg.study_ablation != "no-time-model":
            self.resolve_event_anchor()
        level = "strong" if self.contract.period.kind != "now" or self.contract.temporal_side == "recent" else "normal"
        self.contract.time_weight = self.r._weight_levels().weights(level, "none").time
        if self.cfg.study_ablation in {"no-time-reference", "no-time-model"}:
            self.contract.time_weight = 0.0

    def node(self, eid):
        return self.searcher._identity(eid) if eid >= 0 else -1

    def similarities(self):
        self.vector = self.r.ctx.embedder.encode([self.search_text])[0]
        return self.memory.fact_vectors @ self.vector if self.memory.fact_vectors.size else np.zeros(len(self.memory.facts))

    def grounding_ids(self, atom):
        return self.by_relation.get(canonical_symbol(atom.relation), [])

    def find_anchors(self):
        q = canonical_symbol(self.text)
        matches = []
        for name, nodes in self.entity_nodes.items():
            if len(name) >= 3 and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", q):
                matches.append((name, nodes))
        # Prefer full names over overlapping pieces; keep independent mentions.
        picked = []
        for name, nodes in sorted(matches, key=lambda x: (-len(x[0]), x[0])):
            if not any(name in full for full, _ in picked):
                picked.append((name, nodes))
        if picked:
            return sorted({n for _, nodes in picked[:6] for n in nodes if n >= 0})
        # Unknown entity/no named entity: factual seed hypotheses, never invented
        # entities. Diagnostics distinguish this from literal entity anchoring.
        order = sorted(self.available, key=lambda i: (-self.sim[i], i))[:3]
        return list(dict.fromkeys(self.node(self.memory.facts[i].subj_id) for i in order))

    def resolve_event_anchor(self):
        if "event_relative_anchor_unresolved" not in self.contract.pending:
            return
        m = re.search(r"\b(before|after|since|prior to)\s+(.+?)[?.]?$", self.text, re.I)
        if not m:
            return
        clause = terms(m[2])
        matches = []
        for i in self.available:
            f = self.memory.facts[i]
            if self.dated.fact_time_source[i] != "expressao" or not self.dated.fact_interval[i]:
                continue
            if self.node(f.subj_id) not in self.anchors:
                continue
            overlap = clause & terms(f.verbalize())
            if len(overlap) >= min(2, len(clause)) and clause:
                matches.append(i)
        # Ambiguous event references are not silently resolved to the newest one.
        dates = {self.dated.fact_interval[i] for i in matches}
        if len(dates) == 1:
            index = max(matches, key=lambda i: float(self.sim[i]))
            interval = self.dated.fact_interval[index]
            self.contract.period = Period("window", interval, m[2])
            self.contract.temporal_side = "before" if m[1].lower() in {"before", "prior to"} else "after"
            self.contract.reference_fact = index
            self.contract.pending.remove("event_relative_anchor_unresolved")
            self.contract.pending.append("event_anchor_is_a_lexical_hypothesis")

    def temporal_ok(self, fid, scoped):
        c = self.contract
        if not scoped or c.temporal_side not in {"within", "before", "after"}:
            return True
        fact = self.memory.facts[fid]
        # A session date or an ongoing state's start is not an event's exact
        # occurrence/validity interval. Preserve it as uncertain for the reader.
        if (self.dated.fact_time_source[fid] != "expressao" or fact.kind in {"ongoing", "said"}):
            return True
        value, anchor = self.dated.fact_interval[fid], c.period.interval
        if value is None or anchor is None:
            return True
        ok = (value.end < anchor.start if c.temporal_side == "before" else
              value.start > anchor.end if c.temporal_side == "after" else
              value.start <= anchor.end and value.end >= anchor.start)
        if not ok:
            self.rejected_temporal += 1
        return ok

    def execute(self, query):
        signature = tuple((a.subject, a.relation, a.object, a.time) for a in query.atoms)
        if signature in self.memo:
            return self.memo[signature]
        groundings = []
        scale = self.r.scorer.scale(self.contract.period)
        for pos, atom in enumerate(query.atoms):
            found = []
            for i in self.grounding_ids(atom):
                f = self.memory.facts[i]
                compatible = True
                for term, eid in ((atom.subject, f.subj_id), (atom.object, f.obj_id)):
                    if not is_var(term) and self.node(eid) not in self.entity_nodes.get(canonical_symbol(term), set()):
                        compatible = False
                if not compatible or not self.temporal_ok(i, pos == 0):
                    continue
                recency = proximity(self.dated.fact_interval[i], self.contract.period, scale) if pos == 0 else 0.0
                weight = self.contract.time_weight if pos == 0 else 0.0
                found.append(Grounding(i, (1-weight) + weight*recency, match=1.0))
            # Chronological preference precedes the cut; facts cannot win by file
            # order. Directed relation/constant eligibility is already checked.
            found.sort(key=lambda g: (-g.score, -float(self.sim[g.fact_index]), g.fact_index))
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

    def candidate(self, atoms):
        if self.executions >= self.generation_budget:
            self.truncations.append("execution_budget")
            return None
        operation = self.contract.operation
        query = ConjunctiveQuery(atoms=atoms, answer_var="t" if operation == "date" else "x",
                                 aggregation="set" if operation in {"set", "count"} else "none",
                                 expected_type=self.contract.expected_type, source="local-program-search",
                                 fallback=self.text, conditions=[self.text])
        witnesses = self.execute(query)
        if not witnesses:
            return None
        candidate = Candidate(query, witnesses)
        self.score_cheap(candidate)
        return candidate

    def score_cheap(self, c):
        # Rank partial programs by coverage as well as evidence relevance. A
        # short program must not crowd out the prefix of a necessary chain.
        best = -1.0
        for w in c.witnesses:
            ids = list(w.facts)
            words = set().union(*(terms(self.memory.facts[i].verbalize()) for i in ids))
            cov = len(self.tokens & words) / max(1, len(self.tokens))
            sem = max(0.0, float(np.mean(self.sim[ids])))
            temp = self.temporal_score(c, w)
            value = .65*sem + .30*cov + .05*temp*self.contract.time_weight - .01*(len(c.query.atoms)-1)
            if value > best:
                best, c.coverage, c.temporal = value, cov, temp
        c.score = best

    def temporal_score(self, candidate, witness):
        if self.cfg.study_ablation in {"no-time-reference", "no-time-model"}:
            return 0.0
        # Witness.facts is sorted by source index, not by atom position.
        atom = candidate.query.atoms[0]
        ids = []
        for i in witness.facts:
            f = self.memory.facts[i]
            if canonical_symbol(f.relation) != canonical_symbol(atom.relation):
                continue
            compatible = True
            for term, eid in ((atom.subject, f.subj_id), (atom.object, f.obj_id)):
                value = witness.bindings.get(term[1:], "") if is_var(term) else term
                if self.node(eid) not in self.entity_nodes.get(canonical_symbol(value), set()):
                    compatible = False
            if compatible:
                ids.append(i)
        values = [proximity(self.dated.fact_interval[i], self.contract.period,
                            self.r.scorer.scale(self.contract.period)) for i in ids]
        return max(values, default=0.0)

    def seeds(self):
        candidates, seen = [], set()
        for anchor in self.anchors:
            incident = sorted(set(self.incident.get(anchor, [])), key=lambda i: (-self.sim[i], i))
            for i in incident[:self.cfg.local_plan_candidates]:
                f = self.memory.facts[i]
                if self.node(f.subj_id) == anchor:
                    atom = Atom(f.relation, f.subject, "?x")
                else:
                    atom = Atom(f.relation, "?x", f.object)
                if self.contract.operation == "date":
                    atom.time = "?t"
                key = (atom.subject, atom.relation, atom.object)
                if key in seen:
                    continue
                seen.add(key)
                candidate = self.candidate([atom])
                if candidate:
                    candidates.append(candidate)
        return candidates

    def expand(self, c):
        out = []
        terminal = "?x"
        intermediate = f"?y{len(c.query.atoms)}"
        prefix = [replace(a, subject=intermediate if a.subject == terminal else a.subject,
                          object=intermediate if a.object == terminal else a.object) for a in c.query.atoms]
        bound_nodes = {n for w in c.witnesses for n in self.entity_nodes.get(canonical_symbol(w.bindings.get("x", "")), set())}
        seen = set()
        incident = {i for n in bound_nodes for i in self.incident.get(n, [])}
        for i in sorted(incident, key=lambda i: (-self.sim[i], i))[:self.cfg.local_plan_candidates]:
            f = self.memory.facts[i]
            directions = []
            if self.node(f.subj_id) in bound_nodes:
                directions.append(Atom(f.relation, intermediate, "?x"))
            if self.node(f.obj_id) in bound_nodes:
                directions.append(Atom(f.relation, "?x", intermediate))
            for atom in directions:
                key = (atom.subject, atom.relation, atom.object)
                if key in seen:
                    continue
                seen.add(key)
                candidate = self.candidate(prefix + [atom])
                if candidate:
                    # Reject cycles that merely reuse a seed fact backwards.
                    candidate.witnesses = [w for w in candidate.witnesses if len(w.facts) == len(candidate.query.atoms)]
                    if candidate.witnesses:
                        self.score_cheap(candidate)
                        out.append(candidate)
        return out

    def diverse(self, candidates, limit):
        ordered = sorted(candidates, key=lambda c: (-c.score, c.signature()))
        out, signatures, families = [], set(), set()
        for c in ordered:
            family = tuple(a.relation for a in c.query.atoms)
            if c.signature() not in signatures and family not in families:
                out.append(c); signatures.add(c.signature()); families.add(family)
                if len(out) == limit:
                    return out
        for c in ordered:
            if c.signature() not in signatures:
                out.append(c); signatures.add(c.signature())
                if len(out) == limit:
                    break
        return out

    def build(self):
        seeds = self.seeds()
        frontier = self.diverse(seeds, self.cfg.local_plan_beam)
        all_candidates = {c.signature(): c for c in seeds}
        depth = min(self.cfg.max_atoms, self.cfg.local_plan_depth)
        for _ in range(1, depth):
            extensions = [n for c in frontier for n in self.expand(c)]
            # Intersections share the same answer variable across independently
            # anchored predicates. No union of incompatible interpretations.
            for a in frontier:
                for b in seeds:
                    if (a.signature() >= b.signature() or
                            len(a.query.atoms)+len(b.query.atoms) > depth or
                            set(a.query.constants()) == set(b.query.constants())):
                        continue
                    if set(w.bindings.get("x") for w in a.witnesses) & set(w.bindings.get("x") for w in b.witnesses):
                        merged = self.candidate(a.query.atoms + b.query.atoms)
                        if merged:
                            extensions.append(merged)
            new = [c for c in extensions if c.signature() not in all_candidates]
            for c in new:
                all_candidates[c.signature()] = c
            if not new:
                break
            frontier = self.diverse(new, self.cfg.local_plan_beam)
        candidates = self.diverse(list(all_candidates.values()), self.cfg.local_plan_candidates)
        self.generated_total = len(all_candidates)
        if len(all_candidates) > len(candidates):
            self.truncations.append("program_candidates")
        return candidates

    def rerank(self, candidates):
        pairs = [(c, w) for c in candidates for w in c.witnesses[:2]]
        texts = []
        for c, w in pairs:
            program = " AND ".join(f"{a.relation}({a.subject}, {a.object})" for a in c.query.atoms)
            evidence = "\n".join(self.memory.facts[i].verbalize() for i in w.facts)
            # Direct quotes keep missing qualifiers and modality visible. No
            # graph answer is advertised as the answer to the user's question.
            sources = "\n".join(self.dated.excerpt(i)[:360] for i in w.facts)
            texts.append(f"Directed retrieval program: {program}\nEvidence:\n{evidence}\nSource:\n{sources}")
        if self.cfg.fact_rerank and pairs:
            from wrag.witness.rerank import get_reranker
            semantic = get_reranker(self.cfg.fact_rerank).score(self.text, texts)
            scorer = "cross_encoder_relevance_experimental"
        else:
            semantic = [max(0.0, float(np.mean(self.sim[list(w.facts)]))) for _, w in pairs]
            scorer = "bi_encoder_fallback"
        results = []
        for (c, w), sem in zip(pairs, semantic):
            temp = self.temporal_score(c, w)
            weight = self.contract.time_weight
            score = (1-weight)*float(sem) + weight*temp - .015*(len(c.query.atoms)-1)
            results.append((score, c, w, float(sem), temp))
        results.sort(key=lambda x: (-x[0], x[1].signature(), x[2].facts))
        return results, scorer


def retrieve_local(r, question, k, pool_pids, pool_scores):
    cfg = r.ctx.run.witness
    if cfg.local_plan_version == "lite":
        from wrag.witness.local_lite import retrieve_lite
        return retrieve_lite(r, question, k, pool_pids, pool_scores)
    if cfg.local_plan_version == "v2":
        from wrag.witness.local_plans_v2 import retrieve_local_v2
        return retrieve_local_v2(r, question, k, pool_pids, pool_scores)
    if cfg.local_plan_version != "v1":
        raise ValueError("Unknown local plan version")
    if cfg.fact_delivery != "facts" or cfg.summary_reflection or cfg.plan_router or cfg.multiplan_portfolio:
        raise ValueError("Local plans require facts only, no summaries or LLM router")
    if min(cfg.local_plan_beam, cfg.local_plan_candidates, cfg.local_plan_depth, cfg.local_plan_keep) < 1:
        raise ValueError("Local plan budgets must be positive")
    planner = LocalPlanner(r, question)
    candidates = planner.build()
    ranked, scorer = planner.rerank(candidates)
    selected, signatures = [], set()
    covered_facts = set()
    for item in ranked:
        _, candidate, witness, _, _ = item
        # Preserve a small portfolio; for lists a second member of the same
        # query is useful, while value queries prefer different logical plans.
        key = (candidate.signature(), witness.answer) if planner.contract.operation in {"count", "set"} else candidate.signature()
        if key in signatures:
            continue
        if set(witness.facts) <= covered_facts:
            continue  # same evidence must not occupy every portfolio slot
        signatures.add(key)
        selected.append(item)
        covered_facts.update(witness.facts)
        if len(selected) >= cfg.local_plan_keep:
            break
    packages = [list(item[2].facts) for item in selected]
    if planner.contract.reference_fact is not None:
        packages = [list(dict.fromkeys(ids + [planner.contract.reference_fact])) for ids in packages]
    # _fact_context atomically preserves each admitted chain under the fact
    # budget. These are retrieval candidates, not textually confirmed proofs.
    chosen_pairs = [(ProofPlan(query=c.query, valid=True), w) for _, c, w, _, _ in selected]
    accepted = {"selecionadas": chosen_pairs, "pacotes": packages}
    facts, summary, info = r._fact_context(question, None, accepted, None, pool_pids[:k])
    if summary:
        raise AssertionError("Local retrieval unexpectedly generated a summary")
    delivered = set(info.get("indices", []))
    retained = [item for item, ids in zip(selected, packages) if set(ids) <= delivered]
    source_ids = [i for ids in packages if set(ids) <= delivered for i in ids]
    source_ids += [i for i in info.get("indices", []) if i not in source_ids][:6]
    # Short original turns replace generated summaries. An incomplete excerpt
    # does not invalidate the graph package; its facts are delivered separately.
    excerpts, turn_ids = r._dialogue_block([(i,) for i in dict.fromkeys(source_ids)], 0)
    text = "Retrieved facts (graph joins are evidence candidates, not verified answers):\n" + facts
    if excerpts:
        text += "\n\nOriginal source turns:\n" + excerpts
    pending = planner.contract.pending + ["question_semantics_and_qualifiers_require_reader_check"]
    text += "\n\nPending checks: " + "; ".join(pending)
    diagnostics = {
        "controlador": "local-multiplan-v1", "rota": "local", "classe_prova": "candidata_nao_verificada",
        "motivo_parada": "planos_locais_entregues" if retained else "recuperacao_local_sem_pacote",
        "planejamento": {"chamadas": 0, "chamadas_plano": 0, "chamadas_verificacao": 0,
                         "planos_distintos": len(candidates), "replanejamentos": 0},
        "local_plans": {"contract": planner.contract.to_dict(), "anchors": planner.anchors,
                        "generated": planner.generated_total, "reranked_plans": len(candidates),
                        "executed": planner.executions,
                        "ranked_pairs": len(ranked), "scorer": scorer,
                        "temporal_rejections": planner.rejected_temporal,
                        "truncations": sorted(set(planner.truncations)),
                        "selected": [{"query": c.query.to_dict(), "facts": list(w.facts),
                                      "answer_binding": w.answer, "score": round(score, 6),
                                      "semantic": round(sem, 6), "temporal": round(temp, 6),
                                      "verified": False} for score, c, w, sem, temp in retained],
                        "top_candidates": [{"query": c.query.to_dict(), "facts": list(w.facts),
                                            "score": round(score, 6), "semantic": round(sem, 6),
                                            "temporal": round(temp, 6)}
                                           for score, c, w, sem, temp in ranked[:8]],
                        "source_turns": turn_ids},
        "fatos_entregues": info,
        "trechos_extras": [{"title": "Local logical retrieval", "text": text}],
        "leitura_fatos": "bitemporal" if cfg.fact_time == "both" else True,
    }
    return RetrievalResult(pids=pool_pids[:k], scores=pool_scores[:k], diagnostics=diagnostics)
