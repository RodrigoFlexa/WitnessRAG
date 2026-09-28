"""Lexical multi-plans and deterministic support inspection, with no inference model.

This experimental online method reuses a previously constructed graph. Its
construction cost is NOT zero. A support check establishes structural coverage,
not textual entailment or completeness of the question's interpretation.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from wrag.util import canonical_symbol
from wrag.witness.local_contract import stems
from wrag.witness.local_plans_v2 import MultiOriginPlanner, retrieve_local_v2
from wrag.witness.query import is_var


class LexicalIndex:
    """Posting-list BM25 over literal text; scores are relative, not probabilities."""

    def __init__(self, texts):
        bags = [Counter(word for token in re.findall(r'[^\W\d_]+', text)
                        for word in stems(token)) for text in texts]
        self.lengths = [sum(b.values()) for b in bags]
        self.average = sum(self.lengths) / max(1, len(bags))
        self.postings = defaultdict(list)
        for i, bag in enumerate(bags):
            for term, frequency in bag.items():
                self.postings[term].append((i, frequency))
        self.idf = {term: math.log(1 + (len(bags) - len(rows) + .5) / (len(rows) + .5))
                    for term, rows in self.postings.items()}

    def scores(self, text, normalized=False):
        values = np.zeros(len(self.lengths), dtype=np.float32)
        for term in stems(text):
            for i, frequency in self.postings.get(term, ()):
                denom = frequency + 1.2 * (.25 + .75 * self.lengths[i] / max(1, self.average))
                values[i] += self.idf[term] * frequency * 2.2 / denom
        if normalized and len(values) and values.max() > 0:
            values /= values.max()
        return values


class LexicalMemory:
    def __init__(self, r):
        self.facts = LexicalIndex([f.verbalize() + '\n' + r.dated.excerpt(i, window=0)
                                  for i, f in enumerate(r.memory.facts)])
        self.pids = [p.pid for p in r.corpus.passages]
        self.passages = LexicalIndex([p.full for p in r.corpus.passages])

    def search(self, text, k):
        values = self.passages.scores(text)
        order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))[:k]
        return [self.pids[i] for i in order], [float(values[i]) for i in order]


class FrozenOnlyEmbedder:
    """Validate a frozen checkpoint's embedding identity without loading weights.

    Even an embedding cache hit is disallowed: this tests absence of the online
    embedding path rather than an accidentally warm model cache.
    """
    dim = 0
    name = 'frozen-only'

    def __init__(self, key):
        self.key = key

    def cache_key(self):
        return self.key

    def encode(self, *args, **kwargs):
        raise AssertionError('Lite called an embedding model')


class LitePlanner(MultiOriginPlanner):
    def similarities(self):
        # No vector, embedding call, dense matrix multiplication or reranker.
        return self.r._local_lexical.facts.scores(self.text, normalized=True)

    def grounding_ids(self, atom):
        # Start from the smallest applicable index instead of scanning an entire
        # relation when a constant already determines the candidate neighbourhood.
        relation = self.by_relation.get(canonical_symbol(atom.relation), [])
        buckets = [relation]
        for term in (atom.subject, atom.object):
            if not is_var(term):
                nodes = self.entity_nodes.get(canonical_symbol(term), ())
                buckets.append({i for n in nodes for i in self.incident.get(n, ())})
        eligible = set(min(buckets, key=len))
        for bucket in buckets:
            eligible.intersection_update(bucket)
        return sorted(eligible)

    def rerank(self, candidates):
        ranked, _ = super().rerank(candidates)
        return ranked, 'lexical_structural_temporal'


def inspect_support(planner, candidate, witness, delivered):
    """Mechanical reflector: inspect each atom, roles, provenance and packaging.

    Never consumes gold, assumes source entailment, or certifies that the bounded
    question grammar captured every condition. Unknowns remain explicit.
    """
    missing = []
    unknown = list(planner.contract.pending)
    checks = planner.checks(candidate, witness)
    atoms = [planner.bound_fact_ids(a, witness) for a in candidate.query.atoms]
    if not atoms or any(not ids for ids in atoms):
        missing.append('atom_without_matching_fact')
    if not checks['projection_compatible']:
        missing.append('answer_role_or_scope_incompatible')
    if not checks['required_conjunction']:
        missing.append('participant_conjunction_not_supported')
    if not set(witness.facts) <= set(delivered):
        missing.append('witness_not_fully_delivered')
    corpus_pids = set(planner.r.corpus.pids)
    if any(planner.memory.facts[i].pid not in corpus_pids for i in witness.facts):
        missing.append('source_provenance_missing')
    if checks['predicate_coverage'] < 1:
        unknown.append('requested_predicate_or_qualifier_not_fully_matched')
    if planner.contract.temporal_side in {'within', 'before', 'after'}:
        for i in planner.event_fact_ids(candidate, witness):
            if not planner.temporal_ok(i, True):
                missing.append('event_outside_requested_time_scope')
            elif planner.dated.fact_time_source[i] != 'expressao':
                unknown.append('event_time_supported_only_by_session_or_unknown')
    if re.search(r'\b(?:not|never|except|excluding|unless)\b', planner.text, re.I):
        unknown.append('negation_or_exclusion_requires_semantic_check')
    # Structural identity may itself have been inferred during offline indexing.
    if planner.cfg.grounding_mode != 'exact':
        unknown.append('entity_identity_inherits_offline_graph_clusters')
    unknown += ['question_interpretation_not_verified', 'source_entailment_not_verified']
    return {'status': 'missing_support' if missing else 'structural_support_only',
            'complete_witness_delivered': set(witness.facts) <= set(delivered),
            'atom_facts': atoms, 'checks': checks,
            'missing': sorted(set(missing)), 'unknown': sorted(set(unknown)),
            'semantic_sufficiency': 'undetermined', 'generative_calls': 0}


def retrieve_lite(r, question, k, pool_pids, pool_scores):
    if r.ctx.run.witness.fact_rerank:
        raise ValueError('Lite must disable --fact-rerank; use --fact-rerank ""')
    # Reuse the same packaging, role guards and original-source rescue as v2.
    # Capture the query-local planner without keeping gold-bearing Question data.
    planners = []
    class InspectedPlanner(LitePlanner):
        def __init__(self, retriever, q):
            super().__init__(retriever, q)
            planners.append(self)
    result = retrieve_local_v2(r, question, k, pool_pids, pool_scores, InspectedPlanner)
    planner = planners[0]
    diag = result.diagnostics
    local = diag['local_plans']
    local['version'] = 'lite'
    diag['controlador'] = 'local-multiplan-lite'
    by_signature = {c.signature(): c for c in planner._all.values()}
    inspections = []
    for row in local['selected']:
        signature = tuple((a['subject'], a['relation'], a['object'], a.get('time', ''))
                          for a in row['query']['atoms'])
        c = by_signature[signature]
        w = next(w for w in c.witnesses if list(w.facts) == row['facts'] and w.answer == row['answer_binding'])
        inspections.append(inspect_support(planner, c, w, diag['fatos_entregues']['indices']))
    diag['support_reflector'] = {'mode': 'deterministic', 'generative_calls': 0,
                                'checks': inspections,
                                'semantic_sufficiency': 'undetermined',
                                'status': 'structural_candidates' if inspections else 'no_structural_witness'}
    unresolved = sorted({p for report in inspections for p in report['unknown']
                         if p not in {'question_interpretation_not_verified', 'source_entailment_not_verified'}})
    # One compact instruction, never a second generative reflection. The reader
    # still performs inference where justified and emits only its short answer.
    diag['trechos_extras'][0]['text'] += ('\nSupport inspection: graph joins checked; semantic sufficiency unresolved. '
        + ('Unresolved checks: ' + '; '.join(unresolved) + '.' if unresolved else
           'Verify the full question against the quoted sources before answering.'))
    diag['online_inference'] = {'llm_calls': 0, 'query_embedding_calls': 0, 'cross_encoder_pairs': 0,
                               'memory_construction_excluded': True}
    return result
