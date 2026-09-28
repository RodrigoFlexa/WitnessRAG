"""Lite uses no inference models, even when all caches are warm."""
from dataclasses import replace

import pytest

from test_local_plans_v2 import toy, question
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.local_lite import FrozenOnlyEmbedder, LexicalIndex, LitePlanner, inspect_support
from wrag.witness.query import Atom


def lite(rows, **options):
    original = toy(rows)
    original.ctx.run.witness = replace(original.ctx.run.witness, local_plan_version='lite',
        local_plan_beam=8, local_plan_candidates=24, local_plan_starts=6,
        local_plan_executions=256, fact_rerank='', **options)
    original.ctx.embedder = FrozenOnlyEmbedder('forbidden')
    r = WitnessRAGRetriever(original.ctx)
    r.index()
    return r


def test_no_dense_embedding_cross_encoder_or_generative_call(monkeypatch):
    r = lite([('Iris', 'paint', 'river', 'I painted a river.')])
    def forbidden(*args, **kwargs):raise AssertionError('Neural inference was called')
    monkeypatch.setattr(r._dense, 'search', forbidden)
    monkeypatch.setattr('wrag.witness.rerank.get_reranker', forbidden)
    result = r.retrieve(question('What did Iris paint?'))
    assert 'river' in result.diagnostics['trechos_extras'][0]['text']
    assert result.diagnostics['online_inference']['query_embedding_calls'] == 0
    assert result.diagnostics['support_reflector']['checks']
    assert result.diagnostics['local_plans']['executed'] <= 256


def test_both_participants_need_requested_action_not_just_a_shared_node():
    r = lite([('Iris', 'paint', 'river', 'I painted a river.'),
              ('Leo', 'paint', 'river', 'I painted a river.'),
              ('Leo', 'like', 'river', 'I like the river.')])
    result = r.retrieve(question('What did both Iris and Leo paint?'))
    selected = result.diagnostics['local_plans']['selected']
    assert any(row['answer_binding'] == 'river' and len(row['query']['atoms']) == 2 for row in selected)
    assert all(check['status'] == 'structural_support_only'
               for check in result.diagnostics['support_reflector']['checks'])


def test_reflector_reports_missing_bridge_and_preserves_semantic_uncertainty():
    r = lite([('Iris', 'paint', 'river', 'I painted a river.')])
    p = LitePlanner(r, question('What did Iris paint?'))
    c = p.candidate([Atom('paint', 'Iris', '?x')])
    w = c.witnesses[0]
    report = inspect_support(p, c, w, [])
    assert report['status'] == 'missing_support'
    assert 'witness_not_fully_delivered' in report['missing']
    assert report['semantic_sufficiency'] == 'undetermined'
    assert 'source_entailment_not_verified' in report['unknown']


def test_reflector_does_not_claim_negation_or_exhaustive_count():
    r = lite([('Iris', 'paint', 'river', 'I painted a river.')])
    p = LitePlanner(r, question('How many paintings did Iris not paint?'))
    c = p.candidate([Atom('paint', 'Iris', '?x')])
    report = inspect_support(p, c, c.witnesses[0], [0])
    assert 'negation_or_exclusion_requires_semantic_check' in report['unknown']
    assert 'retrieval_does_not_prove_global_completeness' in report['unknown']


def test_literal_score_morphology_and_unknown_words():
    index = LexicalIndex(['Iris painted a river.', 'Leo read a book.'])
    assert index.scores('Iris paints')[0] > index.scores('Iris paints')[1]
    assert not index.scores('unseenword').any()
    assert index.scores('painting', normalized=True).max() == 1


def test_lite_grounding_matches_unoptimized_exact_executor():
    from wrag.witness.local_plans import LocalPlanner
    r = lite([('Iris', 'paint', 'river', 'I painted a river.'),
              ('Leo', 'paint', 'forest', 'I painted a forest.')])
    p = LitePlanner(r, question('What did Iris paint?'))
    for atom in [Atom('paint', 'Iris', '?x'), Atom('paint', '?x', 'forest'),
                 Atom('paint', 'Iris', 'forest'), Atom('paint', '?x', '?y')]:
        indexed = list(p.grounding_ids(atom))
        expected = [i for i in LocalPlanner.grounding_ids(p, atom)
                    if all(term.startswith('?') or p.node(eid) in p.entity_nodes.get(term.casefold(), ())
                           for term, eid in [(atom.subject, p.memory.facts[i].subj_id),
                                             (atom.object, p.memory.facts[i].obj_id)])]
        assert indexed == expected


def test_lite_does_not_need_fact_vectors_and_ignores_gold_metadata():
    import numpy as np
    r = lite([('Iris', 'play', 'flute', 'I play flute.')])
    r.memory.fact_vectors = np.zeros((0, 0))
    clean = question('What instruments does Iris play?')
    poisoned = replace(clean, qtype='fake', answers=['UNRELATED'], gold_pids=['ghost'],
                       decomposition=[{'answer': 'UNRELATED'}])
    before = r.retrieve(clean)
    after = r.retrieve(poisoned)
    assert before.diagnostics['trechos_extras'] == after.diagnostics['trechos_extras']
    assert before.diagnostics['local_plans']['selected'] == after.diagnostics['local_plans']['selected']


def test_support_reflector_checks_explicit_temporal_scope_and_delivery():
    r = lite([('Iris', 'visit', 'harbor', 'I visited the harbor on 4 June 2023.')])
    r.memory.facts[0].kind = 'ongoing'
    r.dated.fact_time_source[0] = 'expressao'
    p = LitePlanner(r, question('When did Iris visit the harbor after 1 October 2023?'))
    c = p.candidate([Atom('visit', 'Iris', '?x')])
    assert c
    report = inspect_support(p, c, c.witnesses[0], [0])
    assert 'answer_role_or_scope_incompatible' in report['missing']


def test_answerer_runs_once_and_returns_only_the_answer_field():
    from wrag import config as C
    from wrag.eval.runner import _answer_standard
    from wrag.llm.base import LLMResult, UsageLedger
    r = lite([('Iris', 'paint', 'river', 'I painted a river.')])
    class Reader:
        def __init__(self):self.usage = UsageLedger();self.calls = 0
        def chat(self, prompt, **kwargs):
            assert kwargs['stage'] == 'qa'
            assert 'Support inspection' in prompt
            assert 'do not output your reflection' in prompt
            assert 'SECRET_GOLD' not in prompt
            self.calls += 1
            answer = LLMResult(text='{"answer":"river"}', prompt_tokens=180, completion_tokens=6)
            self.usage.record('qa', answer)
            return answer
    reader = Reader();r.ctx.llm = reader
    r.ctx.run.qa = C.QAConfig(evidence_reader=True, reader_reflection=True, answer_guard=False)
    row = _answer_standard('witnessrag-lite', r, r.corpus, question('What did Iris paint?'), r.ctx.run)
    assert reader.calls == 1 and row['resposta'] == 'river'
    assert set(row['uso_llm']['por_estagio']) == {'qa'}
    assert row['diagnosticos']['support_reflector']['generative_calls'] == 0
