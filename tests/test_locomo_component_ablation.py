"""Behavioral controls for the local-v2 component study, without paid APIs."""
from dataclasses import replace
import pytest

from wrag import config as C, prompts
from wrag.data import Corpus, Passage
from wrag.ie import ExtractionResult, Fact
from wrag.methods import build_context
from wrag.methods.hipporag import ensure_graph
from wrag.pilot import parser, make_plan, _run_config, _validate_resume
from wrag.witness.dated_memory import DatedMemory
from wrag.witness.local_plans_v2 import MultiOriginPlanner
from test_local_plans_v2 import toy, question
from test_local_plans import NoLLM
from test_witness import ExactEmbedder


def settings(variant):
    args = parser().parse_args(['--gpu','0','--dataset','locomo','--local-plans',
        '--proof-controller','--fact-delivery','facts','--local-plan-version','v2',
        '--reader-reflection','--evidence-reader','--answer-set','--fact-time','both',
        '--study-ablation',variant])
    return make_plan(args, __import__('pathlib').Path('test-output'))


@pytest.mark.parametrize('variant',['full','no-witness','no-time-reference','no-time-model','no-reflection'])
def test_cli_roundtrip_and_resume_identity(variant):
    plan = settings(variant)
    cfg = _run_config(plan['settings'], 1)
    assert cfg.witness.study_ablation == variant
    assert cfg.qa.reader_reflection == (variant != 'no-reflection')
    _validate_resume(plan, plan)
    if variant != 'full':
        with pytest.raises(ValueError, match='study_ablation'):
            _validate_resume(settings('full'), plan)


def test_no_witness_does_not_compile_or_execute_plans(monkeypatch):
    r = toy([('Iris','paint','river','I painted a river.'),
             ('Leo','paint','river','I painted a river.')], study_ablation='no-witness')
    def forbidden(*args, **kwargs):
        raise AssertionError('Planner must be bypassed')
    monkeypatch.setattr(MultiOriginPlanner, '__init__', forbidden)
    result = r.retrieve(question('What did Iris and Leo both paint?'))
    d = result.diagnostics
    assert d['local_plans']['executed'] == 0
    assert d['fatos_entregues']['prova'] == 0
    assert d['fatos_entregues']['n'] == 2
    assert 'Reading task' not in d['trechos_extras'][0]['text']


def test_no_reference_preserves_calendar_filter_and_disables_all_proximity():
    r = toy([('Iris','visit','harbor','I visited the harbor.')], study_ablation='no-time-reference')
    r.memory.facts[0].time = '4 June 2023'
    r.memory.facts[0].kind = 'past'
    r.dated.fact_time_source[0] = 'expressao'
    p = MultiOriginPlanner(r, question('Where did Iris visit before 1 June 2023?'))
    assert p.contract.time_weight == 0
    assert p.contract.period.interval is not None
    assert p.contract.temporal_side == 'before'
    assert not p.temporal_ok(0, True)  # fact dated June 4; restrictions remain
    assert p.temporal_score(None, None) == 0


def test_no_time_model_keeps_sources_and_importance_but_never_resolves_dates(monkeypatch):
    import wrag.witness.dated_memory as dm
    corpus = Corpus(name='toy', passages=[Passage(pid='p',title='p',text=
        'Session date: 4 June 2023\n[D1:1] Iris: I loved visiting on 1 June 2023!')], questions=[])
    f = Fact(fid='f', subject='Iris', relation='visit', object='harbor',pid='p',time='last week')
    def forbidden(*args, **kwargs):
        raise AssertionError('Temporal resolver must not run')
    monkeypatch.setattr(dm,'parse_date',forbidden)
    monkeypatch.setattr(dm,'resolve_expression',forbidden)
    dated = DatedMemory(corpus,[f],model_time=False)
    assert dated.fact_interval == [None] and dated.last is None
    assert dated.fact_turn == [('p',0)] and dated.fact_importance[0] > 0
    assert '1 June 2023' in dated.turns['p'][0].text
    dated.extend([f],1)
    assert dated.fact_interval == [None,None]


def test_no_time_model_builds_a_new_graph_without_mutating_acquisition():
    corpus = Corpus(name='toy',passages=[Passage(pid='p',title='p',text='Iris visited the harbor.')],questions=[])
    f = Fact(fid='f',subject='Iris',relation='visit',object='harbor',pid='p',time='last week',kind='past',
             statement='Iris visited on 1 June 2023.')
    extraction = ExtractionResult(facts=[f])
    cfg = C.RunConfig(witness=C.WitnessConfig(study_ablation='no-time-model'))
    ctx = build_context(corpus,cfg,llm=NoLLM(),embedder=ExactEmbedder())
    ctx.extraction = extraction
    ensure_graph(ctx)
    assert extraction.facts[0].time == 'last week'
    assert ctx.extraction.facts[0].time == '' and ctx.extraction.facts[0].kind == ''
    assert ctx.extraction.facts[0].statement == f.statement
    assert ctx.kg is not None


def test_no_time_contract_and_reader_do_not_claim_resolved_metadata():
    r = toy([('Iris','visit','harbor','I visited on 1 June 2023.')], study_ablation='no-time-model')
    p = MultiOriginPlanner(r,question('When did Iris visit before 4 June 2023?'))
    assert p.contract.period.interval is None and p.contract.reference_fact is None
    assert not p.contract.temporal_side and p.contract.operation == 'value'
    result = r.retrieve(question('When did Iris visit?'))
    assert result.diagnostics['leitura_fatos'] == 'atemporal'
    text = r._render_fact_ids([0])
    assert 'Session of' not in text and 'event:' not in text
    template = prompts.qa_atemporal_facts_template()
    assert 'already resolved' not in template and 'event date given' not in template
    assert 'QUESTION: {question}' in template


def test_no_reflection_does_not_change_retrieval_context():
    r = toy([('Iris','paint','river','I painted a river.')], study_ablation='full')
    q = question('What did Iris paint?')
    before = r.retrieve(q)
    r.ctx.run.witness.study_ablation = 'no-reflection'
    after = r.retrieve(q)
    assert before.diagnostics['trechos_extras'] == after.diagnostics['trechos_extras']
    assert before.diagnostics['fatos_entregues'] == after.diagnostics['fatos_entregues']
