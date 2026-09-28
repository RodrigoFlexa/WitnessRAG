"""Logical and delivery invariants; no paid services or benchmark gold."""
from test_local_plans import NoLLM
from test_witness import ExactEmbedder
from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.local_contract import SourceIndex
from wrag.witness.local_plans_v2 import MultiOriginPlanner
from wrag.witness.query import Atom


def toy(rows, **kwargs):
    passages=[];facts=[]
    for i,(subject,relation,obj,text) in enumerate(rows):
        pid=f'p{i}'
        passages.append(Passage(pid=pid,title=pid,text=f'Session date: 4 June, 2023\n[D1:{i}] {subject}: {text}'))
        facts.append(Fact(fid=f'f{i}',subject=subject,relation=relation,object=obj,pid=pid,turn_id=f'D1:{i}'))
    corpus=Corpus(name='toy',passages=passages,questions=[])
    embedder=ExactEmbedder()
    kg=build_graph(corpus,ExtractionResult(facts=facts),embedder,C.GraphConfig(),with_passage_nodes=True)
    cfg=C.RunConfig(witness=C.WitnessConfig(local_plans=True,local_plan_version='v2',proof_controller=True,
          grounding_mode='exact',fact_delivery='facts',fact_fill='question',answer_set=True,
          local_plan_beam=16,local_plan_candidates=48,**kwargs))
    r=WitnessRAGRetriever(IndexContext(corpus,NoLLM(),embedder,cfg,kg=kg));r.index()
    return r


def question(text):
    return Question(qid='q',dataset='toy',question=text,answers=['SECRET_GOLD'])


def test_intersection_reserved_before_walks_and_both_owners_are_required():
    rows=[('Iris','paint','river','I painted a river.'),('Leo','paint','river','I painted a river.')]
    rows += [('Iris','discuss',f'chatter{i}',f'I discussed chatter{i} with Leo.') for i in range(30)]
    r=toy(rows,local_plan_executions=120)
    p=MultiOriginPlanner(r,question('What subject have Iris and Leo both painted?'))
    candidates=p.build()
    good=[c for c in candidates if len(c.query.atoms)==2 and any(w.answer=='river' for w in c.witnesses)]
    assert good and p.phase_counts['intersections']>0
    assert p.executions<=120
    assert all(p.checks(c,w)['required_conjunction'] for c in good for w in c.witnesses)
    single=p.candidate([Atom('paint','Iris','?x')])
    if single:
        assert not p.checks(single,single.witnesses[0])['required_conjunction']


def test_read_projection_does_not_return_the_reader_even_with_high_relevance():
    r=toy([('Iris','read','The Sea Garden','I read "The Sea Garden".'),
           ('Leo','recommend','The Sea Garden','I recommend "The Sea Garden" to Iris.'),
           ('Iris','admire','Leo','I admire Leo.')])
    p=MultiOriginPlanner(r,question("What book did Iris read from Leo's suggestion?"))
    good=p.candidate([Atom('read','Iris','?x'),Atom('recommend','Leo','?x')])
    bad=p.candidate([Atom('admire','?x','Leo'),Atom('read','?x','The Sea Garden')])
    assert good and p.checks(good,good.witnesses[0])['projection_compatible']
    assert bad and not p.checks(bad,bad.witnesses[0])['projection_compatible']


def test_factual_origins_are_used_even_when_a_literal_entity_is_found():
    r=toy([('Iris','paint','river','I painted a river.'),('Leo','read','novel','I read a novel.')])
    p=MultiOriginPlanner(r,question('What did Iris paint?'))
    literal=set().union(*(nodes for _,nodes in p.mentions))
    assert literal and any(node not in literal for node in p.anchors)


def test_shared_predicate_keeps_independent_origins_in_frontier():
    r=toy([('Iris','paint','river','I painted a river.'),('Leo','paint','forest','I painted a forest.')])
    p=MultiOriginPlanner(r,question('What have Iris and Leo painted?'))
    seeds=p.consume(p.seed_proposals(),50,'test')
    frontier=p.frontier(seeds)
    assert {'Iris','Leo'}<={constant for c in frontier for constant in c.query.constants()}


def test_complete_set_members_are_delivered_together_without_summary():
    r=toy([('Iris','play','flute','I play flute.'),('Iris','play','piano','I play piano.')])
    result=r.retrieve(question('What instruments does Iris play?'))
    text=result.diagnostics['trechos_extras'][0]['text']
    assert 'flute' in text and 'piano' in text
    assert result.diagnostics['planejamento']['chamadas']==0
    assert 'Chunk summaries' not in text and 'SECRET_GOLD' not in text
    assert result.diagnostics['local_plans']['version']=='v2'
    assert all(not row['query']['uses_annotations'] for row in result.diagnostics['local_plans']['selected'])


def test_event_identity_unites_duplicate_predicates_but_not_unknown_dates():
    r=toy([('Iris','visit','harbor','I visited the harbor on 4 June 2023.'),
           ('Iris','went to','harbor','I went to the harbor on 4 June 2023.')])
    for i in range(2):
        r.memory.facts[i].time='4 June 2023'
        r.dated.fact_time_source[i]='expressao'
    p=MultiOriginPlanner(r,question('How many times did Iris visit the harbor?'))
    assert p.occurrence_key(0)==p.occurrence_key(1)
    r.dated.fact_time_source[1]='sessao'
    assert p.occurrence_key(0)!=p.occurrence_key(1)
    assert p.reading.count_unit=='occurrence'


def test_future_plan_cannot_witness_a_past_occurrence():
    r=toy([('Iris','visit','harbor','I plan to visit the harbor.')])
    r.memory.facts[0].kind='plan'
    p=MultiOriginPlanner(r,question('How many times did Iris visit the harbor?'))
    assert not p.temporal_ok(0,True)


def test_source_antecedents_are_literal_hypotheses_not_new_facts():
    r=toy([('Leo','read','The Sea Garden','I enjoyed "The Sea Garden", you should read it.'),
           ('Iris','read','book','I am reading that book you recommended.')])
    p=MultiOriginPlanner(r,question("What book did Iris read from Leo's suggestion?"))
    before=len(r.memory.facts)
    support=SourceIndex(r.dated).support(p.text,p.reading)
    assert any('The Sea Garden' in row['text'] for row in support)
    assert any(row['reason']=='candidate_antecedent_not_resolved' for row in support)
    assert len(r.memory.facts)==before


def test_small_fact_budget_does_not_silently_cut_a_join():
    r=toy([('Iris','paint','river','I painted a river.'),('Leo','paint','river','I painted a river.')],fact_budget=1)
    result=r.retrieve(question('What subject have Iris and Leo both painted?'))
    d=result.diagnostics
    assert not d['local_plans']['selected']
    assert len(d['fatos_entregues']['indices'])<=1


def test_new_cli_and_config_round_trip():
    from wrag.pilot import parser,make_plan,_run_config
    from pathlib import Path
    args=parser().parse_args(['--gpu','0','--local-plans','--proof-controller','--fact-delivery','facts',
                             '--answer-set','--local-plan-version','v2','--local-plan-executions','500',
                             '--local-plan-starts','4'])
    plan=make_plan(args,Path('.'))
    cfg=_run_config(plan['settings'],1)
    assert cfg.witness.local_plan_version=='v2'
    assert cfg.witness.local_plan_executions==500 and cfg.witness.local_plan_starts==4


def test_poisoned_metadata_never_reaches_v2_planner():
    class QuestionOnly:
        question='What did Iris paint?'
        def __getattr__(self,name):raise AssertionError(name)
    r=toy([('Iris','paint','river','I painted a river.')])
    result=r.retrieve(QuestionOnly())
    assert result.diagnostics['local_plans']['generated']>0


def test_parent_preferences_do_not_stand_for_child_preferences():
    r=toy([('Iris','like','pottery','I enjoy pottery.'),
           ('Iris','has','children','I have children.'),
           ('children','like','birds','The children like birds.')])
    p=MultiOriginPlanner(r,question("What do Iris's children like?"))
    bad=p.candidate([Atom('like','Iris','?x')])
    good=p.candidate([Atom('has','Iris','?y'),Atom('like','?y','?x')])
    assert bad and not p.checks(bad,bad.witnesses[0])['subject_scope_supported']
    assert good and p.checks(good,good.witnesses[0])['subject_scope_supported']


def test_broad_alias_cluster_does_not_establish_a_shared_specific_value():
    r=toy([('Iris','paint','painting','I painted a painting.'),
           ('Leo','paint','river','I painted a river.')])
    r.ctx.run.witness.grounding_mode='semantic'
    first=r.memory.entity_id('painting');second=r.memory.entity_id('river')
    r.memory._cluster_of[second]=r.memory._cluster_of[first]
    p=MultiOriginPlanner(r,question('What subject have Iris and Leo both painted?'))
    c=p.candidate([Atom('paint','Iris','?x'),Atom('paint','Leo','?x')])
    assert c
    assert not p.checks(c,c.witnesses[0])['shared_value_literal_support']


def test_date_of_a_contextual_bridge_cannot_replace_the_requested_event():
    r=toy([('Iris','go','club','I went to the club.'),('Iris','go','harbor','I went to the harbor.')])
    p=MultiOriginPlanner(r,question('When did Iris go to the club?'))
    good=p.candidate([Atom('go','Iris','?x')])
    assert good
    checks={w.bindings['x']:p.checks(good,w) for w in good.witnesses}
    assert checks['club']['predicate_coverage']>checks['harbor']['predicate_coverage']


def test_v2_uses_one_reflective_reader_call_and_no_other_generative_stage():
    from wrag.llm.base import LLMResult,UsageLedger
    from wrag.eval.runner import _answer_standard
    r=toy([('Iris','paint','river','I painted a river.')])
    class Reader:
        def __init__(self):self.usage=UsageLedger();self.calls=0
        def chat(self,prompt,**kwargs):
            assert kwargs['stage']=='qa' and 'Perform reflection' in prompt
            assert 'Reading task' in prompt and 'SECRET_GOLD' not in prompt
            self.calls+=1
            result=LLMResult(text='{"answer":"river"}',prompt_tokens=200,completion_tokens=8)
            self.usage.record('qa',result)
            return result
    reader=Reader();r.ctx.llm=reader
    r.ctx.run.qa=C.QAConfig(evidence_reader=True,reader_reflection=True)
    row=_answer_standard('witnessrag-local',r,r.corpus,question('What did Iris paint?'),r.ctx.run)
    assert reader.calls==1 and set(row['uso_llm']['por_estagio'])=={'qa'}


def test_uncertain_state_timestamp_outside_scope_cannot_be_the_answer_date():
    r=toy([('Iris','hike','forest','I hiked in the forest on 4 June 2023.')])
    r.memory.facts[0].kind='ongoing'
    r.dated.fact_time_source[0]='expressao'
    p=MultiOriginPlanner(r,question('When did Iris hike after 1 October 2023?'))
    # State validity stays uncertain, but its June timestamp cannot answer after
    # October. The reader receives the resolved point explicitly.
    assert p.temporal_ok(0,True)
    c=p.candidate([Atom('hike','Iris','?x')])
    assert c and not p.checks(c,c.witnesses[0])['projection_compatible']
    assert '2023-10-01' in p.reading.instructions()


def test_source_time_prior_is_per_query_and_does_not_mutate_the_shared_index():
    r=toy([('Iris','paint','river','I paint a river.')])
    index=SourceIndex(r.dated)
    first=MultiOriginPlanner(r,question('What did Iris paint?')).reading
    later=MultiOriginPlanner(r,question('What did Iris paint recently?')).reading
    one=index.support(first.temporal.question,first)
    index.support(later.temporal.question,later)
    assert index.support(first.temporal.question,first)==one
    assert not hasattr(index,'temporal')


def test_both_participants_must_satisfy_the_requested_predicate():
    r=toy([('Iris','paint','river','I painted a river.'),('Leo','love','river','I love the river.')])
    p=MultiOriginPlanner(r,question('What subject have Iris and Leo both painted?'))
    c=p.candidate([Atom('paint','Iris','?x'),Atom('love','Leo','?x')])
    assert c and p.checks(c,c.witnesses[0])['owner_coverage']==1
    assert not p.checks(c,c.witnesses[0])['required_conjunction']


def test_reranker_can_see_later_eligible_witnesses_of_one_program(monkeypatch):
    rows=[(name,'paint',value,f'I painted a {value}.') for value in ['river','forest','lake'] for name in ['Iris','Leo']]
    r=toy(rows,fact_rerank='test')
    p=MultiOriginPlanner(r,question('What subject have Iris and Leo both painted?'))
    c=p.candidate([Atom('paint','Iris','?x'),Atom('paint','Leo','?x')])
    class Ranker:
        def score(self,text,values):
            assert len(values)>=3
            return [.99 if 'lake' in v else .1 for v in values]
    monkeypatch.setattr('wrag.witness.rerank.get_reranker',lambda _:Ranker())
    ranked,_=p.rerank([c])
    assert ranked[0][2].answer=='lake'


def test_round_robin_keeps_each_witness_attached_to_its_own_program():
    r=toy([('Iris','paint','river','I painted a river.'),('Leo','read','novel','I read a novel.')])
    p=MultiOriginPlanner(r,question('What did Iris and Leo do?'))
    first=p.candidate([Atom('paint','Iris','?x')])
    second=p.candidate([Atom('read','Leo','?x')])
    ranked,_=p.rerank([first,second])
    assert {c.signature() for _,c,_,_,_ in ranked}=={first.signature(),second.signature()}
    for _,c,w,_,_ in ranked:
        atom=c.query.atoms[0]
        assert all(r.memory.facts[i].subject==atom.subject and r.memory.facts[i].relation==atom.relation for i in w.facts)
