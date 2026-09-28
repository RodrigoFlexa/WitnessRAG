"""Behavioral invariants for local plans; no model server or paid calls."""
from dataclasses import replace
from datetime import date

import pytest

from test_proof_controller import POOL, QUESTION, retriever
from wrag.witness.local_plans import LocalPlanner, contract
from wrag.witness.query import Atom, ConjunctiveQuery


class NoLLM:
    def chat(self, *args, **kwargs):
        raise AssertionError("Local retrieval called a generative model")


def local(**options):
    return retriever(NoLLM(), local_plans=True, fact_delivery="facts",
                     fact_fill="question", **options)


def test_bottom_up_search_produces_many_plans_and_a_real_join():
    r = local()
    planner = LocalPlanner(r, QUESTION)
    candidates = planner.build()
    assert len(candidates) > 1
    chain = [c for c in candidates if len(c.query.atoms) == 2 and
             any(w.bindings.get("x") == "Recife" for w in c.witnesses)]
    assert chain
    assert chain[0].query.atoms[0].subject == "Ana"
    assert chain[0].query.atoms[0].object == chain[0].query.atoms[1].subject
    assert planner.executions <= planner.generation_budget


def test_reverse_traversal_keeps_predicate_direction():
    r = local()
    question = replace(QUESTION, question="Who works at Atlas?")
    planner = LocalPlanner(r, question)
    candidates = planner.build()
    assert any(c.query.atoms[0].subject == "?x" and c.query.atoms[0].object == "Atlas"
               and any(w.answer == "Ana" for w in c.witnesses) for c in candidates)
    assert not planner.execute(ConjunctiveQuery(atoms=[Atom("trabalha em", "Atlas", "?x")]))


def test_join_does_not_accept_a_disconnected_bridge():
    planner = LocalPlanner(local(), QUESTION)
    query = ConjunctiveQuery(atoms=[Atom("trabalha em", "Ana", "?y"),
                                   Atom("acampou em", "?y", "?x")])
    assert not planner.execute(query)


def test_final_reranker_scores_program_with_source_and_keeps_multiple(monkeypatch):
    class Ranker:
        def score(self, question, texts):
            if texts and texts[0].startswith("Directed retrieval program:"):
                assert all("Source:" in t and "Evidence:" in t for t in texts)
            assert all("SECRET_GOLD" not in t for t in texts)
            return [.95 if " AND " in t and "Recife" in t else .2 for t in texts]

    monkeypatch.setattr("wrag.witness.rerank.get_reranker", lambda _: Ranker())
    r = local(fact_rerank="test")
    result = r._retrieve_proof(QUESTION, 5, *POOL)
    d = result.diagnostics
    assert d["planejamento"]["chamadas"] == 0
    assert d["local_plans"]["ranked_pairs"] > 1
    assert d["local_plans"]["selected"]
    assert len(d["local_plans"]["selected"][0]["query"]["atoms"]) == 2
    assert all(not row["verified"] for row in d["local_plans"]["selected"])
    assert d["leitura_fatos"]
    assert "Original source turns" in d["trechos_extras"][0]["text"]
    assert "SECRET_GOLD" not in str(d)


def test_fact_budget_never_advertises_a_partial_chain_as_delivered(monkeypatch):
    class Ranker:
        def score(self, q, texts):
            return [.99 if " AND " in t else .01 for t in texts]
    monkeypatch.setattr("wrag.witness.rerank.get_reranker", lambda _: Ranker())
    r = local(fact_budget=1, local_plan_keep=1, fact_rerank="test")
    result = r._retrieve_proof(QUESTION, 5, *POOL)
    assert not result.diagnostics["local_plans"]["selected"]
    assert len(result.diagnostics["fatos_entregues"]["indices"]) <= 1


def test_question_metadata_is_not_used_by_local_planning():
    class QuestionOnly:
        question = "Where is the company Ana works at located?"
        def __getattr__(self, name):
            raise AssertionError(f"Privileged question field read: {name}")
    r = local()
    result = r._retrieve_proof(QuestionOnly(), 5, *POOL)
    assert result.diagnostics["local_plans"]["generated"] > 1


def test_calendar_filter_beats_recency_and_session_dates_are_not_event_dates():
    r = local()
    planner = LocalPlanner(r, replace(QUESTION, question="Where did Bruno camp in June 2023?"))
    assert planner.contract.period.interval.start == date(2023, 6, 1)
    beach = next(i for i, f in enumerate(r.memory.facts) if f.object == "praia")
    mountains = next(i for i, f in enumerate(r.memory.facts) if f.object == "montanhas")
    # Session-only August remains uncertain, not falsely proven outside June.
    assert planner.temporal_ok(beach, True)
    r.memory.facts[beach].time = "6 August 2023"
    r.memory.facts[beach].kind = "past"
    r.dated.fact_time_source[beach] = "expressao"
    assert not planner.temporal_ok(beach, True)
    assert planner.temporal_ok(mountains, True)
    assert planner.temporal_ok(beach, False)  # independent bridge atom


def test_before_point_is_a_directional_constraint_not_the_previous_week():
    r = local()
    c = contract("Where did Bruno camp before 1 August 2023?", r.dated)
    assert c.temporal_side == "before"
    assert c.period.interval.start == c.period.interval.end == date(2023, 8, 1)


def test_recency_compares_to_memory_clock_and_first_is_not_claimed_complete():
    r = local()
    c = contract("Where is Ana currently working?", r.dated)
    assert c.temporal_side == "recent" and c.time_weight > 0
    assert c.period.interval.start == r.dated.last
    c = contract("What was Ana's first job?", r.dated)
    assert c.period.interval.start == r.dated.first
    assert "retrieval_does_not_prove_global_completeness" in c.pending


@pytest.mark.parametrize("options", [{"fact_delivery": "facts+summary"}, {"plan_router": "llm"},
                                      {"summary_reflection": True}])
def test_incompatible_options_fail_before_any_generative_call(options):
    r = local()
    for key, value in options.items():
        setattr(r.ctx.run.witness, key, value)
    with pytest.raises(ValueError):
        r._retrieve_proof(QUESTION, 5, *POOL)


def test_cli_round_trip_sets_local_flags_without_changing_historical_defaults(tmp_path):
    from wrag.pilot import parser, make_plan, _run_config
    args = parser().parse_args(["--gpu", "0", "--backend", "openai", "--dataset", "locomo",
                               "--proof-controller", "--local-plans", "--fact-delivery", "facts",
                               "--evidence-reader", "--reader-reflection", "--answer-set", "--local-plan-beam", "7"])
    plan = make_plan(args, tmp_path)
    cfg = _run_config(plan["settings"], 12)
    assert cfg.witness.local_plans and cfg.witness.local_plan_beam == 7
    assert not cfg.witness.proof_verify
    assert cfg.witness.proof_importance_strong == 0
    assert cfg.qa.reader_reflection
    from wrag import config as C
    assert not C.WitnessConfig().local_plans


def test_event_anchor_and_temporal_score_follow_event_atom_not_sorted_fact_ids():
    r = local()
    planner = LocalPlanner(r, replace(QUESTION, question="Where did Bruno camp before his beach trip?"))
    assert "event_relative_anchor_unresolved" in planner.contract.pending
    # In a two-atom chain, recency must refer to the anchored employment, even
    # if the location fact is first in sorted Witness.facts.
    from wrag.witness.local_plans import Candidate
    from wrag.witness.search import Witness
    q = ConjunctiveQuery(atoms=[Atom("trabalha em", "Ana", "?y"), Atom("localizada em", "?y", "?x")])
    a = next(i for i,f in enumerate(r.memory.facts) if f.subject == "Ana")
    b = next(i for i,f in enumerate(r.memory.facts) if f.subject == "Atlas")
    w = Witness(facts=(b,a), bindings={"y":"Atlas", "x":"Recife"}, score=1,cost=0,pids=())
    c = Candidate(q, [w])
    from wrag.witness.scoring import proximity
    expected = proximity(r.dated.fact_interval[a], planner.contract.period, r.scorer.scale(planner.contract.period))
    assert planner.temporal_score(c,w) == expected


def test_standard_reader_pipeline_makes_exactly_one_qa_call():
    from wrag import config as C
    from wrag.eval.runner import _answer_standard
    from wrag.llm.base import LLMResult, UsageLedger
    class Reader:
        def __init__(self):
            self.usage = UsageLedger()
            self.calls = []
        def chat(self, prompt, **kwargs):
            assert kwargs["stage"] == "qa"
            assert "SECRET_GOLD" not in prompt and "Chunk summaries" not in prompt
            assert "Original source turns" in prompt and "Perform reflection" in prompt
            self.calls.append(prompt)
            result = LLMResult(text='{"answer":"Recife"}', prompt_tokens=100, completion_tokens=5)
            self.usage.record("qa", result)
            return result
    r = local()
    reader = Reader()
    r.ctx.llm = reader
    r.ctx.run.qa = C.QAConfig(evidence_reader=True, reader_reflection=True)
    row = _answer_standard("witnessrag-local", r, r.corpus, QUESTION, r.ctx.run)
    assert len(reader.calls) == 1
    assert set(row["uso_llm"]["por_estagio"]) == {"qa"}
    assert row["resposta"] == "Recife"
    assert row["diagnosticos"]["motivo_parada"] == "planos_locais_entregues"


def test_frozen_pilot_reuses_only_identical_literal_passages_and_checks_checksum(tmp_path):
    import hashlib
    import importlib.util
    import json
    import pickle
    from pathlib import Path
    from types import SimpleNamespace
    r = local()
    r.corpus.questions = [QUESTION]
    raw = pickle.dumps((SimpleNamespace(corpus=r.corpus, facts=r.memory.facts), None))
    folder = tmp_path / "controlled" / "toy"
    folder.mkdir(parents=True)
    (folder / "memory.pkl").write_bytes(raw)
    manifest = {"identity": {"corpus": {"question_ids": [QUESTION.qid]},
                             "embedder": "test", "model": "extraction-model"},
                "sha256": hashlib.sha256(raw).hexdigest()}
    (folder / "memory.json").write_text(json.dumps(manifest))
    spec = importlib.util.spec_from_file_location("local_pilot", Path("scripts/run-local-plans.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _, _, provenance = module.load_memory(tmp_path, r.corpus, "test")
    assert provenance["extraction_model"] == "extraction-model"
    changed = replace(r.corpus, passages=[replace(p, text=p.text+" changed") for p in r.corpus.passages])
    with pytest.raises(FileNotFoundError):
        module.load_memory(tmp_path, changed, "test")
    (folder / "memory.pkl").write_bytes(raw+b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        module.load_memory(tmp_path, r.corpus, "test")
    path = tmp_path / "manifest.json"
    module.check_manifest(path, {"config": (1, 2), "beam": 12})
    module.check_manifest(path, {"config": (1, 2), "beam": 12}, resume=True)
    with pytest.raises(ValueError):
        module.check_manifest(path, {"config": (1, 2), "beam": 13}, resume=True)
