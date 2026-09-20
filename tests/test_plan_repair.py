"""Focused checks for plan-repair safety and join diagnostics."""
import json
from dataclasses import replace
import pytest

from test_regressions import custom_memory
from wrag import config as C
from wrag.data import Question
from wrag.llm.base import LLMResult
from wrag.methods.witnessrag import _coverage_gap, _compact_plan_feedback
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.query import _query_from_data
from wrag.witness.research import assess_plan_repair
from wrag.witness.search import WitnessSearcher
from wrag.witness.verification import verify_witnesses
from wrag.eval.reader import _guard_answer


class Responses:
    def __init__(self, *values):
        self.values = iter(values)
        self.stages = []

    def chat(self, _prompt, **kwargs):
        self.stages.append(kwargs["stage"])
        return LLMResult(text=json.dumps(next(self.values)))


def test_invalid_coverage_is_retried_without_promoting_to_full():
    question = Question("q", "Which basketball goal did John mention?", ["x"], [])
    plan = ConjunctiveQuery(atoms=[Atom("goals", "John", "?x")])
    llm = Responses(
        {"covers_question": True, "missing": ["short condition"], "reason": ""},
        {"covers_question": False, "missing": ["basketball qualifier"], "reason": ""})
    check = assess_plan_repair(llm, question, plan, "locomo", "witnessrag")
    assert not check.covers and check.missing == ["basketball qualifier"]
    assert llm.stages == ["witness.obligations.v3", "witness.obligations.repair"]


def test_coverage_gap_retains_direction_and_constant():
    plan = ConjunctiveQuery(atoms=[Atom("goals", "John", "?x")])
    gap = _coverage_gap(plan, ["basketball career"])
    assert gap.anchor() == "John"
    assert gap.atom.subject == "John" and gap.atom.object == "?x"
    assert gap.atom.relation == "goals"
    assert "basketball career" in gap.probe()


def test_join_trace_identifies_binding_failure():
    memory, embedder = custom_memory([
        ("Joanna", "recommended", "Little Women"),
        ("Different Book", "given to", "Nate")])
    cfg = C.WitnessConfig(grounding_mode="exact", plan_repair=True)
    plan = ConjunctiveQuery(atoms=[
        Atom("recommended", "Joanna", "?x"), Atom("given to", "?x", "Nate")])
    result = WitnessSearcher(memory, embedder, cfg).join(plan)
    assert result.n_candidates == [1, 1]
    assert not result.complete
    assert any(step["binding_rejections"] for step in result.trace["stages"])
    assert result.trace["top_candidates"][0][0]["triple"][0] == "Joanna"


def test_compiler_keeps_text_conditions_outside_graph_atoms():
    question = Question("q", "What basketball goal did John mention?", ["x"])
    plan = _query_from_data({"atoms": [{"subject": "John", "relation": "goals",
                                        "object": "?x"}], "answer_var": "x",
                             "source_conditions": ["goal concerns basketball career"]},
                            question, 4, "llm-plan")
    assert plan.atoms[0].relation == "goals"
    assert plan.conditions == ["goal concerns basketball career"]
    assert plan.to_dict()["source_conditions"] == plan.conditions


def test_conditional_witness_needs_literal_condition_quote():
    memory, embedder = custom_memory([("John", "goals", "play professionally")])
    plan = ConjunctiveQuery(atoms=[Atom("goals", "John", "?x")],
                            conditions=["goal concerns basketball career"])
    witnesses = WitnessSearcher(memory, embedder,
                                C.WitnessConfig(grounding_mode="exact")).join(plan).witnesses
    base = {"supported": True, "answers_question": True,
            "evidence": [{"atom": 0, "pid": "p0",
                          "quote": "John goals play professionally"}]}
    accepted, _ = verify_witnesses(
        Responses(base), memory.corpus, memory,
        Question("q", "What basketball goal did John mention?", ["SECRET_GOLD"]),
        plan, witnesses, 1, "toy")
    assert not accepted
    memory.corpus._by_pid["p0"] = replace(
        memory.corpus.get("p0"),
        text="John goals play professionally for a basketball career")
    with_quote = {**base, "condition_evidence": [{"condition": 0, "pid": "p0",
                                                   "quote": "play professionally for a basketball career"}]}
    accepted, _ = verify_witnesses(
        Responses(with_quote), memory.corpus, memory,
        Question("q", "What basketball goal did John mention?", ["SECRET_GOLD"]),
        plan, witnesses, 1, "toy")
    assert accepted


def test_answer_guard_repairs_only_with_source_quote():
    memory, _ = custom_memory([("Trip", "visited", "Brazil")])
    question = Question("q", "Which country did the trip visit?", ["SECRET_GOLD"])
    answer, audit, *_ = _guard_answer(
        Responses({"valid": False, "corrected_answer": "Brazil", "evidence": [
            {"pid": "p0", "quote": "Trip visited Brazil"}]}),
        memory.corpus, question, ["p0"], "Rio de Janeiro")
    assert answer == "Brazil" and audit["alterada"]
    answer, audit, *_ = _guard_answer(
        Responses({"valid": False, "corrected_answer": "Brazil", "evidence": [
            {"pid": "p0", "quote": "Trip visited Portugal"}]}),
        memory.corpus, question, ["p0"], "Rio de Janeiro")
    assert answer == "Rio de Janeiro" and not audit["alterada"]


def test_answer_guard_never_counts_two_events_from_one_quote():
    memory, _ = custom_memory([("Ana", "visited", "museum")])
    answer, audit, *_ = _guard_answer(
        Responses({"valid": False, "corrected_answer": "2", "explicit_total": False,
                   "evidence": [{"pid": "p0", "quote": "Ana visited museum"}]}),
        memory.corpus, Question("q", "How many visits did Ana make?", ["SECRET"]),
        ["p0"], "1")
    assert answer == "1" and not audit["alterada"]


def test_answer_guard_rejects_unsupported_list_item():
    memory, _ = custom_memory([("Ana", "visited", "Brazil")])
    answer, audit, *_ = _guard_answer(
        Responses({"valid": False, "corrected_answer": "Brazil, Portugal",
                   "evidence": [{"pid": "p0", "quote": "Ana visited Brazil"}]}),
        memory.corpus,
        Question("q", "Which countries did Ana visit?", ["SECRET"]),
        ["p0"], "Brazil")
    assert answer == "Brazil" and not audit["alterada"]


def test_replanner_feedback_keeps_every_plan_and_failure():
    logs = [{"indice": i, "consulta": {"atoms": [{"relation": f"r{i}"}]},
             "executavel": True, "fechou": False,
             "candidatos_por_atomo": [0], "lacuna": None,
             "trilha_juncao": {"top_candidates": [[]], "stages": []}}
            for i in range(5)]
    feedback = _compact_plan_feedback(logs)
    assert [row["index"] for row in feedback] == list(range(5))
    assert all(row["failure"] == "atom_without_candidate" for row in feedback)


def test_controller_retries_duplicate_with_failure_feedback(monkeypatch):
    from wrag.methods.base import IndexContext
    from wrag.methods.witnessrag import WitnessRAGRetriever
    from wrag.witness.research import PlanAssessment

    memory, embedder = custom_memory([("Ana", "works", "Atlas")])
    cfg = C.WitnessConfig(grounding_mode="exact", plan_repair=True,
                          query_plans=True, max_query_plans=3,
                          active_obligations=True, active_frontier=True,
                          enable_acquisition=False)
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, Responses(),
                                                 embedder, run))
    retriever.memory = memory
    retriever.searcher = WitnessSearcher(memory, embedder, cfg)
    retriever._dense.search = lambda *_: ([], [])
    bad = ConjunctiveQuery(atoms=[Atom("invented", "Ana", "?x")])
    good = ConjunctiveQuery(atoms=[Atom("works", "Ana", "?x")])
    responses = iter([[bad], [bad], [good]])
    calls = []

    def compile_stub(*_args, **kwargs):
        calls.append(kwargs.get("feedback", ""))
        return next(responses)

    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans", compile_stub)
    monkeypatch.setattr("wrag.methods.witnessrag.assess_plan_repair",
                        lambda *_: PlanAssessment(True, [], "covered"))
    result = retriever._retrieve_active(
        Question("q", "Where does Ana work?", ["SECRET_GOLD"]),
        1, ["p0"], [1.0])
    assert result.diagnostics["classe_prova"] == "full"
    assert result.diagnostics["planejamento"]["planos_distintos"] == 2
    assert "REPAIR AFTER DUPLICATE" in calls[2]
    assert "invented" in calls[2]


@pytest.mark.parametrize("with_condition_quote", [True, False])
def test_controller_never_promotes_unchecked_source_condition(
        monkeypatch, with_condition_quote):
    from wrag.methods.base import IndexContext
    from wrag.methods.witnessrag import WitnessRAGRetriever
    from wrag.witness.research import PlanAssessment

    memory, embedder = custom_memory([("John", "goals", "play professionally")])
    memory.corpus._by_pid["p0"] = replace(
        memory.corpus.get("p0"),
        text="John goals play professionally for a basketball career")
    plan = ConjunctiveQuery(atoms=[Atom("goals", "John", "?x")],
                            conditions=["goal concerns basketball career"])
    verdict = {"supported": True, "answers_question": True,
               "evidence": [{"atom": 0, "pid": "p0",
                             "quote": "John goals play professionally"}]}
    if with_condition_quote:
        verdict["condition_evidence"] = [
            {"condition": 0, "pid": "p0",
             "quote": "play professionally for a basketball career"}]
    cfg = C.WitnessConfig(grounding_mode="exact", plan_repair=True,
                          query_plans=True, max_query_plans=1,
                          active_obligations=True, enable_acquisition=False)
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, Responses(verdict),
                                                 embedder, run))
    retriever.memory = memory
    retriever.searcher = WitnessSearcher(memory, embedder, cfg)
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans",
                        lambda *_args, **_kwargs: [plan])
    monkeypatch.setattr("wrag.methods.witnessrag.assess_plan_repair",
                        lambda *_args: PlanAssessment(True, [], "preserved"))
    result = retriever._retrieve_active(
        Question("q", "What basketball goal did John mention?", ["SECRET"]),
        1, ["p0"], [1.0])
    assert (result.diagnostics["classe_prova"] == "full") is with_condition_quote


def test_rejected_broad_plan_can_supply_checked_context_without_full_proof(monkeypatch):
    from wrag.methods.base import IndexContext
    from wrag.methods.witnessrag import WitnessRAGRetriever
    from wrag.witness.research import PlanAssessment

    memory, embedder = custom_memory([("John", "goals", "play professionally")])
    memory.corpus._by_pid["p0"] = replace(
        memory.corpus.get("p0"),
        text="John goals play professionally for a basketball career")
    plan = ConjunctiveQuery(atoms=[Atom("goals", "John", "?x")])
    verdict = {"supported": True, "answers_question": True,
               "evidence": [{"atom": 0, "pid": "p0",
                             "quote": "John goals play professionally"}],
               "condition_evidence": [
                   {"condition": 0, "pid": "p0",
                    "quote": "play professionally for a basketball career"}]}
    cfg = C.WitnessConfig(grounding_mode="exact", plan_repair=True,
                          query_plans=True, max_query_plans=1,
                          active_obligations=True, enable_acquisition=False)
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, Responses(verdict),
                                                 embedder, run))
    retriever.memory = memory
    retriever.searcher = WitnessSearcher(memory, embedder, cfg)
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans",
                        lambda *_args, **_kwargs: [plan])
    monkeypatch.setattr("wrag.methods.witnessrag.assess_plan_repair",
                        lambda *_args: PlanAssessment(False, ["basketball career"],
                                                      "missing qualifier"))
    result = retriever._retrieve_active(
        Question("q", "What basketball goal did John mention?", ["SECRET"]),
        1, ["p0"], [1.0])
    assert result.diagnostics["classe_prova"] == "provisional"
    assert result.diagnostics["motivo_parada"] == "source_conditions_checked_context"
    assert result.diagnostics.get("resposta_estrutural", "") == ""
