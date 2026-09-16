"""Checks the new controller at its failure boundaries without a model server."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from test_witness import build_toy
from wrag import config as C
from wrag.data import Question
from wrag.llm.base import LLMResult
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.research import assess_plan, frontier_probes, select_evidence


def test_obligation_check_rejects_missing_qualifier():
    class Judge:
        def chat(self, prompt, **kwargs):
            assert "SECRET_GOLD" not in prompt
            return LLMResult(text=json.dumps({"covers_question": False,
                "missing": ["employer location"], "reason": "only employer retrieved"}))

    question = Question("q", "Where is Ana's employer located?", ["SECRET_GOLD"])
    query = ConjunctiveQuery(atoms=[Atom("trabalha em", "Ana", "?x")])
    result = assess_plan(Judge(), question, query, "toy", "witnessrag")
    assert not result.covers and result.missing == ["employer location"]


def test_active_controller_replans_after_incomplete_query(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.query_plans = True
    cfg.max_query_plans = 2
    cfg.active_obligations = True
    cfg.enable_acquisition = False
    cfg.answer_set = True
    run = C.RunConfig()
    run.witness = cfg

    class Judge:
        def chat(self, prompt, **kwargs):
            covers = 'localizada em' in prompt
            return LLMResult(text=json.dumps({"covers_question": covers,
                "missing": [] if covers else ["location"], "reason": "checked"}))

    ctx = IndexContext(memory.corpus, Judge(), embedder, run)
    retriever = WitnessRAGRetriever(ctx)
    retriever.memory = memory
    retriever.searcher = searcher
    retriever._dense.search = lambda _q, n: (["p0", "p1", "p2", "p3", "p4"][:n],
                                            [1.0] * min(n, 5))
    weak = ConjunctiveQuery(answer_var="x", atoms=[Atom("trabalha em", "Ana", "?x")])
    full = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"),
        Atom("localizada em", "?y", "?x")])
    calls = []

    def compile_plans(*_args, **kwargs):
        calls.append(kwargs.get("feedback", ""))
        return [full] if kwargs.get("feedback") else [weak]

    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans", compile_plans)
    result = retriever._retrieve_inner(
        Question("q", "Where is Ana's employer located?", []), 2,
        ["p0", "p1"], [1.0, 0.5])
    assert len(calls) == 2
    assert result.diagnostics["plano_escolhido"] == 1
    assert result.diagnostics["testemunha_no_contexto"]
    assert set(result.pids) == {"p0", "p1"}


def test_frontier_and_packing_keep_whole_proof():
    memory, searcher, _embedder, cfg = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"), Atom("localizada em", "?y", "?x")])
    from wrag.witness.provenance import score_answers

    candidates = score_answers(searcher.join(query).witnesses, memory, cfg)
    pids, _, diagnostics = select_evidence(candidates, ["p2", "p3", "p4"], 2, "none")
    assert pids == ["p0", "p1"]
    assert diagnostics["respostas_cobertas"] == 1
    assert len(frontier_probes("Where is Ana's employer?", [query], 5)) == 3


def test_complete_set_gets_one_breadth_pass_before_commit(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.active_frontier = True
    cfg.answer_set = True
    cfg.acquisition_rounds = 2
    run = C.RunConfig()
    run.witness = cfg
    ctx = IndexContext(memory.corpus, object(), embedder, run)
    retriever = WitnessRAGRetriever(ctx)
    retriever.memory = memory
    retriever.searcher = searcher
    retriever._dense.search = lambda _q, n: (["p0", "p1", "p2"][:n],
                                            [1.0] * min(n, 3))
    query = ConjunctiveQuery(answer_var="x", aggregation="set",
                             atoms=[Atom("trabalha em", "Ana", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: query)
    acquired = []
    monkeypatch.setattr(retriever, "_acquire", lambda actions, _q:
                        acquired.append(actions) or 0)
    result = retriever._retrieve_inner(Question("q", "Where does Ana work?", []),
                                      2, ["p0", "p1"], [1.0, 0.5])
    assert len(acquired) == 1
    assert result.diagnostics["pesquisa_provas"]["buscas_dirigidas"] == 1
    assert result.diagnostics["testemunha_no_contexto"]


def test_complete_set_without_acquisition_keeps_proof(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.active_frontier = True
    cfg.enable_acquisition = False
    cfg.answer_set = True
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    retriever._dense.search = lambda _q, n: (["p0", "p1"][:n], [1.0] * min(n, 2))
    query = ConjunctiveQuery(answer_var="x", aggregation="set",
                             atoms=[Atom("trabalha em", "Ana", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: query)
    result = retriever._retrieve_inner(Question("q", "Where does Ana work?", []),
                                      2, ["p0", "p1"], [1.0, 0.5])
    assert result.diagnostics["testemunha_no_contexto"]
    assert result.diagnostics["pesquisa_provas"]["buscas_dirigidas"] == 0


def test_numeric_diagnostic_distinguishes_format_from_count_error():
    spec = importlib.util.spec_from_file_location(
        "compare_active", Path(__file__).resolve().parents[1] /
        "scripts" / "compare-active-research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.numeric_value("two") == module.numeric_value("2") == 2
    assert module.numeric_value("two or three") is None
