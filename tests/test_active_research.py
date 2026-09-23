"""Checks the new controller at its failure boundaries without a model server."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from test_witness import build_toy
from wrag import config as C
from wrag.data import Question
from wrag.llm.base import LLMResult
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.eval.reader import read
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.research import (PlanAssessment, assess_plan, assess_plan_soft,
                                   frontier_probes, proof_hints, select_evidence)


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


def test_selective_controller_skips_graph_for_non_multihop(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.selective_witness = True
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query",
                        lambda *_a, **_k: pytest.fail("compiler must not run"))
    result = retriever._retrieve_selective(
        Question("q", "Where does Ana work?", [], qtype="single-hop"), 3,
        ["p2", "p3", "p4"], [1.0, .5, .25])
    assert result.pids == ["p2", "p3", "p4"]
    assert result.diagnostics["rota"] == "hybrid_only"
    assert result.diagnostics["planejamento"]["chamadas"] == 0


def test_selective_controller_rejects_one_atom_and_accepts_whole_join(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.selective_witness = True
    cfg.answer_set = True
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    question = Question("q", "Where is Ana's employer located?", [],
                        qtype="multi-hop")
    weak = ConjunctiveQuery(answer_var="x",
                            atoms=[Atom("trabalha em", "Ana", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: weak)
    unchanged = retriever._retrieve_selective(
        question, 3, ["p2", "p3", "p4"], [1.0, .5, .25])
    assert unchanged.pids == ["p2", "p3", "p4"]
    assert unchanged.diagnostics["motivo_parada"] == "plan_not_compositional"

    full = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"),
        Atom("localizada em", "?y", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: full)
    changed = retriever._retrieve_selective(
        question, 3, ["p2", "p3", "p4"], [1.0, .5, .25])
    assert set(changed.pids) == {"p2", "p0", "p1"}
    assert changed.diagnostics["testemunha_no_contexto"]
    assert changed.diagnostics["contexto_alterado_pelo_witness"]


def test_selective_multi_probe_requires_agreement_and_only_replaces_tail(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.selective_witness = True
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    weak = ConjunctiveQuery(answer_var="x", fallback="Ana employer city",
                            atoms=[Atom("work at", "Ana", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: weak)
    rankings = {
        "Ana employer city": ["p0", "p3", "p4"],
        "Ana work at": ["p1", "p3", "p0"],
    }
    retriever._dense.search = lambda probe, _n: (rankings[probe], [1.0, .5, .25])
    result = retriever._retrieve_selective(
        Question("q", "Where is Ana's employer located?", [], qtype="multi-hop"),
        3, ["p0", "p1", "p2"], [1.0, .5, .25])
    assert result.pids == ["p0", "p1", "p3"]
    assert result.diagnostics["rota"] == "multi_probe"
    assert result.diagnostics["sondas_recuperacao"]["votos"] == 2


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


def test_soft_assessment_explains_directed_atoms_and_preserves_partial():
    class Judge:
        def chat(self, prompt, **kwargs):
            assert "(?x, supports, Calvin) asks who supports Calvin" in prompt
            return LLMResult(text=json.dumps({"tier": "partial",
                "missing": ["verify support in the passage"], "reason": "related plan"}))

    query = ConjunctiveQuery(answer_var="x", atoms=[Atom("supports", "?x", "Calvin")])
    result = assess_plan_soft(Judge(), Question("q", "Who supports Calvin?", []),
                              query, "toy", "witnessrag")
    assert result.tier == "partial" and not result.covers
    assert result.to_dict()["grau"] == "partial"


def test_provisional_plan_keeps_fallback_and_proof_hints(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.active_obligations = True
    cfg.active_context = True
    cfg.soft_obligations = True
    cfg.proof_reader = True
    cfg.enable_acquisition = False
    cfg.query_plans = False
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    retriever._dense.search = lambda _q, n: (["p2", "p3", "p4"][:n], [1.0] * min(n, 3))
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"), Atom("localizada em", "?y", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: query)
    monkeypatch.setattr("wrag.methods.witnessrag.assess_plan_soft", lambda *_a, **_k:
                        PlanAssessment(False, ["location"], "partial", tier="partial"))
    result = retriever._retrieve_inner(Question("q", "Where is Ana's employer?", []),
                                      3, ["p2", "p3", "p4"], [1.0, 0.5, 0.25])
    assert result.diagnostics["classe_prova"] == "provisional"
    assert result.diagnostics["prova_provisoria"]
    assert result.pids[:2] == ["p0", "p1"]
    assert result.pids[2] == "p2"
    assert result.diagnostics["resposta_estrutural"] == ""
    assert "[1]" in result.diagnostics["leitura_provas"]["hipoteses"]
    assert "[2]" in result.diagnostics["leitura_provas"]["hipoteses"]


def test_full_plan_precedes_cheaper_provisional_plan(monkeypatch):
    memory, searcher, embedder, cfg = build_toy()
    cfg.active_obligations = True
    cfg.active_context = True
    cfg.soft_obligations = True
    cfg.proof_reader = True
    cfg.enable_acquisition = False
    cfg.query_plans = True
    cfg.max_query_plans = 2
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, object(), embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    retriever._dense.search = lambda _q, n: (["p2", "p3", "p4"][:n], [1.0] * min(n, 3))
    weak = ConjunctiveQuery(answer_var="x", atoms=[Atom("trabalha em", "Ana", "?x")])
    full = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"), Atom("localizada em", "?y", "?x")])
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans",
                        lambda *_a, **_k: [weak, full])
    monkeypatch.setattr("wrag.methods.witnessrag.assess_plan_soft", lambda _l, _q, p,
                        *_a: PlanAssessment(p.n_atoms == 2, [], "checked",
                                           tier="full" if p.n_atoms == 2 else "partial"))
    result = retriever._retrieve_inner(Question("q", "Where is Ana's employer?", []),
                                      3, ["p2", "p3", "p4"], [1.0, 0.5, 0.25])
    assert result.diagnostics["plano_escolhido"] == 1
    assert result.diagnostics["classe_prova"] == "full"
    assert "fallback" not in result.diagnostics


def test_proof_hints_exclude_incomplete_witnesses():
    memory, searcher, _embedder, cfg = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?y"), Atom("localizada em", "?y", "?x")])
    from wrag.witness.provenance import score_answers
    candidates = score_answers(searcher.join(query).witnesses, memory, cfg)
    assert proof_hints(candidates, memory, ["p0"]) == ""
    assert "Candidate 1" in proof_hints(candidates, memory, ["p0", "p1"])


def test_reader_checks_proof_notes_against_source_passages():
    memory, _searcher, _embedder, _cfg = build_toy()

    class Judge:
        def chat(self, prompt, **kwargs):
            assert "EVIDENCE MAP" in prompt
            assert "check the cited passage text" in prompt
            assert "[1] Ana | trabalha em" in prompt
            assert "SOURCE PASSAGES" in prompt
            return LLMResult(text='{"answer": "Paris"}')

    qa = C.QAConfig(proof_reader=True, answer_set=True)
    answer = read(Judge(), memory.corpus,
                  Question("q", "Where is Ana's employer?", []), ["p0", "p1"],
                  qa, proof_context={"hipoteses": "Candidate 1: Paris\n"
                                     "  [1] Ana | trabalha em | Acme",
                                     "grau": "provisional",
                                     "condicoes_pendentes": ["location"]})
    assert answer.answer == "Paris"
