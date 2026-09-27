"""Regression cases from the robust-plan review; no model/server calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_proof_controller import CHAIN, POOL, QUESTION, FakeLLM, retriever


def report_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "paired-report.py"
    spec = importlib.util.spec_from_file_location("paired_report_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(qid="q1", answer="Recife"):
    return {"qid": qid, "tipo": "single-hop", "pergunta": "Where?",
            "respostas_ouro": ["Recife"], "resposta": answer}


def test_report_loads_exported_jsonl_and_rescores(tmp_path):
    path = tmp_path / "variant.jsonl"
    path.write_text(json.dumps(dict(row(), f1_locomo=-1)), encoding="utf-8")
    rows = report_module().load(path)
    assert rows["q1"]["f1_locomo"] == 1
    assert rows["q1"]["bleu1_locomo"] == 1


def test_report_refuses_duplicate_question_ids(tmp_path):
    folder = tmp_path / "locomo"
    folder.mkdir()
    for name in ["first", "second"]:
        (folder / (name + ".jsonl")).write_text(json.dumps(row()), encoding="utf-8")
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        report_module().load(tmp_path)


def test_report_refuses_missing_questions_or_changed_gold():
    report = report_module()
    with pytest.raises(ValueError, match="sets differ"):
        report.paired_ids({}, {"q1": row()})
    with pytest.raises(ValueError, match="respostas_ouro"):
        report.paired_ids({"q1": dict(row(), respostas_ouro=["London"])}, {"q1": row()})


def test_bootstrap_resamples_conversations_together():
    report = report_module()
    # One small conversation wins every question; the large one loses every
    # question. Resampling individual questions would falsely narrow the CI.
    ids = ["locomo:one:qa1"] + [f"locomo:two:qa{i}" for i in range(20)]
    rows = {q: {"f1_locomo": float(":one:" in q)} for q in ids}
    ref = {q: {"f1_locomo": float(":two:" in q)} for q in ids}
    low, high, unit = report.paired_interval(rows, ref, ids, "f1_locomo")
    assert (low, high, unit) == (-100, 100, "conversations")


def test_reading_statistics_do_not_count_rejected_or_unproved_readings():
    report = report_module()
    rows = {str(i): {"diagnosticos": {"plano_final": {"leitura": 1},
            "motivo_parada": stop}} for i, stop in enumerate([
                "prova_confirmada", "prova_ja_no_contexto", "prova_recusada", "respostas_demais"])}
    assert report.plan_stats(rows, list(rows))["prova_por_outra_leitura"] == 2


def test_selected_facts_have_traceable_sources():
    r = retriever(FakeLLM([CHAIN]), fact_delivery="facts", fact_budget=6)
    r._build_dated_memory()
    _, _, info = r._fact_context(QUESTION, None, None, None, [])
    assert info["n"] == len(info["fontes"]) == len(info["indices"])
    assert {s["turn_id"] for s in info["fontes"]} == {f"D{i}:1" for i in range(1, 7)}


def test_fact_delivery_checks_a_proof_inside_the_retrieved_passages():
    llm = FakeLLM([CHAIN])
    r = retriever(llm, fact_delivery="facts")
    pids = ["p0", "p1", "p2", "p3", "p4", "p5"]
    result = r._retrieve_proof(QUESTION, 3, pids, [1 / (i + 1) for i in range(6)])
    assert "witness.confirm" in llm.stages()
    assert result.diagnostics["motivo_parada"] == "prova_confirmada"


def test_explicit_no_verify_is_still_respected_for_fact_delivery():
    llm = FakeLLM([CHAIN])
    r = retriever(llm, fact_delivery="facts", proof_verify=False)
    result = r._retrieve_proof(QUESTION, 3, *POOL)
    assert "witness.confirm" not in llm.stages()
    assert result.diagnostics["motivo_parada"] == "prova_sem_verificacao"


def test_explicitly_rejected_candidate_loses_plan_priority():
    llm = FakeLLM([CHAIN], verdict={"supported": [], "rejected": [
        {"id": "A1", "reason": "not_supported"}]})
    r = retriever(llm, fact_delivery="facts", fact_budget=6)
    result = r._retrieve_proof(QUESTION, 3, *POOL)
    assert result.diagnostics["motivo_parada"] == "prova_recusada"
    info = result.diagnostics["fatos_entregues"]
    assert info["plano"] == 0
    # Rejection of a candidate is not a global ban on its source facts.
    assert info["relevancia"] == 6


@pytest.mark.parametrize("same_time,different_kind,expected", [
    (False, False, 2), (True, False, 1), (True, True, 2),
])
def test_fact_dedup_keeps_distinct_events_and_modalities(same_time, different_kind, expected):
    r = retriever(FakeLLM([CHAIN]), fact_delivery="facts", fact_budget=6)
    r._build_dated_memory()
    for index in (2, 3):
        r.memory.facts[index].statement = "Bruno camped outdoors."
    if same_time:
        r.dated.fact_interval[3] = r.dated.fact_interval[2]
    if different_kind:
        r.memory.facts[3].kind = "plan"
    text, _, _ = r._fact_context(QUESTION, None, None, None, [])
    assert text.count("Bruno camped outdoors.") == expected


def test_proof_priority_is_a_tier_even_with_negative_cosine():
    r = retriever(FakeLLM([CHAIN]), fact_delivery="facts", fact_budget=1)
    r._build_dated_memory()
    vector = r.ctx.embedder.encode([QUESTION.question])[0]
    r.memory.fact_vectors[:] = vector
    r.memory.fact_vectors[0] = -vector
    proof = {"selecionadas": [(0, SimpleNamespace(facts=(0,)))]}
    _, _, info = r._fact_context(QUESTION, None, proof, None, ["p1"])
    assert info["prova"] == 1


def test_reranking_cannot_move_an_unproved_fact_ahead_of_a_proof(monkeypatch):
    import wrag.witness.rerank as module
    r = retriever(FakeLLM([CHAIN]), fact_delivery="facts", fact_rerank="fake")
    r._build_dated_memory()
    monkeypatch.setattr(module, "get_reranker", lambda _: SimpleNamespace(
        score=lambda question, texts: [1.0, 0.0]))
    ranked, _ = r._rerank_facts(QUESTION, [1, 0], np.zeros(6), {1: 1.0, 0: 2.0})
    assert ranked == [0, 1]
