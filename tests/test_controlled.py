"""Failure-boundary tests for the server experiment, using the real offline pipeline."""
import json
from dataclasses import replace

import pytest

from test_pipeline import offline
from wrag import config as C
from wrag.data import Question
from wrag.eval import controlled, runner
from wrag.llm.base import GenParams, LLMResult
from wrag.llm.openai_compat import OpenAICompatLLM
from wrag.llm.stub import StubLLM
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.research import assess_plan_soft


def test_placeholder_repair_never_silently_promotes():
    class Judge:
        def __init__(self, valid=False):
            self.calls = 0
            self.valid = valid

        def chat(self, prompt, **kwargs):
            self.calls += 1
            assert "SECRET_GOLD" not in prompt
            return LLMResult(text=json.dumps({"tier": "full", "missing":
                [] if self.valid and self.calls == 2 else ["short condition"], "reason": "test"}))

    q = Question("q", "Where does Ana work?", ["SECRET_GOLD"])
    plan = ConjunctiveQuery(atoms=[Atom("works at", "Ana", "?x")])
    invalid = Judge()
    result = assess_plan_soft(invalid, q, plan, "toy", "witnessrag")
    assert invalid.calls == 2 and result.tier == "unavailable" and not result.covers
    assert result.attempts[0]["error"] == "placeholder"
    repaired = Judge(valid=True)
    result = assess_plan_soft(repaired, q, plan, "toy", "witnessrag")
    assert repaired.calls == 2 and result.covers and result.missing == []


def test_cache_namespace_ignores_port_only_when_explicit(monkeypatch):
    llm = object.__new__(OpenAICompatLLM)
    llm.deployment = "model"
    monkeypatch.delenv("WRAG_EXPERIMENT_CACHE_ID", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8089/v1")
    old = llm.cache_identity()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8090/v1")
    assert llm.cache_identity() != old
    monkeypatch.setenv("WRAG_EXPERIMENT_CACHE_ID", "frozen-run")
    first = llm.cache_identity()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9000/v1")
    assert llm.cache_identity() == first
    llm.deployment = "different-model"
    assert llm.cache_identity() != first


def test_tape_rejects_changed_and_missing_requests():
    llm = StubLLM()
    with controlled.tape(llm) as calls:
        expected = llm.chat("hello", params=GenParams(), stage="qa")
    with controlled.tape(llm, calls):
        assert llm.chat("hello", params=GenParams(), stage="qa").text == expected.text
    with pytest.raises(ValueError, match="diverged"):
        with controlled.tape(llm, calls):
            llm.chat("changed", stage="qa")
    with pytest.raises(ValueError, match="fewer"):
        with controlled.tape(llm, calls):
            pass


def test_controlled_pipeline_freezes_four_cells_and_resumes(offline, monkeypatch):
    tmp, llm, _ = offline
    out = tmp / "controlled-experiment"
    monkeypatch.setenv("WRAG_CONTROLLED_ROOT", str(out))
    data = json.loads((tmp / "sample.json").read_text())
    for index, row in enumerate(data):
        row["type"] = "single-hop" if index == 0 else "multi-hop"
    (tmp / "sample.json").write_text(json.dumps(data), encoding="utf-8")
    cfg = C.RunConfig(n_questions=3, top_k=2)
    cfg.witness.active_obligations = True
    cfg.witness.active_context = True
    cfg.witness.enable_acquisition = False
    cfg.witness.answer_set = True
    cfg.qa.answer_set = True
    first = runner.run(["sample"], ["witnessrag"], cfg)
    folder = next((out / "controlled").iterdir())
    audit = json.loads((folder / "audit.json").read_text())
    assert audit["passed"] and len(audit["sample"]) == 6
    answers = sorted((folder / "answers").glob("*.json"))
    assert len(answers) == 3
    for path in answers:
        cells = json.loads(path.read_text())["cells"]
        assert len(cells) == 4
        for variant in ("evidence", "soft-v2"):
            a, b = [cells[f"{variant}/{mode}"] for mode in ("common", "proof")]
            assert a["recuperadas"] == b["recuperadas"]
            assert a["retrieval_hash"] == b["retrieval_hash"]
    before = {p: p.read_bytes() for p in answers}
    # Simulate a reader-stage interruption, then a new pilot benchmark directory.
    lost = answers[0]
    lost.unlink()
    runner.run(["sample"], ["witnessrag"], cfg)
    assert {p: p.read_bytes() for p in answers} == before
    # A corrupted memory must fail rather than silently rebuild a new graph.
    with (folder / "memory.pkl").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        runner.run(["sample"], ["witnessrag"], cfg)


def test_official_report_validates_denominators_and_snapshots(offline, monkeypatch):
    from wrag.controlled import report, IncompleteExperiment
    tmp, _, _ = offline
    output = tmp / "experiment"
    monkeypatch.setenv("WRAG_CONTROLLED_ROOT", str(output))
    data = json.loads((tmp / "sample.json").read_text())
    for index, item in enumerate(data):
        item["id"] = f"locomo:conv-toy:qa{index}"
        item["type"] = "single-hop" if index == 0 else "multi-hop"
    (tmp / "locomo.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp / "locomo_corpus.json").write_bytes((tmp / "sample_corpus.json").read_bytes())
    cfg = C.RunConfig(n_questions=3, top_k=2)
    cfg.witness.active_obligations = True
    cfg.witness.enable_acquisition = False
    runner.run(["locomo"], ["witnessrag"], cfg)
    controlled.atomic_json(output / "controlled-manifest.json", {"config": {"conversation": "0"}})
    result = report(output)
    assert result["complete"] and len(result["cells"]) == 4
    assert all(cell["n"] == 3 for cell in result["cells"].values())
    path = next((output / "controlled").glob("*/answers/*.json"))
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(IncompleteExperiment):
        report(output)
    path.write_bytes(original)
    snapshot = next((output / "controlled").glob("*/retrieval/evidence/*.json"))
    bad = controlled.load(snapshot)
    bad["retrieval"]["scores"] = [999]
    controlled.atomic_json(snapshot, bad)
    with pytest.raises(ValueError, match="alterado"):
        report(output)
