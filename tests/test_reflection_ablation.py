"""Offline behavioral/experimental controls; no model, GPU or API calls."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wrag import config as C, prompts
from wrag.data import Corpus, Passage, Question
from wrag.eval.reader import read
from wrag.llm.base import LLMResult
from wrag.pilot import _run_config, _validate_resume, make_plan, parser
from wrag.witness.reflection import render_memories, validate_memories
from test_proof_controller import CHAIN, FakeLLM, retriever

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "Session date: 8 May 2023\n[D1:1] Mira: I love inventing new designs for my ceramics.\n[D1:2] Kofi: I prefer solving puzzles."
MEMORY = {"subject": "Mira", "inference": "Mira likely enjoys creative expression.",
          "confidence": "likely", "bridge": "", "basis": [
              {"turn_id": "D1:1", "quote": "I love inventing new designs for my ceramics."}]}


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), ROOT / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_citation_must_belong_to_its_own_turn():
    memories, info = validate_memories({"memories": [MEMORY]}, SOURCE)
    assert len(memories) == 1 and info["accepted"] == 1
    wrong = copy.deepcopy(MEMORY)
    wrong["basis"][0]["turn_id"] = "D1:2"  # quote exists, but in a different turn
    assert validate_memories({"memories": [wrong]}, SOURCE)[0] == []
    wrong["basis"][0] = {"turn_id": "D1:1", "quote": "Mira likes creating ceramics."}
    assert validate_memories({"memories": [wrong]}, SOURCE)[0] == []


@pytest.mark.parametrize("field,value", [("confidence", "certain"), ("subject", "Unknown"),
                                         ("basis", []), ("inference", "word " * 26)])
def test_invalid_or_untraceable_interpretations_are_not_delivered(field, value):
    memory = {**MEMORY, field: value}
    assert validate_memories({"memories": [memory]}, SOURCE)[0] == []


def test_duplicate_and_malformed_memories_do_not_become_facts():
    memories, info = validate_memories({"memories": [MEMORY, MEMORY, None]}, SOURCE)
    assert len(memories) == 1 and info["rejected"] == 2
    assert "tentative interpretations" in render_memories(memories)
    assert "[D1:1]" in render_memories(memories)
    assert validate_memories({"answer": "something"}, SOURCE)[1]["status"] == "invalid_schema"


class MemoryLLM(FakeLLM):
    def __init__(self, response=None):
        super().__init__([CHAIN])
        self.response = response

    def chat(self, prompt, **kwargs):
        if kwargs.get("stage") == "memory.summary":
            self.calls.append(("memory.summary", prompt))
            return LLMResult(text="Ana discussed her new job at Atlas.")
        if kwargs.get("stage") == "memory.reflection":
            self.calls.append(("memory.reflection", prompt))
            if self.response is not None:
                return self.response
            return LLMResult(text=json.dumps({"memories": [{
                "subject": "Ana", "inference": "Ana likely enjoys her current job.", "confidence": "likely",
                "bridge": "", "basis": [{"turn_id": "D1:1", "quote": "I work at Atlas now, I am so happy!"}]}]}))
        return super().chat(prompt, **kwargs)


def test_summary_cache_keeps_baseline_and_reflects_once_without_questions():
    llm = MemoryLLM()
    r = retriever(llm)
    original = r._chunk_summary("p0")
    edges = r.kg.graph.number_of_edges() if hasattr(r.kg, "graph") else len(r.memory.facts)
    r.ctx.run.witness.summary_reflection = True
    enriched = r._chunk_summary("p0")
    assert enriched.startswith(original) and "Ana likely enjoys" in enriched
    assert r._chunk_summary("p0") == enriched
    assert llm.stages().count("memory.summary") == 1
    assert llm.stages().count("memory.reflection") == 1
    assert all("SECRET_GOLD" not in text for _, text in llm.calls)
    assert "Where is the company" not in llm.calls[-1][1]
    assert (r.kg.graph.number_of_edges() if hasattr(r.kg, "graph") else len(r.memory.facts)) == edges
    r.ctx.run.witness.summary_reflection = False
    assert r._chunk_summary("p0") == original


@pytest.mark.parametrize("response", [LLMResult(text="invalid"), LLMResult(filtered=True),
                                     LLMResult(text='{"memories":[]}')])
def test_failed_reflection_preserves_literal_summary(response):
    r = retriever(MemoryLLM(response), summary_reflection=True)
    assert r._chunk_summary("p0") == "Ana discussed her new job at Atlas."
    assert r._reflection_diagnostics["p0"]["accepted"] == 0


def test_joint_reader_preserves_a_named_option_starting_with_yes():
    class Stub:
        calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return LLMResult(text='{"answer_kind":"choice","answer":"Yes, Minister"}')

    question = Question("q", "Would Mira watch Yes, Minister or a horror movie?", ["SECRET_GOLD"], dataset="locomo")
    corpus = Corpus("locomo", [Passage("p", "Memory", SOURCE)], [])
    llm = Stub()
    result = read(llm, corpus, question, ["p"], C.QAConfig(evidence_reader=True,
                         reader_reflection=True, max_tokens=128), facts_mode=True)
    assert result.answer == "Yes, Minister" and result.reflection["schema_valid"]
    assert len(llm.calls) == 1 and llm.calls[0][1]["stage"] == "qa"
    assert llm.calls[0][1]["params"].max_tokens == 128
    assert "SECRET_GOLD" not in llm.calls[0][0]
    assert "Perform reflection" in llm.calls[0][0]


def test_baseline_reader_template_and_behavior_remain_unchanged():
    class Stub:
        def chat(self, prompt, **kwargs):
            self.prompt = prompt
            return LLMResult(text='{"answer":"likely yes"}')

    llm = Stub()
    question = Question("q", "Would Mira enjoy ceramics?", [], dataset="locomo")
    corpus = Corpus("locomo", [Passage("p", "Memory", SOURCE)], [])
    result = read(llm, corpus, question, ["p"], C.QAConfig(evidence_reader=True), facts_mode=True)
    expected = prompts.qa_facts_template().format(passages=prompts.format_passages([("Memory", SOURCE)]),
                                                 question=question.question)
    assert llm.prompt == expected and result.answer == "likely yes" and result.reflection is None


def test_all_cells_have_same_cascade_flags_and_independent_factors(tmp_path):
    run = module("run-reflection-ablation")
    args = run.parser().parse_args(["--output", str(tmp_path), "--device", "cpu"])
    commands = run.build_commands(args)
    configs = {}
    for name, command in commands.items():
        plan = make_plan(parser().parse_args(command[3:]), tmp_path / name)
        cfg = configs[name] = _run_config(plan["settings"], 10)
        assert cfg.witness.plan_router == "llm" and not cfg.witness.multiplan_portfolio
        assert cfg.qa.max_tokens == 128 and cfg.witness.fact_budget == 40
        assert cfg.witness.fact_delivery == "facts+summary"
        assert "--existing-server" in command
    assert [(configs[n].witness.summary_reflection, configs[n].qa.reader_reflection)
            for n in run.VARIANTS] == list(run.VARIANTS.values())
    old = make_plan(parser().parse_args(commands["cascade"][3:]), tmp_path / "cascade")
    new = copy.deepcopy(old)
    new["settings"]["reader_reflection"] = True
    with pytest.raises(ValueError, match="reader_reflection"):
        _validate_resume(old, new)


def test_flags_require_appropriate_delivery_and_reader(tmp_path):
    with pytest.raises(ValueError, match=r"facts\+summary"):
        make_plan(parser().parse_args(["--gpu", "0", "--summary-reflection"]), tmp_path)
    with pytest.raises(ValueError, match="evidence-reader"):
        make_plan(parser().parse_args(["--gpu", "0", "--reader-reflection"]), tmp_path)


@pytest.mark.parametrize("route", ["DIRECT", "PLAN"])
def test_summary_interpretations_reach_reader_in_both_cascade_routes(route):
    from test_proof_controller import QUESTION, POOL

    class RoutedMemoryLLM(MemoryLLM):
        def chat(self, prompt, **kwargs):
            if kwargs.get("stage") == "witness.route":
                self.calls.append(("witness.route", prompt))
                return LLMResult(text=json.dumps({"route": route}))
            return super().chat(prompt, **kwargs)

    llm = RoutedMemoryLLM()
    r = retriever(llm, fact_delivery="facts+summary", summary_reflection=True, plan_router="llm")
    result = r._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert d["roteador"]["rota"] == route
    assert d["trechos_extras"][0]["title"] == "Facts from the memory"
    assert "HIGH-LEVEL MEMORIES" in d["trechos_extras"][1]["text"]
    assert any(item["accepted"] for item in d["fatos_entregues"]["summary_reflection"])


def synthetic_rows(root):
    answers = {"cascade": "green", "summary-reflection": "red", "reader-reflection": "blue", "both": "red blue"}
    for name, answer in answers.items():
        path = root / name / "locomo" / "witnessrag.jsonl"
        path.parent.mkdir(parents=True)
        rows = [{"qid": f"locomo:conv-{conv}:qa1", "pergunta": "Which colors?", "tipo": "open-domain",
                 "respostas_ouro": ["red blue"], "resposta": answer,
                 "diagnosticos": {"roteador": {"rota": "DIRECT"},
                     "fatos_entregues": {"resumos_selecionados": ["p1"]}, "trechos_extras": [
                     {"title": "Facts from the memory", "text": "same facts"},
                     {"title": "Chunk summaries", "text": "inferred" if name in {"summary-reflection", "both"} else "literal"}]}}
                for conv in ("a", "b")]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_factorial_report_checks_retrieval_and_calculates_interaction(tmp_path):
    synthetic_rows(tmp_path)
    report = module("reflection-ablation-report")
    result = report.report(tmp_path)
    assert result["retrieval_control"]["mismatches"] == 0
    stats = result["categories"]["all"]
    assert stats["variants"]["both"]["f1_locomo"] == 100
    interaction = stats["contrasts"]["Interação (ambos − resumos − reader + baseline)"]["delta_f1"]
    assert interaction == pytest.approx(-100 / 3)
    assert (tmp_path / "reflection-ablation.md").exists()
    p = tmp_path / "both/locomo/witnessrag.jsonl"
    p.write_text(p.read_text().replace("same facts", "different facts"), encoding="utf-8")
    with pytest.raises(ValueError, match="recuperação mudou"):
        report.report(tmp_path)


def test_launcher_resumes_and_refuses_changed_protocol(tmp_path, monkeypatch):
    run = module("run-reflection-ablation")
    completed, invocations = set(), []
    monkeypatch.setattr(run.comparison, "completed", lambda p: p.name in completed)
    monkeypatch.setattr(run.comparison, "compare_corpora", lambda *a: {"corpus": "same"})

    def execute(command, **kwargs):
        invocations.append(command)
        if "--output" in command:
            folder = Path(command[command.index("--output") + 1])
            completed.add(folder.name)
        return SimpleNamespace(returncode=0)

    root = tmp_path / "study"
    (root / "cascade").mkdir(parents=True)
    # Without the manifest, pre-existing results cannot silently enter a study.
    monkeypatch.setattr(run.sys, "argv", ["run", "--output", str(root), "--device", "cpu"])
    with pytest.raises(SystemExit, match="sem manifesto"):
        run.main()
    # Use a fresh root for the actual resume scenario.
    root = tmp_path / "fresh"
    monkeypatch.setattr(run.sys, "argv", ["run", "--output", str(root), "--device", "cpu"])
    monkeypatch.setattr(run.subprocess, "run", execute)
    assert run.main() == 0
    assert len([c for c in invocations if "--output" in c]) == 4
    assert run.main() == 0
    assert len([c for c in invocations if "--output" in c]) == 4
    completed.remove("both")
    (root / "both").mkdir()
    (root / "both/pilot.json").write_text("{}")
    assert run.main() == 0
    assert "--resume" in [c for c in invocations if "--output" in c][-1]
    monkeypatch.setattr(run.sys, "argv", ["run", "--output", str(root), "--device", "cpu", "--reflection-limit", "2"])
    with pytest.raises(SystemExit, match="Código/configuração mudou"):
        run.main()
