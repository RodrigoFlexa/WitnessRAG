"""Cross-worker controls and output regressions, entirely offline."""
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from wrag import config as C
from wrag.eval import reflection_study as study
from wrag.eval.reader import _canonicalize_short_answer, read
from wrag.llm.base import LLMResult, UsageLedger
from wrag.methods.base import IndexContext, RetrievalResult
from wrag.methods.witnessrag import WitnessRAGRetriever
from test_proof_controller import QUESTION, POOL, dated_toy, retriever
from test_reflection_ablation import MemoryLLM, module


@pytest.mark.parametrize("answer,expected", [
    ("Saffron,Marble", "Saffron, Marble"),
    ("1,000", "1,000"), ("1,25", "1,25"),
    ("about six years", "about six years"), ("at age 12", "at age 12"),
])
def test_common_serialization_preserves_units_and_numeric_separators(answer, expected):
    assert _canonicalize_short_answer("What was mentioned?", answer) == expected


def test_reflective_reader_uses_simple_schema_and_preserves_complete_answer():
    corpus, _, _ = dated_toy()

    class Model:
        calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            assert "SECRET_GOLD" not in prompt
            assert "answer_kind" not in prompt
            assert prompt.index("Perform reflection") > prompt.index("MEMORY:")
            return LLMResult(text='{"answer":"about six years"}')

    llm = Model()
    question = replace(QUESTION, question="How long has Ana been hiking?")
    result = read(llm, corpus, question, ["p0"],
                  C.QAConfig(evidence_reader=True, reader_reflection=True), facts_mode=True)
    assert result.answer == "about six years"
    assert result.reflection == {"mode": "joint-v2", "answer_kind": "", "schema_valid": True}
    assert len(llm.calls) == 1


def test_two_os_processes_do_not_overwrite_each_others_shared_state(tmp_path):
    target = tmp_path / "counter.json"
    target.write_text('{"n":0}', encoding="utf-8")
    code = """
import json,sys
from pathlib import Path
from wrag.eval.reflection_study import file_lock,atomic_json
p=Path(sys.argv[1])
for _ in range(12):
    with file_lock(p.with_suffix('.lock')):
        value=json.loads(p.read_text(encoding='utf-8'))
        atomic_json(p,{'n':value['n']+1})
"""
    processes = [subprocess.Popen([sys.executable, "-c", code, str(target)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                 for _ in range(2)]
    for process in processes:
        _, error = process.communicate(timeout=45)
        assert process.returncode == 0, error.decode(errors="replace")
    assert json.loads(target.read_text())["n"] == 24


def test_graph_is_built_once_and_checked_across_worker_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("WRAG_REFLECTION_STUDY_ROOT", str(tmp_path))
    corpus, graph, embedder = dated_toy()
    class NamedModel(MemoryLLM):
        name = "offline"
    contexts = [IndexContext(corpus, NamedModel(), embedder, C.RunConfig()) for _ in range(2)]
    builds = []

    def build(ctx, **kwargs):
        builds.append(1)
        ctx.kg = copy.deepcopy(graph)

    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda ctx: study.freeze_graph(ctx, build), contexts))
    assert len(builds) == 1
    assert contexts[0].controlled_memory_hash == contexts[1].controlled_memory_hash
    payload = next(tmp_path.glob("shared/controlled/*/memory.pkl"))
    payload.write_bytes(payload.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        study.freeze_graph(contexts[1], build)
    assert len(builds) == 1


def test_four_cells_share_base_and_enriched_contexts_without_rerouting(tmp_path, monkeypatch):
    monkeypatch.setenv("WRAG_REFLECTION_STUDY_ROOT", str(tmp_path))

    class RoutedModel(MemoryLLM):
        def __init__(self):
            super().__init__()
            self.usage = UsageLedger()

        def chat(self, prompt, **kwargs):
            if kwargs.get("stage") == "witness.route":
                self.calls.append(("witness.route", prompt))
                return LLMResult(text='{"route":"DIRECT"}')
            return super().chat(prompt, **kwargs)

    monkeypatch.setattr(WitnessRAGRetriever, "_retrieve",
                        lambda self, q, k: self._retrieve_proof(q, k, *POOL))
    outputs, models = [], []
    # Start with summary enabled: it must still build a literal base first.
    for summary, reader_flag in [(True, False), (False, False), (False, True), (True, True)]:
        llm = RoutedModel()
        models.append(llm)
        r = retriever(llm, fact_delivery="facts+summary", summary_reflection=summary,
                      summary_reflection_limit=2 if summary else 4, plan_router="llm")
        r.ctx.controlled_memory_hash = "same-frozen-memory"
        r.ctx.run.qa.reader_reflection = reader_flag
        output = study.retrieve(r, r.corpus, QUESTION, r.ctx.run)
        outputs.append(output.diagnostics)
    assert sum(m.stages().count("witness.route") for m in models) == 1
    assert models[1].calls == models[2].calls == models[3].calls == []
    assert len({d["reflection_study"]["base_snapshot_hash"] for d in outputs}) == 1
    assert outputs[0]["trechos_extras"] == outputs[3]["trechos_extras"]
    assert outputs[1]["trechos_extras"] == outputs[2]["trechos_extras"]
    assert outputs[0]["trechos_extras"][0] == outputs[1]["trechos_extras"][0]
    assert outputs[0]["trechos_extras"][1]["text"].startswith(outputs[1]["trechos_extras"][1]["text"])
    assert "HIGH-LEVEL MEMORIES" in outputs[0]["trechos_extras"][1]["text"]
    # A changed retrieval configuration cannot reuse an old question snapshot.
    r.ctx.run.witness.fact_budget += 1
    with pytest.raises(ValueError, match="inputs changed"):
        study.retrieve(r, r.corpus, QUESTION, r.ctx.run)


def test_reflections_are_ranked_individually_with_global_delivery_limit(monkeypatch):
    r = retriever(MemoryLLM(), summary_reflection=True, summary_reflection_limit=2, fact_rerank="test")
    memories = [{"subject": "Ana", "inference": label, "bridge": "", "confidence": "likely",
                 "basis": [{"turn_id": "D1:1", "quote": "I work at Atlas now, I am so happy!"}]}
                for label in ("generic support", "creative hobbies", "specific geography", "relevant work")]
    monkeypatch.setattr(r, "_chunk_reflections", lambda pid: (memories, {"accepted": 4, "status": "validated"}))
    class Ranker:
        def score(self, question, texts):
            assert question == QUESTION.question and "SECRET_GOLD" not in question
            return [0.1, 0.2, 0.9, 0.8]
    monkeypatch.setattr("wrag.witness.rerank.get_reranker", lambda name: Ranker())
    text, info = r._reflect_summary_context(QUESTION, "literal summary", ["p0"])
    assert text.startswith("literal summary")
    assert "specific geography" in text and "relevant work" in text
    assert "generic support" not in text and "creative hobbies" not in text
    assert info[0]["generated"] == 4 and info[0]["accepted"] == 2


def test_snapshot_resume_and_corruption_do_not_trigger_new_retrieval(tmp_path):
    corpus, _, _ = dated_toy()
    path, calls = tmp_path / "q.json", []
    def build():
        calls.append(1)
        return RetrievalResult(pids=["p0"], scores=[.5], diagnostics={"source": "frozen"})
    first, _, hit = study.saved(path, {"q": "q"}, corpus, build)
    assert not hit
    second, _, hit = study.saved(path, {"q": "q"}, corpus, build)
    assert hit and first == second and len(calls) == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    data["retrieval"]["scores"] = [999]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        study.saved(path, {"q": "q"}, corpus, build)
    assert len(calls) == 1


def test_two_launcher_halves_keep_four_variants_and_same_namespace(tmp_path, monkeypatch):
    launch = module("run-reflection-ablation")
    complete, invocations = set(), []
    monkeypatch.setattr(launch.comparison, "completed", lambda p: p.name in complete)
    monkeypatch.setattr(launch.comparison, "compare_corpora", lambda *args: {"corpus": "same"})
    def execute(command, **kwargs):
        if "--output" in command:
            complete.add(Path(command[command.index("--output") + 1]).name)
            invocations.append((command, kwargs["env"]))
        return type("Completed", (), {"returncode": 0})()
    monkeypatch.setattr(launch.subprocess, "run", execute)
    common = ["--output", str(tmp_path), "--device", "cpu", "--ports", "8095,8096", "--gpus", "1,7"]
    for names in ("cascade,reader-reflection", "summary-reflection,both"):
        monkeypatch.setattr(launch.sys, "argv", ["launch", *common, "--only", names])
        assert launch.main() == 0
    assert len(invocations) == 4 and len(complete) == 4
    assert len({env["WRAG_EXPERIMENT_CACHE_ID"] for _, env in invocations}) == 1
    assert all(env["WRAG_REFLECTION_STUDY_ROOT"] == str(tmp_path.resolve()) for _, env in invocations)
    ports = {Path(cmd[cmd.index("--output") + 1]).name: cmd[cmd.index("--port") + 1] for cmd, _ in invocations}
    assert ports == {"cascade": "8095", "reader-reflection": "8095", "summary-reflection": "8096", "both": "8096"}


def test_report_rejects_changed_snapshot_even_if_fact_text_is_identical(tmp_path):
    from test_reflection_ablation import synthetic_rows
    synthetic_rows(tmp_path)
    for name in ("cascade", "summary-reflection", "reader-reflection", "both"):
        path = tmp_path / name / "locomo/witnessrag.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["diagnosticos"]["reflection_study"] = {
                "base_snapshot_hash": "same-base", "memory_hash": "same-memory"}
            if name in {"summary-reflection", "both"}:
                row["diagnosticos"]["trechos_extras"][1]["text"] = "literal\ninferred"
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = module("reflection-ablation-report")
    assert report.report(tmp_path)["retrieval_control"]["mismatches"] == 0
    path = tmp_path / "both/locomo/witnessrag.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("same-memory", "different-memory"), encoding="utf-8")
    with pytest.raises(ValueError, match="recuperação mudou"):
        report.report(tmp_path)


def test_trim_keeps_shared_control_and_cost_fields():
    from wrag.eval.runner import _trim
    control = {"base_snapshot_hash": "same", "memory_hash": "same", "shared_retrieval_usage": {"total": {}}}
    result = _trim({"reflection_study": control, "unused": "x" * 20000})
    assert result["reflection_study"] == control
