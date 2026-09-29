"""Paired protocol and crash-safe resumption, using a deterministic fake model."""
import copy
import json
import re
from pathlib import Path

import pytest

from test_query_memory import fixture_memory, target_data
from wrag.data import Question
from wrag.eval.memory_variant import check_manifest, load_source, run_experiment
from wrag.llm import LLM, LLMResult


class ExperimentLLM(LLM):
    def __init__(self, fail_reflection=None):
        super().__init__()
        self.calls = []
        self.reflections = 0
        self.fail_reflection = fail_reflection

    def _complete(self, messages, params, stage="misc"):
        prompt = messages[-1]["content"]
        self.calls.append((stage, prompt))
        base_stage = stage.removesuffix(".repair")
        if base_stage == "memory.target":
            data = target_data()
        elif base_stage == "memory.reflect":
            self.reflections += 1
            if self.fail_reflection is not None and self.reflections in (self.fail_reflection, self.fail_reflection+1):
                data = {"conclusions":None, "conflicts":[], "missing":[]}
            else:
                label = re.search(r"\n(F\d+) \(", prompt)[1]
                data = {"conclusions":[{"text":"Pomodoro", "kind":"inference", "premises":[label],
                                          "bridge":"The described study and break pattern names this technique."}],
                        "conflicts":[], "missing":[]}
        elif stage == "qa":
            data = {"answer":"Pomodoro"}
        else:
            raise AssertionError(stage)
        return LLMResult(text=json.dumps(data), prompt_tokens=100, completion_tokens=20,
                         finish_reason="stop", latency_s=.01)


def experiment_inputs(n_questions=2):
    corpus, facts, dated, diagnostics = fixture_memory()
    corpus.questions = [Question(f"q{i}", "Which technique does Tim use?", ["Pomodoro"],
                                 dataset="locomo", qtype="open-domain") for i in range(n_questions)]
    diagnostics.update(leitura_fatos="bitemporal", trechos_extras=[
        {"title":"Original complete memory", "text":"\n".join(f.statement for f in facts) + "\nOriginal source turns: literal support."}])
    baseline = {q.qid:{"qid":q.qid, "pergunta":q.question, "tipo":q.qtype,
                       "diagnosticos":copy.deepcopy(diagnostics), "latencia_recuperacao_s":8,
                       "f1_locomo":.5} for q in corpus.questions}
    return corpus, facts, dated, baseline


def test_three_arms_share_target_executor_and_reader_evidence(tmp_path):
    corpus, facts, dated, baseline = experiment_inputs()
    original = copy.deepcopy(baseline)
    llm = ExperimentLLM()
    result = run_experiment(llm, corpus, facts, dated, baseline, {"source":"test"}, tmp_path)
    assert result["complete"] and result["paired_n"] == 2
    assert [s for s,_ in llm.calls].count("memory.target") == 2
    assert [s for s,_ in llm.calls].count("memory.reflect") == 6
    assert [s for s,_ in llm.calls].count("qa") == 6
    assert result["execution_attempt_usage"]["total"]["chamadas"] == 14
    assert baseline == original
    for budget in (10,20,40):
        arm = result["arms"][str(budget)]
        assert arm["facts_delivered"] == budget and arm["memory_tokens"] == 240
        assert arm["reader_tokens"] == 120 and arm["f1_locomo"] == 1
        assert (tmp_path/f"facts-{budget}/predictions.jsonl").exists()
    assert len(result["paired_differences"]) == 3
    for stage,prompt in llm.calls:
        if stage == "qa":
            assert "method 39 with a special qualifier" in prompt
            assert "Memory interpretations" in prompt
            assert "Perform reflection internally" not in prompt
        else:
            assert "respostas_ouro" not in prompt and "SECRET_GOLD" not in prompt


def test_complete_resume_makes_no_calls(tmp_path):
    args = experiment_inputs()
    run_experiment(ExperimentLLM(), *args, {"source":"test"}, tmp_path)
    llm = ExperimentLLM()
    result = run_experiment(llm, *args, {"source":"test"}, tmp_path, resume=True)
    assert result["complete"] and not llm.calls
    assert result["execution_attempt_usage"]["total"]["chamadas"] == 14


def test_partial_resume_keeps_shared_target_and_completed_arm(tmp_path):
    args = experiment_inputs(1)
    with pytest.raises(ValueError, match="malformed reflection envelope"):
        run_experiment(ExperimentLLM(fail_reflection=2), *args, {"source":"test"}, tmp_path)
    partial = json.loads((tmp_path/"comparison.json").read_text(encoding="utf-8"))
    assert not partial["complete"] and partial["paired_n"] == 0
    llm = ExperimentLLM()
    result = run_experiment(llm, *args, {"source":"test"}, tmp_path, resume=True)
    assert result["complete"] and len(llm.calls) == 4
    assert all(s != "memory.target" for s,_ in llm.calls)
    assert result["execution_attempt_usage"]["total"]["chamadas"] == 9


def test_memory_only_never_calls_reader(tmp_path):
    args = experiment_inputs()
    llm = ExperimentLLM()
    result = run_experiment(llm, *args, {"source":"test"}, tmp_path, memory_only=True)
    assert result["complete"] and result["memory_only"]
    assert len(llm.calls) == 8 and all(s != "qa" for s,_ in llm.calls)
    assert all(s["f1_locomo"] is None and s["reader_tokens"] == 0 for s in result["arms"].values())


def test_incompatible_resume_rejected_before_generation(tmp_path):
    args = experiment_inputs(1)
    run_experiment(ExperimentLLM(), *args, {"source":"test"}, tmp_path)
    llm = ExperimentLLM()
    with pytest.raises(ValueError, match="differs"):
        run_experiment(llm, *args, {"source":"test"}, tmp_path, body="triple", resume=True)
    assert not llm.calls


def test_nonempty_output_without_manifest_rejected(tmp_path):
    (tmp_path/"unrelated.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="without"):
        check_manifest(tmp_path, {"identity":1}, True)
    assert (tmp_path/"unrelated.txt").read_text() == "preserve me"


def test_empty_failed_run_may_resume_after_code_only_fix(tmp_path):
    old = {"code_hash":"before", "prompts_hash":"unchanged", "model":"same"}
    check_manifest(tmp_path, old, False)
    (tmp_path/"attempts").mkdir()
    attempt = tmp_path/"attempts/first.json"
    attempt.write_text('{"usage":{"total":{"chamadas":1}}}', encoding="utf-8")
    new = {**old, "code_hash":"after"}
    check_manifest(tmp_path, new, True)
    assert json.loads((tmp_path/"manifest.json").read_text()) == new
    assert list((tmp_path/"manifest-history").glob("*.json"))
    assert json.loads(attempt.read_text())["usage"]["total"]["chamadas"] == 1


@pytest.mark.parametrize("artifact", ["shared/target.json", "facts-10/results/question.json"])
def test_code_fix_cannot_migrate_run_with_results_or_shared_target(tmp_path, artifact):
    old = {"code_hash":"before", "prompts_hash":"unchanged"}
    check_manifest(tmp_path, old, False)
    path = tmp_path/artifact; path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        check_manifest(tmp_path, {**old, "code_hash":"after"}, True)


def test_explicit_code_update_preserves_completed_results(tmp_path):
    args = experiment_inputs(1)
    run_experiment(ExperimentLLM(), *args, {"source":"test"}, tmp_path, code_hash="before")
    existing = {str(p):p.read_bytes() for p in tmp_path.glob("facts-*/results/*.json")}
    llm = ExperimentLLM()
    result = run_experiment(llm, *args, {"source":"test"}, tmp_path, code_hash="after",
                            resume=True, allow_code_update=True)
    assert result["complete"] and not llm.calls
    assert all(Path(p).read_bytes() == data for p,data in existing.items())
    assert list((tmp_path/"manifest-history").glob("*.json"))


def test_gold_answers_never_reach_prompts(tmp_path):
    args = experiment_inputs(1)
    args[0].questions[0].answers = ["GOLD_PRIVATE"]
    llm = ExperimentLLM()
    run_experiment(llm, *args, {"source":"test"}, tmp_path)
    assert all("GOLD_PRIVATE" not in prompt for _,prompt in llm.calls)


def source_fixture(tmp_path):
    from wrag.data import load_dataset
    corpus, facts, _, diagnostics = fixture_memory()
    root = tmp_path/"source"
    conv = root/"conversations/conv00"
    data_dir = conv/"data"; data_dir.mkdir(parents=True)
    raw_passages = [{"title":p.title, "text":p.text} for p in corpus.passages]
    (data_dir/"locomo_corpus.json").write_text(json.dumps(raw_passages), encoding="utf-8")
    (data_dir/"locomo.json").write_text(json.dumps([{
        "id":"q", "question":"Which technique does Tim use?", "answer":"Pomodoro",
        "type":"open-domain", "paragraphs":raw_passages}]), encoding="utf-8")
    loaded = load_dataset("locomo", data_dir=data_dir)
    pid = loaded.passages[0].pid
    for f in facts:
        f.pid = pid
    diagnostics["fatos_entregues"]["fontes"] = [
        {"indice":i, "fid":f.fid, "pid":pid, "turn_id":f.turn_id} for i,f in enumerate(facts)]
    diagnostics.update(leitura_fatos="bitemporal", trechos_extras=[
        {"title":"memory", "text":"\n".join(f.statement for f in facts)}])
    row = {"qid":"q", "pergunta":"Which technique does Tim use?", "respostas_ouro":["Pomodoro"],
           "diagnosticos":diagnostics, "filtrada":False}
    run = conv/"benchmark/test"; (run/"locomo").mkdir(parents=True)
    source_path = run/"locomo/witnessrag.jsonl"
    source_path.write_text(json.dumps(row)+"\n", encoding="utf-8")
    (run/"run.json").write_text(json.dumps({"config":{"qa":{}}}), encoding="utf-8")
    cache_dir = tmp_path/"cache"; (cache_dir/"openie").mkdir(parents=True)
    extraction = cache_dir/"openie/locomo-test.json"
    extraction.write_text(json.dumps({"facts":[{
        "fid":f.fid, "s":f.subject, "r":f.relation, "o":f.object,
        "pid":f.pid, "st":f.statement, "turn":f.turn_id, "kind":f.kind} for f in facts]}), encoding="utf-8")
    return root, cache_dir, extraction, source_path


def test_source_cache_and_literal_provenance_verified(tmp_path):
    root, cache, _, _ = source_fixture(tmp_path)
    (cache/"openie/locomo-unrelated-broken.json").write_text("{", encoding="utf-8")
    corpus, facts, dated, baseline, provenance = load_source(root, cache_dir=cache)
    assert len(facts) == 40 and len(baseline) == 1
    assert provenance["fact_sources_verified"]
    assert facts[0].pid == corpus.passages[0].pid


def test_source_cache_tampering_is_rejected(tmp_path):
    root, cache, path, _ = source_fixture(tmp_path)
    data = json.loads(path.read_text()); data["facts"][0]["fid"] = "different"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="matching"):
        load_source(root, cache_dir=cache)


def test_source_context_truncation_is_rejected(tmp_path):
    root, cache, _, path = source_fixture(tmp_path)
    row = json.loads(path.read_text()); row["diagnosticos"]["trechos_extras"][0]["text"] = "truncated"
    path.write_text(json.dumps(row)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        load_source(root, cache_dir=cache)
