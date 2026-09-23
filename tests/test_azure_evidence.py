"""Offline contracts for the Azure LoCoMo route and evidence-aware reader."""

import json

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.eval.reader import read
from wrag.llm.base import LLMResult
from wrag.llm.filters import Ledger
from wrag.pilot import _run_config, make_plan, parser


def test_azure_plan_keeps_credentials_out_of_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-secret")
    args = parser().parse_args([
        "--gpu", "1", "--backend", "azure", "--model", "gpt-4-1-mini-petrobras",
        "--dataset", "locomo", "--evidence-reader", "--gap-context-rescue",
        "--locomo-ie-window-tokens", "512"])
    plan = make_plan(args, tmp_path)
    cfg = _run_config(plan["settings"], 2)
    assert plan["env"]["WRAG_LLM_BACKEND"] == "azure"
    assert plan["env"]["WRAG_AZURE_DEPLOYMENT"] == "gpt-4-1-mini-petrobras"
    assert "AZURE_OPENAI_API_KEY" not in plan["env"]
    assert "test-secret" not in json.dumps(plan)
    assert cfg.ie.window_tokenizer == "Qwen/Qwen2.5-14B-Instruct"
    assert cfg.qa.evidence_reader and cfg.witness.gap_context_rescue


def test_evidence_reader_uses_one_call_for_any_locomo_category():
    class CaptureLLM:
        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return LLMResult(text='{"answer":"Independence Day"}')

    llm = CaptureLLM()
    corpus = Corpus("locomo", [Passage("p1", "dialogue", "Maria mentioned a holiday.")], [])
    cfg = C.QAConfig(evidence_reader=True, top_k=5)
    for category in ("single-hop", "multi-hop", "temporal", "open-domain"):
        question = Question("q", "Around which US holiday?", ["Independence Day"],
                            dataset="locomo", qtype=category)
        assert read(llm, corpus, question, ["p1"], cfg).answer == "Independence Day"
    assert len(llm.prompts) == 4
    assert all("not yes/no" in prompt for prompt in llm.prompts)


def test_evidence_reader_canonicalizes_only_safe_short_answer_types():
    class VerboseLLM:
        def __init__(self, answer):
            self.answer = answer

        def chat(self, prompt, **kwargs):
            return LLMResult(text=json.dumps({"answer": self.answer}))

    corpus = Corpus("locomo", [Passage("p1", "dialogue", "Melanie has three children.")], [])
    cfg = C.QAConfig(evidence_reader=True, top_k=1)
    count = Question("q1", "How many children does Melanie have?", ["3"],
                     dataset="locomo", qtype="single-hop")
    assert read(VerboseLLM("Melanie has three children."), corpus, count,
                ["p1"], cfg).answer == "3"
    yes_no = Question("q2", "Does Melanie have children?", ["yes"],
                      dataset="locomo", qtype="single-hop")
    assert read(VerboseLLM("Yes, she has three."), corpus, yes_no,
                ["p1"], cfg).answer == "yes"
    entity = Question("q3", "Who has three children?", ["Melanie"],
                      dataset="locomo", qtype="single-hop")
    assert read(VerboseLLM("Melanie has three children."), corpus, entity,
                ["p1"], cfg).answer == "Melanie has three children."


def test_content_policy_block_is_question_specific(tmp_path):
    ledger = Ledger(tmp_path / "filters.jsonl")
    ledger.add("obligations", "locomo", "witnessrag", "q1", "content_filter")
    assert ledger.question_blocked("locomo", "witnessrag", "q1")
    assert not ledger.question_blocked("locomo", "witnessrag", "q2")
    assert not ledger.question_blocked("locomo", "dense", "q1")


def test_azure_cache_is_scoped_to_gateway_and_version(tmp_path, monkeypatch):
    from wrag.llm.azure import AzureLLM
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(AzureLLM, "_make_client", lambda self: None)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://gateway-a.example/openai")
    first = AzureLLM(deployment="gpt-4-1-mini-petrobras")._cache_path({
        "model": "gpt-4-1-mini-petrobras", "messages": []})
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://gateway-b.example/openai")
    second = AzureLLM(deployment="gpt-4-1-mini-petrobras")._cache_path({
        "model": "gpt-4-1-mini-petrobras", "messages": []})
    assert first != second


def test_azure_policy_filter_never_aborts_batch(tmp_path, monkeypatch):
    from wrag.llm.azure import AzureLLM
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(C, "HEALTH_CHECK_CALLS", 1)
    monkeypatch.setattr(C, "MAX_FILTER_RATE", 0.1)
    monkeypatch.setattr(AzureLLM, "_make_client", lambda self: None)
    llm = AzureLLM(deployment="gpt-4-1-mini-petrobras")
    assert llm._filtered_result("content_filter", tmp_path / "unused").filtered
    assert llm._filtered_result("content_filter", tmp_path / "unused").filtered


def test_json_word_error_does_not_disable_structured_mode(monkeypatch):
    from wrag.llm.azure import AzureLLM
    monkeypatch.setattr(AzureLLM, "_make_client", lambda self: None)
    llm = AzureLLM(deployment="gpt-4-1-mini-petrobras")
    class BadRequest(Exception):
        status_code = 400
    error = BadRequest("'messages' must contain the word 'json' in some form, "
                       "to use 'response_format' of type 'json_object'.")
    kwargs = {"response_format": {"type": "json_object"}}
    assert not llm._maybe_drop_parameter(error, kwargs)
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in llm._unsupported
