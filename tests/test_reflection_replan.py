"""Single-retry limits, provenance-only retention, budget, isolation and no gold."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.ie import Fact
from wrag.llm.base import LLM, LLMResult
from wrag.methods.base import IndexContext, RetrievalResult, Retriever
from wrag.witness.reflection_replan import adaptive_read, parse_gate


class Replies(LLM):
    def __init__(self, replies):
        super().__init__()
        self.replies, self.calls = iter(replies), []
    def _complete(self, messages, params, stage="misc"):
        assert "SECRET_GOLD" not in str(messages)
        self.calls.append((messages, params, stage))
        return LLMResult(text=json.dumps(next(self.replies)), prompt_tokens=10, completion_tokens=3,
                         latency_s=.1)


class Memory(Retriever):
    def __init__(self, llm):
        corpus = Corpus("locomo", [Passage(f"p{i}", "", f"Iris visited City{i}.", session_time="2023-07-01")
                                   for i in range(30)], [])
        cfg = C.RunConfig(dataset="locomo", top_k=5)
        cfg.witness.local_plans, cfg.witness.local_plan_version = True, "v2"
        cfg.witness.fact_budget, cfg.witness.fact_delivery = 10, "facts"
        cfg.qa = C.QAConfig(evidence_reader=True, reader_reflection=True, reflection_replan=True, max_tokens=128)
        super().__init__(IndexContext(corpus, llm, None, cfg))
        self.memory = SimpleNamespace(facts=[Fact(f"f{i}", "Iris", "visited", f"City{i}", f"p{i}",
                                       statement=f"Iris visited City{i}.", turn_id=f"t{i}") for i in range(30)])
        self.dated = SimpleNamespace(fact_time_text=lambda i: "2023-07-01")
        self.calls, self.no_new, self.filtered = [], False, False
    def packet(self, ids):
        return RetrievalResult(pids=[f"p{i}" for i in ids[:5]], diagnostics={
            "leitura_fatos": "bitemporal", "planejamento": {"replanejamentos": 0},
            "local_plans": {"source_turns": [f"t{i}" for i in ids], "additional_source_turns": []},
            "trechos_extras": [{"title": "Facts", "text": "\n".join(self.memory.facts[i].statement for i in ids)}],
            "fatos_entregues": {"indices": list(ids), "entregues": len(ids),
                 "fontes": [{"indice": i, "fid": f"f{i}", "pid": f"p{i}", "turn_id": f"t{i}"} for i in ids]}},
            filtered=self.filtered)
    def _retrieve(self, question, k):
        assert question.answers == question.gold_pids == question.decomposition == []
        assert question.qtype == ""
        self.calls.append((question.question, self._reflection_search_hint, self.ctx.run.witness.fact_budget))
        start = 0 if self.no_new else 10
        return self.packet(range(start, start + self.ctx.run.witness.fact_budget))


def gate(**changes):
    return {"answer": "insufficient information", "decision": "replan", "missing": "Iris destination",
            "searches": ["Iris travel destination"], "keep": [0, 1, 999], **changes}


Q = Question("q", "Where did Iris travel?", ["SECRET_GOLD"], gold_pids=["SECRET_GOLD"],
             dataset="locomo", qtype="SECRET_GOLD")


def test_single_retry_keeps_provenance_budget_and_original_question():
    llm = Replies([gate(), {"answer": "City10"}])
    r = Memory(llm)
    initial = r.packet(range(10))
    final, reading = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert reading.answer == "City10" and len(llm.calls) == 2 and len(r.calls) == 1
    assert r.calls[0] == (Q.question, "Iris travel destination", 8)
    assert r.ctx.run.witness.fact_budget == 10 and not hasattr(r, "_reflection_search_hint")
    assert len(final.diagnostics["fatos_entregues"]["indices"]) == 10
    assert len(initial.diagnostics["fatos_entregues"]["indices"]) == 10
    assert final.diagnostics["reflection_replan"]["rejected_keep_ids"] == [999]
    assert final.diagnostics["reflection_replan"]["stop_reason"] == "one_retry_exhausted"
    assert llm.calls[0][1].max_tokens == 384 and llm.calls[0][1].exact_max_tokens
    assert llm.calls[1][1].max_tokens == 128 and llm.calls[1][2] == "qa"
    final_prompt = str(llm.calls[1][0])
    assert "fact=0; source=p0/t0" in final_prompt
    assert "SECRET_GOLD" not in final_prompt and "Iris travel destination" not in final_prompt
    assert reading.prompt_tokens == 20 and reading.completion_tokens == 6


def test_sufficient_evidence_uses_one_joint_call_no_search():
    llm = Replies([gate(decision="continue", answer="City0", missing="", searches=[], keep=[])])
    r = Memory(llm)
    result, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert answer.answer == "City0" and len(llm.calls) == 1 and not r.calls
    assert result.diagnostics["reflection_replan"]["stop_reason"] == "sufficient"


@pytest.mark.parametrize("changes", [{"searches": []}, {"searches": ["x"] * 3}, {"keep": [True]},
                                     {"missing": ""}, {"decision": "loop"}, {"searches": "x"}])
def test_invalid_control_does_not_trigger_unbounded_or_guessed_search(changes):
    assert parse_gate(gate(**changes))["decision"] == "continue"
    llm = Replies([gate(**changes)])
    r = Memory(llm)
    _, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert len(llm.calls) == 1 and not r.calls


def test_no_new_evidence_does_not_spend_second_reader_call():
    llm = Replies([gate()])
    r = Memory(llm)
    r.no_new = True
    result, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert len(r.calls) == len(llm.calls) == 1
    assert result.diagnostics["reflection_replan"]["stop_reason"] == "no_new_evidence"


def test_filter_blocks_final_reader_instead_of_reusing_old_answer():
    llm = Replies([gate()])
    r = Memory(llm)
    initial = r.packet(range(10))
    r.filtered = True
    result, answer = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert answer.filtered and answer.answer == "" and len(llm.calls) == 1


def test_retry_cannot_repeat_and_cannot_grow_persistent_memory():
    llm = Replies([gate(), gate(answer="City10")])
    r = Memory(llm)
    original = list(r.memory.facts)
    _, reading = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert reading.answer == "City10" and len(llm.calls) == 2 and len(r.calls) == 1
    assert r.memory.facts == original


def test_hints_change_search_similarity_but_not_operation_or_temporal_contract():
    from test_local_plans_v2 import toy
    from wrag.witness.local_plans_v2 import MultiOriginPlanner
    r = toy([("Iris", "paint", "river", "Iris painted a river.")])
    question = Question("q", "What did Iris paint?", ["SECRET_GOLD"])
    baseline = MultiOriginPlanner(r, question)
    r._reflection_search_hint = "How many cats did Iris own before 2019?"
    guided = MultiOriginPlanner(r, question)
    assert guided.text == baseline.text == question.question
    assert guided.contract.to_dict() == baseline.contract.to_dict()
    assert guided.reading.to_dict() == baseline.reading.to_dict()
    assert "Search focus" in guided.search_text


def test_option_round_trip_and_resume_separates_standard_from_variant(tmp_path):
    from wrag.pilot import parser, make_plan, _run_config, _validate_resume
    flags = ["--gpu", "7", "--dataset", "locomo", "--proof-controller", "--local-plans", "--local-plan-version", "v2",
             "--fact-delivery", "facts", "--reader-reflection", "--evidence-reader", "--answer-set"]
    base = make_plan(parser().parse_args(flags), tmp_path)
    variant = make_plan(parser().parse_args(flags + ["--reflection-replan"]), tmp_path)
    assert _run_config(variant["settings"], 12).qa.reflection_replan
    with pytest.raises(ValueError, match="reflection_replan"):
        _validate_resume(base, variant)
    with pytest.raises(ValueError, match="requires LoCoMo"):
        make_plan(parser().parse_args(["--gpu", "7", "--reflection-replan"]), tmp_path)


def test_full_runner_records_both_calls_and_replan_trace():
    from wrag.eval.runner import _answer_standard
    class PipelineMemory(Memory):
        def _retrieve(self, question, k):
            if not getattr(self, "_reflection_search_hint", ""):
                return self.packet(range(10))
            return super()._retrieve(question, k)
    llm = Replies([gate(), {"answer": "City10"}])
    r = PipelineMemory(llm)
    question = replace(Q, qid="full-runner-synthetic", qtype="single-hop")
    row = _answer_standard("witnessrag", r, r.corpus, question, r.ctx.run)
    assert row["resposta"] == "City10"
    assert row["uso_llm"]["total"]["chamadas"] == 2
    assert set(row["uso_llm"]["por_estagio"]) == {"qa.replan_gate", "qa"}
    assert row["diagnosticos"]["reflection_replan"]["max_replans"] == 1
    assert row["diagnosticos"]["reflection_replan"]["performed"]


def test_retry_quotes_receive_source_session_date_not_current_clock():
    llm = Replies([gate(), {"answer": "2022"}])
    r = Memory(llm)
    r.dated.turns = {"p10": [SimpleNamespace(turn_id="t10", when="2023-08-17")]}
    original_packet = r.packet
    def packet(ids):
        result = original_packet(ids)
        result.diagnostics["trechos_extras"][0]["text"] += "\n[t10] Iris: We went last year."
        return result
    r.packet = packet
    adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert "[t10; session=2023-08-17]" in str(llm.calls[-1][0])
