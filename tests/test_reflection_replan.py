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
        reply = next(self.replies)
        if isinstance(reply, LLMResult):
            return reply
        return LLMResult(text=json.dumps(reply), prompt_tokens=10, completion_tokens=3,
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
    def _render_fact_ids(self, ids):
        return "\n".join(self.memory.facts[i].statement for i in ids)
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
    return {"decision": "replan", "missing": "Iris destination",
            "searches": ["Iris travel destination"], "keep": [0, 1, 999], **changes}


def cover(**changes):
    return {"covered": True, "retain": list(range(6)),
            "use": [f"fact:{i}" for i in range(10, 14)],
            "irrelevant": [f"fact:{i}" for i in range(14, 20)], "sources": [], **changes}


Q = Question("q", "Where did Iris travel?", ["SECRET_GOLD"], gold_pids=["SECRET_GOLD"],
             dataset="locomo", qtype="SECRET_GOLD")


def test_single_retry_keeps_provenance_budget_and_original_question():
    llm = Replies([gate(), cover(), {"answer": "City10"}])
    r = Memory(llm)
    initial = r.packet(range(10))
    final, reading = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert reading.answer == "City10" and len(llm.calls) == 3 and len(r.calls) == 1
    assert r.calls[0] == (Q.question, "Iris travel destination", 10)
    assert r.ctx.run.witness.fact_budget == 10 and not hasattr(r, "_reflection_search_hint")
    assert len(final.diagnostics["fatos_entregues"]["indices"]) == 10
    assert final.diagnostics["fatos_entregues"]["n"] == 10
    assert len(initial.diagnostics["fatos_entregues"]["indices"]) == 10
    assert final.diagnostics["reflection_replan"]["rejected_keep_ids"] == [999]
    assert final.diagnostics["reflection_replan"]["stop_reason"] == "one_retry_exhausted"
    assert llm.calls[0][1].max_tokens == 384 and llm.calls[0][1].exact_max_tokens
    assert llm.calls[0][2] == "memory.sufficiency"
    assert llm.calls[1][2] == "memory.gap_coverage"
    assert llm.calls[2][1].max_tokens == 128 and llm.calls[2][2] == "qa"
    final_prompt = str(llm.calls[2][0])
    assert "Iris visited City0." in final_prompt
    assert "SECRET_GOLD" not in final_prompt and "Iris travel destination" not in final_prompt
    assert "Are these facts minimally sufficient" not in final_prompt
    assert "Final retry reading contract" not in final_prompt
    assert reading.reflection["mode"] == "joint-v2"
    assert reading.prompt_tokens == 30 and reading.completion_tokens == 9


def test_sufficient_evidence_always_calls_the_standard_reader_without_search():
    from wrag.eval.reader import read
    llm = Replies([gate(decision="continue", missing="", searches=[], keep=[]), {"answer": "City0"}])
    r = Memory(llm)
    initial = r.packet(range(10))
    result, answer = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert answer.answer == "City0" and len(llm.calls) == 2 and not r.calls
    assert result.diagnostics["reflection_replan"]["stop_reason"] == "sufficient"
    baseline = Replies([{"answer": "City0"}])
    read(baseline, r.corpus, Q, [], replace(r.ctx.run.qa, reflection_replan=False),
         extra_passages=initial.diagnostics["trechos_extras"], facts_mode="bitemporal")
    assert llm.calls[1] == baseline.calls[0]
    assert result.diagnostics["trechos_extras"] == initial.diagnostics["trechos_extras"]
    assert answer.prompt_tokens == 20 and answer.completion_tokens == 6


@pytest.mark.parametrize("changes", [{"searches": []}, {"searches": ["x", None]}, {"keep": [True]},
                                     {"missing": ""}, {"decision": "loop"}, {"searches": "x"},
                                     {"decision": []}, {"keep": None}])
def test_invalid_control_does_not_trigger_unbounded_or_guessed_search(changes):
    assert parse_gate(gate(**changes))["decision"] == "continue"
    llm = Replies([gate(**changes), {"answer": "City0"}])
    r = Memory(llm)
    _, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert answer.answer == "City0" and len(llm.calls) == 2 and not r.calls


def test_no_new_evidence_still_reaches_the_standard_reader_with_original_context():
    llm = Replies([gate(), cover(), {"answer": "City0"}])
    r = Memory(llm)
    r.no_new = True
    # There are no novel candidates, so no coverage call is needed.
    llm = Replies([gate(), {"answer": "City0"}])
    r.ctx = replace(r.ctx, llm=llm)
    initial = r.packet(range(10))
    result, answer = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert len(r.calls) == 1 and len(llm.calls) == 2 and answer.answer == "City0"
    assert result.diagnostics["reflection_replan"]["stop_reason"] == "no_new_evidence"
    assert result.diagnostics["trechos_extras"] == initial.diagnostics["trechos_extras"]
    assert result.diagnostics["fatos_entregues"]["entregues"] == 10
    assert result.diagnostics["planejamento"]["replanejamentos"] == 1


def test_filter_blocks_final_reader_instead_of_reusing_old_answer():
    llm = Replies([gate()])
    r = Memory(llm)
    initial = r.packet(range(10))
    r.filtered = True
    result, answer = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert answer.filtered and answer.answer == "" and len(llm.calls) == 1
    assert result.diagnostics["planejamento"]["replanejamentos"] == 1


def test_retry_cannot_repeat_and_cannot_grow_persistent_memory():
    llm = Replies([gate(), cover(), gate(answer="City10")])
    r = Memory(llm)
    original = list(r.memory.facts)
    _, reading = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert reading.answer == "City10" and len(llm.calls) == 3 and len(r.calls) == 1
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
    llm = Replies([gate(), cover(), {"answer": "City10"}])
    r = PipelineMemory(llm)
    question = replace(Q, qid="full-runner-synthetic", qtype="single-hop")
    row = _answer_standard("witnessrag", r, r.corpus, question, r.ctx.run)
    assert row["resposta"] == "City10"
    assert row["uso_llm"]["total"]["chamadas"] == 3
    assert set(row["uso_llm"]["por_estagio"]) == {"memory.sufficiency", "memory.gap_coverage", "qa"}
    assert row["diagnosticos"]["reflection_replan"]["max_replans"] == 1
    assert row["diagnosticos"]["reflection_replan"]["performed"]


def test_retry_quotes_receive_source_session_date_not_current_clock():
    from wrag.witness.dated_memory import DatedMemory
    llm = Replies([gate(), cover(), {"answer": "2022"}])
    r = Memory(llm)
    passages = list(r.corpus.passages)
    passages[10] = replace(passages[10], text="Session date: 17 August 2023\n[t10] Iris: We went last year.")
    r.ctx = replace(r.ctx, corpus=Corpus("locomo", passages, []))
    r.dated = DatedMemory(r.corpus, r.memory.facts)
    original_packet = r.packet
    def packet(ids):
        result = original_packet(ids)
        if 10 in result.diagnostics["fatos_entregues"]["indices"]:
            result.diagnostics["trechos_extras"][0]["text"] += "\n\nOriginal source turns:\nSession date: 2023-08-17\n[t10] Iris: We went last year."
        return result
    r.packet = packet
    llm = Replies([gate(), cover(sources=["t10"]), {"answer": "2022"}])
    r.ctx = replace(r.ctx, llm=llm)
    adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert "Session date: 2023-08-17" in str(llm.calls[-1][0])


def test_checker_sees_only_query_and_delivered_dated_triples():
    llm = Replies([gate(decision="continue", missing="", searches=[], keep=[]), {"answer": "City0"}])
    r = Memory(llm)
    packet = r.packet(range(10))
    packet.diagnostics["trechos_extras"].append({"title": "Source", "text": "SOURCE_LITERAL_ONLY"})
    adaptive_read(r, r.corpus, Q, packet, r.ctx.run.qa)
    verifier = str(llm.calls[0][0])
    assert Q.question in verifier and "Iris visited City0." in verifier
    assert "event=2023-07-01; session=2023-07-01" in verifier
    assert "City29" not in verifier and "SOURCE_LITERAL_ONLY" not in verifier
    assert "SOURCE_LITERAL_ONLY" in str(llm.calls[1][0])


def test_unrequested_checker_answer_is_discarded_before_final_reader():
    llm = Replies([gate(decision="continue", missing="", searches=[], keep=[], answer="ANSWER_POISON"),
                   {"answer": "City0"}])
    r = Memory(llm)
    result, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert answer.answer == "City0" and "ANSWER_POISON" not in str(llm.calls[1][0])
    assert "initial_answer" not in result.diagnostics["reflection_replan"]
    assert "ANSWER_POISON" not in str(result.diagnostics)


def test_continue_discards_unneeded_control_fields_without_rejecting_decision():
    assert parse_gate({"decision": "continue", "keep": [0], "missing": "No gap", "searches": None}) == {
        "decision": "continue", "valid": True, "missing": "", "searches": [], "keep": [], "irrelevant": []}


def test_qwen_oversized_control_is_bounded_without_losing_the_retry():
    # The server run returned valid JSON with 9+ retained IDs, or 3 searches.
    # Previously this silently disabled a requested retry in hundreds of rows.
    control = parse_gate(gate(keep=list(range(12)), searches=["first", "second", "third"],
                              missing="details " * 70))
    assert control["valid"] and control["decision"] == "replan"
    assert control["keep"] == list(range(8))
    assert control["searches"] == ["first", "second"]
    assert len(control["missing"]) == 400
    llm = Replies([gate(keep=list(range(12)), searches=["first", "second", "third"]),
                   cover(), {"answer": "City10"}])
    r = Memory(llm)
    result, answer = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert answer.answer == "City10" and len(r.calls) == 1
    assert len(result.diagnostics["fatos_entregues"]["indices"]) <= 10
    assert result.diagnostics["reflection_replan"]["kept_fact_indices"] == list(range(6))


def test_checker_and_retention_date_later_source_session_and_preserve_modality():
    from wrag.witness.dated_memory import DatedMemory
    from wrag.witness.reflection_replan import catalog, cached_fact_block
    corpus = Corpus("locomo", [Passage("p", "", "Session date: 1 May 2023\n"
        "[D1:1] Iris: Hello.\nSession date: 10 June 2023\n"
        "[D2:1] Iris: I plan to visit Rome next week.", session_time="1 May 2023")], [])
    facts = [Fact("f", "Iris", "visit", "Rome", "p", time="next week",
                  turn_id="D2:1", kind="plan", statement="Iris plans to visit Rome next week.")]
    r = SimpleNamespace(corpus=corpus, memory=SimpleNamespace(facts=facts), dated=DatedMemory(corpus,facts))
    retrieval = SimpleNamespace(diagnostics={"fatos_entregues":{"indices":[0]}})
    assert "session=2023-06-10" in catalog(r,retrieval)
    assert "session=1 May 2023" not in catalog(r,retrieval)
    assert "kind=plan" in catalog(r,retrieval)
    retained = cached_fact_block(r,[0])["text"]
    assert "session=2023-06-10" in retained and "kind=plan" in retained
    assert r.dated.fact_time_text(0) != "10 June 2023"  # event != source date


def test_unmatched_multisession_fact_does_not_invent_first_session_date():
    from wrag.witness.dated_memory import DatedMemory
    corpus=Corpus("locomo",[Passage("p","","Session date: 1 May 2023\n[D1:1] Iris: Hello.\n"
        "Session date: 10 June 2023\n[D2:1] Iris: Welcome.",session_time="1 May 2023")],[])
    dated=DatedMemory(corpus,[Fact("f","unmatched","other","elsewhere","p")])
    assert dated.fact_session_time(0) == ""
    assert dated.fact_session_time(-1) == dated.fact_session_time(2) == ""


def test_filtered_verifier_blocks_answer_and_preserves_usage():
    llm = Replies([LLMResult(filtered=True, prompt_tokens=12, latency_s=.3)])
    r = Memory(llm)
    result, answer = adaptive_read(r, r.corpus, replace(Q, qid="filtered-verifier"),
                                  r.packet(range(10)), r.ctx.run.qa)
    assert result.filtered and answer.filtered and len(llm.calls) == 1 and not r.calls
    assert answer.prompt_tokens == 12 and answer.latency_s == .3
    assert result.diagnostics["reflection_replan"]["stop_reason"] == "verifier_filtered"


def test_twenty_fact_union_preserves_complete_initial_chain_and_adds_eight():
    from wrag.witness.replan_merge import select_union
    r = Memory(Replies([]))
    initial, retry = r.packet(range(20)), r.packet(range(10, 30))
    # Relevant bridge outside the first five IDs must survive as a package.
    initial.diagnostics["local_plans"]["selected"] = [{"package_facts": [2, 15, 19]}]
    retry.diagnostics["local_plans"]["selected"] = [{"package_facts": [20, 21, 22]}]
    chosen = select_union(initial, retry, [0, 1], 20)
    assert len(chosen) == len(set(chosen)) == 20
    assert {2, 15, 19} <= set(chosen)
    assert {20, 21, 22} <= set(chosen)
    assert len(set(chosen) & set(range(20))) == 12
    assert len(set(chosen) - set(range(20))) == 8


def test_candidate_join_that_does_not_fit_is_not_sliced_by_singleton_fill():
    from wrag.witness.replan_merge import select_union
    r = Memory(Replies([]))
    initial, retry = r.packet(range(20)), r.packet(range(10, 30))
    retry.diagnostics["local_plans"]["selected"] = [{"package_facts": list(range(20, 29))}]
    chosen = select_union(initial, retry, [], 20)
    assert not set(chosen) & set(range(20, 29))
    assert 29 in chosen and len(chosen) == 20


def test_bounded_literal_union_preserves_old_cue_and_records_only_printed_turns():
    from wrag.witness.replan_merge import compose_union
    r = Memory(Replies([]))
    r.ctx.run.witness.excerpt_max_chars = 140
    initial, retry = r.packet(range(10)), r.packet(range(10, 20))
    marker = "\n\nAdditional original turns (candidate support, not inferred facts):\n"
    initial.diagnostics["trechos_extras"][0]["text"] += marker + "[D4:3] Iris (2023-07-01): My home country, Sweden."
    retry.diagnostics["trechos_extras"][0]["text"] += marker + "[D9:1] Iris (2023-08-01): A new cue.\n[D9:2] Iris: " + "x" * 200
    result, trace = compose_union(r, initial, retry, list(range(6)) + list(range(10, 14)))
    text = result.diagnostics["trechos_extras"][0]["text"]
    assert "My home country, Sweden." in text and "A new cue." in text
    assert "[D9:2]" not in text
    assert trace["delivered_source_turns"] == ["D4:3", "D9:1"]
    assert trace["additional_source_chars"] <= 140
    assert len(result.diagnostics["fatos_entregues"]["fontes"]) == 10


def test_checker_keeps_qualifiers_in_fact_statement_without_source_catalog_overhead():
    from wrag.witness.reflection_replan import catalog
    r = Memory(Replies([]))
    r.memory.facts[0] = replace(r.memory.facts[0], statement="Iris listened to Tupac during childhood, with her father.")
    text = catalog(r, r.packet([0]))
    assert "during childhood, with her father" in text
    assert "source=p0/t0" not in text


def test_initial_printed_bridge_quote_has_priority_over_other_retained_facts():
    from wrag.witness.dated_memory import DatedMemory
    from wrag.witness.replan_merge import compose_union
    r = Memory(Replies([]))
    passages = [replace(p, text=f"Session date: 1 July 2023\n[t{i}] Iris: " +
                        ("My home country, Sweden." if i == 5 else "A filler remark. " * 3))
                for i, p in enumerate(r.corpus.passages)]
    r.ctx = replace(r.ctx, corpus=Corpus("locomo", passages, []))
    r.dated = DatedMemory(r.corpus, r.memory.facts)
    r.ctx.run.witness.excerpt_max_chars = 110
    initial, retry = r.packet(range(10)), r.packet(range(10, 20))
    initial.diagnostics["local_plans"]["source_turns"] = ["t5"]
    result, trace = compose_union(r, initial, retry, list(range(6)) + list(range(10, 14)))
    assert "My home country, Sweden." in result.diagnostics["trechos_extras"][0]["text"]
    assert trace["primary_source_chars"] <= 110
    assert trace["delivered_source_turns"][0] == "t5"
