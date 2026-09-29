"""Memory-stage contracts: cited premises, temporal fidelity and bounded packets."""
import copy
import json

import pytest

from wrag.data import Corpus, Passage
from wrag.ie import Fact
from wrag.llm import LLM, LLMResult
from wrag.witness.dated_memory import DatedMemory
from wrag.witness.query_memory import (compact_packet, reflect, target_query,
                                      validate_reflection, validate_target)


def target_data():
    return {"answer_var": "x", "atoms": [{"relation": "use", "subject": "Tim", "object": "?x"}],
            "operation": "inference", "expected_type": "technique", "constraints": ["for studying"]}


def fixture_memory(n=40, packages=None):
    passage = Passage("p", "history", "Session date: 16 November, 2023\n" +
        "\n".join(f"[D1:{i}] Tim: Fact number {i}." for i in range(n)))
    facts = [Fact(f"f{i}", "Tim", "use", f"method {i}", "p",
                  statement=f"Tim uses method {i} with a special qualifier.",
                  turn_id=f"D1:{i}", kind="ongoing") for i in range(n)]
    corpus = Corpus("locomo", [passage], [])
    dated = DatedMemory(corpus, facts)
    diagnostics = {"local_plans": {"version": "v2", "selected": [
        {"package_facts": group} for group in (packages or [])]},
        "fatos_entregues": {"indices": list(range(n))}}
    return corpus, facts, dated, diagnostics


class RecordingLLM(LLM):
    def __init__(self, reply=None, reason="stop", filtered=False):
        super().__init__()
        self.reply = reply
        self.reason = reason
        self.filtered = filtered
        self.requests = []

    def _complete(self, messages, params, stage="misc"):
        self.requests.append((stage, messages, params))
        return LLMResult(text=json.dumps(self.reply), prompt_tokens=100, completion_tokens=25,
                         finish_reason=self.reason, filtered=self.filtered)


@pytest.mark.parametrize("budget", [10,20,40])
def test_fact_cap_and_no_mutation(budget):
    _, facts, dated, diagnostics = fixture_memory()
    original = copy.deepcopy((facts, diagnostics))
    packet = compact_packet(facts, dated, diagnostics, budget)
    assert len(packet.records) == budget
    assert {r["index"] for r in packet.records} == set(range(budget))
    assert (facts, diagnostics) == original
    assert "special qualifier" in packet.text
    assert all(r["event"] is None for r in packet.records)
    assert "event=" not in packet.text


def test_complete_packages_before_filling_without_partial_dropped_chain():
    _, facts, dated, diagnostics = fixture_memory(packages=[[0,1,2], list(range(3,13))])
    packet = compact_packet(facts, dated, diagnostics, 10)
    chosen = {r["index"] for r in packet.records}
    assert {0,1,2} <= chosen and len(chosen) == 10
    assert not chosen & set(range(3,13))
    assert packet.dropped_packages == [list(range(3,13))]
    larger = compact_packet(facts, dated, diagnostics, 20)
    assert set(range(13)) <= {r["index"] for r in larger.records}


def test_time_slot_modalities_and_qualifiers_survive():
    corpus, facts, _, diagnostics = fixture_memory()
    facts[0].time = "yesterday"
    facts[0].kind = "past"
    facts[0].statement = "Tim studies for 25 minutes and then takes a 5 minute break."
    dated = DatedMemory(corpus, facts)
    packet = compact_packet(facts, dated, diagnostics, 10)
    assert "25 minutes" in packet.text and "5 minute" in packet.text
    assert "event=2023-11-15; said=yesterday" in packet.text
    assert "F0 (past)" in packet.text
    assert packet.records[0]["turn_id"] == "D1:0"
    triple = compact_packet(facts, dated, diagnostics, 10, "triple")
    assert "25 minutes" not in triple.text


def test_out_of_context_selected_package_is_rejected():
    _, facts, dated, diagnostics = fixture_memory(packages=[[100]])
    with pytest.raises(ValueError, match="absent"):
        compact_packet(facts, dated, diagnostics, 10)


def test_unresolved_event_expression_is_preserved_in_triple_mode():
    corpus, facts, _, diagnostics = fixture_memory()
    facts[0].time = "sometime during college"
    dated = DatedMemory(corpus, facts)
    packet = compact_packet(facts, dated, diagnostics, 10, "triple")
    assert packet.records[0]["event"] is None
    assert "stated time=sometime during college (unresolved)" in packet.text


@pytest.mark.parametrize("change", [
    {"answer_var": "unbound", "atoms": [{"relation":"use", "subject":"?person", "object":"?technique"}]},
    {"atoms": []}, {"operation": "made-up"},
    {"atoms": [{"relation":"use","subject":"Tim","object":"?bad variable"}]},
])
def test_invalid_target_rejected(change):
    data = target_data(); data.update(change)
    with pytest.raises(ValueError):
        validate_target(data)


def test_target_is_advisory_and_only_question_reaches_prompt():
    llm = RecordingLLM(target_data())
    result = target_query(llm, "Which technique does Tim use?")
    assert result["status"] == "semantic_hypothesis_not_executed"
    assert result["uses_annotations"] is False
    stage, messages, params = llm.requests[0]
    assert stage == "memory.target" and params.exact_max_tokens
    assert len(messages) == 1
    assert "Which technique does Tim use?" in messages[0]["content"]


def test_actual_temporal_model_response_normalizes_unique_projection():
    data = {"answer_var":"x", "atoms":[{"relation":"went_to", "subject":"Caroline",
            "object":"LGBTQ support group", "time":"?time"}],
            "expected_type":"date", "operation":"value", "constraints":[]}
    llm = RecordingLLM(data)
    normalized = target_query(llm, "When did Caroline go to the LGBTQ support group?")
    assert normalized["answer_var"] == "time"
    assert normalized["atoms"] == data["atoms"]
    assert normalized["normalizations"] == [{"kind":"unique_bound_variable_projection", "from":"x", "to":"time"}]
    assert validate_target(normalized) == normalized
    assert len(llm.requests) == 1


def test_unique_entity_variable_is_normalized_without_changing_predicates():
    data = target_data()
    data["atoms"][0]["object"] = "?technique"
    normalized = validate_target(data)
    assert normalized["answer_var"] == "technique"
    assert normalized["atoms"] == data["atoms"]


def test_bare_declared_head_and_optional_time_placeholder_are_canonicalized():
    data = {**target_data(), "atoms":[{"relation":"hasIdentity", "subject":"Caroline",
            "object":"x", "time":"optional ?time"}]}
    llm = RecordingLLM(data)
    normalized = target_query(llm, "What is Caroline's identity?")
    assert normalized["answer_var"] == "x"
    assert normalized["atoms"] == [{"relation":"hasIdentity", "subject":"Caroline", "object":"?x"}]
    assert len(llm.requests) == 1
    assert validate_target(normalized) == normalized


def test_concrete_answer_type_is_not_guessed_as_a_variable():
    data = {**target_data(), "atoms":[{"relation":"hasIdentity", "subject":"Caroline", "object":"transgender"}]}
    with pytest.raises(ValueError, match="not bound"):
        validate_target(data)


@pytest.mark.parametrize("answer", ["x", "?x", None])
def test_boolean_conjunctive_query_has_no_projected_head(answer):
    data = {**target_data(), "answer_var":answer, "operation":"yesno", "expected_type":"boolean",
            "atoms":[{"relation":"wants_to_pursue", "subject":"Caroline", "object":"counseling_career",
                      "time":"optional ?time_variable"}]}
    llm = RecordingLLM(data)
    normalized = target_query(llm, "Would Caroline want to pursue counseling?")
    assert normalized["answer_var"] is None
    assert normalized["atoms"] == [{"relation":"wants_to_pursue", "subject":"Caroline", "object":"counseling_career"}]
    assert len(llm.requests) == 1
    assert validate_target(normalized) == normalized


def test_inference_bridge_and_valid_citations_required():
    _, facts, dated, diagnostics = fixture_memory()
    packet = compact_packet(facts, dated, diagnostics, 10)
    valid = {"conclusions": [{"text":"Pomodoro", "kind":"inference", "premises":["F0"],
                               "bridge":"A named technique matching the described intervals"}],
             "conflicts":[], "missing":[]}
    assert validate_reflection(valid, packet)["conclusions"][0]["text"] == "Pomodoro"
    bad = copy.deepcopy(valid); bad["conclusions"][0]["premises"] = ["F39"]
    with pytest.raises(ValueError, match="absent"):
        validate_reflection(bad, packet)
    bad = copy.deepcopy(valid); bad["conclusions"][0]["bridge"] = ""
    with pytest.raises(ValueError, match="bridge"):
        validate_reflection(bad, packet)
    bad = copy.deepcopy(valid); bad["conflicts"] = [{"text":"conflict", "premises":["F0","F0"]}]
    with pytest.raises(ValueError, match="two"):
        validate_reflection(bad, packet)


def test_reflector_sees_only_packet_and_original_question():
    _, facts, dated, diagnostics = fixture_memory()
    packet = compact_packet(facts, dated, diagnostics, 10)
    llm = RecordingLLM({"conclusions":[], "conflicts":[], "missing":["insufficient evidence"]})
    reflect(llm, "Which technique?", validate_target(target_data()), packet)
    prompt = llm.requests[0][1][0]["content"]
    assert "method 9 with" in prompt and "method 39 with" not in prompt
    assert "SECRET_GOLD" not in prompt
    assert llm.requests[0][0] == "memory.reflect"


@pytest.mark.parametrize("reason,filtered,error", [("length",False,ValueError), ("content_filter",True,RuntimeError)])
def test_truncated_or_filtered_generation_does_not_continue(reason, filtered, error):
    llm = RecordingLLM(target_data(), reason=reason, filtered=filtered)
    with pytest.raises(error):
        target_query(llm, "Which technique?")
    assert len(llm.requests) == (1 if filtered else 2)


def test_ambiguous_projection_uses_one_audited_model_repair():
    ambiguous = target_data()
    ambiguous["atoms"] = [{"relation":"pursue", "subject":"Caroline", "object":"?field", "time":"?time"},
                           {"relation":"education_level", "subject":"Caroline", "object":"?level"}]
    class RepairLLM(RecordingLLM):
        def _complete(self, messages, params, stage="misc"):
            self.reply = {"answer_var":"field"} if stage.endswith(".repair") else ambiguous
            return super()._complete(messages, params, stage)
    llm = RepairLLM()
    value = target_query(llm, "What fields could Caroline pursue?")
    assert value["answer_var"] == "field" and value["atoms"] == ambiguous["atoms"]
    assert [r[0] for r in llm.requests] == ["memory.target", "memory.target.repair"]
    assert value["generation_repairs"][0]["previous_output"]
    assert "Allowed choices:" in llm.requests[1][1][0]["content"]
    assert validate_target(value) == value


def test_projection_repair_cannot_invent_a_variable_or_rewrite_atoms():
    ambiguous = {**target_data(), "atoms":[{"relation":"pursue", "subject":"Caroline", "object":"?field", "time":"?time"}]}
    class BadRepairLLM(RecordingLLM):
        def _complete(self, messages, params, stage="misc"):
            self.reply = {"answer_var":"invented"} if stage.endswith(".repair") else ambiguous
            return super()._complete(messages, params, stage)
    llm = BadRepairLLM()
    with pytest.raises(ValueError, match="allowed answer variable"):
        target_query(llm, "Which field?")
    assert len(llm.requests) == 2


def test_reflector_discards_uncited_claim_after_failed_repair_without_inventing_ids():
    _, facts, dated, diagnostics = fixture_memory()
    packet = compact_packet(facts, dated, diagnostics, 20)
    valid = {"text":"Tim uses a method", "kind":"observation", "premises":["F0"], "bridge":""}
    invalid = {"text":"The Four Seasons is classical and uplifting", "kind":"observation", "premises":[], "bridge":"ordinary knowledge"}
    llm = RecordingLLM({"conclusions":[valid, invalid], "conflicts":[], "missing":[]})
    value = reflect(llm, "Would Tim enjoy this piece?", validate_target(target_data()), packet)
    assert value["conclusions"] == [valid]
    assert value["discarded_entries"][0]["entry"] == invalid
    assert value["generation_repairs"][0]["repair_output"]
    assert value["missing"] and len(llm.requests) == 2
    assert validate_reflection(value, packet)["conclusions"] == [valid]


def test_reflector_rejects_absent_ids_inferences_without_bridges_and_single_premise_conflicts():
    _, facts, dated, diagnostics = fixture_memory()
    packet = compact_packet(facts, dated, diagnostics, 10)
    llm = RecordingLLM({"conclusions":[
        {"text":"absent", "kind":"observation", "premises":["F39"]},
        {"text":"no bridge", "kind":"inference", "premises":["F0"], "bridge":""}],
        "conflicts":[{"text":"not a conflict", "premises":["F0"]}], "missing":[]})
    value = reflect(llm, "Which technique?", validate_target(target_data()), packet)
    assert value["conclusions"] == [] and value["conflicts"] == []
    assert len(value["discarded_entries"]) == 3
