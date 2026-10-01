"""Coverage decisions belong to the model; code enforces only their contract."""
from dataclasses import replace

import pytest

from test_reflection_replan import Memory, Q, Replies, cover, gate
from wrag.witness.gap_coverage import candidate_units, parse_coverage
from wrag.witness.reflection_replan import adaptive_read, parse_gate


def test_no_confirmed_coverage_preserves_exact_initial_reader_packet():
    llm = Replies([gate(), {"covered": False}, {"answer": "City0"}])
    r = Memory(llm)
    initial = r.packet(range(10))
    final, answer = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert answer.answer == "City0"
    assert final.diagnostics["trechos_extras"] == initial.diagnostics["trechos_extras"]
    assert final.diagnostics["fatos_entregues"] == initial.diagnostics["fatos_entregues"]
    assert final.diagnostics["reflection_replan"]["stop_reason"] == "gap_not_covered"
    assert final.diagnostics["planejamento"]["replanejamentos"] == 1
    assert [call[2] for call in llm.calls] == ["memory.sufficiency", "memory.gap_coverage", "qa"]
    assert answer.prompt_tokens == 30


def test_model_explicitly_selects_old_premises_and_rejects_new_distraction():
    verdict = cover(retain=[0, 1, 2, 3, 5], use=[f"fact:{i}" for i in range(10, 15)],
                    irrelevant=[f"fact:{i}" for i in range(15, 20)])
    llm = Replies([gate(irrelevant=[4]), verdict, {"answer": "City10"}])
    r = Memory(llm)
    final, _ = adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert final.diagnostics["fatos_entregues"]["indices"] == [0, 1, 2, 3, 5, 10, 11, 12, 13, 14]
    assert final.diagnostics["reflection_replan"]["initial_irrelevant_fact_indices"] == [4]
    assert "City15" not in str(llm.calls[-1][0])
    assert "Iris destination" in str(llm.calls[1][0])
    assert "Iris destination" not in str(llm.calls[-1][0])


@pytest.mark.parametrize("changes", [{"retain": [999]}, {"retain": [True]},
    {"use": ["fact:999"]}, {"sources": ["made-up-source"]}, {"covered": "yes"},
    {"irrelevant": ["fact:10"]}, {"retain": list(range(10))}])
def test_invalid_or_overbudget_coverage_cannot_change_original_evidence(changes):
    llm = Replies([gate(), cover(**changes), {"answer": "City0"}])
    r = Memory(llm)
    initial = r.packet(range(10))
    final, _ = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert final.diagnostics["trechos_extras"] == initial.diagnostics["trechos_extras"]
    assert final.diagnostics["reflection_replan"]["stop_reason"] == "invalid_coverage_control"
    assert len(r.calls) == 1


def test_whole_candidate_package_is_selected_without_automatic_member_slicing():
    r = Memory(Replies([]))
    initial, retry = r.packet(range(10)), r.packet([0] + list(range(10, 19)))
    retry.diagnostics["local_plans"]["selected"] = [{"package_facts": [0, 10, 11]}]
    units, sources = candidate_units(initial, retry)
    assert units["package:0"] == [0, 10, 11]
    assert "fact:10" not in units and "fact:11" not in units
    result = parse_coverage(cover(retain=[1, 2], use=["package:0"], irrelevant=[]),
                            set(range(10)), units, sources, 10)
    assert result["final_fact_indices"] == [1, 2, 0, 10, 11]
    assert not parse_coverage(cover(retain=list(range(10)), use=["package:0"], irrelevant=[]),
                              set(range(10)), units, sources, 10)["valid"]


def test_unapproved_literal_from_retry_never_bypasses_the_checker():
    llm = Replies([gate(), cover(), {"answer": "City10"}])
    r = Memory(llm)
    packet = r.packet
    def with_sources(ids):
        result = packet(ids)
        if 10 in result.diagnostics["fatos_entregues"]["indices"]:
            result.diagnostics["trechos_extras"][0]["text"] += (
                "\n\nAdditional original turns (candidate support, not inferred facts):\n"
                "[t99] Iris (2023-07-01): UNAPPROVED_LITERAL")
        return result
    r.packet = with_sources
    adaptive_read(r, r.corpus, Q, r.packet(range(10)), r.ctx.run.qa)
    assert "UNAPPROVED_LITERAL" in str(llm.calls[1][0])
    assert "UNAPPROVED_LITERAL" not in str(llm.calls[-1][0])


def test_first_checker_irrelevant_ids_are_typed_and_nonconflicting():
    assert parse_gate(gate(irrelevant=[3, 4]))["irrelevant"] == [3, 4]
    assert not parse_gate(gate(irrelevant=[0]))["valid"]
    assert not parse_gate(gate(irrelevant=[True]))["valid"]


def test_positive_coverage_is_rejected_if_selected_source_cannot_be_delivered():
    llm = Replies([gate(), cover(sources=["t99"]), {"answer": "City0"}])
    r = Memory(llm)
    r.ctx.run.witness.excerpt_max_chars = 5
    packet = r.packet
    def with_sources(ids):
        result = packet(ids)
        if 10 in result.diagnostics["fatos_entregues"]["indices"]:
            result.diagnostics["trechos_extras"][0]["text"] += (
                "\n\nAdditional original turns (candidate support, not inferred facts):\n"
                "[t99] Iris (2023-07-01): A necessary literal premise.")
        return result
    r.packet = with_sources
    initial = r.packet(range(10))
    final, _ = adaptive_read(r, r.corpus, Q, initial, r.ctx.run.qa)
    assert final.diagnostics["trechos_extras"] == initial.diagnostics["trechos_extras"]
    assert final.diagnostics["reflection_replan"]["stop_reason"] == "coverage_sources_do_not_fit"
    assert final.diagnostics["reflection_replan"]["new_fact_indices"] == []


def test_filtered_coverage_stops_before_reader_and_counts_both_checks():
    from wrag.llm.base import LLMResult
    llm = Replies([gate(), LLMResult(filtered=True, prompt_tokens=12, completion_tokens=1)])
    r = Memory(llm)
    result, answer = adaptive_read(r, r.corpus, replace(Q, qid="coverage-filtered"),
                                  r.packet(range(10)), r.ctx.run.qa)
    assert result.filtered and answer.filtered and len(llm.calls) == 2
    assert answer.prompt_tokens == 22 and answer.completion_tokens == 4
