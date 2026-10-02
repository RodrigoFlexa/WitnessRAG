"""Proof requirements: planner -> executor support -> reflection completion.

A scripted LLM and a word-overlap reranker replace the models; no benchmark
gold, supporting passage or category is visible to any component.
"""
import re
from datetime import date

import pytest

from test_reflection_time_v3 import Script, build, q
from wrag.witness import rerank
from wrag.witness.requirements import Requirement, parse_plan, time_compatible


class OverlapReranker:
    def score(self, question, texts):
        words = lambda t: set(re.findall(r"[a-z]+", t.lower())) - {"the", "a", "did", "what", "who", "when", "his", "her"}
        need = words(question)
        return [len(need & words(t)) / max(1, len(need)) for t in texts]


@pytest.fixture(autouse=True)
def fake_reranker(monkeypatch):
    monkeypatch.setattr(rerank, "get_reranker", lambda name: OverlapReranker())


def plan(*needs):
    return {"requirements": [{"need": n, "time": t, "all": "Which places" in n} for n, t in needs]}


def judge(prompt):
    """Scripted membership judge: every listed fact that says 'travelled'."""
    ids = [int(m.group(1)) for m in re.finditer(r"^(\d+): .*travelled", prompt, re.M)]
    return {"members": ids}


def run(needs, question, **witness):
    options = dict(requirements=True, fact_rerank="fake", requirement_threshold=0.6, temporal_reference=False)
    options.update(witness)
    llm = Script({"plan.requirements": plan(*needs), "reflect.members": judge})
    r = build(llm, **options)
    return r, llm, r.retrieve(q(question))


def test_parse_plan_vocabulary_time_and_limits():
    now = date(2023, 8, 15)
    p = parse_plan({"requirements": ["Who is Lina's mentor?", {"need": "Where did the mentor move?",
                                     "time": "the summer of 2023"}, {"need": "x"}] + ["Extra need?"] * 5}, now)
    assert p.valid and len(p.requirements) == 3 and "empty_need" in p.repairs
    assert str(p.requirements[1].window.start) == "2023-06-01"
    assert parse_plan({"requirements": [{"need": "Where did he travel?", "time": "July 2022"}]}, now,
                      use_time=False).requirements[0].window is None
    assert not parse_plan("garbage", now).valid


def test_planner_sees_only_the_question():
    r, llm, _ = run([("Did John return from Chicago?", "")], "What did John do after returning from Chicago?")
    prompts = [p for s, p in llm.calls if s == "plan.requirements"]
    assert len(prompts) == 1 and "SECRET_GOLD" not in prompts[0] and "museum" not in prompts[0]


def test_supported_requirement_changes_nothing():
    r, _, result = run([("Did John visit a museum?", "")], "What did John visit?")
    info = result.diagnostics["requisitos"]
    assert info["requirements"][0]["supported"] and not info["acquired"]


def test_unsupported_requirement_is_completed_from_memory_without_growing_the_budget():
    # Budget 2: the question-relevant facts fill the context; the requirement's
    # fact (Woodhaven) is outside it until the reflection acquires it.
    r, _, result = run([("Did Joanna travel to Woodhaven?", "")], "What did John visit after Chicago?",
                       fact_budget=2)
    info = result.diagnostics["requisitos"]
    row = info["requirements"][0]
    assert not row["supported"] and row.get("acquired") and row["memory_fact"] == 3
    delivered = result.diagnostics["fatos_entregues"]["indices"]
    assert 3 in delivered and len(delivered) <= 2 and result.diagnostics["fatos_entregues"]["adquiridos"] == [3]
    assert "Woodhaven" in result.diagnostics["trechos_extras"][0]["text"]


def test_no_reflection_measures_but_does_not_complete():
    r, _, result = run([("Did Joanna travel to Woodhaven?", "")], "What did John visit after Chicago?",
                       fact_budget=2, study_ablation="no-reflection")
    info = result.diagnostics["requisitos"]
    assert not info["requirements"][0]["supported"] and info["requirements"][0]["memory_fact"] == 3
    assert info["acquired"] == [] and 3 not in result.diagnostics["fatos_entregues"]["indices"]


def test_window_rejects_a_contradicting_stated_time_but_never_a_session_date():
    r, _, _ = run([("Where did Joanna travel?", "")], "Where did Joanna travel?")
    july = Requirement("Where did Joanna travel?", "July 2022",
                       window=__import__("wrag.witness.timeline", fromlist=["parse_anchor"]).parse_anchor("July 2022"))
    assert time_compatible(r, 3, july)        # "July 2022" stated by the fact
    assert not time_compatible(r, 4, july)    # "March 2023" stated by the fact
    assert time_compatible(r, 1, july)        # only a session date: never contradicts


def test_time_reference_ablation_ignores_requirement_windows():
    r, _, result = run([("Where did Joanna travel?", "March 2023")], "Where did Joanna travel in early 2023?",
                       study_ablation="no-time-reference")
    assert result.diagnostics["requisitos"]["requirements"][0]["window"] == ""


def test_witness_ablation_still_runs_the_reflection():
    r, _, result = run([("Did Joanna travel to Woodhaven?", "")], "What did John visit after Chicago?",
                       fact_budget=2, study_ablation="no-witness")
    assert result.diagnostics["requisitos"]["acquired"] == [3]


def test_set_members_are_judged_among_the_named_persons_facts_and_fill_filler_slots():
    from wrag.witness.requirements import check_and_complete, member_candidates, Requirement
    r, llm, _ = run([("Which places has Joanna travelled to?", "")], "Which places has Joanna travelled to?")
    need = Requirement("Which places has Joanna travelled to?", every=True)
    candidates = member_candidates(r, need, "Which places has Joanna travelled to?", set(), 80)
    assert set(candidates) == {3, 4}  # only Joanna's facts, one per source turn
    # Context of John's facts; fact 0 is a proof fact and is never displaced.
    final, acquired, diag = check_and_complete(r, "Which places has Joanna travelled to?", [0, 1, 2], {0}, True)
    row = diag["requirements"][0]
    assert row["all"] and row["judge"]["called"] and sorted(acquired) == [3, 4]
    assert 0 in final and len(final) <= 40
    prompt = [p for s_, p in llm.calls if s_ == "reflect.members"][-1]
    assert "SECRET_GOLD" not in prompt and "John" not in prompt


def test_set_members_are_measured_but_not_added_without_reflection():
    from wrag.witness.requirements import check_and_complete
    r, _, _ = run([("Which places has Joanna travelled to?", "")], "Which places has Joanna travelled to?",
                  study_ablation="no-reflection")
    final, acquired, diag = check_and_complete(r, "Which places has Joanna travelled to?", [0, 1, 2], {0}, False)
    assert final == [0, 1, 2] and acquired == [] and len(diag["requirements"][0]["members"]) == 2


def test_budget_is_respected_when_members_exceed_free_slots():
    from wrag.witness.requirements import check_and_complete
    r, _, _ = run([("Which places has Joanna travelled to?", "")], "Which places has Joanna travelled to?",
                  fact_budget=2)
    final, acquired, diag = check_and_complete(r, "Which places has Joanna travelled to?", [0, 1], {0}, True)
    assert len(final) == 2 and 0 in final and len(acquired) == 1


def _llm_calls(r):
    return r.ctx.llm.calls


def test_member_scan_covers_every_turn_of_the_person_and_cites_only_shown_turns():
    from wrag.witness.member_scan import scan, source_turns, render_table

    r, _, _ = run([("Which places has Joanna travelled to?", "")], "Which places has Joanna travelled to?")
    turns = source_turns(r, ["Joanna"])
    assert [t["turn"] for t in turns] == ["D3:1", "D4:1"]     # only Joanna's turns, in order

    def scanner(prompt):
        found = re.findall(r"^\[(D\d+:\d+)\].*travelled to (\w+)", prompt, re.M)
        return {"members": [{"value": city, "turn": tid} for tid, city in found] +
                           [{"value": "Atlantis", "turn": "D99:1"}]}   # cites a turn not shown: dropped
    r.ctx.llm.replies["reflect.member_scan"] = scanner
    result = scan(r, "Which places has Joanna travelled to?", ["Joanna"], None, batch=1)
    assert result["batches"] == 2 and result["complete_scan"]
    assert [m["value"] for m in result["members"]] == ["Woodhaven", "Lisbon"]
    text = render_table([result], 12)
    assert "[D3:1" in text and "(instance: Woodhaven)" in text and "Atlantis" not in text
    assert "every turn of Joanna was scanned" in text and "They are evidence, not the answer" in text


def test_member_table_reaches_the_reader_and_is_skipped_without_reflection():
    def scanner(prompt):
        found = re.findall(r"^\[(D\d+:\d+)\].*travelled to (\w+)", prompt, re.M)
        return {"members": [{"value": city, "turn": tid} for tid, city in found]}
    for ablation, expected in (("full", True), ("no-reflection", False)):
        llm = Script({"plan.requirements": plan(("Which places has Joanna travelled to?", "")),
                      "reflect.member_scan": scanner, "reflect.members": judge})
        r = build(llm, requirements=True, fact_rerank="fake", member_scan=True, study_ablation=ablation)
        result = r.retrieve(q("Which places has Joanna travelled to?"))
        text = result.diagnostics["trechos_extras"][0]["text"]
        assert ("by scanning the original turns" in text) is expected
        assert ("reflect.member_scan" in llm.stages()) is expected
