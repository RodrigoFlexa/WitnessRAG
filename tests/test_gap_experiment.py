from wrag.data import Corpus, Passage
from wrag.eval.gap_search import obligations, propose, replace_tail, verify, Candidate
from wrag.gap_experiment import item_diagnostic, prepare
from wrag.llm.base import LLMResult


def test_gap_candidates_use_plan_requirement_and_leave_baseline_unchanged():
    pages = [Passage(f"p{i}", "", f"Session date: 1 May 2023\n[D1:{i}] Ana: unrelated talk")
             for i in range(5)]
    pages.append(Passage("new", "", "Session date: 2 May 2023\n[D2:1] Ana: I visited Sweden after graduation."))
    corpus = Corpus("locomo", pages, [])
    baseline = [p.pid for p in pages[:5]]
    diagnostics = {"planos_compilados": [{"obrigacoes": {"faltas": ["visited Sweden"]}}]}
    found = propose("Where did Ana visit after graduation?", diagnostics, corpus, baseline)
    assert found and found[0].pid == "new"
    assert found[0].obligation == "visited Sweden"
    assert baseline == [f"p{i}" for i in range(5)]
    assert replace_tail(baseline, ["new"]) == ["p0", "p1", "p2", "p3", "new"]


def test_missing_plan_is_not_a_query():
    assert obligations("Who?", {}) == []
    assert replace_tail(["a", "b", "c", "d", "e"], ["x", "y"], max_added=2) == ["a", "b", "c", "x", "y"]


def test_verifier_rejects_quote_not_in_page():
    class Fake:
        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"supports":true,"quote":"invented source quote","reason":"yes"}')

    candidate = Candidate("new", "visited Sweden", 1.0, "", "")
    result = verify(Fake(), "Where?", candidate, "[D2:1] Ana: I visited Sweden.")
    assert result["accepted"] is False


def test_verifier_accepts_literal_source_quote():
    class Fake:
        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"supports":true,"quote":"Ana: I visited Sweden.","reason":"direct"}')

    candidate = Candidate("new", "visited Sweden", 1.0, "", "")
    result = verify(Fake(), "Where?", candidate, "[D2:1] Ana: I visited Sweden.")
    assert result["accepted"] is True


def test_list_diagnostic_explicitly_limited_to_literal_matching():
    result = item_diagnostic("What books did Ana read?", "Dune, Foundation", "Ana read Dune.")
    assert result["unmatched_literal"] == ["Foundation"]
    assert "semantic" in result["note"]


def test_manifest_prepare_can_resume(tmp_path):
    prepare(tmp_path / "run", {"arms": ["common", "gap-lexical"]})
    prepare(tmp_path / "run", {"arms": ["common", "gap-lexical"]})
