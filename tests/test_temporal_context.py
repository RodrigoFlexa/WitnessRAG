from wrag.data import Corpus, Passage
from wrag.eval.locomo_official import bleu1_score, multi_answer_bleu1
from wrag.witness.context_selection import is_temporal_question, select_complement
from wrag.pilot import _run_config
from wrag.long_benchmark import _config as long_config, _normalize, _ruler_score
from wrag.methods import REGISTRY, GRAPH_METHODS


def _corpus():
    return Corpus("locomo", [
        Passage("p0", "", "Session date: 1 May 2023\n[D1:1] Ana: I enjoy hiking.", "1 May 2023", 0),
        Passage("p1", "", "[D1:2] Bob: unrelated one", "1 May 2023", 1),
        Passage("p2", "", "[D1:3] Bob: unrelated two", "1 May 2023", 2),
        Passage("p3", "", "[D1:4] Bob: unrelated three", "1 May 2023", 3),
        Passage("p4", "", "[D1:5] Bob: unrelated four", "1 May 2023", 4),
        Passage("p5", "", "Session date: 9 June 2023\n[D2:1] Ana: I went hiking yesterday.", "9 June 2023", 5),
    ], [])


def test_temporal_selector_replaces_only_tail_and_uses_latest_evidence():
    corpus = _corpus()
    base = ["p0", "p1", "p2", "p3", "p4"]
    result = select_complement(corpus, "When did Ana last go hiking?", base, {}, 5,
                               temporal=True, complementary=False)
    assert result.pids == ["p0", "p1", "p2", "p3", "p5"]
    assert result.reason == "temporal_endpoint"
    assert base == ["p0", "p1", "p2", "p3", "p4"]


def test_non_temporal_ablation_does_not_change_context():
    result = select_complement(_corpus(), "What does Ana enjoy?",
                               ["p0", "p1", "p2", "p3", "p4"], {}, 5,
                               temporal=True, complementary=False)
    assert not result.changed
    assert is_temporal_question("How long ago was the race?")


def test_bleu1_is_clipped_and_multi_answer_matches_items():
    assert bleu1_score("blue car", "blue car") == 1.0
    assert bleu1_score("blue blue", "blue") == 0.5
    assert multi_answer_bleu1("red, blue", "blue, red") == 1.0


def test_pilot_profile_flags_reach_retriever_config():
    settings = {"questions": 1, "seed": 42, "dataset": "locomo", "no_acquisition": True,
                "model": "Qwen/Qwen2.5-14B-Instruct", "temporal_memory": True,
                "complementary_context": True}
    cfg = _run_config(settings, 1)
    assert cfg.witness.temporal_memory is True
    assert cfg.witness.complementary_context is True


def test_long_context_adapter_and_aggregation_profile():
    qid, context, question, answers, task = _normalize(
        {"id": "x", "context": "needle context", "question": "Find it",
         "answers": ["needle"], "task": "agg"}, 0, "ruler")
    assert (qid, context, question, answers, task) == (
        "x", "needle context", "Find it", ["needle"], "agg")
    cfg = long_config("full", 5, "agg")
    assert cfg.qa.answer_set and cfg.witness.complementary_context
    assert "witnessrag-lite" in REGISTRY and "witnessrag-lite" not in GRAPH_METHODS


def test_ruler_official_combined_input_and_task_metrics():
    qid, context, question, answers, _task = _normalize(
        {"id": "r", "input": "noise and key value\n\nWhat is the key?",
         "outputs": ["alpha", "beta"]}, 0, "ruler")
    assert qid == "r" and context == "noise and key value"
    assert question == "What is the key?" and answers == ["alpha", "beta"]
    assert _ruler_score("Alpha only", answers, "qa") == 1.0
    assert _ruler_score("Alpha only", answers, "mt") == .5
