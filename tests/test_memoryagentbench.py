"""Official protocol parity, gold isolation, state isolation and durable resume."""
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from nltk.metrics.distance import edit_distance as nltk_distance

from benchmarks.memoryagentbench import data as D, metrics as M, protocol as P
from benchmarks.memoryagentbench import reader as R, runner as B
from benchmarks.memoryagentbench.engine import WitnessEngine, annotate_serials, standard_config
from benchmarks.memoryagentbench.judge import evaluate, parse_summary_json
from benchmarks.memoryagentbench.vendor import reference_helpers as official
from wrag import config as C
from wrag.llm.base import LLM, LLMResult, UsageLedger


@pytest.mark.parametrize("prediction,answer", [
    ("Answer: The France.", "France"), ("the", "a"), ("Paris, France", "France"),
    ("yes indeed", "yes"), ("no", "yes"), ("", "noanswer"), ("Line one\nAnswer: 43", "43"),
    ("label: 43", "43"), ("43", "43"), ("Answer: Answer: Belgium\nextra", "Belgium"),
    (" café—thé! ", "café—thé"), ("42 43", "43"), ("", ""),
])
def test_primary_metric_and_parsing_parity(prediction, answer):
    assert M.normalize_answer(prediction) == official.normalize_answer(prediction)
    assert M.parse_output(prediction) == official.parse_output(prediction)
    expected = {"exact_match": float(official.drqa_exact_match_score(prediction, answer)),
                "substring_exact_match": float(official.substring_exact_match_score(prediction, answer)),
                "f1": official.f1_score(prediction, answer)[0]}
    assert M.basic_metrics(prediction, [answer]) == pytest.approx(expected)


def test_classification_is_not_given_a_more_permissive_parser():
    assert M.score("icl_banking77_5900shot_balance", "label: 43", ["43"])[0]["exact_match"] == 0
    assert M.score("icl_banking77_5900shot_balance", "43", ["43"])[0]["exact_match"] == 1
    assert M.score("detective_qa", "The culprit is 43", ["43"])[0]["exact_match"] == 0
    # Release's default postprocessor also tries Answer: first-line parsing.
    assert M.score("detective_qa", "Answer: 43\nReason", ["43"])[0]["exact_match"] == 1


def test_max_over_aliases_and_eventqa_all_elements():
    answers = [["Paris", "the city of Paris"], ["France"]]
    assert M.basic_metrics("France", answers)["exact_match"] == 1
    assert M.score("eventqa_full", "Paris", ["Paris", "France"])[0]["substring_exact_match"] == 1
    assert M.score("eventqa_full", "Paris", ["Paris", "France"])[0]["eventqa_recall"] == 0


class WordEncoding:
    def encode(self, text, **kwargs):
        return text.split()


@pytest.mark.parametrize("sentences,size", [
    (["One sentence.", "Two sentences here.", "The last."], 4),
    (["overlong first sentence has six words", "Short."], 2),
    (["<|endoftext|> A.", "B."], 3), ([], 4), (["Only one."], 10),
])
def test_chunker_is_identical_to_official_including_oversized_sentences(monkeypatch, sentences, size):
    monkeypatch.setattr(official.nltk, "download", lambda *a, **kw: True)
    monkeypatch.setattr(official.nltk, "sent_tokenize", lambda text: sentences)
    monkeypatch.setattr(official.tiktoken, "encoding_for_model", lambda name: WordEncoding())
    expected = official.chunk_text_into_sentences("ignored", chunk_size=size)
    assert D.chunk_text("ignored", size, sentences=sentences, encoding=WordEncoding()) == expected


def test_edit_distance_matches_independent_implementation():
    rng = random.Random(41)
    for _ in range(120):
        a = "".join(rng.choices("abc è-", k=rng.randrange(25)))
        b = "".join(rng.choices("abc è-", k=rng.randrange(25)))
        assert M.edit_distance(a, b) == nltk_distance(a, b)


def test_recommendation_postprocessing_and_recall_parity():
    mapping = {"http://db/A_Movie_(1990)": 1, "http://db/Zebra_(2000)": 2,
               "http://db/The_Sea": 3, "http://db/Marsh": 4}
    scorer = M.MovieScorer(mapping)
    text = "1. A Movie (1990)\n2. Zebra (2000)\n3. The Sea\n4. Marsh"
    expected, _ = official.extract_recommendation_list(text, list(scorer.id_to_name.values()))
    names = [x["nearest_movie"] for x in expected]
    metrics, actual = scorer.score(text, ["1", "3"])
    assert actual == names
    assert metrics == {f"recsys_recall@{k}": sum(x in names[:k] for x in ("A Movie", "The Sea")) / 2
                       for k in (1, 5, 10)}


def sample(source="factconsolidation_sh_6k", suffix="", answers=None):
    return D.Sample("Conflict_Resolution", source, 0,
                    f"0. Iris lives in Paris.\n1. Iris lives in Rome.\n{suffix}",
                    ["Where does Iris live?", "Where is Iris located?"],
                    answers or [["SECRET_GOLD"], ["SECRET_GOLD"]],
                    {"source": source, "qa_pair_ids": ["official0", "official1"]})


def test_official_query_is_exact_and_has_no_answer_or_private_metadata():
    s = sample()
    s.metadata.update(keypoints=["KEYPOINT_SECRET"], question_types=["QTYPE_SECRET"])
    expected = D.get_template(s.source, "query", "rag_bm25").format(question=s.questions[0])
    assert s.query(0) == expected
    assert all(secret not in s.query(0) for secret in ("SECRET_GOLD", "KEYPOINT_SECRET", "QTYPE_SECRET"))
    assert "larger serial number" in expected


def test_released_settings_and_paper_differences_are_explicit():
    official_conf = P.task_settings("factconsolidation_sh_6k", "official")
    paper_conf = P.task_settings("factconsolidation_sh_6k", "paper")
    assert official_conf["chunk_size"] == 4096 and paper_conf["chunk_size"] == 512
    assert official_conf["generation_max_length"] == paper_conf["generation_max_length"] == 10
    assert P.task_settings("detective_qa", "official")["generation_max_length"] == 2000
    assert P.task_settings("detective_qa", "paper")["generation_max_length"] == 500
    assert P.task_settings("longmemeval_s*", "paper")["generation_max_length"] == 100
    assert len(P.PAPER_SOURCES) == 10


def test_profile_matches_standard_local_v2():
    cfg = standard_config()
    assert cfg.witness.local_plan_version == "v2" and cfg.witness.local_plans
    assert cfg.witness.local_plan_beam == 32 and cfg.witness.local_plan_candidates == 96
    assert cfg.witness.local_plan_executions == 4000
    assert cfg.witness.fact_budget == 40 and cfg.witness.fact_time == "both"
    assert cfg.witness.fact_rerank == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert cfg.ie.style == "memory" and cfg.ie.window_tokens == 512
    assert cfg.ie.max_tokens == 4000 and cfg.qa.reader_reflection
    assert cfg.top_k == 10 and not cfg.witness.proof_verify


def test_source_serials_do_not_become_fake_calendar_dates():
    raw = "0. Iris was born in 2000.\n1. Iris lives in Rome."
    chunk = "0. Iris was born in 2000. 1. Iris lives in Rome."
    annotated = annotate_serials(chunk, raw)
    assert "[MAB:0]" in annotated and "[MAB:1]" in annotated
    assert "[MAB:2000]" not in annotated
    assert "Session date:" not in D.memorize_message("factconsolidation_sh_6k", annotated, 2)


class RecordingLLM(LLM):
    name = "recording"
    deployment = "test-model"

    def __init__(self):
        super().__init__()
        self.messages = []
        self.fail_reader = False

    def _complete(self, messages, params, stage):
        self.messages.append((messages, params, stage))
        if stage == "index.openie":
            text = messages[-1]["content"]
            found = []
            for serial, city in ((0, "Paris"), (1, "Rome")):
                if f"[MAB:{serial}]" in text:
                    found.append({"subject": "Iris", "relation": "lives in", "object": city,
                                  "statement": f"Iris lives in {city}.", "turn": f"MAB:{serial}"})
            return LLMResult(text=json.dumps({"memories": found}), prompt_tokens=20, completion_tokens=20)
        if self.fail_reader:
            raise RuntimeError("simulated interrupted API call")
        return LLMResult(text="Rome", prompt_tokens=11, completion_tokens=1, finish_reason="stop")


def test_reader_keeps_exact_budget_and_plain_official_output():
    llm = RecordingLLM()
    result = R.answer(llm, "[source serial 2] Iris lives in Rome", "Question?",
                      P.task_settings("factconsolidation_sh_6k", "paper"))
    messages, params, _ = llm.messages[-1]
    assert result.text == "Rome"
    assert params.max_tokens == 10 and params.exact_max_tokens
    assert params.temperature == 0.7 and params.seed is None and not params.json_mode
    assert "EVERY step" in messages[-1]["content"]
    assert "ordinary world knowledge" not in messages[-1]["content"]
    assert "Lisbon" not in messages[-1]["content"]
    assert 'Return the original JSON shape' not in messages[-1]["content"]
    assert '{{"answer"' not in messages[-1]["content"]


def test_reflection_does_not_override_benchmark_list_or_classification_formats():
    prompt = R.build_prompt("hypothesis", "Output label: 43", True, False)
    assert "comma AND" not in prompt and "Synthetic examples" not in prompt
    assert "original JSON shape" not in prompt
    assert "task instructions below define" in prompt
    assert prompt.endswith("Output label: 43")


def test_backend_ceiling_cannot_expand_benchmark_response(monkeypatch, tmp_path):
    from wrag.llm.azure import AzureLLM
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(AzureLLM, "_make_client", lambda _: None)
    backend = AzureLLM(deployment="gpt-4o-mini", max_tokens=2048)
    llm = RecordingLLM()
    R.answer(llm, "context", "query", P.task_settings("factconsolidation_sh_6k", "paper"))
    messages, params, _ = llm.messages[-1]
    kwargs = backend.build_kwargs(messages, params)
    assert kwargs["max_tokens"] == 10 and kwargs["temperature"] == 0.7


def test_reasoning_backend_is_not_silently_called_with_larger_budget():
    llm = RecordingLLM()
    llm.reasoning = True
    with pytest.raises(ValueError, match="Reasoning"):
        R.answer(llm, "context", "query", P.task_settings("factconsolidation_sh_6k"))
    assert not llm.messages


class DummyEngine:
    cfg = standard_config(rerank="")

    def __init__(self):
        self.prepared = []
        self.selected = []

    def prepare(self, key, source, chunks, context):
        self.prepared.append((key, source, chunks))
        return {"usage": {}, "seconds": 0.1, "chunks": len(chunks)}

    def select(self, qid, text):
        self.selected.append((qid, text))
        return SimpleNamespace(context="[source serial 1] Iris lives in Rome.",
                               filtered=False, retrieve_s=0.01, diagnostics={})


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "verify_data", lambda cache: {"verified": True})
    monkeypatch.setattr(B, "tokenizer_identity", lambda cache: {"tokenizer": "test"})
    monkeypatch.setattr(B, "chunks_for", lambda sample, cache, settings: [sample.context])
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "core-cache")
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    return tmp_path


def test_single_registration_multiple_questions_resume_and_gold_isolation(isolated_runtime):
    root = isolated_runtime
    llm, engine = RecordingLLM(), DummyEngine()
    s = sample()
    first = B.run([s], engine, llm, root, root / "run", protocol="paper", max_questions=1)
    assert len(engine.prepared) == 1 and len(engine.selected) == 1
    assert not first["generation_complete"] and first["paper_overall"] is None
    second = B.run([s], engine, llm, root, root / "run", protocol="paper", resume=True)
    assert second["generation_complete"] and second["evaluation_complete"]
    assert len(B.read_rows(root / "run/results.jsonl")) == 2
    third = B.run([s], engine, llm, root, root / "run", protocol="paper", resume=True)
    assert third == second  # no calls or altered report on a complete resume
    assert len(engine.selected) == 2
    assert all("SECRET_GOLD" not in str(messages) for messages, _, _ in llm.messages)


def test_changing_protocol_or_gold_reference_rejects_resume(isolated_runtime):
    root = isolated_runtime
    B.run([sample()], DummyEngine(), RecordingLLM(), root, root / "run", max_questions=1)
    for kwargs in ({"protocol": "paper"}, {}):
        s = sample(answers=[["changed gold"], ["changed gold"]]) if not kwargs else sample()
        with pytest.raises(ValueError, match="identity mismatch"):
            B.run([s], DummyEngine(), RecordingLLM(), root, root / "run", resume=True, **kwargs)
    assert not (root / "run/run.lock").exists()


def test_interrupted_reader_leaves_question_pending_and_preserves_lock_cleanup(isolated_runtime):
    root = isolated_runtime
    llm = RecordingLLM()
    llm.fail_reader = True
    with pytest.raises(RuntimeError, match="interrupted"):
        B.run([sample()], DummyEngine(), llm, root, root / "run")
    assert not B.read_rows(root / "run/results.jsonl")
    assert not (root / "run/run.lock").exists()
    llm.fail_reader = False
    result = B.run([sample()], DummyEngine(), llm, root, root / "run", resume=True)
    assert result["generation_complete"]


def test_live_lock_and_corrupt_checkpoint_are_not_discarded(tmp_path):
    (tmp_path / "run.lock").write_text('{"pid":42}')
    with pytest.raises(RuntimeError, match="locked"):
        with B.lock(tmp_path):
            pass
    assert (tmp_path / "run.lock").exists()
    bad = tmp_path / "broken.jsonl"
    bad.write_text('{"qid":"x"}\n{"incomplete"')
    with pytest.raises(ValueError, match="Damaged"):
        B.read_rows(bad)


def test_real_standard_engine_ingests_in_order_without_future_context_or_gold(isolated_runtime):
    from test_witness import ExactEmbedder
    cfg = standard_config(rerank="")
    cfg.ie.window_tokens = 0  # no external model/tokenizer in this offline test
    llm = RecordingLLM()
    engine = WitnessEngine(llm, ExactEmbedder(), cfg)
    engine.prepare("first", "factconsolidation_sh_6k",
                   ["0. Iris lives in Paris.", "1. Iris lives in Rome."], sample().context)
    calls = [m for m, _, stage in llm.messages if stage == "index.openie"]
    assert len(calls) == 2 and "Rome" not in str(calls[0])
    assert "Paris" not in str(calls[1])
    selected = engine.select("q", "Where does Iris live?")
    assert "Rome" in selected.context and "Paris" in selected.context
    assert "source serial 1" in selected.context and "source serial 0" in selected.context
    assert selected.diagnostics["local_plans"]["version"] == "v2"
    selected_ids = selected.diagnostics["local_plans"]["selected"][0]["facts"]
    assert all(engine.retriever.memory.facts[i].turn_id == "MAB:1" for i in selected_ids)
    assert selected.diagnostics["local_plans"]["selected"][0]["temporal"] == 1
    assert selected.context.startswith("Question-relevant original records, newest first.")
    assert selected.context.index("source serial 1") < selected.context.index("source serial 0")
    assert selected.diagnostics["planejamento"]["chamadas"] == 0
    assert "SECRET_GOLD" not in str(llm.messages)
    engine.prepare("second", "factconsolidation_sh_6k", ["0. Iris lives in Paris."], sample().context)
    assert "Rome" not in engine.select("other", "Where does Iris live?").context


def test_serial_policy_validates_provenance_and_never_boosts_unrelated_facts():
    from wrag.data import Corpus, Passage
    from wrag.ie import Fact
    from benchmarks.memoryagentbench.recency import SerialPolicy
    corpus = Corpus("serials", [Passage("p", "", annotate_serials(
        "12. Iris lives in Paris. 100. Iris lives in Rome.",
        "12. Iris lives in Paris.\n100. Iris lives in Rome."))], [])
    facts = [Fact("a", "Iris", "lives in", "Paris", "p", turn_id="MAB:12"),
             Fact("b", "Iris", "lives in", "Rome", "p", turn_id="MAB:100"),
             Fact("c", "Bob", "lives in", "Milan", "p", turn_id="MAB:999")]
    policy = SerialPolicy(corpus, facts)
    assert policy.serials == [12, 100, None]
    assert list(policy.freshness) == [0, 1, .5]
    assert policy.blend([1, .9, 0], [0, 1, 2]) == pytest.approx([.35, .9, 0])
    assert policy.witness_score([0, 1]) == 0  # latest last hop cannot hide stale first hop
    assert len(facts) == 3 and not policy.describe()["old_facts_deleted"]
    # Weight zero removes ordering preference; bad settings fail explicitly.
    assert SerialPolicy(corpus, facts, 0).blend([1, .9], [0, 1]) == pytest.approx([1, .9])
    with pytest.raises(ValueError, match="between"):
        SerialPolicy(corpus, facts, 2)


def test_recency_applies_before_grounding_cut_at_every_chain_hop(isolated_runtime):
    from test_witness import ExactEmbedder
    from wrag.witness.query import Atom, ConjunctiveQuery
    from benchmarks.memoryagentbench.recency import SerialPlanner
    triples = [(0, "Iris", "works at", "Atlas"), (1, "Atlas", "located in", "Paris"),
               (2, "Iris", "works at", "Boreal"), (3, "Boreal", "located in", "Berlin"),
               (9, "Boreal", "located in", "Rome")]
    raw = "\n".join(f"{n}. {s} {r} {o}." for n, s, r, o in triples)
    class ChainLLM(RecordingLLM):
        def _complete(self, messages, params, stage):
            self.messages.append((messages, params, stage))
            return LLMResult(text=json.dumps({"memories": [
                {"subject": s, "relation": r, "object": o, "statement": f"{s} {r} {o}.", "turn": f"MAB:{n}"}
                for n, s, r, o in triples]}))
    cfg = standard_config(rerank="")
    cfg.ie.window_tokens = 0
    engine = WitnessEngine(ChainLLM(), ExactEmbedder(), cfg)
    engine.prepare("chain", "factconsolidation_mh_6k", [raw], raw)
    planner = SerialPlanner(engine.retriever, SimpleNamespace(question="Where does Iris work?"))
    query = ConjunctiveQuery(answer_var="x", atoms=[Atom("works at", "Iris", "?y"),
                                                   Atom("located in", "?y", "?x")])
    witnesses = planner.execute(query)
    assert witnesses[0].answer == "Rome"
    assert {engine.retriever.memory.facts[i].turn_id for i in witnesses[0].facts} == {"MAB:2", "MAB:9"}
    assert any(w.answer == "Berlin" for w in witnesses)  # kept as historical counterevidence
    cfg.witness.candidates_per_atom = 1
    cut = SerialPlanner(engine.retriever, SimpleNamespace(question="Where does Iris work?"))
    assert [w.answer for w in cut.execute(query)] == ["Rome"]
    assert "groundings" in cut.truncations


def test_source_updates_not_extracted_as_triples_still_age_old_premises():
    from wrag.data import Corpus, Passage
    from wrag.ie import Fact
    from benchmarks.memoryagentbench.recency import SerialPolicy
    raw = ("3. Iris lives in Paris.\n70. Iris lives in Rome.\n"
           "500. Bob lives in Madrid.\n900. Iris works in Milan.")
    corpus = Corpus("missed", [Passage("p", "", annotate_serials(raw, raw))], [])
    old = Fact("old", "Iris", "lives in", "Paris", "p", turn_id="MAB:3")
    policy = SerialPolicy(corpus, [old])  # the update is genuinely absent from triples
    assert policy.serials == [3] and policy.source_versions[0] == [3, 70]
    assert policy.freshness[0] == 0 and policy.newest[0] == 70
    assert "OLDER VERSION" in policy.version_label(3)
    assert "LATEST MATCHING VERSION" in policy.version_label(70)
    assert policy.version_label(900) == "[source serial 900]"  # different predicate
    assert policy.version_label(500) == "[source serial 500]"  # different subject
    assert policy.describe()["versions_found_only_in_source"] == 1


def test_source_version_templates_preserve_negation_and_other_qualifiers():
    from wrag.data import Corpus, Passage
    from wrag.ie import Fact
    from benchmarks.memoryagentbench.recency import SerialPolicy
    raw = ("1. Iris lives in Paris during summer.\n9. Iris lives in Rome during summer.\n"
           "10. Iris lives in Milan during winter.\n99. Iris does not live in Athens during summer.")
    corpus = Corpus("scopes", [Passage("p", "", annotate_serials(raw, raw))], [])
    old = Fact("old", "Iris", "lives in", "Paris", "p", turn_id="MAB:1")
    policy = SerialPolicy(corpus, [old])
    assert policy.source_versions[0] == [1, 9]
    assert policy.newest[0] == 9


def test_filler_and_source_rescue_prefer_relevant_new_versions(isolated_runtime, monkeypatch):
    from test_witness import ExactEmbedder
    from benchmarks.memoryagentbench.recency import SerialSourceIndex
    from wrag.witness.local_contract import stems
    import wrag.witness.rerank as rerank
    cfg = standard_config(rerank="test-reranker")
    cfg.ie.window_tokens = 0
    engine = WitnessEngine(RecordingLLM(), ExactEmbedder(), cfg)
    engine.prepare("versions", "factconsolidation_sh_6k", [sample().context], sample().context)
    facts = engine.retriever.memory.facts
    old = next(i for i, f in enumerate(facts) if f.object == "Paris")
    new = next(i for i, f in enumerate(facts) if f.object == "Rome")
    monkeypatch.setattr(rerank, "get_reranker", lambda _: SimpleNamespace(score=lambda q, texts: [1, .8]))
    order, _ = engine.retriever._rerank_facts(SimpleNamespace(question="Where does Iris live?"),
                                           [old, new], None, {})
    assert order == [new, old]
    source = engine.retriever._local_source_index
    assert isinstance(source, SerialSourceIndex)
    ranked = source.rank(stems("Where does Iris live?"))
    assert source.turns[ranked[0][1]][2].turn_id == "MAB:1"


def test_conflict_adaptation_weight_is_part_of_resume_identity(isolated_runtime):
    root = isolated_runtime
    engine = DummyEngine()
    engine.protocol_identity = lambda: {"conflict_recency_weight": .65}
    B.run([sample()], engine, RecordingLLM(), root, root / "run", max_questions=1)
    engine.protocol_identity = lambda: {"conflict_recency_weight": 0}
    with pytest.raises(ValueError, match="identity mismatch"):
        B.run([sample()], engine, RecordingLLM(), root, root / "run", resume=True)


class JudgeLLM(RecordingLLM):
    def _complete(self, messages, params, stage):
        self.messages.append((messages, params, stage))
        outputs = {"mab.judge.summary.fluency": '{"fluency":1}',
                   "mab.judge.summary.recall": '{"recall":2}',
                   "mab.judge.summary.precision": '{"precision":3,"sentence_count":4}'}
        return LLMResult(text=outputs.get(stage, "yes"))


def test_summary_judge_uses_three_official_prompts_and_fluency_weighted_f1():
    s = D.Sample("Long_Range_Understanding", "infbench_sum_eng_shots2", 0, "ctx",
                 ["Summarize"], [["REFERENCE_EXPERT_SUMMARY"]], {"keypoints": ["k1", "k2", "k3", "k4"]})
    llm = JudgeLLM()
    evaluated = evaluate(s, 0, "Generated plot", llm)
    assert evaluated["metrics"]["summary_judge_f1"] == pytest.approx(0.6)
    assert len(llm.messages) == 3
    assert all(p.temperature == 0.1 and p.top_p == 0.9 and p.seed == 42
               and p.max_tokens == 4096 and p.exact_max_tokens for _, p, _ in llm.messages)
    assert "REFERENCE_EXPERT_SUMMARY" in str(llm.messages[-1])


def test_longmem_judge_has_official_temporal_tolerance_and_abstention():
    for qid, phrase in (("date0", "off-by-one"), ("date_abs", "unanswerable")):
        s = D.Sample("Accurate_Retrieval", "longmemeval_s*", 0, "ctx", ["How many days?"], [["18 days"]],
                     {"question_types": ["temporal-reasoning"], "question_ids": [qid]})
        llm = JudgeLLM()
        assert evaluate(s, 0, "19 days", llm)["metrics"]["longmemeval_judge_accuracy"] == 1
        assert phrase in str(llm.messages[-1])
        assert llm.messages[-1][1].max_tokens == 10 and llm.messages[-1][1].seed is None


def test_summary_parser_uses_last_json_object():
    assert parse_summary_json('Reason {"recall": 1} Final {"recall": 2}') == {"recall": 2}
    assert parse_summary_json("invalid") is None


def test_single_question_reference_shape_is_preserved_for_official_judges():
    s = D.Sample("Long_Range_Understanding", "infbench_sum_eng_shots2", 0, "text",
                 ["Summarize"], [["Reference"]], {})
    assert s.evaluation_answers(0) == [["Reference"]]
    assert M.basic_metrics("Reference", s.evaluation_answers(0))["exact_match"] == 1


def test_filtered_response_counts_zero_and_stays_in_denominator(isolated_runtime):
    class FilteredLLM(RecordingLLM):
        def _complete(self, messages, params, stage):
            return LLMResult(filtered=True, text="")
    root = isolated_runtime
    # Article-only gold normalizes to empty, but a filtered answer must be zero.
    s = sample(answers=[["the"], ["the"]])
    result = B.run([s], DummyEngine(), FilteredLLM(), root, root / "filtered")
    source = result["sources"][s.source]
    assert source["score"] == 0 and source["scored"] == 2 and source["filtered"] == 2


def test_judge_stage_is_resumable_and_never_repeats_completed_judgments(isolated_runtime):
    root = isolated_runtime
    s = D.Sample("Accurate_Retrieval", "longmemeval_s*", 0, "context",
                 ["Where?", "Which city?"], [["Rome"], ["Rome"]],
                 {"question_ids": ["a", "b"], "question_types": ["multi-session", "single-session-user"]})
    before = B.run([s], DummyEngine(), RecordingLLM(), root, root / "judge")
    assert not before["evaluation_complete"] and before["sources"][s.source]["judge_pending"] == 2
    llm = JudgeLLM()
    clients = {"longmemeval": llm, "summary": llm}
    after = B.judge_run([s], root / "judge", clients, code_hash="test")
    assert after["evaluation_complete"] and after["sources"][s.source]["score"] == 1
    assert len(llm.messages) == 2
    B.judge_run([s], root / "judge", clients, code_hash="test")
    assert len(llm.messages) == 2
    with pytest.raises(ValueError, match="identity changed"):
        B.judge_run([s], root / "judge", clients, code_hash="different")


def test_failed_judge_retains_pending_score_and_known_costs(isolated_runtime):
    class BrokenJudge(JudgeLLM):
        def _complete(self, messages, params, stage):
            return LLMResult(text="invalid JSON", prompt_tokens=7, completion_tokens=3)
    root = isolated_runtime
    s = D.Sample("Long_Range_Understanding", "infbench_sum_eng_shots2", 0, "context",
                 ["Summarize"], [["Reference"]], {"keypoints": ["k1"]})
    B.run([s], DummyEngine(), RecordingLLM(), root, root / "broken-judge")
    clients = {"summary": BrokenJudge(), "longmemeval": JudgeLLM()}
    with pytest.raises(ValueError, match="Malformed"):
        B.judge_run([s], root / "broken-judge", clients)
    assert not B.read_rows(root / "broken-judge/judgments.jsonl")
    from benchmarks.memoryagentbench.report import report
    result = report(root / "broken-judge")
    assert result["sources"][s.source]["judge_pending"] == 1
    assert result["costs"]["failed_judge_attempts"] == 1
    assert result["costs"]["judge"]["total"]["tokens_prompt"] == 21


def test_no_paper_overall_until_all_main_tasks_and_judges_are_complete(isolated_runtime):
    from benchmarks.memoryagentbench.report import report
    root = isolated_runtime / "macro"
    root.mkdir()
    selection = [{"source": s, "split": P.CONFIGS[s]["dataset"], "questions": i + 1}
                 for i, s in enumerate(P.PAPER_SOURCES)]
    B.atomic_json(root / "manifest.json", {"identity": {"selection": selection,
                  "settings": {s: P.task_settings(s) for s in P.PAPER_SOURCES}, "model": "test"}})
    for i, source in enumerate(P.PAPER_SOURCES):
        for j in range(i + 1):
            # Each task contributes one equal column, despite different sizes.
            B.append(root / "results.jsonl", {"qid": f"{i}-{j}", "source": source, "query": "q", "answer": ["a"],
                "output": "a", "qa_pair_id": "id", "metrics": {P.primary_metric(source): i / 10},
                "filtered": False, "completion_tokens": 1, "query_time_len": 1, "usage": {}})
    result = report(root)
    assert result["paper_overall"] == pytest.approx(((0 + .1 + .2 + .3) / 4 + (.4 + .5) / 2
                                                   + (.6 + .7) / 2 + (.8 + .9) / 2) / 4)
    assert result["paper_overall"] != pytest.approx(sum(i * (i + 1) / 10 for i in range(10)) / 55)
