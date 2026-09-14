"""Contraprovas e comparação independente: sem APIs, sem pesos de modelos."""
import itertools
import json
import random
from dataclasses import replace

import pytest

from test_witness import ExactEmbedder, build_toy
from wrag import config as C
from wrag.data import Corpus, Passage, Question, _PassageIndex, load_dataset
from wrag.embed import TfidfEmbedder
from wrag.eval import metrics as M
from wrag.eval.report import _dataset_block
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact, _cache_path, _dedupe, extract_targeted
from wrag.llm.base import LLMResult, UsageLedger
from wrag.llm.filters import LEDGER, configure_ledger
from wrag.llm.stub import StubLLM
from wrag.methods.base import IndexContext
from wrag.methods.hipporag2 import HippoRAG2Retriever
from wrag.util import canonical_symbol
from wrag.witness.budget import Demand, select_ilp, toy_instance, utility
from wrag.witness.demands import synthesize_demands
from wrag.witness.memory import MemoryView
from wrag.witness.provenance import AnswerCandidate, rank_passages, score_answers
from wrag.witness.query import Atom, ConjunctiveQuery, compile_query, compile_with_llm, from_decomposition
from wrag.witness.relational import SQLWitnessSearcher
from wrag.witness.search import Witness, WitnessSearcher


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    monkeypatch.setenv("WRAG_NO_PROGRESS", "1")
    configure_ledger(tmp_path / "filters.jsonl")


def custom_memory(triples):
    passages = [Passage(f"p{i}", f"T{i}", " ".join(t)) for i, t in enumerate(triples)]
    facts = [Fact(f"f{i}", *t, pid=f"p{i}") for i, t in enumerate(triples)]
    emb = ExactEmbedder()
    kg = build_graph(Corpus("toy", passages, []), ExtractionResult(facts=facts), emb)
    return MemoryView(kg), emb


def query(*atoms, answer="x"):
    return ConjunctiveQuery(answer_var=answer, atoms=[Atom(*a) for a in atoms])


def test_wrong_relation_never_becomes_a_proof():
    memory, searcher, _, _ = build_toy()
    assert not searcher.join(query(("fundada por", "Ana", "?x"))).complete


def test_reverse_edge_does_not_make_relation_symmetric():
    memory, emb = custom_memory([("Ana", "mother of", "Bruno")])
    for mode in ("exact", "semantic"):
        searcher = WitnessSearcher(memory, emb, C.WitnessConfig(grounding_mode=mode))
        assert not searcher.join(query(("mother of", "Bruno", "?x"))).complete


def test_answer_identity_preserves_punctuation():
    memory, emb = custom_memory([("Ana", "uses", "C++"), ("Ana", "uses", "C#")])
    result = WitnessSearcher(memory, emb).join(query(("uses", "Ana", "?x")))
    answers = score_answers(result.witnesses, memory, C.WitnessConfig())
    assert {a.answer for a in answers} == {"C++", "C#"}


def test_reacquisition_of_pruned_fact_is_query_local(monkeypatch):
    from wrag.methods import witnessrag as module
    memory, emb = custom_memory([("Ana", "works", "Atlas")])
    ctx = IndexContext(corpus=memory.corpus, llm=StubLLM(), embedder=emb, run=C.RunConfig())
    retriever = module.WitnessRAGRetriever(ctx)
    retriever.memory = memory
    retriever.searcher = WitnessSearcher(memory, emb)
    retriever.searcher.allowed = set()
    q = query(("works", "Ana", "?x"))
    assert not retriever.searcher.join(q).complete
    monkeypatch.setattr(module, "extract_targeted", lambda *a, **kw: [memory.facts[0]])
    action = module.AcquisitionAction("p0", "T0", "Ana works Atlas", "works", "Ana", 1, 0)
    assert retriever._acquire([action], Question("q", "Where?", ["Atlas"])) == 1
    assert len(memory.facts) == 1
    assert retriever.searcher.join(q).complete
    retriever.searcher.rollback()
    memory.reset()
    assert not retriever.searcher.join(q).complete


def test_answer_normalization_removes_punctuation_without_splitting_tokens():
    assert M.exact_match("The U.S.", ["US"]) == 1
    assert M.token_f1("Jean-Luc", ["JeanLuc"]) == 1


def test_reader_receives_evidence_beyond_old_character_limit():
    from wrag.eval.reader import read
    class RecordingLLM(StubLLM):
        def _complete(self, messages, params, stage="misc"):
            self.messages = messages
            return LLMResult(text='{"answer": "Atlas"}')
    llm = RecordingLLM()
    body = "Background. " * 200 + "Ana works at Atlas."
    corpus = Corpus("toy", [Passage("p", "Long passage", body)], [])
    read(llm, corpus, Question("q", "Where does Ana work?", ["Atlas"]), ["p"])
    assert "Ana works at Atlas." in llm.messages[-1]["content"]


def test_semantic_mode_gates_on_relation_and_is_not_certified():
    memory, _, emb, cfg = build_toy()
    cfg.grounding_mode = "semantic"
    searcher = WitnessSearcher(memory, emb, cfg)
    assert not searcher.join(query(("fundada por", "Ana", "?x"))).complete
    assert not searcher.join(query(("trabalha em", "Ana", "?x"))).exhaustive


def test_exact_joins_ignore_similarity_clusters():
    memory, emb = custom_memory([("Ana", "works", "Atlas"), ("Bruno", "researches", "Optics")])
    memory._cluster_of[memory.entity_id("Bruno")] = memory.cluster(memory.entity_id("Ana"))
    searcher = WitnessSearcher(memory, emb, C.WitnessConfig(grounding_mode="exact"))
    assert not searcher.join(query(("works", "?x", "Atlas"), ("researches", "?x", "Optics"))).complete


def test_same_fact_set_can_support_different_answer_bindings():
    memory, emb = custom_memory([("Ana", "follows", "Bruno"), ("Bruno", "follows", "Ana")])
    result = WitnessSearcher(memory, emb).join(query(("follows", "?x", "?y"), ("follows", "?y", "?x")))
    assert {w.answer for w in result.witnesses} == {"Ana", "Bruno"}


def test_repeated_atom_uses_one_evidence_event():
    memory, searcher, _, cfg = build_toy()
    result = searcher.join(query(("trabalha em", "Ana", "?x"), ("trabalha em", "Ana", "?x")))
    assert result.witnesses[0].facts == (0,)
    candidate = score_answers(result.witnesses, memory, cfg)[0]
    assert candidate.risk_raw == pytest.approx(cfg.delta_interpretation + cfg.delta_verbalization + 0.1)


@pytest.mark.parametrize("setting,reason", [("candidates_per_atom", "candidatos"),
                                          ("beam_width", "feixe"), ("max_witnesses", "testemunhas")])
def test_every_pruning_stage_is_reported(setting, reason):
    memory, _, emb, cfg = build_toy()
    setattr(cfg, setting, 1)
    result = WitnessSearcher(memory, emb, cfg).join(query(("trabalha em", "?x", "Atlas")))
    assert not result.exhaustive
    assert reason in result.truncations


def test_sql_and_exhaustive_search_agree_on_random_finite_databases():
    rng = random.Random(123)
    cfg = C.WitnessConfig(grounding_mode="exact", candidates_per_atom=0, beam_width=0, max_witnesses=0)
    universe = list(itertools.product(["Ana", "Bruno", "Clara"], ["r", "s"], ["Ana", "Bruno", "Clara"]))
    patterns = [query(("r", "?x", "?y"), ("s", "?y", "Clara")),
                query(("r", "?x", "?y"), ("r", "?y", "?x")),
                query(("r", "?x", "?x")),
                query(("r", "?x", "Bruno"), ("s", "?x", "Clara")),
                query(("r", "?x", "?y"), ("s", "?u", "?v"))]
    for _ in range(12):
        memory, emb = custom_memory(rng.sample(universe, 8))
        searcher, sql = WitnessSearcher(memory, emb, cfg), SQLWitnessSearcher(memory, cfg)
        for q in patterns:
            got, expected = searcher.join(q), sql.join(q)
            assert got.exhaustive
            assert {(w.answer, w.facts) for w in got.witnesses} == {(w.answer, w.facts) for w in expected.witnesses}
        sql.close()


def test_sql_constants_are_bound_parameters():
    name = "Ana'; DROP TABLE facts; --"
    memory, emb = custom_memory([(name, "works", "Atlas")])
    sql = SQLWitnessSearcher(memory, C.WitnessConfig())
    assert sql.join(query(("works", name, "?x"))).witnesses[0].answer == "Atlas"
    assert sql.db.execute("SELECT count(*) FROM facts").fetchone()[0] == 1
    sql.close()


def test_packing_never_splits_a_proof_to_fill_context():
    big = Witness((0, 1, 2), {}, 1, 0, ("a", "b", "c"), "first")
    small = Witness((3, 4), {}, 1, 1, ("d", "e"), "second")
    pids, _ = rank_passages([AnswerCandidate("first", [big], 1), AnswerCandidate("second", [small], .9)], 2)
    assert pids == ["d", "e"]


def test_duplicate_proofs_do_not_increase_score():
    memory, searcher, _, cfg = build_toy()
    witnesses = searcher.join(query(("trabalha em", "Ana", "?x"))).witnesses
    one = score_answers(witnesses, memory, cfg)[0].score
    assert score_answers(witnesses * 4, memory, cfg)[0].score == one
    copies = [Fact("a", "Ana", "works", "Atlas", "p0"), Fact("b", "Ana", "works", "Atlas", "p1")]
    assert [f.confidence for f in _dedupe(copies)] == [.9, .9]


def test_acquisition_does_not_mutate_shared_indexes_even_before_reset():
    memory, searcher, emb, cfg = build_toy()
    original = {k: list(v) for k, v in memory.base.facts_by_subject.items()}
    added = memory.add_facts([Fact("new", "Ana", "founded", "Nova", "p0")], emb)
    searcher.register_facts(added)
    assert dict(memory.base.facts_by_subject) == original
    assert memory.add_facts([Fact("new", "Ana", "founded", "Nova", "p0")], emb) == []
    searcher.rollback()
    memory.reset()
    assert dict(memory.base.facts_by_subject) == original


def test_distinct_symbols_do_not_collapse():
    assert len({canonical_symbol(s) for s in ["C", "C++", "C#"]}) == 3
    memory, emb = custom_memory([("Ana", "uses", "C++"), ("Bruno", "uses", "C#")])
    assert memory.entity_id("C++") != memory.entity_id("C#")


def test_passages_with_same_prefix_and_title_remain_distinct():
    index = _PassageIndex()
    first = index.add("Title", "prefix " * 50 + " first ending")
    second = index.add("Title", "prefix " * 50 + " second ending")
    assert first != second
    assert index.find("Title") is None
    assert index.find("Title", "another text") is None
    reversed_index = _PassageIndex()
    assert reversed_index.add("Title", "prefix " * 50 + " second ending") == second


def test_extraction_cache_depends_on_content_and_parameters():
    c1 = Corpus("test", [Passage("p0", "Title", "old")], [])
    c2 = Corpus("test", [Passage("p0", "Title", "new")], [])
    llm, cfg = StubLLM(), C.IEConfig()
    assert _cache_path(c1, llm, cfg) != _cache_path(c2, llm, cfg)
    assert _cache_path(c1, llm, cfg) != _cache_path(c1, llm, replace(cfg, max_tokens=42))


def test_tfidf_cache_tracks_tail_and_one_word_corpus():
    emb = TfidfEmbedder()
    emb._corpus = ["same"] * 2000 + ["old"]
    key = emb.cache_key()
    emb._corpus[-1] = "new"
    assert emb.cache_key() != key
    emb.fit(["word"])
    assert emb.encode(["word"]).shape == (1, 1)


def test_witness_metric_requires_relation_and_direction():
    gold = [("Ana", "mother of", "Bruno")]
    assert M.witness_coverage([("Ana", "works with", "Bruno")], gold) == 0
    assert M.witness_coverage([("Bruno", "mother of", "Ana")], gold) == 0
    assert M.endpoint_coverage([("Bruno", "works with", "Ana")], gold) == 1


def test_threshold_cannot_split_equal_risk():
    curve = M.risk_coverage_curve([1] * 12, [0, 1] * 6)
    assert len(curve) == 1 and curve[0].coverage == 1 and curve[0].accuracy == .5


def metric_row(qid, score=1, filtered=False):
    return {"qid": qid, "recall@2": score, "recall@5": score, "all_recall@5": score,
            "em": score, "f1": score, "filtrada": filtered, "n_hops": 1}


def test_partial_reports_use_same_question_ids_and_filters():
    records = {"witnessrag": [metric_row("a"), metric_row("b"), metric_row("c", filtered=True)],
               "dense": [metric_row("b", 0), metric_row("c")]}
    _, data = _dataset_block("test", records, {"c"}, {})
    assert data["ids_comparados"] == ["b"]
    assert {m["n_avaliadas"] for m in data["metodos"].values()} == {1}
    assert data["ausentes_por_metodo"]["dense"] == ["a"]
    with pytest.raises(ValueError):
        _dataset_block("test", {"dense": [metric_row("a"), metric_row("a")]}, set(), {})


def test_missing_method_cannot_disappear_from_comparison():
    _, data = _dataset_block("test", {"witnessrag": [metric_row("a")], "dense": []}, set(), {})
    assert data["ids_comparados"] == []


def test_annotated_mode_does_not_silently_use_llm():
    llm = StubLLM()
    q = compile_query(llm, Question("q", "Where?", ["Recife"]), mode="oracle")
    assert not q.atoms and q.source == "oracle-indisponivel"
    assert llm.usage.snapshot()["total"]["chamadas"] == 0
    q = from_decomposition(Question("q", "Who?", ["x"], decomposition=[{"question": "Who joined #1 and #2?"}]))
    assert not q.atoms


def test_compiler_does_not_silently_drop_constraints():
    class Fixed(StubLLM):
        def _complete(self, messages, params, stage="misc"):
            return LLMResult(text=json.dumps({"answer_var": "x", "atoms": [
                {"relation": "r", "subject": "Ana", "object": "?x"} for _ in range(5)]}))
    result = compile_with_llm(Fixed(), Question("q", "Who?", ["x"]), max_atoms=4)
    assert not result.atoms and result.validation_error == "limite_de_atomos"


def test_empty_recognition_result_really_triggers_dense_fallback():
    memory, _, emb, cfg = build_toy()
    class Empty(StubLLM):
        def _complete(self, messages, params, stage="misc"):
            return LLMResult(text='{"fact": []}')
    retriever = HippoRAG2Retriever(IndexContext(memory.corpus, Empty(), emb, C.RunConfig(), kg=memory.base))
    assert retriever.recognition_memory(Question("q", "Who?", ["x"]), [0, 1], 5) == ([], False)


def test_filter_acquisition_records_question_not_passage():
    llm = StubLLM(filter_rate=1)
    extract_targeted(llm, "works", [("p0", "title", "body")], question_id="q1")
    assert LEDGER.events[-1].item_id == "q1"


def test_cache_hits_are_not_new_api_tokens():
    usage = UsageLedger()
    usage.record("qa", LLMResult(prompt_tokens=10, completion_tokens=2, cached=True))
    usage.record("qa", LLMResult(prompt_tokens=20, completion_tokens=3))
    total = usage.snapshot()["total"]
    assert total["tokens_prompt"] == 30 and total["tokens_prompt_sem_cache"] == 20


def test_ilp_matches_bruteforce_for_every_toy_budget():
    pytest.importorskip("pulp")
    demands, costs = toy_instance()
    all_sets = [set(s) for n in range(6) for s in itertools.combinations(costs, n)]
    for budget in range(6):
        result = select_ilp(demands, costs, budget)
        optimum = max(utility(demands, s) for s in all_sets if len(s) <= budget)
        assert result.optimal and result.value == optimum and result.cost <= budget


def test_time_limited_incumbent_is_not_labeled_optimal(monkeypatch):
    pulp = pytest.importorskip("pulp")
    def feasible(problem, solver):
        for var in problem.variables():
            var.varValue = float(var.name in {"x_1", "x_2", "x_3"})
        problem.status = pulp.LpStatusOptimal
        problem.sol_status = pulp.LpSolutionIntegerFeasible
    monkeypatch.setattr(pulp.LpProblem, "solve", feasible)
    demands, costs = toy_instance()
    result = select_ilp(demands, costs, 3)
    assert not result.optimal and result.status.startswith("Feasible")


def test_invalid_budget_input_is_rejected():
    demands, costs = toy_instance()
    with pytest.raises(ValueError):
        select_ilp(demands, costs, -1)
    with pytest.raises(ValueError):
        select_ilp(demands + demands[:1], costs, 3)


def test_synthesized_demands_group_alternative_witnesses():
    memory, _ = custom_memory([("Ana", "works", "Atlas"), ("Atlas", "located", "Recife"),
                               ("Ana", "works", "Nova"), ("Nova", "located", "Recife")])
    demands = synthesize_demands(memory, n_demands=50)
    assert any({frozenset({0, 1}), frozenset({2, 3})} <= set(d.witnesses) for d in demands)


def test_loader_keeps_full_corpus_when_sampling_questions(tmp_path):
    rows = [{"id": "q1", "question": "Who?", "answer": "Ana", "paragraphs": [
        {"title": "T1", "text": "Ana works Atlas.", "is_supporting": True}]}]
    (tmp_path / "sample.json").write_text(json.dumps(rows))
    (tmp_path / "sample_corpus.json").write_text(json.dumps([
        {"title": "T1", "text": "Ana works Atlas."}, {"title": "Distractor", "text": "Bruno plays tennis."}]))
    assert len(load_dataset("sample", n_questions=1, data_dir=tmp_path).passages) == 2
