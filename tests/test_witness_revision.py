"""Contraprovas: corte global, identidade ligada e verificação sem citações."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_regressions import custom_memory
from wrag import config as C
from wrag.data import Question
from wrag.llm.base import LLMResult
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.search import Grounding, WitnessSearcher
from wrag.witness.verification import verify_witnesses


def chain(monkeypatch):
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    memory, embedder = custom_memory([
        ("Ana", "works", "Atlas"), ("Other", "located", "Paris"),
        ("Atlas", "located", "Recife")])
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("works", "Ana", "?y"), Atom("located", "?y", "?x")])
    # Distratores dominam a sonda global "located". A aresta correta ainda
    # existe e tem relação exata, mas cai fora do pool global de tamanho 2.
    memory.fact_vectors[0] = embedder.encode(["located"])[0]
    memory.fact_vectors[1] = memory.fact_vectors[0]
    memory.fact_vectors[2] = 0
    cfg = C.WitnessConfig(candidates_per_atom=1, enable_acquisition=False)
    return memory, embedder, query, cfg


def test_bound_join_recovers_edge_excluded_by_global_topk(monkeypatch):
    memory, embedder, query, cfg = chain(monkeypatch)
    searcher = WitnessSearcher(memory, embedder, cfg)
    assert not searcher.join(query).complete
    cfg.binding_aware_grounding = True
    result = searcher.join(query)
    assert [w.answer for w in result.witnesses] == ["Recife"]
    assert not result.exhaustive
    # Uma ligação nunca pula para Other; nem contorna a máscara de memória.
    searcher.allowed = {0, 1}
    assert not searcher.join(query).complete


def test_external_candidates_are_not_silently_expanded(monkeypatch):
    memory, embedder, query, cfg = chain(monkeypatch)
    cfg.binding_aware_grounding = True
    result = WitnessSearcher(memory, embedder, cfg).join(query, [[Grounding(0, 1)], []])
    assert not result.complete


def test_exact_mode_keeps_symbolic_semantics(monkeypatch):
    memory, embedder, query, cfg = chain(monkeypatch)
    cfg.grounding_mode = "exact"
    cfg.binding_aware_grounding = True
    cfg.candidates_per_atom = 0
    result = WitnessSearcher(memory, embedder, cfg).join(query)
    assert result.complete and result.exhaustive
    assert result.witnesses[0].answer == "Recife"


class Judge:
    def __init__(self, output, **kwargs):
        self.output, self.kwargs, self.calls = output, kwargs, []

    def chat(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return LLMResult(text=json.dumps(self.output), **self.kwargs)


@pytest.mark.parametrize("failure", [None, "missing_atom", "invented_quote", "wrong_pid",
                                     "wrong_type", "string_bool", "filtered", "empty", "budget"])
def test_verifier_requires_complete_source_quotes_and_real_booleans(monkeypatch, failure):
    memory, embedder, query, cfg = chain(monkeypatch)
    cfg.binding_aware_grounding = True
    witnesses = WitnessSearcher(memory, embedder, cfg).join(query).witnesses
    output = {"supported": True, "answers_question": True, "evidence": [
        {"atom": 0, "pid": "p0", "quote": "Ana works Atlas"},
        {"atom": 1, "pid": "p2", "quote": "Atlas located Recife"}]}
    if failure == "missing_atom": output["evidence"].pop()
    if failure == "invented_quote": output["evidence"][1]["quote"] = "Atlas located Berlin"
    if failure == "wrong_pid": output["evidence"][1]["pid"] = "p1"
    if failure == "wrong_type": output["answers_question"] = False
    if failure == "string_bool": output["supported"] = "true"
    if failure == "empty": output = []
    judge = Judge(output, filtered=failure == "filtered")
    accepted, diag = verify_witnesses(judge, memory.corpus, memory,
        Question("q", "Where does Ana work?", ["SECRET_GOLD"]), query,
        witnesses, 0 if failure == "budget" else 1, "toy")
    assert bool(accepted) == (failure is None)
    assert diag["avaliadas"] == (0 if failure == "budget" else 1)
    for prompt, kwargs in judge.calls:
        assert "SECRET_GOLD" not in prompt
        assert kwargs["stage"] == "witness.verify"


def comparator():
    spec = importlib.util.spec_from_file_location("compare_runs",
        Path(__file__).resolve().parents[1] / "scripts" / "compare-runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fallback", [True, False])
def test_rejected_witness_does_not_replace_dense(monkeypatch, fallback):
    from wrag.methods.base import IndexContext
    from wrag.methods.witnessrag import WitnessRAGRetriever
    memory, embedder, query, cfg = chain(monkeypatch)
    cfg.binding_aware_grounding = cfg.verify_witnesses = True
    cfg.dense_fallback = fallback
    run = C.RunConfig()
    run.witness = cfg
    ctx = IndexContext(memory.corpus, Judge({"supported": False}), embedder, run)
    retriever = WitnessRAGRetriever(ctx)
    retriever.memory = memory
    retriever.searcher = WitnessSearcher(memory, embedder, cfg)
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *a, **k: query)
    result = retriever._retrieve_inner(Question("q", "Where?", []), 2, ["p1", "p0"], [.8, .7])
    assert result.pids == (["p1", "p0"] if fallback else [])
    assert result.diagnostics["verificacao"]["aceitas"] == 0
    assert result.diagnostics["n_testemunhas_propostas"] == 1


def test_compare_rejects_different_corpus_same_size_and_duplicate_ids(tmp_path):
    compare = comparator()
    for name in ("a", "b"):
        folder = tmp_path / name
        (folder / "toy").mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps({"datasets": ["toy"]}))
        (folder / "toy" / "corpus.json").write_text(json.dumps({
            "corpus_hash": name, "questions_hash": "same", "n_passages": 10}))
    assert any("corpus_hash" in s for s in compare.guard(tmp_path / "a", tmp_path / "b"))
    (tmp_path / "a/toy/witnessrag.jsonl").write_text('{"qid":"x"}\n{"qid":"x"}\n')
    with pytest.raises(ValueError, match="duplicados"):
        compare.load(tmp_path / "a", "toy")
