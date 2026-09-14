"""Integração offline: execução, custos, retomada e protocolo de treino."""
import json
from dataclasses import replace

import pytest

from wrag import config as C
from wrag import methods
from wrag.embed import TfidfEmbedder
from wrag.eval import runner
from wrag.llm.stub import StubLLM
from wrag.util import read_json, read_jsonl


@pytest.fixture
def offline(monkeypatch, tmp_path):
    pytest.importorskip("rank_bm25")
    monkeypatch.setenv("WRAG_NO_PROGRESS", "1")
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(C, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    llm, emb = StubLLM(filter_rate=0), TfidfEmbedder()
    for module in (runner, methods):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
        monkeypatch.setattr(module, "get_embedder", lambda: emb)
    corpus = [{"title": "Employment", "text": "Ana works at Atlas."},
              {"title": "Location", "text": "Atlas is located in Recife."},
              {"title": "Research", "text": "Ana researches Optics."},
              {"title": "Other", "text": "Bruno lives in Natal."}]
    questions = [
        {"id": "q1", "question": "Where does Ana work?", "answer": "Atlas", "supports": [0]},
        {"id": "q2", "question": "Where is the employer of Ana located?", "answer": "Recife", "supports": [0, 1]},
        {"id": "q3", "question": "Who works at Atlas and researches Optics?", "answer": "Ana", "supports": [0, 2]},
    ]
    for q in questions:
        q["paragraphs"] = [
            {**p, "is_supporting": i in q["supports"]} for i, p in enumerate(corpus)]
        q.pop("supports")
    (tmp_path / "sample.json").write_text(json.dumps(questions), encoding="utf-8")
    (tmp_path / "sample_corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    return tmp_path, llm, questions


def test_all_methods_and_resume_preserve_records_and_original_index_cost(offline):
    tmp, llm, _ = offline
    names = ["dense", "bm25", "graphrag", "hipporag", "hipporag2", "relational", "witnessrag"]
    cfg = C.RunConfig(n_questions=3, top_k=2)
    root = runner.run(["sample"], names, cfg, tag="offline")
    before = {m: (root / "sample" / f"{m}.jsonl").read_bytes() for m in names}
    initial = read_json(root / "sample" / "summary.json")
    assert initial["corpus"]["n_passages"] == 4
    for m in names:
        rows = read_jsonl(root / "sample" / f"{m}.jsonl")
        assert len(rows) == 3 and all("uso_llm" in r for r in rows)
        assert all(len(r["recuperadas"]) <= 2 for r in rows)
    resumed = runner.run(["sample"], names, cfg, resume_dir=root)
    assert resumed == root
    summary = read_json(root / "sample" / "summary.json")
    assert len(summary["tentativas_indexacao"]) == 2
    assert summary["uso_indexacao"] == initial["uso_indexacao"]
    for m in names:
        assert (root / "sample" / f"{m}.jsonl").read_bytes() == before[m]
        assert summary["metodos"][m]["uso_consulta"] == initial["metodos"][m]["uso_consulta"]
    report = read_json(root / "report.json")
    assert report["datasets"]["sample"]["ids_comparados"] == ["q1", "q2", "q3"]
    with pytest.raises(ValueError, match="config"):
        runner.run(["sample"], names, replace(cfg, top_k=1), resume_dir=root)


def test_budget_demands_do_not_use_remaining_evaluation_questions(offline):
    cfg = C.RunConfig(n_questions=1, top_k=2)
    cfg.witness.budget_fraction = 0.5
    cfg.witness.enable_acquisition = False
    root = runner.run(["sample"], ["relational", "witnessrag"], cfg)
    summary = read_json(root / "sample" / "summary.json")
    for info in summary["metodos"].values():
        sources = info["indice"]["fontes_demandas"]
        assert sources["treino"] == 0
        assert sources["sintetizadas"] > 0
        assert sources["anotacoes_avaliacao_usadas"] is False


def test_explicit_training_overlap_is_rejected(offline):
    tmp, _, questions = offline
    train = tmp / "training.json"
    train.write_text(json.dumps(questions), encoding="utf-8")
    cfg = C.RunConfig(n_questions=3, train_questions_path=str(train))
    cfg.witness.budget_fraction = 0.5
    with pytest.raises(ValueError, match="sobrepostas"):
        runner.run(["sample"], ["witnessrag"], cfg)


def test_fresh_replaces_records_instead_of_appending(offline):
    cfg = C.RunConfig(n_questions=3)
    root = runner.run(["sample"], ["dense"], cfg)
    runner.run(["sample"], ["dense"], cfg, resume=False, resume_dir=root)
    assert len(read_jsonl(root / "sample" / "dense.jsonl")) == 3


def test_interleaved_failure_preserves_paired_prefix_and_can_resume(offline, monkeypatch):
    cfg = C.RunConfig(n_questions=3, interleave_methods=True)
    real = runner._answer_one
    calls = []
    def interrupted(name, retriever, corpus, question, run_cfg):
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        calls.append((question.qid, name))
        return real(name, retriever, corpus, question, run_cfg)
    monkeypatch.setattr(runner, "_answer_one", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.run(["sample"], ["dense", "witnessrag"], cfg)
    roots = list((offline[0] / "runs").iterdir())
    root = roots[0]
    assert calls[0][0] == calls[1][0]
    assert {calls[0][1], calls[1][1]} == {"dense", "witnessrag"}
    from wrag.eval.report import build_report
    build_report(root)
    report = read_json(root / "report.json")
    assert len(report["datasets"]["sample"]["ids_comparados"]) == 1
    monkeypatch.setattr(runner, "_answer_one", real)
    runner.run(["sample"], ["dense", "witnessrag"], cfg, resume_dir=root)
    for name in ("dense", "witnessrag"):
        rows = read_jsonl(root / "sample" / f"{name}.jsonl")
        assert len(rows) == len({r["qid"] for r in rows}) == 3
