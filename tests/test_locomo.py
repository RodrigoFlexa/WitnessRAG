import copy
import json

import pytest

from wrag.data import load_dataset
from wrag.locomo import convert
from wrag.pilot import make_plan, parser, prepare_data
from wrag.util import read_json


def sample():
    return [{"sample_id": "conv-test", "conversation": {
        "session_2_date_time": "2023-05-10", "session_2": [
            {"dia_id": "D2:1", "speaker": "Ana", "text": "I live in Recife."}],
        "session_1_date_time": "2023-05-01", "session_1": [
            {"dia_id": "D1:1", "speaker": "Ana", "text": "I work at Atlas."},
            {"dia_id": "D1:2", "speaker": "Bruno", "text": "Here is my drawing.",
             "blip_caption": "A blue bird"}],
        "speaker_a": "Ana", "speaker_b": "Bruno"},
        "qa": [{"question": f"Question category {c}?", "answer": "SECRET_GOLD",
                "category": c, "evidence": ["D1:1; D2:1"]} for c in range(1, 6)],
        "session_summary": "SECRET_SUMMARY", "observation": "SECRET_OBSERVATION"},
        {"sample_id": "second", "conversation": {}, "qa": []}]


def test_categories_complete_conversation_and_evidence_mapping(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(sample()), encoding="utf-8")
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--locomo-file", str(source)])
    plan = make_plan(args, tmp_path / "output")
    assert plan["methods"] == ["witnessrag"] and plan["settings"]["questions"] is None
    meta = prepare_data(plan)
    corpus = load_dataset("locomo", data_dir=tmp_path / "output/data")
    assert [q.qtype for q in corpus.questions] == ["multi-hop", "single-hop"]
    assert len(corpus.passages) == 2
    assert all(len(q.gold_pids) == 2 for q in corpus.questions)
    assert meta["questions_by_type"] == {"multi-hop": 1, "single-hop": 1}
    text = "\n".join(corpus.texts())
    assert "Ana: I work at Atlas." in text and "2023-05-01" in text and "A blue bird" in text
    assert "SECRET" not in text and "Question category" not in text
    assert "session_1" in corpus.passages[0].title


def test_sampling_never_removes_dialogue_and_ignores_qa_when_chunking():
    raw = sample()
    q, passages, _ = convert(raw, turns_per_passage=1)
    raw[0]["qa"][0]["answer"] = "different answer"
    one, other_passages, _ = convert(raw, turns_per_passage=1, n_questions=1)
    assert len(one) == 1 and len(q) == 2 and passages == other_passages
    assert len(passages) == 3


@pytest.mark.parametrize("error", ["missing", "duplicate", "empty_evidence", "index"])
def test_invalid_annotations_fail_explicitly(error):
    raw = copy.deepcopy(sample())
    if error == "missing": raw[0]["qa"][0]["evidence"] = ["D999:1"]
    if error == "empty_evidence": raw[0]["qa"][0]["evidence"] = []
    if error == "duplicate": raw[0]["conversation"]["session_2"][0]["dia_id"] = "D1:1"
    with pytest.raises(ValueError):
        convert(raw, conversation_index=99 if error == "index" else 0)


def test_other_pilot_defaults_unchanged(tmp_path):
    plan = make_plan(parser().parse_args(["--gpu", "3"]), tmp_path)
    assert plan["settings"]["questions"] == 100
    assert "hipporag2" in plan["methods"] and len(plan["methods"]) == 7


def test_locomo_witness_pipeline_and_category_report(tmp_path, monkeypatch):
    from wrag import config as C, methods
    from wrag.eval import runner
    from wrag.embed import TfidfEmbedder
    from wrag.llm.stub import StubLLM
    from wrag.util import write_json
    questions, passages, _ = convert(sample())
    write_json(tmp_path / "locomo.json", questions)
    write_json(tmp_path / "locomo_corpus.json", passages)
    for name, path in (("DATA_DIR", tmp_path), ("CACHE_DIR", tmp_path / "cache"),
                       ("RUNS_DIR", tmp_path / "runs")):
        monkeypatch.setattr(C, name, path)
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    llm, emb = StubLLM(filter_rate=0), TfidfEmbedder()
    for module in (runner, methods):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
        monkeypatch.setattr(module, "get_embedder", lambda: emb)
    cfg = C.RunConfig(n_questions=2, corpus_scope="locomo_full_selected_conversation")
    cfg.witness.enable_acquisition = False
    root = runner.run(["locomo"], ["witnessrag"], cfg, tag="locomo-offline")
    report = read_json(root / "report.json")["datasets"]["locomo"]
    assert set(report["metodos"]) == {"witnessrag"}
    assert report["por_categoria"]["witnessrag"]["single-hop"]["n"] == 1
    assert report["por_categoria"]["witnessrag"]["multi-hop"]["n"] == 1
