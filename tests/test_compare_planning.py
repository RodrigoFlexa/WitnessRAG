from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare-planning.py"
    spec = importlib.util.spec_from_file_location("compare_planning", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def make_run(root: Path, query_plans: bool, qids=("q1", "q2"), *, top_k=5) -> Path:
    run = root / ("multi" if query_plans else "simple")
    (run / "locomo").mkdir(parents=True)
    manifest = {
        "datasets": ["locomo"],
        "config": {
            "dataset": "locomo", "n_questions": 2, "top_k": top_k,
            "witness": {"query_plans": query_plans, "max_query_plans": 3},
        },
        "llm": {"backend": "openai", "deployment": "model"},
        "embedder": {"chave": "embedder"},
        "codigo_hash": "same-code",
    }
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "locomo" / "corpus.json").write_text(json.dumps({
        "corpus_hash": "corpus", "questions_hash": "questions",
        "question_ids": ["q1", "q2"],
    }), encoding="utf-8")
    records = [{
        "qid": qid, "categoria_locomo": "single-hop", "f1_locomo": 1.0,
        "em_locomo": 1.0, "f1": 1.0, "em": 1.0, "recall@5": 1.0,
        "all_recall@5": 1.0, "diagnosticos": {},
    } for qid in qids]
    (run / "locomo" / "witnessrag.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return run


def test_strict_comparison_accepts_only_planning_difference(tmp_path):
    compare = module()
    simple = make_run(tmp_path, False)
    multi = make_run(tmp_path, True)
    paired, notices = compare.validate(simple, multi, False)
    assert paired == {"q1", "q2"}
    assert notices == []


def test_strict_comparison_rejects_partial_or_other_config_change(tmp_path):
    compare = module()
    simple = make_run(tmp_path, False)
    multi = make_run(tmp_path, True, qids=("q1",))
    with pytest.raises(ValueError, match="rodada incompleta"):
        compare.validate(simple, multi, False)
    paired, notices = compare.validate(simple, multi, True)
    assert paired == {"q1"}
    assert notices and "PARCIAL" in notices[0]

    manifest = json.loads((multi / "run.json").read_text(encoding="utf-8"))
    manifest["config"]["top_k"] = 15
    (multi / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="configuração diferente"):
        compare.validate(simple, multi, True)
