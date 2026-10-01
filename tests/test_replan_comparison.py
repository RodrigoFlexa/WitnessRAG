"""A paired report must not silently combine versions or duplicate questions."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("replan_comparison", Path(__file__).parents[1] / "scripts/analyze-replan-frozen.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_case(root, name="conv00", qid="q", new_score=.5, version="v4"):
    folder = root / name
    folder.mkdir(parents=True)
    manifest = {"code_hash": "hash", "controller_version": version, "model": "qwen",
                "selection_rule": "pilot", "selected_ids": [qid]}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    row = {"qid": qid, "tipo": "multi-hop", "historical_f1": 0., "f1_locomo": new_score,
           "trace": {"performed": True, "gate": {"valid": True}, "kept_fact_indices": list(range(12))},
           "historical_usage": {"total": {"tokens_prompt": 100, "tokens_resposta": 10}},
           "uso_llm": {"total": {"tokens_prompt": 150, "tokens_resposta": 20}}}
    (folder / "results.jsonl").write_text(json.dumps(row) + "\n")


def test_pilot_report_has_paired_metrics_and_cannot_claim_full_locomo(tmp_path):
    write_case(tmp_path)
    result = module.summarize(tmp_path)
    assert result["complete_selected_cohort"] and result["development_only"]
    assert not result["full_locomo"]
    assert result["groups"]["all"]["standard_f1"] == 0
    assert result["groups"]["all"]["new_replan_f1"] == 50
    assert result["logical_tokens"] == {"standard": 110, "new_replan": 170}


def test_duplicate_questions_and_mixed_versions_are_rejected(tmp_path):
    write_case(tmp_path)
    write_case(tmp_path, "conv01")
    with pytest.raises(ValueError, match="duplicate"):
        module.summarize(tmp_path)
    (tmp_path / "conv01" / "results.jsonl").unlink()
    (tmp_path / "conv01" / "manifest.json").unlink()
    (tmp_path / "conv01").rmdir()
    write_case(tmp_path, "conv01", qid="other", version="v2")
    with pytest.raises(ValueError, match="Mixed"):
        module.summarize(tmp_path)
