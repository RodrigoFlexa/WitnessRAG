"""The queue must distinguish a completed SF benchmark from a dead process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run-memoryagentbench-after-sf-qwen.sh"
PROBE = SCRIPT.read_text(encoding="utf-8").split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]


def probe(path):
    return subprocess.run([sys.executable, "-c", PROBE],
        env={**os.environ, "SF_OUTPUT_DIR": str(path)}, capture_output=True, text=True).returncode


def complete(path):
    sources = ["factconsolidation_sh_262k", "factconsolidation_mh_262k"]
    (path / "summary.json").write_text(json.dumps({"sources": dict.fromkeys(sources, {}),
        "generation_complete": True, "evaluation_complete": True}))
    (path / "manifest.json").write_text(json.dumps({"identity": {
        "model": "Qwen/Qwen2.5-14B-Instruct", "engine": {"witness": {"fact_budget": 40}},
        "settings": {source: {"protocol": "paper"} for source in sources}}}))
    rows = [{"qid": f"{source}:{i}", "source": source} for source in sources for i in range(100)]
    (path / "results.jsonl").write_text("\n".join(map(json.dumps, rows)))


def test_queue_waits_without_completed_sf(tmp_path):
    assert probe(tmp_path) == 1


def test_queue_accepts_exactly_two_complete_hundred_question_tasks(tmp_path):
    complete(tmp_path)
    assert probe(tmp_path) == 0


@pytest.mark.parametrize("field", ["generation_complete", "evaluation_complete"])
def test_queue_waits_for_incomplete_predictions_or_evaluation(tmp_path, field):
    complete(tmp_path)
    path = tmp_path / "summary.json"
    summary = json.loads(path.read_text())
    summary[field] = False
    path.write_text(json.dumps(summary))
    assert probe(tmp_path) == 1


def test_queue_rejects_wrong_budget_and_duplicate_predictions(tmp_path):
    complete(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["engine"]["witness"]["fact_budget"] = 20
    path.write_text(json.dumps(manifest))
    assert probe(tmp_path) == 2
    complete(tmp_path)
    path = tmp_path / "results.jsonl"
    rows = path.read_text().splitlines()
    rows[-1] = rows[0]
    path.write_text("\n".join(rows))
    assert probe(tmp_path) == 2
