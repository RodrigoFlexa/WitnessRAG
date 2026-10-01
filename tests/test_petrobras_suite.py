"""Exercise queue safety and official selections without network/model calls."""
import importlib.util
import json
from pathlib import Path
from contextlib import nullcontext

import pytest

from benchmarks.memoryagentbench.__main__ import parser as mab_parser
from wrag import pilot

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("petrobras_suite", ROOT / "scripts/run-petrobras-suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def args_for(tmp_path, *extra):
    return suite.parser().parse_args(["--output", str(tmp_path / "outputs"),
                                    "--cache", str(tmp_path / "cache"), *extra])


def test_official_task_order_profiles_and_azure_embedding(tmp_path):
    args = args_for(tmp_path)
    commands = suite.stage_commands(args)
    assert list(commands) == ["locomo", "sf", "ar"]
    c = commands["locomo"]
    locomo = pilot.parser().parse_args(c[c.index("wrag.pilot") + 1:])
    assert locomo.backend == "azure" and locomo.locomo_conversation == "all"
    assert locomo.local_plans and locomo.local_plan_version == "v2" and locomo.reader_reflection
    assert not locomo.reflection_replan
    assert locomo.fact_budget == 40 and locomo.top_k == 5 and locomo.qa_max_tokens == 128
    assert locomo.locomo_chunk_tokens == 2048 and locomo.locomo_ie_window_tokens == 512
    plan = pilot.make_plan(locomo, args.output / "locomo")
    assert plan["env"]["WRAG_EMBED_BACKEND"] == "azure"
    assert plan["env"]["WRAG_AZURE_EMBED_DEPLOYMENT"] == "embedding-3-small-global"
    for stage, expected, adaptation in [("sf", list(suite.SF), "standard"), ("ar", list(suite.AR), "ar-source-v2")]:
        c = commands[stage]
        mab = mab_parser().parse_args(c[c.index("benchmarks.memoryagentbench") + 1:])
        assert mab.backend == mab.embed_backend == "azure" and mab.sources == expected
        assert mab.adaptation == adaptation and mab.protocol == mab.suite == "paper"
        assert mab.resume and mab.fact_budget == 40 and mab.max_questions == mab.max_contexts == 0
        assert not mab.no_reflection and "judge" not in c


def test_dotenv_is_data_environment_wins_and_secrets_never_enter_identity(tmp_path):
    args = args_for(tmp_path)
    args.env_file = tmp_path / ".env"
    args.env_file.write_text('AZURE_OPENAI_API_KEY="do-not-persist"\nAZURE_OPENAI_ENDPOINT=https://example.test\n'
                             'WRAG_FROZEN_MEMORY_SOURCE=/old\nWRAG_AZURE_DEPLOYMENT=old-model\n'
                             'WRAG_EMBED_BACKEND=tfidf\n', encoding="utf-8")
    env = suite.runtime_environment(args, {"AZURE_OPENAI_API_KEY": "environment-wins"})
    assert env["AZURE_OPENAI_API_KEY"] == "environment-wins"
    assert env["WRAG_AZURE_DEPLOYMENT"] == "gpt-4-1-mini-petrobras"
    assert env["WRAG_EMBED_BACKEND"] == "azure" and env["WRAG_FROZEN_MEMORY_SOURCE"] == ""
    saved = json.dumps(suite.identity(args, env))
    assert "environment-wins" not in saved and "do-not-persist" not in saved and "https://example.test" not in saved


def test_loсomo_partial_receives_resume(tmp_path):
    args = args_for(tmp_path)
    (args.output / "locomo").mkdir(parents=True)
    (args.output / "locomo/pilot.json").write_text("{}")
    assert "--resume" in suite.stage_commands(args)["locomo"]


def test_two_models_run_entire_suite_in_order_with_shared_cache(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(suite, "suite_lock", lambda _: nullcontext())
    monkeypatch.setattr(suite, "execute", lambda a, e, c, i: captured.append((a, e, c)))
    suite.main(["--output", str(tmp_path / "results"), "--cache", str(tmp_path / "cache"),
                "--models", "gpt-4-1-mini-petrobras", "gpt-4o-mini-example"])
    assert [a.model for a, _, _ in captured] == ["gpt-4-1-mini-petrobras", "gpt-4o-mini-example"]
    assert captured[0][0].cache == captured[1][0].cache
    assert captured[0][0].output != captured[1][0].output
    assert all(list(c) == ["locomo", "sf", "ar"] for _, _, c in captured)


def test_failure_stops_the_queue_and_preserves_current_stage(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    calls = []
    monkeypatch.setattr(suite, "suite_lock", lambda _: nullcontext())
    def fail(command, *rest):
        calls.append(command)
        if command == ["sf"]:
            raise RuntimeError("intentional SF failure")
    monkeypatch.setattr(suite, "run_command", fail)
    monkeypatch.setattr(suite, "verify_complete", lambda *a: None)
    monkeypatch.setattr("benchmarks.memoryagentbench.data.verify_data", lambda *a: {})
    env = {"AZURE_OPENAI_API_KEY": "fake", "AZURE_OPENAI_ENDPOINT": "https://test.invalid"}
    with pytest.raises(RuntimeError, match="SF failure"):
        suite.execute(args, env, {"locomo": ["locomo"], "sf": ["sf"], "ar": ["ar"]}, {"test": 1})
    state = json.loads((args.output / "suite.json").read_text())
    assert state["completed"] == ["locomo"] and state["current_stage"] == "sf"
    assert ["ar"] not in calls


def test_completed_stage_is_revalidated_without_inference(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.output.mkdir()
    (args.output / "suite.json").write_text(json.dumps({"identity": {"test": 1}, "completed": ["locomo"]}))
    monkeypatch.setattr(suite, "suite_lock", lambda _: nullcontext())
    checks = []
    monkeypatch.setattr(suite, "verify_complete", lambda *a: checks.append(a[0]))
    monkeypatch.setattr(suite, "run_command", lambda *a: pytest.fail("No command should run"))
    suite.execute(args, {}, {"locomo": ["locomo"]}, {"test": 1})
    assert checks == ["locomo"]


def test_live_pid_prevents_resume_before_any_calls(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    args.output.mkdir()
    (args.output / "suite.json").write_text(json.dumps({"identity": {}, "completed": [], "active_pid": 123}))
    monkeypatch.setattr(suite, "suite_lock", lambda _: nullcontext())
    monkeypatch.setattr(suite, "pid_alive", lambda p: True)
    monkeypatch.setattr(suite, "run_command", lambda *a: pytest.fail("Duplicate writer"))
    with pytest.raises(RuntimeError, match="still active"):
        suite.execute(args, {}, {"locomo": []}, {})


def test_stale_benchmark_lock_checks_pid_before_removal(tmp_path, monkeypatch):
    lock = tmp_path / "run.lock"
    lock.write_text('{"pid": 123}')
    monkeypatch.setattr(suite, "pid_alive", lambda p: True)
    with pytest.raises(RuntimeError, match="still alive"):
        suite.clear_abandoned_mab_lock(tmp_path)
    assert lock.exists()
    monkeypatch.setattr(suite, "pid_alive", lambda p: False)
    suite.clear_abandoned_mab_lock(tmp_path)
    assert not lock.exists()


def sf_checkpoint(tmp_path):
    selection, rows = [], []
    for source, count in suite.SF.items():
        selection.append({"key": source, "source": source, "questions": count})
        rows.extend({"qid": f"{source}:q{i}", "source": source,
                     "primary_metric": "substring_exact_match", "metrics": {"substring_exact_match": 1}}
                    for i in range(count))
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {"selection": selection}}))
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return rows


@pytest.mark.parametrize("damage", ["duplicate", "missing", "foreign", "metric"])
def test_completion_rejects_bad_checkpoints(tmp_path, damage):
    rows = sf_checkpoint(tmp_path)
    if damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "missing":
        rows.pop()
    elif damage == "foreign":
        rows[-1]["qid"] = "foreign:q99"
    else:
        rows[-1]["metrics"] = {}
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        suite.verify_complete("sf", tmp_path)


def test_complete_sf_checks_all_200_ids(tmp_path):
    sf_checkpoint(tmp_path)
    suite.verify_complete("sf", tmp_path)


def test_dry_run_needs_no_key_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(suite, "execute", lambda *a: pytest.fail("dry run must not execute"))
    suite.main(["--output", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["models"]) == 1 and not (tmp_path / "out").exists()


def test_resume_rejects_embedding_provider_switch_even_for_same_model(tmp_path):
    args = pilot.parser().parse_args(["--gpu", "0", "--backend", "azure"])
    old = pilot.make_plan(args, tmp_path)
    args.embed_backend = "azure"
    new = pilot.make_plan(args, tmp_path)
    with pytest.raises(ValueError, match="embed_backend"):
        pilot._validate_resume(old, new)
    old["settings"].pop("embed_backend")
    args.embed_backend = "st"
    pilot._validate_resume(old, pilot.make_plan(args, tmp_path))
