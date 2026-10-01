"""Exercise Linux launchers without downloading models or starting a GPU job."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from benchmarks.memoryagentbench.__main__ import configure_endpoint, parser as mab_parser
from wrag import pilot

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path("C:/Program Files/Git/bin/bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@pytest.fixture
def launch_workspace(tmp_path):
    if not BASH:
        pytest.skip("bash unavailable")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run-standard-qwen.sh", "run-standard-qwen-variants.sh", "run-local-plans-qwen.sh",
                 "run-witness-proof-locomo.sh", "proof-profiles.sh", "serve-qwen-vllm.sh", "setup-qwen-server.sh",
                 "run-replan-frozen-qwen20.sh", "run-replan-gap-qwen20.sh",
                 "run-memoryagentbench-sf-qwen.sh", "run-memoryagentbench-after-sf-qwen.sh",
                 "run-memoryagentbench-ar-adapted-qwen.sh"):
        (scripts / name).write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"),
                                   encoding="utf-8", newline="\n")
    fake = tmp_path / "fake-python"
    fake.write_text('#!/usr/bin/env bash\nprintf "CALL\\nCUDA=%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$CAPTURE"\n'
                    'printf "%s\\n" "$@" >> "$CAPTURE"\n', encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    # Git Bash prepends its own binaries to PATH; export a function instead.
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("curl() { return 0; }\nexport -f curl\n"
                        "flock() { return ${FLOCK_EXIT:-0}; }\nexport -f flock\n",
                        encoding="utf-8", newline="\n")
    capture = tmp_path / "calls.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WRAG_", "OPENAI_"))}
    env.update({"BENCH_PYTHON": fake.as_posix(), "VLLM_PYTHON": fake.as_posix(),
                "CAPTURE": capture.as_posix(), "PATH": str(tmp_path) + os.pathsep + env["PATH"],
                "GPU": "7", "PORT": "8095", "MODEL": "Qwen/Qwen2.5-14B-Instruct",
                "BASH_ENV": bash_env.as_posix()})
    for key in ("FACT_BUDGET", "EXTRA_FLAGS", "LOCOMO_CONVERSATION", "EMBED_GPU", "CACHE_DIR",
                "MODEL_REVISION", "VLLM_API_KEY", "PROTOCOL", "SUITE", "REFLECTION_REPLAN"):
        env.pop(key, None)
    return tmp_path, capture, env


def launch(workspace, script, *args, **extra_env):
    root, capture, env = workspace
    result = subprocess.run([BASH, f"scripts/{script}", *args], cwd=root, env={**env, **extra_env},
                            capture_output=True, text=True, timeout=30)
    lines = capture.read_text().splitlines() if capture.exists() else []
    calls = []
    for line in lines:
        if line == "CALL":
            calls.append([])
        else:
            calls[-1].append(line)
    return result, calls


def test_adapted_ar_launcher_uses_only_authorized_tasks_and_gpu_zero(launch_workspace):
    result, calls = launch(launch_workspace, "run-memoryagentbench-ar-adapted-qwen.sh")
    assert result.returncode == 0, result.stderr
    evaluations = [c for c in calls if "run" in c and "benchmarks.memoryagentbench" in c]
    assert len(evaluations) == 1
    c = evaluations[0]
    args = mab_parser().parse_args(c[c.index("benchmarks.memoryagentbench") + 1:])
    assert args.adaptation == "ar-source-v2" and args.fact_budget == 40
    assert args.splits == ["Accurate_Retrieval"]
    assert args.sources == ["ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*", "eventqa_full"]
    assert args.model == "Qwen/Qwen2.5-14B-Instruct" and args.protocol == args.suite == "paper"
    assert args.resume and args.source_excerpt_chars == 2400 and c[0] == "CUDA=0"
    assert all("judge" not in c for c in calls)


def test_adapted_ar_launcher_rejects_duplicate_before_model_calls(launch_workspace):
    result, calls = launch(launch_workspace, "run-memoryagentbench-ar-adapted-qwen.sh", FLOCK_EXIT="1")
    assert result.returncode == 2 and "already running" in result.stderr and not calls


def test_frozen_replan_runs_only_twenty_facts_all_ten_conversations(launch_workspace):
    result, calls = launch(launch_workspace, "run-replan-frozen-qwen20.sh", "full")
    assert result.returncode == 0, result.stderr
    evaluations = [call for call in calls if "scripts/test-reflection-replan.py" in call]
    assert len(evaluations) == 10
    assert [call[call.index("--conversation") + 1] for call in evaluations] == list(map(str, range(10)))
    for call in evaluations:
        assert "CUDA=7" in call
        assert call[call.index("--expected-fact-budget") + 1] == "20"
        assert call[call.index("--embed-device") + 1] == "cuda:0"
        assert "--all-questions" in call and "--resume" in call
    assert "scripts/analyze-replan-frozen.py" in calls[-1]


def test_frozen_replan_pilot_declares_cases_and_does_not_select_full_benchmark(launch_workspace):
    result, calls = launch(launch_workspace, "run-replan-frozen-qwen20.sh", "pilot")
    assert result.returncode == 0, result.stderr
    evaluations = [call for call in calls if "scripts/test-reflection-replan.py" in call]
    assert all("--all-questions" not in call for call in evaluations)
    assert "locomo:conv-26:qa11" in evaluations[0]
    assert "locomo:conv-50:qa40" in evaluations[9] and "locomo:conv-50:qa56" in evaluations[9]


def test_gap_replan_launcher_limits_evaluation_to_first_conversation_and_twenty(launch_workspace):
    result, calls = launch(launch_workspace, "run-replan-gap-qwen20.sh")
    assert result.returncode == 0, result.stderr
    evaluations = [call for call in calls if "scripts/test-reflection-replan.py" in call]
    assert len(evaluations) == 1
    call = evaluations[0]
    assert call[call.index("--conversation") + 1] == "0"
    assert call[call.index("--expected-fact-budget") + 1] == "20"
    assert "--all-questions" in call and "--resume" in call


def test_gap_replan_lock_rejects_a_second_writer_before_inference(launch_workspace):
    result, calls = launch(launch_workspace, "run-replan-gap-qwen20.sh", FLOCK_EXIT="1")
    assert result.returncode == 2 and "already running" in result.stderr
    assert not calls


def test_locomo_two_budgets_preserve_the_standard_and_resume(launch_workspace):
    root, _, _ = launch_workspace
    # Simulate a partial second variant: only it should receive --resume.
    output = root / "results with spaces"
    (output / "facts40").mkdir(parents=True)
    (output / "facts40" / "pilot.json").write_text("{}")
    result, calls = launch(launch_workspace, "run-standard-qwen-variants.sh", "locomo", output.as_posix())
    assert result.returncode == 0, result.stderr
    assert len(calls) == 4
    for i, budget in enumerate((20, 40)):
        check, invocation = calls[i * 2:i * 2 + 2]
        assert check[0] == invocation[0] == "CUDA=7"
        assert check[check.index("--output") + 1].endswith(f"facts{budget}/cache/server.json")
        args = pilot.parser().parse_args(invocation[3:])
        plan = pilot.make_plan(args, output / f"facts{budget}")
        assert args.gpu == "7" and args.embed_device == "cuda"
        assert args.backend == "vllm" and args.existing_server
        assert args.model == "Qwen/Qwen2.5-14B-Instruct"
        assert args.locomo_conversation == "all" and args.fact_budget == budget
        assert args.fact_rerank == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert args.local_plans and args.local_plan_version == "v2" and args.reader_reflection
        assert args.qa_max_tokens == 128 and args.top_k == 5
        assert args.resume == (budget == 40)
        assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "7"


def test_memoryagentbench_budgets_keep_official_protocol(launch_workspace):
    result, calls = launch(launch_workspace, "run-standard-qwen-variants.sh", "memoryagentbench",
                           "results", MODEL_REVISION="fixed-revision")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 6
    for i, budget in enumerate((20, 40)):
        check, prepare, invocation = calls[i * 3:i * 3 + 3]
        assert check[0] == prepare[0] == invocation[0] == "CUDA=7"
        args = mab_parser().parse_args(invocation[3:])
        assert args.backend == "vllm" and args.base_url == "http://127.0.0.1:8095/v1"
        assert args.model_revision == "fixed-revision"
        assert args.fact_budget == budget and args.embed_device == "cuda"
        assert args.protocol == "paper" and args.suite == "all"
        assert args.resume and not args.no_reflection
        assert args.max_questions == args.max_contexts == 0 and args.seed is None
        assert args.conflict_recency_weight == .65


def test_server_reserves_memory_and_pins_weights_tokenizer(launch_workspace):
    result, calls = launch(launch_workspace, "serve-qwen-vllm.sh", MODEL_REVISION="fixed-revision")
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1 and calls[0][0] == "CUDA=7"
    flags = calls[0]
    assert flags[flags.index("--gpu-memory-utilization") + 1] == "0.80"
    assert flags[flags.index("--max-model-len") + 1] == "32768"
    assert flags[flags.index("--dtype") + 1] == "bfloat16"
    assert flags[flags.index("--generation-config") + 1] == "vllm"
    assert flags[flags.index("--tensor-parallel-size") + 1] == "1"
    assert flags[flags.index("--revision") + 1] == flags[flags.index("--tokenizer-revision") + 1] == "fixed-revision"


def test_invalid_budget_does_not_start_any_job(launch_workspace):
    result, calls = launch(launch_workspace, "run-standard-qwen.sh", "locomo", FACT_BUDGET="0")
    assert result.returncode == 2 and not calls


def test_replan_script_uses_separate_outputs_and_real_pilot_option(launch_workspace):
    result, calls = launch(launch_workspace, "run-standard-qwen-variants.sh", "locomo", REFLECTION_REPLAN="1")
    assert result.returncode == 0, result.stderr
    for i, budget in enumerate((20, 40)):
        args = pilot.parser().parse_args(calls[i*2+1][3:])
        assert args.reflection_replan and args.fact_budget == budget
        assert args.output.as_posix() == f"runs/replan2-locomo-qwen14b/facts{budget}"
        assert pilot._run_config(pilot.make_plan(args, args.output)["settings"], 152).qa.reflection_replan


def test_unvalidated_benchmark_does_not_silently_enable_replanning(launch_workspace):
    result, calls = launch(launch_workspace, "run-standard-qwen.sh", "memoryagentbench", REFLECTION_REPLAN="1")
    assert result.returncode == 2 and not calls
    assert "LoCoMo only" in result.stderr


def test_failed_preflight_stops_variants(launch_workspace):
    root, _, _ = launch_workspace
    fake = root / "fake-python"
    fake.write_text("#!/usr/bin/env bash\nexit 3\n", encoding="utf-8", newline="\n")
    result, calls = launch(launch_workspace, "run-standard-qwen-variants.sh", "memoryagentbench")
    assert result.returncode == 3 and not calls
    assert "with 40 facts" not in result.stdout


def test_vllm_endpoint_is_local_by_default_and_explicit_when_given(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = mab_parser().parse_args(["run", "--backend", "vllm", "--output", "unused"])
    configure_endpoint(args)
    assert os.environ["OPENAI_BASE_URL"] == "http://127.0.0.1:8095/v1"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://old:1111/v1")
    args.base_url, args.model_revision = "http://localhost:8096/v1/", "fixed"
    configure_endpoint(args)
    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:8096/v1"
    assert os.environ["WRAG_MODEL_REVISION"] == "fixed"


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "file:///tmp/model", "not-a-url"])
def test_vllm_does_not_silently_call_public_api_or_invalid_endpoint(monkeypatch, url):
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    args = mab_parser().parse_args(["run", "--backend", "vllm", "--output", "unused"])
    with pytest.raises(ValueError, match="non-public"):
        configure_endpoint(args)


@pytest.fixture
def preflight():
    spec = importlib.util.spec_from_file_location("qwen_preflight", ROOT / "scripts/check-qwen-server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_checks_model_and_records_server_version(preflight, monkeypatch):
    def response(request, timeout):
        payload = ({"data": [{"id": "qwen", "max_model_len": 16384}]} if request.full_url.endswith("/models")
                   else {"version": "test-vllm"})
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(preflight.urllib.request, "urlopen", response)
    info = preflight.server_info("http://localhost:8095/v1", "qwen")
    assert info["server_version"] == {"version": "test-vllm"}
    with pytest.raises(ValueError, match="not served"):
        preflight.server_info("http://localhost:8095/v1", "wrong-model")


def test_preflight_requires_cuda_in_benchmark_environment(preflight, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        preflight.cuda_info()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="A100 80GB", total_memory=80 * 1024**3))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert preflight.cuda_info()["visible_devices"] == "7"


def test_shell_syntax(launch_workspace):
    root, _, env = launch_workspace
    for script in (root / "scripts").glob("*.sh"):
        result = subprocess.run([BASH, "-n", script.as_posix()], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_memoryagentbench_queue_only_remaining_paper_tasks_on_gpu_zero(launch_workspace):
    result, calls = launch(launch_workspace, "run-memoryagentbench-after-sf-qwen.sh")
    assert result.returncode == 0, result.stderr
    runs = [call for call in calls if "benchmarks.memoryagentbench" in call and "run" in call]
    assert len(runs) == 1
    call = runs[0]
    assert "CUDA=0" in call
    splits = call[call.index("--splits") + 1:call.index("--cache")]
    assert splits == ["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding"]
    assert call[call.index("--suite") + 1] == "paper"
    assert call[call.index("--fact-budget") + 1] == "40"
    assert "--resume" in call
    assert all("judge" not in call for call in calls)


def test_memoryagentbench_queue_lock_prevents_duplicate_writer(launch_workspace):
    result, calls = launch(launch_workspace, "run-memoryagentbench-after-sf-qwen.sh", FLOCK_EXIT="1")
    assert result.returncode == 2
    assert calls == []
