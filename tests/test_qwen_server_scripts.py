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
                 "run-witness-proof-locomo.sh", "proof-profiles.sh", "serve-qwen-vllm.sh", "setup-qwen-server.sh"):
        (scripts / name).write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"),
                                   encoding="utf-8", newline="\n")
    fake = tmp_path / "fake-python"
    fake.write_text('#!/usr/bin/env bash\nprintf "CALL\\nCUDA=%s\\n" "$CUDA_VISIBLE_DEVICES" >> "$CAPTURE"\n'
                    'printf "%s\\n" "$@" >> "$CAPTURE"\n', encoding="utf-8", newline="\n")
    fake.chmod(0o755)
    # Git Bash prepends its own binaries to PATH; export a function instead.
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("curl() { return 0; }\nexport -f curl\n", encoding="utf-8", newline="\n")
    capture = tmp_path / "calls.txt"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("WRAG_", "OPENAI_"))}
    env.update({"BENCH_PYTHON": fake.as_posix(), "VLLM_PYTHON": fake.as_posix(),
                "CAPTURE": capture.as_posix(), "PATH": str(tmp_path) + os.pathsep + env["PATH"],
                "GPU": "7", "PORT": "8095", "MODEL": "Qwen/Qwen2.5-14B-Instruct",
                "BASH_ENV": bash_env.as_posix()})
    for key in ("FACT_BUDGET", "EXTRA_FLAGS", "LOCOMO_CONVERSATION", "EMBED_GPU", "CACHE_DIR",
                "MODEL_REVISION", "VLLM_API_KEY", "PROTOCOL", "SUITE"):
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
