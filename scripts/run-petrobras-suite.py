"""Sequential Azure suite: LoCoMo -> selective forgetting -> adapted AR.

Run through run-petrobras-suite.sh in WSL2. No dataset or credential is bundled.
Repeating the command resumes checkpoints; completed stages are revalidated.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SF = {"factconsolidation_sh_262k": 100, "factconsolidation_mh_262k": 100}
AR = {"ruler_qa1_197K": 100, "ruler_qa2_421K": 100,
      "longmemeval_s*": 300, "eventqa_full": 500}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    models = p.add_mutually_exclusive_group()
    models.add_argument("--model", default="gpt-4-1-mini-petrobras",
                   help="Exact Azure deployment name, not necessarily the model's public name")
    models.add_argument("--models", nargs="+", help="Run full suites in this deployment order")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--output", type=Path, default=ROOT / "runs/petrobras-suite")
    p.add_argument("--cache", type=Path, default=ROOT / "runs/.cache/petrobras-suite")
    p.add_argument("--embed-backend", choices=("azure", "st"), default="azure")
    p.add_argument("--embed-model", default="embedding-3-small-global",
                   help="Azure deployment ID; for --embed-backend st, Hugging Face model ID")
    p.add_argument("--embed-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--embed-gpu", default="0")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--fact-budget", type=int, default=40)
    p.add_argument("--locomo-hours", type=float, default=168)
    p.add_argument("--dry-run", action="store_true", help="Print commands; no API calls/downloads/writes")
    return p


def load_environment(path: Path, existing: dict) -> dict:
    """Read dotenv values as data, without executing shell code or echoing keys."""
    env = dict(existing)
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def runtime_environment(args, existing: dict) -> dict:
    env = load_environment(args.env_file, existing)
    # Retain gateway retries/timeouts, but isolate every experimental switch.
    tuning = {k: v for k, v in env.items() if k.startswith("WRAG_AZURE_")}
    env = {k: v for k, v in env.items() if not k.startswith("WRAG_")}
    env.update(tuning)
    env.update(WRAG_LLM_BACKEND="azure", WRAG_AZURE_DEPLOYMENT=args.model,
               WRAG_AZURE_CONCURRENCY=str(args.concurrency), WRAG_AZURE_MAX_TOKENS="2048",
               WRAG_EMBED_BACKEND=args.embed_backend, WRAG_EMBED_MODEL=args.embed_model,
               WRAG_AZURE_EMBED_DEPLOYMENT=args.embed_model, WRAG_EMBED_BATCH_SIZE="32",
               WRAG_EMBED_DEVICE=args.embed_device, WRAG_EMBED_MAX_SEQ_LENGTH="0",
               WRAG_EMBED_STRICT_DEVICE="1", WRAG_LLM_CACHE="1", WRAG_EMBED_CACHE="1",
               WRAG_CACHE_DIR=str(args.cache / "locomo"), WRAG_CONTINUE_ON_CONTENT_FILTER="1",
               WRAG_REFLECTION_STUDY_ROOT="", WRAG_CONTROLLED_ROOT="",
               WRAG_FROZEN_MEMORY_SOURCE="", WRAG_MODEL_REVISION="",
               PYTHONUNBUFFERED="1", PYTHONHASHSEED="42", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=args.embed_gpu)
    env.setdefault("AZURE_OPENAI_API_VERSION", "2024-10-21")
    return env


def identity(args, env):
    # Includes code and endpoint identity, never the API key or raw .env.
    from benchmarks.memoryagentbench.protocol import code_hash
    endpoint = [env.get(k, "") for k in ("AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_ENDPOINT",
                                          "AZURE_OPENAI_API_VERSION")]
    return {"schema": 1, "code_hash": code_hash(ROOT),
            "launcher_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider": "azure", "model": args.model,
            "endpoint_hash": hashlib.sha256(json.dumps(endpoint).encode()).hexdigest(),
            "fact_budget": args.fact_budget, "embed_device": args.embed_device,
            "embed_backend": args.embed_backend, "embed_model": args.embed_model,
            "embed_gpu": args.embed_gpu,
            "embed_api_version": env.get("WRAG_AZURE_EMBED_API_VERSION", env.get("AZURE_OPENAI_API_VERSION")),
            "embedding_pooling": "token-weighted, 8191-token pieces; no text truncation",
            "concurrency": args.concurrency, "cache": str(args.cache),
            "order": ["locomo", "sf", "ar"], "ar_adapter": "ar-source-v2",
            "sources": {"sf": SF, "ar": AR}, "judges": "not scheduled"}


def stage_commands(args):
    py = [sys.executable, "-u"]
    locomo = py + ["-m", "wrag.pilot", "--backend", "azure", "--model", args.model,
        "--gpu", args.embed_gpu, "--concurrency", str(args.concurrency),
        "--dataset", "locomo", "--locomo-conversation", "all", "--methods", "witnessrag",
        "--tokenizer-model", "Qwen/Qwen2.5-14B-Instruct", "--embed-model", args.embed_model,
        "--embed-backend", args.embed_backend,
        "--embed-device", args.embed_device, "--locomo-chunk-tokens", "2048",
        "--locomo-ie-window-tokens", "512", "--top-k", "5", "--qa-max-tokens", "128",
        "--witness-candidate-pool", "20", "--answer-set", "--temporal-annotations",
        "--evidence-reader", "--binding-aware-grounding", "--vocab-compile", "--hybrid-fallback",
        "--dialogue-ie", "--gap-context-rescue", "--proof-controller", "--ie-style", "memory",
        "--fact-delivery", "facts", "--fact-fill", "question", "--fact-time", "both",
        "--fact-budget", str(args.fact_budget), "--fact-rerank", "cross-encoder/ms-marco-MiniLM-L6-v2",
        "--local-plans", "--reader-reflection", "--no-proof-verify", "--local-plan-version", "v2",
        "--local-plan-beam", "32", "--local-plan-candidates", "96",
        "--local-plan-executions", "4000", "--local-plan-starts", "12",
        "--cache-dir", str(args.cache / "locomo"), "--hours", str(args.locomo_hours),
        "--output", str(args.output / "locomo")]
    if (args.output / "locomo/pilot.json").exists():
        locomo.append("--resume")
    common = py + ["-m", "benchmarks.memoryagentbench", "run", "--backend", "azure",
        "--model", args.model, "--embed-device", args.embed_device, "--embed-model", args.embed_model,
        "--embed-backend", args.embed_backend,
        "--fact-budget", str(args.fact_budget), "--fact-rerank", "cross-encoder/ms-marco-MiniLM-L6-v2",
        "--protocol", "paper", "--suite", "paper", "--cache", str(args.cache / "memoryagentbench"),
        "--resume"]
    return {
        "locomo": locomo,
        "sf": common + ["--output", str(args.output / "sf"), "--adaptation", "standard",
            "--conflict-recency-weight", "0.65", "--splits", "Conflict_Resolution",
            "--sources", *SF],
        "ar": common + ["--output", str(args.output / "ar"), "--adaptation", "ar-source-v2",
            "--source-excerpt-chars", "2400", "--splits", "Accurate_Retrieval", "--sources", *AR],
    }


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        raise ValueError("Invalid checkpoint PID")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def suite_lock(path):
    # An OS lock survives no crashed owner; the file alone does not mean locked.
    if os.name != "posix":
        raise RuntimeError("Use this launcher inside WSL2/Linux")
    import fcntl
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This suite already has an active launcher") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_complete(stage, output):
    if stage == "locomo":
        from wrag.pilot import _completed_locomo_run
        status = json.loads((output / "status.json").read_text())
        items = json.loads((output / "conversations.json").read_text())["conversas"]
        aggregate = json.loads((output / "locomo_agregado.json").read_text())
        if (status.get("status") != "complete" or len(items) != 10
                or {x["conversa"] for x in items} != set(range(10))
                or sum(x["perguntas"] for x in items) != 1540
                or aggregate.get("conversas") != 10
                or aggregate["metodos"]["witnessrag"]["n"] != 1540
                or not all(_completed_locomo_run(x, ["witnessrag"]) for x in items)):
            raise ValueError("LoCoMo is incomplete; repeat the same command to resume")
        return
    expected = SF if stage == "sf" else AR
    manifest = json.loads((output / "manifest.json").read_text())["identity"]
    selection = manifest["selection"]
    intended = {f"{s['key']}:q{i}" for s in selection for i in range(s["questions"])}
    counts = Counter()
    for item in selection:
        counts[item["source"]] += item["questions"]
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines() if line.strip()]
    ids = [row["qid"] for row in rows]
    if (dict(counts) != expected or len(ids) != len(set(ids)) or set(ids) != intended
            or dict(Counter(row["source"] for row in rows)) != expected):
        raise ValueError(f"{stage}: missing, duplicate or foreign predictions")
    for row in rows:
        if row["source"] != "longmemeval_s*":
            if row["primary_metric"] != "substring_exact_match" or row["metrics"].get("substring_exact_match") not in (0, 1):
                raise ValueError(f"{stage}: official accuracy metric missing or invalid")


def clear_abandoned_mab_lock(output):
    lock = output / "run.lock"
    if lock.exists():
        pid = json.loads(lock.read_text())["pid"]
        if pid_alive(pid):
            raise RuntimeError(f"A benchmark worker is still alive (PID {pid}); no duplicate will be started")
        lock.unlink()
        print(f"Removed abandoned benchmark lock after checking PID {pid}.", flush=True)


PREFLIGHT = """
import importlib, os
for name in ('openai', 'httpx', 'numpy', 'scipy', 'pandas', 'pyarrow', 'nltk',
             'huggingface_hub', 'tiktoken', 'sentence_transformers', 'transformers', 'pulp'):
    importlib.import_module(name)
if os.environ['WRAG_EMBED_DEVICE'] == 'cuda':
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in WSL2; choose --embed-device cpu')
from wrag.llm import get_llm, GenParams
llm = get_llm('azure', deployment=os.environ['WRAG_AZURE_DEPLOYMENT'])
r = llm.chat('Reply with OK.', params=GenParams(max_tokens=16, exact_max_tokens=True), stage='suite.preflight')
if not r.ok or r.error:
    raise RuntimeError('Azure preflight produced no usable response; check deployment and gateway configuration')
print('Azure deployment verified. No local generative model server is needed.', flush=True)
if os.environ['WRAG_EMBED_BACKEND'] == 'azure':
    from wrag.embed import get_embedder
    embedder = get_embedder('azure', force_new=True)
    vectors = embedder.encode(['Embedding connection check.'])
    if vectors.shape[0] != 1 or vectors.shape[1] < 1:
        raise RuntimeError('Embedding preflight returned invalid dimensions')
    print('Azure embedding deployment verified, dimensions:', vectors.shape[1], flush=True)
"""


def run_command(command, env, state, state_path, log_path):
    print(f"Log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        state.update(active_pid=process.pid, status="running", updated=time.time())
        write_json(state_path, state)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            # The LoCoMo pilot catches SIGINT and stops its owned worker too.
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass  # Keep PID recorded; resume refuses to duplicate a live writer.
            raise
    state["active_pid"] = None
    write_json(state_path, state)
    if code:
        raise RuntimeError(f"Stage failed (exit {code}). Read {log_path}; repeat the same command to resume")


def execute(args, env, commands, suite_identity):
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "suite.json"
    with suite_lock(args.output / "suite.lock"):
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "identity": suite_identity, "completed": [], "started": time.time()}
        if state["identity"] != suite_identity:
            raise ValueError("Suite configuration/code/model changed. Use a new output folder")
        if state.get("active_pid") and pid_alive(state["active_pid"]):
            raise RuntimeError(f"Prior worker PID {state['active_pid']} is still active")
        write_json(state_path, state)
        try:
            # Validate existing data once, download it only if absent/invalid.
            prepared = False
            for stage, command in commands.items():
                state["current_stage"] = stage
                if stage in state["completed"]:
                    verify_complete(stage, args.output / stage)
                    print(f"{stage}: already complete and validated.", flush=True)
                    continue
                if not prepared:
                    if not env.get("AZURE_OPENAI_API_KEY") or not (env.get("AZURE_OPENAI_BASE_URL") or env.get("AZURE_OPENAI_ENDPOINT")):
                        raise ValueError("Configure AZURE_OPENAI_API_KEY and BASE_URL/ENDPOINT in .env")
                    run_command([sys.executable, "-u", "-c", PREFLIGHT], env, state, state_path,
                                args.output / "preflight.log")
                    prepared = True
                if stage != "locomo":
                    clear_abandoned_mab_lock(args.output / stage)
                    from benchmarks.memoryagentbench.data import verify_data
                    try:
                        verify_data(args.cache / "memoryagentbench")
                    except (FileNotFoundError, ValueError):
                        run_command([sys.executable, "-u", "-m", "benchmarks.memoryagentbench", "prepare",
                                     "--cache", str(args.cache / "memoryagentbench")], env, state, state_path,
                                    args.output / "prepare-memoryagentbench.log")
                print(f"Starting {stage}: Azure deployment={args.model}, {args.fact_budget} facts.", flush=True)
                run_command(command, env, state, state_path, args.output / f"{stage}.log")
                verify_complete(stage, args.output / stage)
                if stage != "locomo":
                    run_command([sys.executable, "-u", "-m", "benchmarks.memoryagentbench", "report",
                                 "--output", str(args.output / stage)], env, state, state_path,
                                args.output / f"{stage}.log")
                state["completed"].append(stage)
                write_json(state_path, state)
            state.update(status="generation_complete", current_stage=None, updated=time.time())
            write_json(state_path, state)
            print("Suite generation complete. LME official judgment remains pending; no judge was invoked.", flush=True)
        except (Exception, KeyboardInterrupt):
            state.update(status="interrupted_or_failed", updated=time.time())
            write_json(state_path, state)
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    if args.fact_budget < 1 or args.concurrency < 1 or args.locomo_hours <= 2 / 60:
        raise ValueError("Positive budgets/concurrency and LoCoMo deadline > 2 minutes required")
    if not args.embed_gpu.strip() or "," in args.embed_gpu:
        raise ValueError("Choose one embedding GPU index/UUID")
    args.output, args.cache, args.env_file = args.output.resolve(), args.cache.resolve(), args.env_file.resolve()
    if args.output == args.cache or args.output in args.cache.parents or args.cache in args.output.parents:
        raise ValueError("Keep cache and output in separate directories")
    models = args.models or [args.model]
    if len(models) != len(set(models)) or any(not m.strip() or "/" in m or "\\" in m or m in (".", "..") for m in models):
        raise ValueError("Provide unique Azure deployment IDs, without path separators")
    root_output = args.output
    plans = []
    for model in models:
        args.model = model
        args.output = root_output / model
        env = runtime_environment(args, os.environ)
        plans.append((argparse.Namespace(**vars(args)), env, stage_commands(args), identity(args, env)))
    if args.dry_run:
        print(json.dumps({"models": [{"identity": i, "commands": c} for a, e, c, i in plans],
                          "note": "Dry run: no downloads, API calls or checkpoint writes"}, indent=2))
        return 0
    root_output.mkdir(parents=True, exist_ok=True)
    with suite_lock(root_output / "queue.lock"):
        for model_args, env, commands, suite_identity in plans:
            execute(model_args, env, commands, suite_identity)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Petrobras suite: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted. Repeat the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
