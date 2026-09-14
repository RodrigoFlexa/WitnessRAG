"""Piloto Linux/vLLM com GPU explícita, prazo e finalização de resultados parciais.

Executar: python -m wrag.pilot --gpu 5
Este módulo não importa torch/config antes de fixar o ambiente dos subprocessos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METHODS = "dense,bm25,graphrag,hipporag,hipporag2,relational,witnessrag"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, help="índice físico NVIDIA ou UUID; ex.: 5")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="repo HF ou diretório de pesos HF")
    p.add_argument("--model-revision", default="", help="commit HF para fixar pesos/tokenizer")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "half"])
    p.add_argument("--quantization", default="", help="opcional, conforme checkpoint e suporte vLLM")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--vllm-python", default=sys.executable, help="Python do ambiente que contém vLLM")
    p.add_argument("--existing-server", action="store_true", help="usa servidor já ativo em localhost:port")
    p.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--embed-device", default="cuda:0", help="cuda:0 é a GPU selecionada após remapeamento; ou cpu")
    p.add_argument("--dataset", choices=["2wikimultihopqa", "musique", "hotpotqa", "sample"], default="2wikimultihopqa")
    p.add_argument("-n", "--questions", type=int, default=100)
    p.add_argument("--distractors", type=int, default=300, help="passagens aleatórias adicionais ao corpus candidato")
    p.add_argument("--max-passages", type=int, default=1500, help="falha se o corpus candidato exceder este teto")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--no-acquisition", action="store_true", help="ablação sem aquisição dirigida")
    p.add_argument("--hours", type=float, default=6.5, help="janela total; reserva 2 min para finalização")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="reaproveita caches (OpenIE, embeddings) de outra rodada; a extração é a parte cara")
    p.add_argument("--dry-run", action="store_true", help="mostra comandos sem baixar dados ou iniciar processos")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def make_plan(args, output):
    if not args.gpu.strip() or "," in args.gpu:
        raise ValueError("selecione uma GPU; este piloto usa tensor-parallel-size=1")
    if args.hours <= 2 / 60 or args.questions < 1 or args.max_passages < 1 or args.distractors < 0:
        raise ValueError("prazo deve exceder 2 minutos; n/teto positivos; distratores >= 0")
    if not 0 < args.gpu_memory_utilization < 1 or args.concurrency < 1 or not 1 <= args.port <= 65535:
        raise ValueError("memória, concorrência ou porta inválidas")
    if args.max_model_len < 4096:
        raise ValueError("use contexto >= 4096; o leitor recebe passagens completas")
    if args.embed_device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("embed-device deve ser cpu ou cuda:0 na GPU remapeada")
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    available = {"dense", "bm25", "graphrag", "hipporag", "hipporag2", "relational", "witnessrag",
                 "witnessrag-annotated", "witnessrag-oracle"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - available:
        raise ValueError("lista de métodos inválida ou duplicada")
    env = {
        "CUDA_VISIBLE_DEVICES": args.gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "WRAG_LLM_BACKEND": "vllm", "OPENAI_MODEL": args.model,
        "OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1", "OPENAI_API_KEY": "local-pilot",
        "WRAG_MODEL_REVISION": args.model_revision,
        "WRAG_AZURE_CONCURRENCY": str(args.concurrency), "WRAG_AZURE_MAX_RETRIES": "2",
        "WRAG_AZURE_TIMEOUT_S": "120", "WRAG_EMBED_BACKEND": "st",
        "WRAG_EMBED_MODEL": args.embed_model, "WRAG_EMBED_DEVICE": args.embed_device,
        "WRAG_EMBED_BATCH_SIZE": "32", "WRAG_SEED": str(args.seed),
        "WRAG_DATA_DIR": str(output / "data"), "WRAG_RUNS_DIR": str(output / "benchmark"),
        "WRAG_CACHE_DIR": str(args.cache_dir.resolve() if args.cache_dir else output / "cache"), "WRAG_NO_PROGRESS": "1", "PYTHONUNBUFFERED": "1",
        # O sampler FlashInfer compila kernels via JIT e resolve o nvcc por `which nvcc`,
        # que aqui aponta para CUDA 11.5 e rejeita --compress-mode=size (exige >= 12.8).
        # A decodificação é gulosa (temperature=0), logo o sampler nativo não altera saídas.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    command = [args.vllm_python, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", args.dtype, "--max-model-len", str(args.max_model_len),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-num-seqs", str(args.concurrency), "--tensor-parallel-size", "1",
               "--generation-config", "vllm"]
    if args.model_revision:
        command += ["--revision", args.model_revision, "--tokenizer-revision", args.model_revision]
    if args.quantization:
        command += ["--quantization", args.quantization]
    return {"output": str(output), "env": env, "server_command": command, "methods": methods,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scope": "piloto com corpus candidato reduzido e distratores; adaptações locais dos artigos"}


def prepare_data(plan):
    """Corpus fixo para todos; seleção não usa desempenho dos métodos."""
    from wrag.data import load_dataset
    from wrag.util import read_json, write_json
    settings = plan["settings"]
    data = Path(plan["output"]) / "data"
    source = Path(plan["output"]) / "source-data"
    source.mkdir(parents=True, exist_ok=True)
    name, seed = settings["dataset"], settings["seed"]
    sources = {}
    for filename in (f"{name}.json", f"{name}_corpus.json"):
        path = source / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/{filename}"
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            json.loads(payload)  # não guardar uma página HTML de erro
            path.write_bytes(payload)
        sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    n = settings["questions"]
    reduced = load_dataset(name, n_questions=n, seed=seed, data_dir=source, subset_corpus=True)
    full = load_dataset(name, n_questions=n, seed=seed, data_dir=source, subset_corpus=False)
    keep = {p.pid for p in reduced.passages}
    cap = settings["max_passages"]
    if len(keep) > cap:
        raise ValueError(f"corpus candidato tem {len(keep)} passagens > teto {cap}; reduza -n ou aumente --max-passages")
    extras = [p for p in full.passages if p.pid not in keep]
    random.Random(seed).shuffle(extras)
    chosen = reduced.passages + extras[:min(settings["distractors"], cap - len(keep))]
    ids = {q.qid for q in reduced.questions}
    raw = read_json(source / f"{name}.json")
    selected = [q for q in raw if str(q.get("id") or q.get("_id") or "") in ids]
    write_json(data / f"{name}.json", selected)
    write_json(data / f"{name}_corpus.json", [{"title": p.title, "text": p.text} for p in chosen])
    metadata = {"source_sha256": sources, "source_repo": "https://github.com/OSU-NLP-Group/HippoRAG",
                "seed": seed, "full_passages": len(full.passages), "candidate_passages": len(keep),
                "additional_distractors": len(chosen) - len(keep), "selected_passages": len(chosen),
                "questions": len(selected), "question_ids": sorted(ids),
                "corpus_reduced": len(chosen) < len(full.passages)}
    write_json(Path(plan["output"]) / "data_selection.json", metadata)
    return metadata


def worker(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # Ambiente já foi aplicado pelo processo pai, antes de importar config.
    from wrag import config as C
    from wrag.eval.runner import run
    from wrag.llm import get_llm, GenParams
    from wrag.util import setup_logging, write_json
    setup_logging()
    settings = plan["settings"]
    print("Preparando corpus do piloto...", flush=True)
    metadata = prepare_data(plan)
    print(json.dumps({k: v for k, v in metadata.items() if k != "question_ids"}, ensure_ascii=False), flush=True)
    import importlib.metadata
    versions = {}
    for package in ("openai", "torch", "transformers", "sentence-transformers", "numpy", "scipy", "networkx", "igraph"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    write_json(Path(plan["output"]) / "benchmark_versions.json", versions)
    llm = get_llm()
    probe = llm.chat('Return only JSON: {"ok": true}', params=GenParams(max_tokens=32, json_mode=True),
                     stage="pilot.preflight")
    if not isinstance(probe.json(), dict) or probe.json().get("ok") is not True:
        raise RuntimeError("preflight não retornou o JSON esperado; confira modelo/endpoint")
    cfg = C.RunConfig(n_questions=settings["questions"], seed=settings["seed"], top_k=5,
                      interleave_methods=True, corpus_scope="pilot_candidates_plus_random_distractors")
    cfg.witness.enable_acquisition = not settings["no_acquisition"]
    # Mantém os demais hiperparâmetros consolidados: sem corte oculto de tokens/fatos.
    root = run([settings["dataset"]], plan["methods"], cfg, tag="qwen-pilot")
    write_json(Path(plan["output"]) / "completed.json", {"run_dir": str(root)})


def stop_owned(process):
    """Encerra somente o grupo de processos iniciado por este launcher."""
    if process is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_ready(server, port, model, deadline):
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError("vLLM encerrou durante a inicialização; consulte vllm.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                models = json.load(response)
            if model in {x.get("id") for x in models.get("data", [])}:
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("servidor não ficou pronto dentro do prazo; consulte vllm.log")


def launch(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "runs" / f"local-pilot-{stamp}").resolve()
    plan = make_plan(args, output)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if os.name != "posix" and not args.existing_server:
        raise RuntimeError("execute o launcher no servidor Linux com NVIDIA/vLLM; use --dry-run para inspecionar")
    if output.exists() and any(output.iterdir()):
        raise ValueError("--output deve ser novo ou vazio; resultados existentes não serão sobrescritos")
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "pilot.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {**os.environ, **plan["env"]}
    # 120s reservados para encerrar processos e gerar relatórios/gráficos.
    deadline = time.monotonic() + args.hours * 3600 - 120
    server = job = None
    status = "failed"
    print(f"Saída: {output}\nGPU: {args.gpu}; prazo: {args.hours} h; modelo: {args.model}", flush=True)
    try:
        try:
            hardware = subprocess.run(["nvidia-smi", "-i", args.gpu,
                                       "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
                                      capture_output=True, text=True, timeout=10)
            (output / "gpu.txt").write_text(hardware.stdout + hardware.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError):
            pass
        if not args.existing_server:
            with socket.socket() as check:
                check.bind(("127.0.0.1", args.port))
            with (output / "vllm.log").open("w", encoding="utf-8") as log:
                server = subprocess.Popen(plan["server_command"], env=env, cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("Aguardando vLLM (pesos ausentes podem exigir download)...", flush=True)
        wait_ready(server, args.port, args.model, min(deadline, time.monotonic() + 3600))
        if not args.existing_server:
            try:
                version = subprocess.run([args.vllm_python, "-c",
                                          "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
                                         capture_output=True, text=True, timeout=10)
                (output / "vllm-version.txt").write_text(version.stdout, encoding="utf-8")
            except (OSError, subprocess.SubprocessError):
                pass
        with (output / "benchmark.log").open("w", encoding="utf-8") as log:
            job = subprocess.Popen([sys.executable, "-m", "wrag.pilot", "--gpu", args.gpu,
                                    "--worker", str(plan_path)], env=env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        print(f"Benchmark em execução. Acompanhe: tail -f {output / 'benchmark.log'}", flush=True)
        while job.poll() is None:
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            if server is not None and server.poll() is not None:
                raise RuntimeError("vLLM encerrou durante o benchmark")
            time.sleep(2)
        if job.poll() is not None:
            status = "complete" if job.returncode == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        print(f"Falha: {exc}", file=sys.stderr)
        (output / "error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        stop_owned(job)
        stop_owned(server)
        (output / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        manifests = sorted((output / "benchmark").glob("*/run.json"))
        for manifest in manifests:
            try:
                subprocess.run([sys.executable, "-m", "wrag.eval.plots", str(manifest.parent)],
                               env=env, cwd=ROOT, check=True, timeout=80)
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"Falha na finalização: {exc}. Reexecute python -m wrag.eval.plots {manifest.parent}",
                      file=sys.stderr)
        if not manifests:
            print("Nenhuma rodada chegou à avaliação. Consulte logs; não há resultados de qualidade.")
    print(f"Estado: {status}. Resultados: {output}", flush=True)
    return 0 if status == "complete" else 2


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        worker(args.worker)
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
