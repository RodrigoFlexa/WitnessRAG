"""python -m benchmarks.memoryagentbench {prepare,inspect,run,judge,report}."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import protocol as P

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "runs" / ".cache" / "memoryagentbench"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pinned MemoryAgentBench evaluation of standard WitnessRAG")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "run", "judge", "report"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        if name in ("run", "judge", "report"):
            cmd.add_argument("--output", type=Path, required=True)
        if name in ("inspect", "run"):
            cmd.add_argument("--suite", choices=("all", "paper"), default="all")
            cmd.add_argument("--splits", nargs="+", choices=P.SPLITS, default=list(P.SPLITS))
            cmd.add_argument("--sources", nargs="+", default=[])
            cmd.add_argument("--protocol", choices=("official", "paper"), default="paper")
        if name == "inspect":
            cmd.add_argument("--chunks", action="store_true", help="Validate complete chunking without LLM/embeddings")
        if name in ("run", "judge"):
            backends = ("openai", "azure", "stub", "vllm") if name == "run" else ("openai", "azure", "stub")
            cmd.add_argument("--backend", choices=backends, default="openai")
        if name == "run":
            cmd.add_argument("--model", default="gpt-4o-mini")
            cmd.add_argument("--base-url", help="OpenAI-compatible endpoint; vLLM defaults to localhost:8095/v1")
            cmd.add_argument("--model-revision", default="", help="HF revision used by the already running server")
            cmd.add_argument("--embed-device", choices=("auto", "cpu", "cuda"), default="auto")
            cmd.add_argument("--embed-model", default="BAAI/bge-m3")
            cmd.add_argument("--embed-backend", choices=("st", "azure", "tfidf"), default="st",
                             help="azure uses --embed-model as the embedding deployment ID")
            cmd.add_argument("--fact-budget", type=int, default=40)
            cmd.add_argument("--fact-rerank", default="cross-encoder/ms-marco-MiniLM-L6-v2")
            cmd.add_argument("--conflict-recency-weight", type=float, default=.65)
            cmd.add_argument("--adaptation", choices=("standard", "ar-source-v2"), default="standard")
            cmd.add_argument("--source-excerpt-chars", type=int, default=2400)
            cmd.add_argument("--no-reflection", action="store_true")
            cmd.add_argument("--resume", action="store_true")
            cmd.add_argument("--max-questions", type=int, default=0)
            cmd.add_argument("--max-contexts", type=int, default=0)
            cmd.add_argument("--seed", type=int, default=None)
        if name == "judge":
            cmd.add_argument("--longmem-model", default="gpt-4o")
            cmd.add_argument("--summary-model", default="gpt-4o-2024-05-13")
    return p


def configure_runtime(args):
    # This is a separate OS process. No env/global state of another run changes.
    for name in ("WRAG_REFLECTION_STUDY_ROOT", "WRAG_CONTROLLED_ROOT", "WRAG_FROZEN_MEMORY_SOURCE"):
        os.environ.pop(name, None)
    os.environ["WRAG_CACHE_DIR"] = str((args.cache / "runtime").resolve())
    os.environ["WRAG_LLM_CACHE"] = "1"
    os.environ["WRAG_EMBED_CACHE"] = "1"
    from wrag import config as C
    # .env may contain experimental paths: isolate after dotenv too.
    for name in ("WRAG_REFLECTION_STUDY_ROOT", "WRAG_CONTROLLED_ROOT", "WRAG_FROZEN_MEMORY_SOURCE"):
        os.environ.pop(name, None)
    C.CACHE_DIR = Path(os.environ["WRAG_CACHE_DIR"])
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        configure_endpoint(args)
    return C


def configure_endpoint(args):
    """An explicit vLLM selection must never silently use the public API."""
    from urllib.parse import urlparse
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url.rstrip("/")
    if args.backend == "vllm":
        endpoint = os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8095/v1"
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.hostname == "api.openai.com":
            raise ValueError("--backend vllm requires a non-public OpenAI-compatible endpoint")
        os.environ["OPENAI_BASE_URL"] = endpoint.rstrip("/")
        os.environ.setdefault("OPENAI_API_KEY", "local-pilot")
    if args.model_revision:
        os.environ["WRAG_MODEL_REVISION"] = args.model_revision


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    args.cache = args.cache.resolve()
    if hasattr(args, "output"):
        args.output = args.output.resolve()
    from . import data
    if args.command == "prepare":
        result = data.prepare(args.cache)
        print(f"Prepared {result['contexts']} contexts / {result['questions']} questions, "
              f"revision={P.DATASET_REVISION}")
        return
    if args.command == "report":
        from .report import report
        result = report(args.output)
        print(json.dumps({k: result[k] for k in ("generation_complete", "evaluation_complete", "paper_overall")}, indent=2))
        return
    data.verify_data(args.cache)
    samples = list(data.iter_samples(args.cache, tuple(args.splits), tuple(args.sources), args.suite)
                   if args.command in ("inspect", "run") else data.iter_samples(args.cache))
    if not samples:
        raise ValueError("Empty selection: check source names and suite")
    if args.command in ("inspect", "run"):
        missing = set(args.sources) - {s.source for s in samples}
        if missing:
            raise ValueError(f"Requested sources not present in selected splits/suite: {sorted(missing)}")
    if args.command == "run" and args.adaptation == "ar-source-v2":
        from .source_memory import order_samples
        samples = order_samples(samples)
    if args.command == "inspect":
        groups = {}
        for sample in samples:
            row = groups.setdefault(sample.source, {"contexts": 0, "questions": 0,
                    "chunk_size": P.task_settings(sample.source, args.protocol)["chunk_size"],
                    "max_output_tokens": P.task_settings(sample.source, args.protocol)["generation_max_length"],
                    "metric": P.primary_metric(sample.source)})
            row["contexts"] += 1
            row["questions"] += len(sample.questions)
            # Validate every official query, including long demo-bearing queries.
            for i in range(len(sample.questions)):
                sample.query(i)
            if args.chunks:
                chunks = data.chunks_for(sample, args.cache, P.task_settings(sample.source, args.protocol))
                row["chunks"] = row.get("chunks", 0) + len(chunks)
                if not chunks:
                    raise ValueError(f"Empty chunk list: {sample.key}")
                print(f"Validated {sample.source} row={sample.row_index} chunks={len(chunks)}", flush=True)
        print(json.dumps({"contexts": len(samples), "questions": sum(len(s.questions) for s in samples),
                          "sources": groups}, indent=2))
        return
    C = configure_runtime(args)
    from wrag.llm import get_llm
    from .runner import judge_run, run
    if args.command == "judge":
        clients = {"longmemeval": get_llm(args.backend, deployment=args.longmem_model),
                   "summary": get_llm(args.backend, deployment=args.summary_model)}
        result = judge_run(samples, args.output, clients, code_hash=P.code_hash(ROOT))
    else:
        if args.fact_budget < 1 or args.max_questions < 0 or args.max_contexts < 0:
            raise ValueError("Budgets must be positive; pilot limits must be nonnegative")
        if args.backend == "stub" or args.embed_backend == "tfidf":
            print("OFFLINE DEBUG RUN: this backend is not a publishable benchmark result.", flush=True)
        device = args.embed_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; choose --embed-device cpu or auto")
        os.environ["WRAG_EMBED_DEVICE"] = device
        C.EMBED_DEVICE, C.EMBED_MODEL = device, args.embed_model
        C.EMBED_BACKEND = args.embed_backend
        if args.embed_backend == "azure":
            os.environ["WRAG_AZURE_EMBED_DEPLOYMENT"] = args.embed_model
            C.AZURE_EMBED_DEPLOYMENT = args.embed_model
        C.EMBED_STRICT_DEVICE = device == "cuda"
        C.EMBED_MAX_SEQ_LENGTH = 0
        from wrag.embed import get_embedder
        from .engine import WitnessEngine, standard_config
        import logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        llm = get_llm(args.backend, deployment=args.model)
        embedder = get_embedder(args.embed_backend, force_new=True)
        actual_device = ("azure" if args.embed_backend == "azure" else
                         str(getattr(getattr(embedder, "_model", None), "device", device)))
        if args.embed_backend == "st" and device == "cuda" and not actual_device.startswith("cuda"):
            raise RuntimeError("Embedder fell back to CPU although CUDA was requested")
        cfg = standard_config(fact_budget=args.fact_budget, rerank=args.fact_rerank)
        if args.source_excerpt_chars < 1:
            raise ValueError("Source excerpt budget must be positive")
        cfg.witness.excerpt_max_chars = args.source_excerpt_chars
        cfg.qa.reader_reflection = not args.no_reflection
        print(f"Standard local-v2, {args.fact_budget} facts, embedder={embedder.name}/{args.embed_model} "
              f"device={actual_device}, protocol={args.protocol}", flush=True)
        print(f"Source adaptation: {args.adaptation}; source excerpt budget={args.source_excerpt_chars} chars", flush=True)
        result = run(samples, WitnessEngine(llm, embedder, cfg, args.conflict_recency_weight,
                                           adaptation=args.adaptation), llm, args.cache, args.output,
                     protocol=args.protocol, reflection=not args.no_reflection, resume=args.resume,
                     max_questions=args.max_questions, max_contexts=args.max_contexts, seed=args.seed,
                     identity_extra={"code_hash": P.code_hash(ROOT), "backend": args.backend,
                                     "model_revision": os.environ.get("WRAG_MODEL_REVISION", ""),
                                     "embedder": {"backend": embedder.name, "model": args.embed_model,
                                                  "device": actual_device},
                                     "debug_only": args.backend == "stub" or args.embed_backend == "tfidf",
                                     "dependency_versions": dependency_versions()})
    print(f"Saved {args.output / 'report.md'}; evaluation_complete={result['evaluation_complete']}")


def dependency_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    import platform
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "transformers", "sentence-transformers", "torch", "openai"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"MemoryAgentBench: {error}", file=sys.stderr)
        raise SystemExit(1)
