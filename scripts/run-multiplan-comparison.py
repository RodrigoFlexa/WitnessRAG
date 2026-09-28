"""One command, two matched LoCoMo runs, a strict paired report. No shell needed.

python scripts/run-multiplan-comparison.py
python scripts/run-multiplan-comparison.py --conversation 5 --questions 8
python scripts/run-multiplan-comparison.py --include-controls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/multiplan-comparison"))
    p.add_argument("--conversation", default="all", help="all ou índice 0..9")
    p.add_argument("--questions", type=int, help="limite por conversa; omita para todas")
    p.add_argument("--backend", choices=["openai", "vllm"], default="openai")
    p.add_argument("--model", default=None, help="padrão: gpt-4o-mini (openai) ou Qwen/Qwen2.5-14B-Instruct (vllm)")
    p.add_argument("--ports", default=None,
                    help="vllm: uma porta por variante, na ordem de build_commands (ex.: 8095,8096); "
                         "reaproveita em round-robin se houver mais variantes que portas")
    p.add_argument("--gpus", default=None,
                    help="vllm: um índice/UUID de GPU por variante, mesma ordem/round-robin de --ports")
    p.add_argument("--parallel", action="store_true",
                    help="lança todas as variantes pendentes ao mesmo tempo (subprocess.Popen) em vez de uma por vez; "
                         "use com --ports para não competir pelo mesmo servidor vLLM")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="auto")
    p.add_argument("--embed-model", default="BAAI/bge-m3")
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-plans", type=int, choices=range(1, 5), default=3)
    p.add_argument("--cycles", type=int, choices=range(1, 5), default=2)
    p.add_argument("--fact-budget", type=int, default=40)
    p.add_argument("--hours", type=float, default=72, help="prazo POR variante")
    p.add_argument("--locomo-file", type=Path)
    p.add_argument("--cache-dir", type=Path, default=None,
                    help="padrão: runs/.cache/witness-openai ou runs/.cache/witness-qwen (conforme --backend)")
    p.add_argument("--include-controls", action="store_true", help="inclui WitnessRAG antigo e robusto sem plano")
    p.add_argument("--dry-run", action="store_true", help="mostra comandos; nenhuma chamada de API")
    p.add_argument("--report-only", action="store_true", help="recalcula relatório de rodada completa")
    return p


def device_choice(device):
    if device != "auto":
        return "cuda:0" if device == "cuda" else device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_defaults(args):
    """--model/--cache-dir depend on --backend; fill them in once, after parsing."""
    if args.model is None:
        args.model = "Qwen/Qwen2.5-14B-Instruct" if args.backend == "vllm" else "gpt-4o-mini"
    if args.cache_dir is None:
        args.cache_dir = ROOT / ("runs/.cache/witness-qwen" if args.backend == "vllm" else "runs/.cache/witness-openai")
    return args


def backend_flags_for(args, index):
    """--backend openai: one shared client, no port. --backend vllm: round-robin
    over --ports/--gpus so each variant hits its own already-running server
    (scripts/serve-qwen-vllm.sh) instead of competing for one."""
    if args.backend == "openai":
        return ["--backend", "openai", "--model", args.model, "--gpu", "0"]
    ports = [p.strip() for p in (args.ports or "8095").split(",") if p.strip()]
    gpus = [g.strip() for g in (args.gpus or "1").split(",") if g.strip()]
    port = ports[index % len(ports)]
    gpu = gpus[index % len(gpus)]
    return ["--backend", "vllm", "--existing-server", "--port", port, "--model", args.model, "--gpu", gpu]


def build_commands(args, device, variant_index=0):
    args = resolve_defaults(args)
    root = args.output.resolve()

    def base_for(index):
        cmd = [sys.executable, "-m", "wrag.pilot", *backend_flags_for(args, index),
               "--concurrency", str(args.concurrency), "--dataset", "locomo",
               "--locomo-conversation", args.conversation, "--methods", "witnessrag",
               "--embed-model", args.embed_model, "--embed-device", device,
               "--tokenizer-model", "Qwen/Qwen2.5-14B-Instruct", "--locomo-chunk-tokens", "2048",
               "--locomo-ie-window-tokens", "512", "--top-k", "5", "--qa-max-tokens", "128",
               "--witness-candidate-pool", "20", "--answer-set", "--temporal-annotations",
               "--evidence-reader", "--binding-aware-grounding", "--vocab-compile", "--hybrid-fallback",
               "--dialogue-ie", "--gap-context-rescue", "--proof-controller", "--typed-variables",
               "--item-set-proofs", "--witness-delivery", "mixed", "--abductive-premises",
               "--ie-style", "memory", "--fact-delivery", "facts+summary", "--fact-budget", str(args.fact_budget),
               "--proof-cycles", str(args.cycles), "--cache-dir", str(args.cache_dir.resolve()),
               "--hours", str(args.hours), "--seed", "42"]
        if args.questions:
            cmd += ["--questions", str(args.questions)]
        if args.locomo_file:
            cmd += ["--locomo-file", str(args.locomo_file.resolve())]
        return cmd

    robust = ["--relation-alternatives", "--plan-readings", "2", "--fact-fill", "question",
              "--fact-rerank", args.reranker, "--fact-time", "both"]
    variants = {"witnessrag-robust": robust,
                "witnessrag-multiplan": robust + ["--multiplan-portfolio", "--portfolio-max-plans", str(args.max_plans)]}
    if args.include_controls:
        variants["witnessrag-original"] = []
        variants["robust-no-plan"] = robust + ["--ablation", "no-plan"]
    return {name: base_for(i + variant_index) + flags + ["--output", str(root / name)]
            for i, (name, flags) in enumerate(variants.items())}


def source_hashes():
    paths = [*sorted((ROOT / "wrag").rglob("*.py")), Path(__file__).resolve(), ROOT / "scripts/paired-report.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def completed(folder):
    try:
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        roots = result.get("run_dirs", [])
        if state.get("status") != "complete" or not roots:
            return False
        for root in roots:
            manifest = json.loads((Path(root) / "run.json").read_text(encoding="utf-8"))
            if not manifest.get("terminado_em"):
                return False
        progress = folder / "conversations.json"
        if progress.exists():
            from wrag.pilot import _completed_conversation
            conversations = json.loads(progress.read_text(encoding="utf-8"))["conversas"]
            if len(conversations) != len(roots) or not all(
                    _completed_conversation(item, ["witnessrag"]) for item in conversations):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def compare_corpora(root, names):
    """Question pairing alone cannot prove the same retrieval corpus was used."""
    signatures = []
    for name in names:
        folder = root / name
        paths = sorted(folder.glob("**/data/locomo_corpus.json"))
        if not paths:
            raise ValueError(f"Corpus ausente em {folder}")
        signatures.append({str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Corpora diferentes entre variantes; comparação recusada")
    return signatures[0]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = resolve_defaults(parser().parse_args())
    if args.questions is not None and args.questions < 1:
        raise SystemExit("--questions deve ser >= 1")
    if args.conversation != "all" and (not args.conversation.isdigit() or not 0 <= int(args.conversation) <= 9):
        raise SystemExit("--conversation deve ser all ou 0..9")
    if args.fact_budget < 1 or args.concurrency < 1:
        raise SystemExit("orçamento e concorrência devem ser >= 1")
    device = device_choice(args.device)
    commands = build_commands(args, device)
    root = args.output.resolve()
    manifest = {"commands": commands, "sources": source_hashes(), "device": device}
    # Validate every command with the actual parser before spending API tokens.
    sys.path.insert(0, str(ROOT))
    from wrag.pilot import parser as pilot_parser, make_plan
    for name, command in commands.items():
        parsed = pilot_parser().parse_args(command[3:])
        make_plan(parsed, root / name)
    if args.dry_run:
        print("Nenhuma API será chamada. Comparação com a mesma memória e entrega robusta:")
        for name, command in commands.items():
            print(name + ": " + subprocess.list2cmdline(command))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    path = root / "comparison-manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(key) != manifest[key] for key in manifest):
            raise SystemExit("Configuração/código mudou. Use outro --output para não misturar resultados.")
    elif any((root / name).exists() for name in commands):
        raise SystemExit("Saídas existentes sem manifesto da comparação. Use outro --output.")
    else:
        path.write_text(json.dumps(dict(manifest, created=datetime.now().isoformat()), indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="42", WRAG_LLM_CACHE="1",
               WRAG_EMBED_CACHE="1", WRAG_CONTINUE_ON_CONTENT_FILTER="1")
    pending = {}
    for name, command in commands.items():
        folder = root / name
        if completed(folder):
            print(f"{name}: rodada completa preservada.", flush=True)
            continue
        if args.report_only:
            raise SystemExit(f"Rodada incompleta: {folder}; execute sem --report-only para retomar.")
        resume = ["--resume"] if (folder / "pilot.json").exists() else []
        pending[name] = command + resume

    if args.parallel:
        # Uma variante por servidor vLLM (scripts/serve-qwen-vllm.sh): lança todas
        # de uma vez em vez de esperar cada wrag.pilot terminar antes da próxima.
        procs = {}
        for name, command in pending.items():
            folder = root / name
            print(f"Executando {name} ({device}), em paralelo. Log: {folder / 'benchmark.log'}", flush=True)
            procs[name] = subprocess.Popen(command, cwd=ROOT, env=env)
        failed = []
        for name, proc in procs.items():
            if proc.wait() or not completed(root / name):
                failed.append(name)
        if failed:
            raise SystemExit(f"{', '.join(failed)} incompleto(s). Corrija a falha e repita o mesmo comando para retomar.")
    else:
        for name, command in pending.items():
            folder = root / name
            print(f"Executando {name} ({device}). Log: {folder / 'benchmark.log'}", flush=True)
            result = subprocess.run(command, cwd=ROOT, env=env)
            if result.returncode or not completed(folder):
                raise SystemExit(f"{name} incompleto. Corrija a falha e repita o mesmo comando para retomar.")
    corpora = compare_corpora(root, list(commands))
    (root / "corpus-hashes.json").write_text(json.dumps(corpora, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "scripts/paired-report.py"), str(root), *commands],
                   cwd=ROOT, env=env, check=True)
    print(f"Relatório: {root / 'compare.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
