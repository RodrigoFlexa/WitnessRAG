"""Factorial ablation: cascade, summary reflection, reader reflection, both.

Defaults to an existing GPU Qwen2.5-14B vLLM server; repeat to resume safely.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("comparison", ROOT / "scripts/run-multiplan-comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

VARIANTS = {"cascade": (False, False), "summary-reflection": (True, False),
            "reader-reflection": (False, True), "both": (True, True)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/cascade-reflection-ablation"))
    p.add_argument("--conversation", default="all", help="all, índice 0..9 ou lista como 0,3,5")
    p.add_argument("--questions", type=int, help="limite por conversa; padrão: todas")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--gpu", default="1", help="GPU física/UUID para embeddings e reranker")
    p.add_argument("--ports", default=None,
                    help="uma porta por variante, round-robin na ordem cascade/summary-reflection/"
                         "reader-reflection/both (ex.: 8095,8096); sobrepõe --port")
    p.add_argument("--gpus", default=None, help="idem, para --gpu; sobrepõe --gpu")
    p.add_argument("--only", default=None,
                    help="roda só estas variantes agora (ex.: cascade,summary-reflection); as outras "
                         "ficam de fora desta execução, mas o manifesto continua cobrindo as 4 — "
                         "rode a outra metade em paralelo com --only complementar e o mesmo --output")
    p.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda")
    p.add_argument("--embed-model", default="BAAI/bge-m3")
    p.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--fact-budget", type=int, default=40)
    p.add_argument("--reflection-limit", type=int, choices=range(1, 7), default=4)
    p.add_argument("--hours", type=float, default=72, help="prazo por variante")
    p.add_argument("--locomo-file", type=Path)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "runs/.cache/witness-qwen")
    p.add_argument("--dry-run", action="store_true", help="valida as quatro configurações sem chamar o servidor")
    p.add_argument("--report-only", action="store_true", help="recalcula relatórios de estudo completo")
    return p


def build_commands(args):
    # Reuse the actual robust configuration, then add the same cascade router
    # to every cell. No separate copy of the scientific baseline's flags.
    # --ports/--gpus (round-robin per variant, fixed VARIANTS order) let two
    # halves of this same manifest run against two different vLLM servers;
    # the manifest always covers all 4 regardless of which ones --only runs.
    ports = str(args.ports if args.ports else args.port)
    gpus = str(args.gpus if args.gpus else args.gpu)
    device = comparison.device_choice(args.device)
    base_kwargs = {k: v for k, v in vars(args).items() if k not in ("port", "gpu", "ports", "gpus", "only")}
    commands = {}
    for index, (name, (summary, reader)) in enumerate(VARIANTS.items()):
        settings = argparse.Namespace(**base_kwargs, backend="vllm", ports=ports, gpus=gpus,
                                      max_plans=3, include_controls=False)
        base = comparison.build_commands(settings, device, variant_index=index)["witnessrag-robust"]
        base = base[:-2] + ["--plan-router", "llm"]  # remove only the old --output
        flags = (["--summary-reflection", "--summary-reflection-limit", str(args.reflection_limit)]
                 if summary else [])
        if reader:
            flags += ["--reader-reflection"]
        commands[name] = base + flags + ["--output", str((args.output / name).resolve())]
    return commands


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    commands = build_commands(args)
    from wrag.pilot import make_plan, parser as pilot_parser
    for name, command in commands.items():
        make_plan(pilot_parser().parse_args(command[3:]), args.output.resolve() / name)
    if args.fact_budget < 1:
        raise SystemExit("--fact-budget deve ser positivo")
    if args.dry_run:
        print("Quatro variantes validadas; nenhuma chamada de modelo:")
        for name, command in commands.items():
            print(name + ": " + subprocess.list2cmdline(command))
        return 0
    root = args.output.resolve()
    sources = comparison.source_hashes()
    for relative in ["scripts/run-reflection-ablation.py", "scripts/reflection-ablation-report.py",
                     "scripts/run-reflection-ablation-qwen.sh"]:
        sources[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    manifest = {"protocol": "cascade-reflection-2x2-v1", "commands": commands,
                "sources": sources, "factors": VARIANTS}
    # JSON turns tuples into lists: compare canonical serializations on resume.
    manifest = json.loads(json.dumps(manifest))
    path = root / "ablation-manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if any(previous.get(k) != v for k, v in manifest.items()):
            raise SystemExit("Código/configuração mudou. Use outro --output para preservar a comparação.")
    elif any((root / name).exists() for name in VARIANTS):
        raise SystemExit("Saídas existentes sem manifesto. Use outro --output.")
    elif args.report_only:
        raise SystemExit("Estudo ausente: execute primeiro sem --report-only.")
    else:
        root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**manifest, "created": datetime.now(timezone.utc).isoformat()},
                                   indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="42", WRAG_LLM_CACHE="1",
               WRAG_EMBED_CACHE="1", WRAG_CONTINUE_ON_CONTENT_FILTER="1")
    only_names = set(n.strip() for n in args.only.split(",")) if args.only else None
    if only_names and not only_names <= set(VARIANTS):
        raise SystemExit(f"--only desconhecido: {only_names - set(VARIANTS)}; use {list(VARIANTS)}")
    for index, (name, command) in enumerate(commands.items(), 1):
        if only_names and name not in only_names:
            continue
        folder = root / name
        if comparison.completed(folder):
            print(f"[{index}/4] {name}: completo, preservado.", flush=True)
            continue
        if args.report_only:
            raise SystemExit(f"{name} incompleto; repita sem --report-only para retomar.")
        resume = ["--resume"] if (folder / "pilot.json").exists() else []
        print(f"[{index}/4] {name}: execução {'retomada' if resume else 'nova'}. "
              f"Log: {folder / 'benchmark.log'}", flush=True)
        result = subprocess.run(command + resume, cwd=ROOT, env=env)
        if result.returncode or not comparison.completed(folder):
            raise SystemExit(f"{name} incompleto. Repita o mesmo comando para retomar; "
                             f"veja {folder / 'benchmark.log'}.")
    if only_names and not all(comparison.completed(root / n) for n in VARIANTS):
        print(f"--only={sorted(only_names)} feito. Rode a(s) outra(s) variante(s) (mesmo --output) "
              f"e depois --report-only para o relatório combinado.", flush=True)
        return 0
    corpora = comparison.compare_corpora(root, list(VARIANTS))
    (root / "corpus-hashes.json").write_text(json.dumps(corpora, indent=2), encoding="utf-8")
    for report, names in [("paired-report.py", list(VARIANTS)), ("reflection-ablation-report.py", [])]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / report), str(root), *names],
                       cwd=ROOT, env=env, check=True)
    print(f"Concluído: {root / 'reflection-ablation.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
