#!/usr/bin/env python3
"""Placar ao vivo de uma rodada em andamento.

Os registros são anexados a <run>/<dataset>/<metodo>.jsonl a cada pergunta, então
dá para ler as métricas parciais sem esperar o relatório final.

    scripts/pilot-watch.py [runs/algum-run]     # uma leitura
    watch -n 20 scripts/pilot-watch.py          # atualizando sozinho
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [("em", "EM"), ("f1", "F1"), ("recall@5", "R@5"), ("all_recall@5", "AR@5")]


def newest_run(base: Path) -> Path | None:
    runs = [p for p in base.glob("*/benchmark/*/") if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main() -> int:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    if target and not (target / "run.json").exists():
        inner = sorted(target.glob("benchmark/*/"), key=lambda p: p.stat().st_mtime)
        target = inner[-1] if inner else None
    run = target or newest_run(ROOT / "runs")
    if run is None:
        print("nenhuma rodada encontrada em runs/")
        return 1

    print(f"rodada: {run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}")
    for dataset_dir in sorted(p for p in run.iterdir() if p.is_dir() and p.name != "figures"):
        files = sorted(dataset_dir.glob("*.jsonl"))
        if not files:
            continue
        total = 0
        corpus = dataset_dir / "corpus.json"
        if corpus.exists():
            try:
                total = len(json.loads(corpus.read_text()).get("questions", []))
            except Exception:
                total = 0
        print(f"\n{dataset_dir.name}")
        print(f"  {'método':<14} {'feitas':>7} " + " ".join(f"{h:>6}" for _, h in FIELDS))
        print("  " + "-" * (14 + 8 + 7 * len(FIELDS)))
        rows = []
        for f in files:
            recs = []
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # última linha pode estar sendo escrita agora
            if recs:
                rows.append((f.stem, recs))
        for name, recs in sorted(rows, key=lambda r: -sum(x.get("f1", 0) for x in r[1]) / len(r[1])):
            vals = " ".join(f"{100 * sum(r.get(k, 0) for r in recs) / len(recs):>6.1f}" for k, _ in FIELDS)
            print(f"  {name:<14} {len(recs):>7} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
