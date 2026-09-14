#!/usr/bin/env python3
"""Compara uma rodada nova (p.ex. só witnessrag) com os métodos de uma rodada
anterior, sem reexecutar os comparadores.

    scripts/compare-runs.py runs/witness-v2 runs/qwen14b-pilot

Só é honesto se o corpus, as perguntas e o leitor forem os mesmos: o script
confere isso e recusa quando divergem. A comparação é pareada — entram apenas
as perguntas presentes em todos os métodos dos dois lados.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = [("em", "EM"), ("f1", "F1"), ("recall@5", "R@5"), ("all_recall@5", "AR@5"),
          ("abstencao", "abst.")]


def bench_dir(run: Path) -> Path:
    if (run / "run.json").exists():
        return run
    inner = sorted((run / "benchmark").glob("*/"), key=lambda p: p.stat().st_mtime)
    if not inner:
        raise SystemExit(f"{run}: nenhuma rodada encontrada")
    return inner[-1]


def load(bench: Path, dataset: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    d = bench / dataset
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[f.stem] = {r["qid"]: r for r in rows}
        if len(out[f.stem]) != len(rows):
            raise ValueError(f"{f}: qids duplicados")
    return out


def guard(new: Path, base: Path) -> list[str]:
    """Diferenças que invalidariam a comparação."""
    a, b = json.loads((new / "run.json").read_text()), json.loads((base / "run.json").read_text())
    bad = []
    if a.get("datasets") != b.get("datasets"):
        bad.append("datasets diferentes")
    for key in ("backend", "deployment", "provider_identity"):
        if a.get("llm", {}).get(key) != b.get("llm", {}).get(key):
            bad.append(f"leitor.{key} diferente")
    if a.get("config", {}).get("qa") != b.get("config", {}).get("qa"):
        bad.append("configuração do leitor diferente")
    if a.get("embedder", {}).get("chave") != b.get("embedder", {}).get("chave"):
        bad.append("embedder diferente")
    for key in ("top_k", "seed", "n_questions"):
        if a.get("config", {}).get(key) != b.get("config", {}).get(key):
            bad.append(f"config.{key} diferente")
    for dataset in a.get("datasets", []):
        left, right = new / dataset / "corpus.json", base / dataset / "corpus.json"
        if not left.exists() or not right.exists():
            bad.append(f"{dataset}: manifesto do corpus ausente")
            continue
        ca, cb = (json.loads(p.read_text(encoding="utf-8")) for p in (left, right))
        for key in ("corpus_hash", "questions_hash"):
            if not ca.get(key) or not cb.get(key) or ca[key] != cb[key]:
                bad.append(f"{dataset}: {key} ausente ou diferente")
    return bad


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    new, base = bench_dir(Path(sys.argv[1]).resolve()), bench_dir(Path(sys.argv[2]).resolve())
    problems = guard(new, base)
    if problems:
        print("COMPARAÇÃO INVÁLIDA: " + "; ".join(problems))
        print("Os números não seriam pareáveis. Rode os comparadores na mesma configuração.")
        return 1

    datasets = json.loads((new / "run.json").read_text()).get("datasets", [])
    for dataset in datasets:
        A, B = load(new, dataset), load(base, dataset)
        if not A:
            continue
        # A rodada nova manda no que existe; os demais vêm da anterior.
        merged = {**{m: r for m, r in B.items() if m not in A}, **A}
        origem = {m: ("nova" if m in A else "anterior") for m in merged}
        paired = set.intersection(*(set(v) for v in merged.values()))
        if not paired:
            raise ValueError(f"{dataset}: nenhuma pergunta pareada")
        for method, rows in merged.items():
            if len(rows) != len(paired):
                print(f"  AVISO: {method}: {len(rows) - len(paired)} perguntas fora da interseção")
        print(f"\n{dataset} · {len(paired)} perguntas pareadas · "
              f"{sum(1 for o in origem.values() if o == 'nova')} método(s) da rodada nova")
        print(f"  {'método':<14} {'origem':<9} " + " ".join(f"{h:>6}" for _, h in FIELDS))
        print("  " + "-" * (24 + 7 * len(FIELDS)))
        rows = []
        for m, recs in merged.items():
            vals = [100 * sum(recs[q].get(k, 0) for q in paired) / len(paired) for k, _ in FIELDS]
            rows.append((m, origem[m], vals))
        for m, o, vals in sorted(rows, key=lambda r: -r[2][1]):
            print(f"  {m:<14} {o:<9} " + " ".join(f"{v:>6.1f}" for v in vals))

        if "witnessrag" in merged:
            print("\n  diferenças pareadas (witnessrag menos comparador):")
            w = merged["witnessrag"]
            for m, recs in sorted(merged.items()):
                if m == "witnessrag":
                    continue
                d = [(w[q]["f1"] - recs[q]["f1"]) for q in paired]
                wins = sum(1 for x in d if x > 0)
                loss = sum(1 for x in d if x < 0)
                print(f"    vs {m:<12} ΔF1 {100 * sum(d) / len(d):>+6.1f}  "
                      f"(ganha {wins:>3}, perde {loss:>3}, empata {len(d) - wins - loss:>3})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
