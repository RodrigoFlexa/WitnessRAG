"""Paired report for the proof controller (design v3).

    python scripts/proof-report.py --proof runs/witness-suite-locomo-azure-proof \\
        --reference runs/witness-suite-locomo-azure-selective \\
        [--hybrid runs/witness-suite-locomo-azure-hybrid]

The LoCoMo category is read here only to split the evaluation tables. Outputs
proof_report.json and proof_report.md inside --proof. The bootstrap resamples
conversations (clusters), not questions, because questions of one conversation
share a memory.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrag.util import write_json  # noqa: E402

CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def load(root: Path, method: str = "witnessrag") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.rglob(f"{method}.jsonl")):
        if "locomo" not in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "f1_locomo" not in row or "bleu1_locomo" not in row:
                try:
                    from wrag.eval.locomo_official import score_record
                    row.update(score_record(row))
                except Exception:  # noqa: BLE001 - NLTK may be absent
                    pass
            rows[row["qid"]] = row
    if not rows:
        raise FileNotFoundError(f"no LoCoMo {method}.jsonl under {root}")
    return rows


def number(row: dict, key: str) -> float:
    value = row.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value


def f1(row: dict) -> float:
    value = number(row, "f1_locomo")
    return value if value == value else number(row, "f1")


def diag(row: dict) -> dict:
    value = row.get("diagnosticos")
    return value if isinstance(value, dict) else {}


def pids(row: dict) -> list[str]:
    value = row.get("recuperadas")
    if isinstance(value, str):
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    return list(value or [])


def calls(row: dict) -> dict[str, int]:
    usage = row.get("uso_llm")
    if isinstance(usage, str):
        try:
            import ast
            usage = ast.literal_eval(usage)
        except (ValueError, SyntaxError):
            usage = {}
    stages = (usage or {}).get("por_estagio") or {}
    return {stage: int(v.get("chamadas", 0)) for stage, v in stages.items()
            if not stage.startswith("index")}


def conversation(qid: str) -> str:
    return qid.split(":")[1] if qid.count(":") >= 2 else "?"


def cluster_ci(deltas: dict[str, float], n_boot: int = 5000, seed: int = 42):
    groups: dict[str, list[float]] = defaultdict(list)
    for qid, value in deltas.items():
        groups[conversation(qid)].append(value)
    if len(groups) < 2:
        return [None, None]
    rng = random.Random(seed)
    values = list(groups.values())
    draws = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        total = sum(len(g) for g in sample)
        draws.append(sum(sum(g) for g in sample) / total)
    draws.sort()
    return [draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]]


def block(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    finite = lambda xs: [x for x in xs if x == x]
    per_call = [sum(calls(r).values()) for r in rows]
    return {"n": len(rows), "f1": mean(finite([f1(r) for r in rows]) or [float("nan")]),
            "bleu1": mean(finite([number(r, "bleu1_locomo") for r in rows]) or [float("nan")]),
            "r5": mean(finite([number(r, "recall@5") for r in rows]) or [float("nan")]),
            "ar5": mean(finite([number(r, "all_recall@5") for r in rows]) or [float("nan")]),
            "changed": mean(float(bool(diag(r).get("contexto_alterado_pelo_witness")))
                            for r in rows),
            "calls_per_question": mean(per_call)}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proof", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="e.g. the selective-v1 run")
    p.add_argument("--hybrid", type=Path, help="optional METHOD=hybrid run")
    args = p.parse_args(argv)
    proof = load(args.proof)
    report: dict = {"proof": {}, "routes": {}, "stops": {}, "plans": {}, "verification": {},
                    "calls_by_stage": {}}
    by_category = lambda rows, c: [r for r in rows if c == "all" or r.get("tipo") == c]
    for category in CATEGORIES + ("all",):
        report["proof"][category] = block(by_category(list(proof.values()), category))
        subset = by_category(list(proof.values()), category)
        report["routes"][category] = dict(Counter(diag(r).get("rota", "?") for r in subset))
        report["stops"][category] = dict(Counter(diag(r).get("motivo_parada", "?")
                                                 for r in subset))
    stage_calls: dict[str, int] = Counter()
    for row in proof.values():
        stage_calls.update(calls(row))
    report["calls_by_stage"] = {s: v / len(proof) for s, v in sorted(stage_calls.items())}

    plans = Counter()
    verification = Counter()
    rejections = Counter()
    for row in proof.values():
        d = diag(row)
        final = d.get("plano_final") or {}
        plans["with_final_plan"] += bool(final)
        plans[f"period:{(final.get('periodo') or {}).get('tipo', 'none')}"] += bool(final)
        plans[f"time_level:{final.get('nivel_tempo', 'none')}"] += bool(final)
        plans[f"importance_level:{final.get('nivel_importancia', 'none')}"] += bool(final)
        plans["composite"] += bool(final.get("composto"))
        plans["cycles_2plus"] += len(d.get("ciclos") or []) >= 2
        plans["context_changed_by_score"] += bool(d.get("contexto_alterado_pela_pontuacao"))
        plans["context_changed_by_proof"] += bool(d.get("contexto_alterado_pela_prova"))
        for cycle in d.get("ciclos") or []:
            check = cycle.get("verificacao")
            if isinstance(check, dict):
                verification["called"] += int(bool(check.get("chamada")))
                verification["confirmed_any"] += int(bool(check.get("confirmadas")))
                for reason in (check.get("recusadas") or {}).values():
                    rejections[reason] += 1
            elif isinstance(check, str):
                verification[check] += 1
    report["plans"] = dict(plans)
    report["verification"] = dict(verification, rejections=dict(rejections))

    def paired(other: dict[str, dict], name: str) -> None:
        common = sorted(set(proof) & set(other))
        report[f"paired_{name}"] = {"n": len(common)}
        for category in CATEGORIES + ("all",):
            qids = [q for q in common if category == "all" or proof[q].get("tipo") == category]
            deltas = {q: f1(proof[q]) - f1(other[q]) for q in qids
                      if f1(proof[q]) == f1(proof[q]) and f1(other[q]) == f1(other[q])}
            same = [q for q in qids if pids(proof[q]) == pids(other[q])]
            report[f"paired_{name}"][category] = {
                "n": len(qids), f"f1_{name}": mean(f1(other[q]) for q in qids) if qids else None,
                "delta_f1": mean(deltas.values()) if deltas else None,
                "ci95_by_conversation": cluster_ci(deltas),
                "wins": sum(v > 0 for v in deltas.values()),
                "losses": sum(v < 0 for v in deltas.values()),
                "same_context": len(same) / len(qids) if qids else None,
                # Sanity check: identical context => cached identical reader answer.
                "same_context_same_answer": (sum(proof[q].get("resposta") == other[q].get("resposta")
                                                 for q in same) / len(same)) if same else None,
            }

    if args.reference:
        paired(load(args.reference), "reference")
    if args.hybrid:
        paired(load(args.hybrid), "hybrid")
    write_json(args.proof / "proof_report.json", report)

    fmt = lambda x: f"{100 * x:.2f}" if isinstance(x, float) and x == x else "-"
    lines = ["# Proof controller report", "",
             "| category | n | F1 | BLEU-1 | R@5 | AR@5 | ctx changed | calls/q |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES + ("all",):
        b = report["proof"][category]
        if b.get("n"):
            lines.append(f"| {category} | {b['n']} | {fmt(b['f1'])} | {fmt(b['bleu1'])} | "
                         f"{fmt(b['r5'])} | {fmt(b['ar5'])} | {fmt(b['changed'])} | "
                         f"{b['calls_per_question']:.2f} |")
    for name in ("reference", "hybrid"):
        block_ = report.get(f"paired_{name}")
        if not block_:
            continue
        lines += ["", f"## Paired with {name} ({block_['n']} questions)", "",
                  f"| category | F1 {name} | ΔF1 | 95% CI (conversations) | W/L | same context |",
                  "|---|---:|---:|---|---|---:|"]
        for category in CATEGORIES + ("all",):
            c = block_.get(category) or {}
            if not c.get("n"):
                continue
            ci = c["ci95_by_conversation"]
            lines.append(f"| {category} | {fmt(c[f'f1_{name}'])} | {fmt(c['delta_f1'])} | "
                         f"[{fmt(ci[0])}, {fmt(ci[1])}] | {c['wins']}/{c['losses']} | "
                         f"{fmt(c['same_context'])} |")
    lines += ["", "Routes: " + json.dumps(report["routes"]["all"]),
              "Stops: " + json.dumps(report["stops"]["all"]),
              "Plans: " + json.dumps(report["plans"]),
              "Verification: " + json.dumps(report["verification"]),
              "Calls by stage (per question): " + json.dumps(
                  {k: round(v, 3) for k, v in report["calls_by_stage"].items()})]
    (args.proof / "proof_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    main()
