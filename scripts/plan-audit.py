"""How the proof controller's plans were written and used, per LoCoMo category.

    python scripts/plan-audit.py v3=runs/v4-qwen/proof-v3 v4=runs/v4-qwen/proof-v4

For each run (only the questions all runs answered): how often a proof was used
(confirmed, or already in the passages), how often the proof changed what the
reader sees (a swapped passage or a block of source turns), why the others
stopped (too many answers, incomplete join, invalid plan), how many plans had a
type and how many of those the PLANNER wrote (the rest came from the kind named
by the question, repair "tipo_da_pergunta"), the verifier's per-item decisions
and the mean number of planning cycles. The category is read only to split the
table.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def load(root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.rglob("witnessrag.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["qid"]] = row
    return rows


def stats(rows: list[dict]) -> dict[str, float]:
    n = len(rows) or 1
    stop = collections.Counter(r["diagnosticos"].get("motivo_parada") for r in rows)
    typed = llm_typed = confirmed = rejected = cycles = changed = 0
    for r in rows:
        d = r["diagnosticos"]
        final = d.get("plano_final") or {}
        if (final.get("consulta") or {}).get("types"):
            typed += 1
            if "tipo_da_pergunta" not in (final.get("reparos") or []):
                llm_typed += 1
        for cycle in d.get("ciclos") or []:
            verdict = cycle.get("verificacao")
            if isinstance(verdict, dict):
                confirmed += len(verdict.get("confirmadas") or [])
                rejected += len(verdict.get("recusadas") or {})
        cycles += len(d.get("ciclos") or [])
        changed += bool(d.get("contexto_alterado_pela_prova") or d.get("contexto_alterado_por_trechos"))
    return {"n": len(rows),
            "proof_used": 100 * (stop["prova_confirmada"] + stop["prova_ja_no_contexto"]) / n,
            "changed": 100 * changed / n,
            "too_many": 100 * stop["respostas_demais"] / n,
            "join_incomplete": 100 * stop["join_incompleto"] / n,
            "invalid": 100 * (stop["plano_invalido"] + stop["plano_filtrado"]) / n,
            "typed": typed, "typed_by_planner": llm_typed,
            "verifier_ok": confirmed, "verifier_no": rejected,
            "cycles": cycles / n}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    runs = []
    for item in args.runs:
        label, _, path = item.partition("=")
        runs.append((label, load(Path(path or label))))
    shared = set.intersection(*(set(rows) for _l, rows in runs))
    lines = [f"# Planos do controlador de prova ({len(shared)} perguntas em comum)", "",
             "| rodada | categoria | n | prova usada % | mudou contexto % | respostas demais % | "
             "junção incompleta % | plano inválido % | tipados (pelo planejador) | "
             "verificador sim/não | ciclos |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for label, rows in runs:
        for category in CATEGORIES + ("todas",):
            items = [rows[q] for q in shared
                     if category == "todas" or rows[q].get("tipo") == category]
            s = stats(items)
            lines.append(
                f"| {label} | {category} | {s['n']} | {s['proof_used']:.1f} | {s['changed']:.1f} | "
                f"{s['too_many']:.1f} | {s['join_incomplete']:.1f} | {s['invalid']:.1f} | "
                f"{s['typed']} ({s['typed_by_planner']}) | {s['verifier_ok']}/{s['verifier_no']} | "
                f"{s['cycles']:.2f} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
