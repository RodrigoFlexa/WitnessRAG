"""Paired report: category-agnostic controller vs. the labelled selective controller.

    python scripts/router-report.py --agnostic runs/witness-suite-locomo-azure-agnostic \\
        --labelled runs/witness-suite-locomo-azure-selective

The LoCoMo category is used here only for evaluation (per-category tables and
the route confusion matrix). Outputs router_report.json/.md in --agnostic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrag.eval.metrics import bootstrap_ci  # noqa: E402
from wrag.util import write_json  # noqa: E402

CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def load(root: Path, method: str = "witnessrag") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(root.rglob(f"{method}.jsonl")):
        if "locomo" not in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "bleu1_locomo" not in row:
                    try:
                        from wrag.eval.locomo_official import score_record
                        row.update(score_record(row))
                    except Exception:  # noqa: BLE001 - NLTK may be absent
                        pass
                rows[row["qid"]] = row
    if not rows:
        raise FileNotFoundError(f"no LoCoMo {method}.jsonl under {root}")
    return rows


def f1(row: dict) -> float:
    return float(row.get("f1_locomo", row.get("f1", 0.0)) or 0.0)


def calls(row: dict) -> tuple[int, dict[str, int]]:
    stages = (row.get("uso_llm") or {}).get("por_estagio") or {}
    per = {stage: int(v.get("chamadas", 0)) for stage, v in stages.items()}
    return sum(per.values()), per


def route(row: dict) -> str:
    diag = row.get("diagnosticos") or {}
    if diag.get("rota_plano"):
        return diag["rota_plano"]
    return "compose" if diag.get("planejamento", {}).get("chamadas", 0) else "direct"


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    total_calls = [calls(r)[0] for r in rows]
    return {"n": len(rows), "f1": mean(f1(r) for r in rows),
            "bleu1": mean(float(r.get("bleu1_locomo", 0.0) or 0.0) for r in rows),
            "llm_calls_per_question": mean(total_calls)}


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agnostic", type=Path, required=True)
    p.add_argument("--labelled", type=Path)
    args = p.parse_args(argv)
    agnostic = load(args.agnostic)
    report: dict = {"agnostic": {}, "confusion_category_x_route": {}, "lenses": {}}
    confusion = Counter((r.get("tipo"), route(r)) for r in agnostic.values())
    report["confusion_category_x_route"] = {f"{c}|{r}": n for (c, r), n in sorted(confusion.items())}
    for category in CATEGORIES + ("all",):
        subset = [r for r in agnostic.values() if category == "all" or r.get("tipo") == category]
        report["agnostic"][category] = summarize(subset)
    stage_calls: dict[str, list[int]] = defaultdict(list)
    for row in agnostic.values():
        for stage, count in calls(row)[1].items():
            stage_calls[stage].append(count)
    report["agnostic_calls_by_stage"] = {s: sum(v) / len(agnostic) for s, v in stage_calls.items()}

    lens_rows = [r for r in agnostic.values()
                 if ((r.get("diagnosticos") or {}).get("lentes") or {}).get("alterou")]
    policies = Counter()
    for row in agnostic.values():
        policy = ((row.get("diagnosticos") or {}).get("lentes") or {}).get("politica") or {}
        for lens in ("temporal", "salience", "confidence"):
            value = policy.get(lens)
            if value not in (None, False, "none"):
                policies[f"{lens}:{value}"] += 1
    report["lenses"] = {"questions_with_active_policy": dict(policies),
                        "context_changed_by_lens": len(lens_rows)}

    if args.labelled:
        labelled = load(args.labelled)
        common = sorted(set(agnostic) & set(labelled))
        report["paired_questions"] = len(common)
        report["labelled"] = {c: summarize([labelled[q] for q in common
                                            if c == "all" or labelled[q].get("tipo") == c])
                              for c in CATEGORIES + ("all",)}
        report["delta_f1"] = {}
        for category in CATEGORIES + ("all",):
            qids = [q for q in common if category == "all" or agnostic[q].get("tipo") == category]
            delta = [f1(agnostic[q]) - f1(labelled[q]) for q in qids]
            report["delta_f1"][category] = {
                "delta": mean(delta) if delta else None,
                "ci95": bootstrap_ci(delta, n_boot=5000) if len(delta) > 1 else [None, None],
                "wins": sum(d > 0 for d in delta), "losses": sum(d < 0 for d in delta)}
        agree = [q for q in common
                 if (route(agnostic[q]) == "compose") == (labelled[q].get("tipo") == "multi-hop")]
        same_context = [q for q in agree
                        if agnostic[q].get("recuperadas") == labelled[q].get("recuperadas")]
        # Questions whose outcome CAN differ: a disagreeing route where either
        # controller actually changed the hybrid context.
        effective = [q for q in common if q not in set(agree) and (
            (labelled[q].get("diagnosticos") or {}).get("contexto_alterado_pelo_witness") or
            (agnostic[q].get("diagnosticos") or {}).get("contexto_alterado_pelo_witness"))]
        report["route_agreement"] = len(agree) / len(common) if common else None
        report["identical_context_given_agreement"] = (len(same_context) / len(agree)
                                                       if agree else None)
        report["effective_disagreements"] = len(effective)
        report["effective_disagreement_rate"] = len(effective) / len(common) if common else None
        report["identical_answers"] = (sum(agnostic[q].get("resposta") == labelled[q].get("resposta")
                                           for q in common) / len(common)) if common else None
    write_json(args.agnostic / "router_report.json", report)
    lines = ["# Agnostic router report", "",
             "| category | n | F1 agnostic | F1 labelled | ΔF1 | 95% CI | calls/q |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for category in CATEGORIES + ("all",):
        a = report["agnostic"][category]
        if not a.get("n"):
            continue
        b = (report.get("labelled") or {}).get(category, {})
        d = (report.get("delta_f1") or {}).get(category, {})
        fmt = (lambda x: f"{100 * x:.2f}" if isinstance(x, float) else "-")
        ci = d.get("ci95") or [None, None]
        lines.append(f"| {category} | {a['n']} | {fmt(a['f1'])} | {fmt(b.get('f1'))} | "
                     f"{fmt(d.get('delta'))} | [{fmt(ci[0])}, {fmt(ci[1])}] | "
                     f"{a['llm_calls_per_question']:.2f} |")
    lines += ["", "Route confusion (category used only for evaluation):", ""]
    lines += [f"- {key}: {value}" for key, value in report["confusion_category_x_route"].items()]
    if args.labelled:
        lines += ["", f"Route agreement: {report['route_agreement']:.3f}; identical context "
                  f"when routes agree: {report['identical_context_given_agreement']:.3f}; "
                  f"effective disagreements: {report['effective_disagreements']} "
                  f"({report['effective_disagreement_rate']:.3f})."]
    lines += ["", f"Lens policies: {report['lenses']['questions_with_active_policy']}; "
              f"contexts changed by a lens: {report['lenses']['context_changed_by_lens']}."]
    (args.agnostic / "router_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    main()
