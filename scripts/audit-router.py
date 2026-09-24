"""Audit the category-agnostic router BEFORE a full LoCoMo run.

Runs only the planning call (the evidence contract) for every LoCoMo question
of categories 1-4 and compares the resulting route with the benchmark label.
The label is read here ONLY to evaluate the router; the planner never sees it.

Why this exists: when the contract route equals the labelled route, the
agnostic controller reproduces the labelled context exactly (same function,
same arguments, same LLM cache keys). Hence

    |F1_agnostic - F1_labelled| <= disagreement rate

and the disagreement rate costs one short LLM call per question to measure.
The contract calls land in the normal LLM cache, so a later full run with the
same WRAG_CACHE_DIR does not pay for them again.

    WRAG_LLM_BACKEND=azure WRAG_AZURE_DEPLOYMENT=gpt-4-1-mini-petrobras \\
    WRAG_CACHE_DIR=runs/.cache/witness-azure \\
    python scripts/audit-router.py --locomo-file .cache/locomo/locomo10.json \\
        --output runs/router-audit
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrag import prompts  # noqa: E402
from wrag.llm import get_llm  # noqa: E402
from wrag.locomo import convert  # noqa: E402
from wrag.util import write_json  # noqa: E402
from wrag.witness.contract import (ROUTE_COMPOSE, contract_from_result,  # noqa: E402
                                   contract_params, contract_prompt)

CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")
# Labelled routing of the selective controller, used only as the reference.
LABELLED_COMPOSE = {"multi-hop"}


def load_questions(path: Path, conversations: list[int] | None) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    indices = conversations if conversations else list(range(len(raw)))
    out = []
    for index in indices:
        questions, _passages, _meta = convert(raw, index, turns_per_passage=8)
        out.extend({"qid": q["id"], "question": q["question"], "category": q["type"],
                    "conversation": index} for q in questions)
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--locomo-file", type=Path, default=ROOT / ".cache/locomo/locomo10.json")
    p.add_argument("--conversations", default="all",
                   help="'all' or comma-separated zero-based indices")
    p.add_argument("--output", type=Path, default=ROOT / "runs/router-audit")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args(argv)

    selected = None if args.conversations == "all" else \
        [int(x) for x in args.conversations.split(",") if x.strip()]
    questions = load_questions(args.locomo_file, selected)
    llm = get_llm()
    results = llm.chat_many([contract_prompt(q["question"]) for q in questions],
                            system=prompts.CONTRACT_SYSTEM,
                            params=contract_params(args.temperature),
                            stage="witness.contract", desc="contracts")
    rows, confusion = [], Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    for question, result in zip(questions, results):
        contract = contract_from_result(result)
        route = contract.route
        labelled = "compose" if question["category"] in LABELLED_COMPOSE else "direct"
        confusion[(question["category"], route)] += 1
        stats = by_category[question["category"]]
        stats["n"] += 1
        stats["agree"] += route == labelled
        stats[f"route:{route}"] += 1
        stats[f"form:{contract.answer_form}"] += 1
        stats[f"operator:{contract.operator}"] += 1
        stats[f"scope:{contract.evidence_scope}"] += 1
        for lens in ("temporal", "salience", "confidence"):
            value = contract.lenses[lens]
            if value not in (False, "none"):
                stats[f"lens:{lens}:{value}"] += 1
        stats["invalid"] += not contract.valid
        stats["filtered"] += contract.filtered
        rows.append({**question, "route": route, "labelled_route": labelled,
                     "agree": route == labelled, **{k: v for k, v in contract.to_dict().items()
                                                    if k in ("answer_form", "operator",
                                                             "evidence_scope", "valid",
                                                             "filtered", "error")},
                     "time_focus": contract.time_focus, "time_anchor": contract.time_anchor,
                     "lenses": json.dumps(contract.lenses)})
    total = len(rows)
    disagreements = [r for r in rows if not r["agree"]]
    summary = {
        "questions": total,
        "agreement": (total - len(disagreements)) / total if total else None,
        "disagreement_rate_bound_on_f1_gap": len(disagreements) / total if total else None,
        "composition_rate": sum(r["route"] == ROUTE_COMPOSE for r in rows) / total if total else None,
        "llm_calls_per_question_expected": 2 + (sum(r["route"] == ROUTE_COMPOSE for r in rows) / total
                                                if total else 0),
        "confusion_category_x_route": {f"{c}|{r}": n for (c, r), n in sorted(confusion.items())},
        "per_category": {c: dict(by_category[c]) for c in CATEGORIES if c in by_category},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "router_audit.json", summary)
    with (args.output / "router_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["qid"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Router audit (planner only)", "",
             f"Questions: {total}. Agreement with the labelled route: "
             f"{summary['agreement']:.3f}. Upper bound on |ΔF1| vs the labelled controller: "
             f"{summary['disagreement_rate_bound_on_f1_gap']:.3f}.", "",
             "| category | n | agree | → compose | → direct |", "|---|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        stats = by_category.get(category)
        if not stats:
            continue
        lines.append(f"| {category} | {stats['n']} | {stats['agree'] / stats['n']:.3f} | "
                     f"{stats['route:compose']} | {stats['route:direct']} |")
    lines += ["", "Disagreements are listed in router_audit.csv (column agree=False)."]
    (args.output / "router_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main()
