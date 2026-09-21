"""Paired report for base/soft/temporal/full suite arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrag.eval.metrics import bootstrap_ci
from wrag.eval.locomo_official import score_record
from wrag.util import write_json

ARMS = ("base", "soft", "temporal", "full")
CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def load_arm(root: Path, arm: str) -> dict[str, dict]:
    paths = sorted((root / arm).glob("benchmark/*/locomo/witnessrag.jsonl"))
    paths += sorted((root / arm).glob("conversations/*/benchmark/*/locomo/witnessrag.jsonl"))
    rows = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "bleu1_locomo" not in row:
                row.update(score_record(row))
            rows[row["qid"]] = row
    if not rows:
        raise FileNotFoundError(f"no LoCoMo predictions under {root / arm}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args(argv)
    arms = {name: load_arm(args.root, name) for name in ARMS}
    common = set.intersection(*(set(rows) for rows in arms.values()))
    report = {"paired_questions": len(common), "arms": {}, "against_base": {}}
    for arm, rows in arms.items():
        report["arms"][arm] = {}
        for category in CATEGORIES:
            subset = [rows[qid] for qid in common if rows[qid].get("tipo") == category]
            report["arms"][arm][category] = {
                "n": len(subset),
                "f1": mean(r["f1_locomo"] for r in subset) if subset else None,
                "bleu1": mean(r["bleu1_locomo"] for r in subset) if subset else None,
            }
        if arm == "base":
            continue
        report["against_base"][arm] = {}
        for category in CATEGORIES:
            qids = [qid for qid in common if rows[qid].get("tipo") == category]
            delta = [rows[qid]["f1_locomo"] - arms["base"][qid]["f1_locomo"] for qid in qids]
            report["against_base"][arm][category] = {
                "delta_f1": mean(delta) if delta else None,
                "question_bootstrap_ci95": bootstrap_ci(delta, n_boot=5000) if len(delta) > 1 else [None, None],
                "wins": sum(x > 0 for x in delta), "losses": sum(x < 0 for x in delta),
                "ties": sum(x == 0 for x in delta),
            }
    write_json(args.root / "ablation_report.json", report)
    lines = ["# Witness suite — paired ablation", "", f"Common questions: {len(common)}", "",
             "| arm | category | n | F1 | BLEU-1 | ΔF1 vs base | 95% question bootstrap |",
             "|---|---|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        for category in CATEGORIES:
            value = report["arms"][arm][category]
            paired = report["against_base"].get(arm, {}).get(category, {})
            ci = paired.get("question_bootstrap_ci95", [None, None])
            fmt = lambda x: "—" if x is None else f"{x:.4f}"
            lines.append(f"| {arm} | {category} | {value['n']} | {fmt(value['f1'])} | "
                         f"{fmt(value['bleu1'])} | {fmt(paired.get('delta_f1'))} | "
                         f"[{fmt(ci[0])}, {fmt(ci[1])}] |")
    (args.root / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
