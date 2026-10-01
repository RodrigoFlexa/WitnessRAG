#!/usr/bin/env python3
"""Summarize matched replan records, without confusing a pilot with full LoCoMo."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(results, old_replan=None):
    rows = [json.loads(line) for p in sorted(Path(results).glob("conv*/results.jsonl"))
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate matched results")
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results).glob("conv*/manifest.json"))]
    identities = {(m["code_hash"], m["controller_version"], m["model"]) for m in manifests}
    if len(identities) != 1:
        raise ValueError("Mixed versions or models")
    old = {}
    if old_replan and Path(old_replan).exists():
        for p in Path(old_replan).glob("conversations/conv*/benchmark/*/locomo/witnessrag.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["qid"] in old:
                    raise ValueError("Ambiguous old replan")
                old[row["qid"]] = row
        if not {r["qid"] for r in rows} <= old.keys():
            raise ValueError("Old replan comparison has missing question IDs")
    groups = {}
    for name, subset in [("all", rows)] + [(kind, [r for r in rows if r["tipo"] == kind])
                                         for kind in sorted({r["tipo"] for r in rows})]:
        groups[name] = {"n": len(subset),
                        "standard_f1": 100 * statistics.mean(r["historical_f1"] for r in subset),
                        "new_replan_f1": 100 * statistics.mean(r["f1_locomo"] for r in subset),
                        "wins_vs_standard": sum(r["f1_locomo"] > r["historical_f1"] for r in subset),
                        "losses_vs_standard": sum(r["f1_locomo"] < r["historical_f1"] for r in subset)}
        if old:
            groups[name]["old_replan_f1"] = 100 * statistics.mean(old[r["qid"]]["f1_locomo"] for r in subset)
    selected = {qid for m in manifests for qid in m["selected_ids"]}
    retries = [r for r in rows if r["trace"]["performed"]]
    tokens = lambda key: sum(r[key]["total"].get(f, 0) for r in rows
                            for f in ("tokens_prompt", "tokens_resposta"))
    return {"questions": len(rows), "complete_selected_cohort": selected == {r["qid"] for r in rows},
            "development_only": any(m["selection_rule"] != "all source questions" for m in manifests),
            "full_locomo": len(rows) == 1540 and len(manifests) == 10,
            "controller_version": manifests[0]["controller_version"], "groups": groups,
            "retries": len(retries), "invalid_controls": sum(not r["trace"]["gate"]["valid"] for r in rows),
            "mean_initial_facts_retained_after_retry": (statistics.mean(len(r["trace"]["kept_fact_indices"]) for r in retries) if retries else None),
            "logical_tokens": {"standard": tokens("historical_usage"), "new_replan": tokens("uso_llm")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--old-replan", type=Path)
    args = parser.parse_args()
    summary = summarize(args.results, args.old_replan)
    (args.results / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
