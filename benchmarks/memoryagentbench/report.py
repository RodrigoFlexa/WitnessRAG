"""Coverage first, task means next, then the paper's competency macro-average."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from wrag.llm.base import sum_usage
from . import protocol as P
from .runner import atomic_json, read_rows


def report(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = defaultdict(int)
    split_for = {}
    for selected in manifest["identity"]["selection"]:
        expected[selected["source"]] += selected["questions"]
        split_for[selected["source"]] = selected["split"]
    rows = read_rows(output / "results.jsonl")
    ids = [r["qid"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate predictions in report checkpoint")
    if all("key" in s for s in manifest["identity"]["selection"]):
        intended = {f"{s['key']}:q{i}" for s in manifest["identity"]["selection"]
                    for i in range(s["questions"])}
        if not set(ids) <= intended:
            raise ValueError("Foreign question IDs in report checkpoint")
    judgments = read_rows(output / "judgments.jsonl")
    judged = {r["qid"]: r for r in judgments}
    if len(judged) != len(judgments) or not set(judged) <= set(ids):
        raise ValueError("Duplicate or foreign judge records")
    groups = defaultdict(list)
    for row in rows:
        current = dict(row)
        current["metrics"] = dict(row["metrics"])
        if row["qid"] in judged:
            judgment = judged[row["qid"]]
            if judgment["output_hash"] != P.digest(row["output"]):
                raise ValueError("Judgment refers to a different prediction")
            current["metrics"].update(judgment["metrics"])
        groups[row["source"]].append(current)
    sources = {}
    for source, total in expected.items():
        primary = P.primary_metric(source)
        records = groups[source]
        scores = [r["metrics"][primary] for r in records if primary in r["metrics"]]
        sources[source] = {"split": split_for[source], "expected": total, "answered": len(records),
                           "scored": len(scores), "filtered": sum(bool(r["filtered"]) for r in records),
                           "judge_pending": len(records) - len(scores), "metric": primary,
                           "score": sum(scores) / len(scores) if scores else None,
                           "complete": len(records) == total and len(scores) == total}
    competency = {}
    for split in P.SPLITS:
        tasks = [s for s in P.PAPER_SOURCES if P.CONFIGS[s]["dataset"] == split]
        available = [sources[s]["score"] for s in tasks if s in sources and sources[s]["complete"]]
        competency[split] = {"tasks": tasks,
                             "complete": len(available) == len(tasks),
                             "score": sum(available) / len(available) if len(available) == len(tasks) else None}
    overall = (sum(c["score"] for c in competency.values()) / len(competency)
               if all(c["complete"] for c in competency.values()) else None)
    index_rows = read_rows(output / "index.jsonl")
    query_failures = read_rows(output / "query-failures.jsonl")
    judge_failures = read_rows(output / "judge-failures.jsonl")
    costs = {"memory": sum_usage(r["cost"]["usage"] for r in index_rows),
             "query": sum_usage(r["usage"] for r in rows + query_failures),
             "judge": sum_usage(r.get("usage", {}) for r in judgments + judge_failures),
             "memory_seconds": sum(r["cost"]["seconds"] for r in index_rows),
             "query_seconds": sum(r["query_time_len"] for r in rows) + sum(r["seconds"] for r in query_failures),
             "failed_memory_attempts": sum(bool(r.get("failed")) for r in index_rows),
             "failed_query_attempts": len(query_failures), "failed_judge_attempts": len(judge_failures)}
    result = {"dataset_revision": P.DATASET_REVISION, "protocol": next(iter(manifest["identity"]["settings"].values()))["protocol"],
              "generation_complete": len(rows) == sum(expected.values()),
              "evaluation_complete": all(r["complete"] for r in sources.values()),
              "sources": sources, "paper_competencies": competency,
              "paper_overall": overall, "costs": costs,
              "notes": ["Scores are fractions; percentages are displayed in report.md.",
                        "Averages from partial tasks are exploratory; no complete paper score is emitted.",
                        "Paper columns are task means, then competency means; not pooled question accuracy.",
                        "Memory construction, querying and external judges are costed separately.",
                        "Memory usage is observed in this run; reused OpenIE caches do not report their prior token cost.",
                        "Method adaptation: stable ingestion timestamps, source serial provenance, plain task outputs."]}
    atomic_json(output / "summary.json", result)
    lines = [f"# MemoryAgentBench — {result['protocol']}", "",
             f"Dataset revision: `{P.DATASET_REVISION}`. Official code: `{P.UPSTREAM_COMMIT}`.", "",
             "Scores below are provisional until each row is complete. Missing judge scores are pending.", "",
             "| Source | Answered / expected | Scored | Primary metric | Score (%) | Complete |",
             "|---|---:|---:|---|---:|---|"]
    for name, stats in sources.items():
        value = f"{100 * stats['score']:.2f}" if stats["score"] is not None else "pending"
        lines.append(f"| {name} | {stats['answered']} / {stats['expected']} | {stats['scored']} | "
                     f"{stats['metric']} | {value} | {stats['complete']} |")
    lines += ["", "Paper competency means (only complete main tasks):", ""]
    for split, stats in competency.items():
        value = f"{stats['score'] * 100:.2f}%" if stats["score"] is not None else "pending"
        lines.append(f"- {split}: {value}")
    lines += ["", f"Paper overall: {overall * 100:.2f}%" if overall is not None else "Paper overall: pending.", "",
              "Memory, query and judge usage: see summary.json; cache hits are recorded separately.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # One export per source, compatible with the official evaluator's data layout.
    exports = output / "official_exports"
    for source, records in groups.items():
        safe = source.replace("*", "_star")
        entries = [{"query": r["query"], "answer": r["answer"], "output": r["output"],
                    "query_id": i, "qa_pair_id": r["qa_pair_id"],
                    "output_len": r["completion_tokens"], "query_time_len": r["query_time_len"]}
                   for i, r in enumerate(records)]
        atomic_json(exports / split_for[source] / f"{safe}_results.json",
                    {"dataset_config": manifest["identity"]["settings"][source],
                     "agent_config": {"agent_name": "WitnessRAG_rag", "model": manifest["identity"]["model"]},
                     "data": entries, "averaged_metrics": {}})
    return result
