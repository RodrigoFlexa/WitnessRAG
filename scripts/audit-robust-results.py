"""Audit saved robust-plan predictions, without embeddings, model calls or API use.

python scripts/audit-robust-results.py runs/conv05-robust --output runs/review-20260927
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("paired_report", Path(__file__).with_name("paired-report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    data, sources, mismatch = {}, {}, {}
    for path in sorted(args.root.glob("*.jsonl")):
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = report.load(path)
        data[path.stem] = rows
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        mismatch[path.stem] = {metric: sum(abs(r.get(metric, -1) - rows[r["qid"]][metric]) > 1e-9
            for r in raw) for metric in ("f1_locomo", "bleu1_locomo")}
    robust, control = data["witnessrag-robust"], data["robust-no-plan"]
    ids = report.paired_ids(robust, control)
    accepted = {"prova_confirmada", "prova_ja_no_contexto", "prova_sem_verificacao"}
    readings, unchecked = [], []
    primary_operators, alternative_operators, structures = Counter(), Counter(), Counter()
    plans = 0
    for qid in ids:
        r = robust[qid]
        d = r.get("diagnosticos", {})
        p = d.get("plano_final", {})
        item = {"qid": qid, "question": r["pergunta"], "category": r["tipo"],
                "stop": d.get("motivo_parada"), "answer": r["resposta"],
                "gold": r["respostas_ouro"], "f1": r["f1_locomo"],
                "delta_vs_no_plan": r["f1_locomo"] - control[qid]["f1_locomo"],
                "query": p.get("consulta"), "accepted": d.get("motivo_parada") in accepted}
        if p.get("leitura"):
            readings.append(item)
        if d.get("motivo_parada") == "prova_ja_no_contexto":
            unchecked.append(item)
        for cycle in d.get("ciclos") or []:
            primary = cycle.get("plano") or {}
            query = primary.get("consulta") or {}
            plans += 1
            primary_operators[query.get("aggregation", "missing")] += 1
            skeleton = lambda query: [(a.get("subject"), a.get("object"), a.get("time"))
                                      for a in query.get("atoms") or []]
            for other in primary.get("outras_leituras") or []:
                alternative = other.get("consulta") or {}
                alternative_operators[alternative.get("aggregation", "missing")] += 1
                structures["same_argument_structure" if skeleton(query) == skeleton(alternative)
                           else "different_argument_structure"] += 1
    subsets = {}
    for name, selected in {
        **{c: [q for q in ids if robust[q]["tipo"] == c] for c in report.CATS},
        "connected_plan_exploratory": [q for q in ids if robust[q]["diagnosticos"].get("plano_final", {}).get("conectado")],
    }.items():
        subsets[name] = {"n": len(selected),
            "delta_f1": 100 * statistics.mean(robust[q]["f1_locomo"] - control[q]["f1_locomo"] for q in selected),
            "ic95": report.paired_interval(robust, control, selected, "f1_locomo")[:2]}
    payload = {"api_calls": 0, "source_sha256": sources, "metric_mismatches": mismatch,
        "other_readings_final": len(readings), "other_readings_accepted": sum(r["accepted"] for r in readings),
        "other_readings_verified": sum(r["stop"] == "prova_confirmada" for r in readings),
        "other_readings": readings, "unchecked_proofs": unchecked,
        "plan_generation": {"calls": plans, "primary_operators": dict(primary_operators),
            "alternative_operators": dict(alternative_operators),
            "alternative_argument_structure": dict(structures),
            "note": "Literal argument/time-slot structure only; equal structure does not establish semantic redundancy."},
        "shape_by_category": {c: dict(Counter(r["diagnosticos"].get("forma", "invalid")
            for r in robust.values() if r["tipo"] == c)) for c in report.CATS},
        "exploratory_subsets": subsets,
        "note": "Saved predictions precede review fixes. Subsets selected by a model-produced plan are exploratory, not causal effects."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audited {len(data)} variants / {len(ids)} paired questions; no API calls.")


if __name__ == "__main__":
    main()
