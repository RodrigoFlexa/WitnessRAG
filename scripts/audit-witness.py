#!/usr/bin/env python3
"""Auditoria descritiva de resultados salvos; não reexecuta leitor nem inferência.

python scripts/audit-witness.py runs/witness-v2 runs/qwen14b-pilot --output docs/audit-witness.json
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("compare_runs",
        Path(__file__).with_name("compare-runs.py"))
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    new, base = compare.bench_dir(args.new), compare.bench_dir(args.base)
    a, b = (json.loads((p / "run.json").read_text(encoding="utf-8")) for p in (new, base))
    problems = compare.guard(new, base)
    audit = {"new": str(new), "base": str(base), "comparison_caveats": problems,
             "scope": "descriptive; no causal attribution; bootstrap does not account for tuning",
             "datasets": {}}
    for dataset in a["datasets"]:
        A, B = compare.load(new, dataset), compare.load(base, dataset)
        w = A["witnessrag"]
        rows = {**B, "witnessrag": w}
        ids = sorted(set.intersection(*(set(v) for v in rows.values())))
        if not ids:
            raise ValueError("no paired questions")
        # Hash mismatch prevents even a descriptive cross-dataset pairing.
        if any("hash" in problem or "manifesto" in problem for problem in problems):
            raise ValueError("corpus/questions cannot be paired")
        metrics = ["f1", "recall@5", "all_recall@5"]
        def means(group):
            return {m: {key: 100 * sum(rr[q][key] for q in group) / len(group)
                        for key in metrics} for m, rr in rows.items()} if group else {}
        strata = {}
        for kind in sorted({w[q]["tipo"] for q in ids}):
            for state in ("all", "complete", "fallback"):
                group = [q for q in ids if w[q]["tipo"] == kind and
                         (state == "all" or bool(w[q]["testemunha_no_contexto"]) == (state == "complete"))]
                if group:
                    strata[f"{kind}/{state}"] = {"n": len(group), "metrics": means(group),
                        "structural_exact_matches": sum(w[q]["em_estrutural"] for q in group)}
        pairs = {}
        for method, rr in rows.items():
            if method == "witnessrag": continue
            delta = [w[q]["f1"] - rr[q]["f1"] for q in ids]
            rng = random.Random(42)
            boot = sorted(100 * sum(rng.choices(delta, k=len(delta))) / len(delta)
                          for _ in range(10000))
            pairs[method] = {"delta_f1_pp": 100 * sum(delta) / len(delta),
                "wins": sum(v > 0 for v in delta), "losses": sum(v < 0 for v in delta),
                "ties": sum(v == 0 for v in delta), "descriptive_bootstrap_95": [boot[249], boot[9749]]}
        complete = [q for q in ids if w[q]["testemunha_no_contexto"]]
        full = [q for q in ids if w[q]["all_recall@5"] == 1]
        entry = {"n_paired": len(ids), "metrics": means(ids), "strata": strata, "pairs": pairs,
            "complete": len(complete), "structural_exact_matches": sum(w[q]["em_estrutural"] for q in complete),
            "full_evidence": len(full), "full_evidence_f1": means(full),
            "full_evidence_zero_f1": sum(w[q]["f1"] == 0 for q in full),
            "extraction": {"new": a["resumos"][dataset]["extracao"],
                           "base": b["resumos"][dataset]["extracao"]},
            "complete_examples": [{k: w[q][k] for k in ("qid", "pergunta", "respostas_ouro",
                "resposta_estrutural", "em_estrutural", "resposta", "f1", "recall@5")}
                for q in complete]}
        audit["datasets"][dataset] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit saved: {args.output}; caveats: {problems}")


if __name__ == "__main__":
    main()
