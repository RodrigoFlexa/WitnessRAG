"""Reader-budget table for LoCoMo runs.

    python scripts/budget-report.py --output runs/locomo-budget/budget_report \\
        proof-k5=runs/witness-suite-locomo-azure-proof \\
        proof-k3=runs/locomo-budget/proof-k3 hybrid-k3=runs/locomo-budget/hybrid-k3

For each run: chunks given to the reader (k), chunk size, mean reader prompt
tokens, mean controller tokens (planning + verification), official F1 and
BLEU-1 overall and per category, and evidence recall inside the reader budget.
The first run is the reference: every other run gets a paired ΔF1 (same
questions) with a 95% interval from a bootstrap over conversations, because
questions of one conversation share a memory. The LoCoMo category is read only
to split the table.

Every row is computed on the questions that ALL runs answered, so a run over the
first three conversations can be compared with the full ten-conversation
reference (--all-questions turns this off).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wrag.util import write_json  # noqa: E402

CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def load(root: Path) -> tuple[str, dict[str, dict]]:
    """Rows of the single method in the run (witnessrag or hybrid), by qid.
    Files are read in path order, so a resumed conversation's latest run wins."""
    rows: dict[str, dict] = {}
    methods = set()
    for path in sorted(root.rglob("*.jsonl")):
        if "locomo" not in path.parts or path.stem not in {"witnessrag", "hybrid"}:
            continue
        methods.add(path.stem)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "f1_locomo" not in row:
                try:
                    from wrag.eval.locomo_official import score_record
                    row.update(score_record(row))
                except Exception:  # noqa: BLE001 - NLTK may be absent
                    pass
            rows[row["qid"]] = row
    if not rows:
        raise FileNotFoundError(f"no LoCoMo results under {root}")
    if len(methods) > 1:
        raise ValueError(f"{root} mixes methods {sorted(methods)}; pass one run per method")
    return methods.pop(), rows


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _avg(values) -> float | None:
    values = [v for v in values if v == v]
    return mean(values) if values else None


def _recall(row: dict, k: int) -> float:
    if "recall@k" in row:
        return _num(row["recall@k"])
    gold = set(row.get("passagens_ouro") or [])
    if not gold:
        return float("nan")
    return len(gold & set((row.get("recuperadas") or [])[:k])) / len(gold)


def _stage_tokens(row: dict, prefix: str) -> float:
    stages = (row.get("uso_llm") or {}).get("por_estagio") or {}
    return sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
               for name, v in stages.items() if name.startswith(prefix))


def summarize(label: str, root: Path, keep: set[str] | None = None) -> dict:
    method, rows = load(root)
    if keep is not None:
        rows = {qid: row for qid, row in rows.items() if qid in keep}
    settings = (json.loads((root / "pilot.json").read_text()).get("settings", {})
                if (root / "pilot.json").exists() else {})
    any_row = next(iter(rows.values()))
    k = int(settings.get("top_k") or any_row.get("k_leitor") or len(any_row.get("recuperadas", [])))
    chunk = int(settings.get("locomo_chunk_tokens") or 2048)

    def block(items: list[dict]) -> dict:
        return {"n": len(items),
                "f1": _avg(100 * _num(r.get("f1_locomo")) for r in items),
                "bleu1": _avg(100 * _num(r.get("bleu1_locomo")) for r in items),
                "recall_k": _avg(100 * _recall(r, k) for r in items)}

    values = list(rows.values())
    return {"label": label, "path": str(root), "method": method, "k": k, "chunk_tokens": chunk,
            "reader_tokens": _avg(((r.get("uso_llm") or {}).get("por_estagio") or {})
                                  .get("qa", {}).get("tokens_prompt", float("nan")) for r in values),
            "controller_tokens": _avg(_stage_tokens(r, "witness.") for r in values),
            # Design v4 may keep the passages and add proof excerpts instead.
            "changed_context": _avg(float(bool((r.get("diagnosticos") or {})
                                               .get("contexto_alterado_pelo_witness")
                                               or (r.get("diagnosticos") or {})
                                               .get("contexto_alterado_por_trechos")))
                                    for r in values),
            "overall": block(values),
            "by_category": {c: block([r for r in values if r.get("tipo") == c]) for c in CATEGORIES},
            "_rows": rows}


def paired_delta(reference: dict, other: dict, resamples: int = 2000, seed: int = 42) -> dict:
    shared = sorted(set(reference["_rows"]) & set(other["_rows"]))
    by_conv: dict[str, list[float]] = {}
    for qid in shared:
        a = _num(reference["_rows"][qid].get("f1_locomo"))
        b = _num(other["_rows"][qid].get("f1_locomo"))
        if a == a and b == b:
            by_conv.setdefault(qid.split(":")[1] if ":" in qid else "all", []).append(100 * (b - a))
    diffs = [d for values in by_conv.values() for d in values]
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    convs = sorted(by_conv)
    samples = []
    for _ in range(resamples):
        pick = [by_conv[rng.choice(convs)] for _ in convs]
        flat = [d for values in pick for d in values]
        samples.append(mean(flat))
    samples.sort()
    return {"n": len(diffs), "delta": mean(diffs),
            "ci95": (samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]),
            "piores": sum(d < 0 for d in diffs), "melhores": sum(d > 0 for d in diffs)}


def fmt(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=path; the first one is the reference")
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    parser.add_argument("--all-questions", action="store_true",
                        help="each row on its own questions instead of the shared set")
    args = parser.parse_args(argv)

    found = []
    for item in args.runs:
        label, _, path = item.partition("=")
        root = Path(path or label)
        if not root.exists():
            print(f"skipping {item}: not found", file=sys.stderr)
            continue
        try:
            load(root)
        except (FileNotFoundError, ValueError) as exc:
            print(f"skipping {item}: {exc}", file=sys.stderr)
            continue
        found.append((label if path else root.name, root))
    if not found:
        raise SystemExit("no runs to report")
    keep = None
    if not args.all_questions:
        keep = set.intersection(*(set(load(root)[1]) for _label, root in found))
        if not keep:
            raise SystemExit("the runs share no question")
    runs = [summarize(label, root, keep) for label, root in found]
    reference = runs[0]
    runs.sort(key=lambda r: (r is not reference, r["method"] != reference["method"],
                             r["method"], -r["k"] * r["chunk_tokens"]))

    lines = ["# LoCoMo: orçamento do leitor", "",
             f"Perguntas: {reference['overall']['n']}"
             + (" (as que todas as rodadas responderam)" if keep is not None else "") + ". "
             f"Referência: {reference['label']}. ΔF1 pareado por pergunta, IC 95% por bootstrap "
             "de conversas. Tokens do leitor = prompt inteiro da resposta (média por pergunta); "
             "tokens do controlador = planejamento + verificação.", "",
             "| rodada | método | leitor | tokens leitor | tokens controlador | F1 ofic. | BLEU-1 | "
             "R@k | ΔF1 [IC 95%] | single | multi | temporal | open |",
             "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    report = {"referencia": reference["label"], "rodadas": []}
    for run in runs:
        delta = {"n": 0} if run is reference else paired_delta(reference, run)
        cats = run["by_category"]
        delta_text = ("—" if not delta.get("n") else
                      f"{delta['delta']:+.2f} [{delta['ci95'][0]:+.2f}, {delta['ci95'][1]:+.2f}]")
        lines.append(
            f"| {run['label']} | {run['method']} | {run['k']} × {run['chunk_tokens']} | "
            f"{fmt(run['reader_tokens'], 0)} | {fmt(run['controller_tokens'], 0)} | "
            f"{fmt(run['overall']['f1'])} | {fmt(run['overall']['bleu1'])} | "
            f"{fmt(run['overall']['recall_k'], 1)} | {delta_text} | "
            + " | ".join(fmt(cats[c]["f1"]) for c in CATEGORIES) + " |")
        report["rodadas"].append({**{k: v for k, v in run.items() if k != "_rows"}, "delta": delta})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output.with_suffix(".json"), report)
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
