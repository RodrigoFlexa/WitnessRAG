"""Read-only run audit; exports derived evidence, never changes predictions."""
import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wrag.eval.locomo_official import score_record


def load(folder):
    rows = {}
    for path in sorted(folder.glob("**/locomo/*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["qid"] in rows:
                raise ValueError(f"duplicate {row['qid']} in {folder}")
            row.update(score_record(row))
            row["_path"], row["_line"] = str(path), number
            rows[row["qid"]] = row
    return rows


def polarity(text):
    match = re.match(r"\s*(?:most\s+)?(?:likely\s+)?(yes|no)\b", text, re.I)
    return match[1].lower() if match else ""


def eval_gold(row):
    gold = row["respostas_ouro"][0]
    return gold.split(";")[0].strip() if row["tipo"] == "open-domain" else gold


def stats(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    return {"n": n, "f1": 100 * statistics.mean(r["f1_locomo"] for r in rows),
        "bleu": 100 * statistics.mean(r["bleu1_locomo"] for r in rows),
        "em": 100 * statistics.mean(r["em_locomo"] for r in rows),
        "empty": sum(not r.get("resposta", "").strip() for r in rows),
        "bare_polarity": sum(r.get("resposta", "").strip().lower() in {"yes", "no"} for r in rows),
        "minimal_polarity": sum(r.get("resposta", "").strip().lower() in {"yes", "no", "likely yes", "likely no"} for r in rows),
        "abstentions": sum("insufficient information" in r.get("resposta", "").lower() for r in rows),
        "zero_f1": sum(r["f1_locomo"] == 0 for r in rows),
        "gold_starts_polarity": sum(bool(polarity(eval_gold(r))) for r in rows),
        "gold_polarity_long": sum(bool(polarity(eval_gold(r))) and len(eval_gold(r).split()) > 2 for r in rows),
        "matching_polarity": sum(bool(polarity(eval_gold(r))) and polarity(r.get("resposta", "")) == polarity(eval_gold(r)) for r in rows),
        "median_answer_words": statistics.median(len(r.get("resposta", "").split()) for r in rows),
        "median_gold_words": statistics.median(len(r["respostas_ouro"][0].split()) for r in rows),
        "median_eval_gold_words": statistics.median(len(eval_gold(r).split()) for r in rows),
        "route": dict(Counter((r.get("diagnosticos", {}).get("roteador") or {}).get("rota", "missing") for r in rows)),
        "stops": dict(Counter(r.get("diagnosticos", {}).get("motivo_parada", "missing") for r in rows)),
        "hypothesis": sum(bool((r.get("diagnosticos", {}).get("plano_final") or {}).get("hipotese")) for r in rows),
        "premise_diagnostic": sum(bool(r.get("diagnosticos", {}).get("premissas")) for r in rows),
        "premise_delivered": sum(any("premise" in e.get("title", "").lower() for e in r.get("diagnosticos", {}).get("trechos_extras", [])) for r in rows),
        "proof": sum(r.get("diagnosticos", {}).get("testemunha_no_contexto", False) for r in rows)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "runs/open-domain-audit")
    p.add_argument("--limit-paired", type=int, help="prefixo cronológico para reproduzir uma tabela parcial")
    args = p.parse_args()
    if args.limit_paired is not None and args.limit_paired <= 0:
        p.error("--limit-paired deve ser positivo")
    folders = {"robust": ROOT / "runs/multiplan-comparison/witnessrag-robust",
               "cascade": ROOT / "runs/multiplan-comparison/witnessrag-cascade"}
    variants = {k: load(v) for k, v in folders.items()}
    common = set.intersection(*(set(v) for v in variants.values()))
    if not common:
        raise ValueError("No paired questions found")
    if args.limit_paired is not None and args.limit_paired > len(common):
        p.error("--limit-paired excede o conjunto pareado disponível")
    if args.limit_paired:
        common = set(sorted(common, key=lambda q: (variants["cascade"][q]["_path"],
                                                  variants["cascade"][q]["_line"]))[:args.limit_paired])
    for qid in common:
        for key in ["pergunta", "respostas_ouro", "tipo"]:
            if variants["robust"][qid].get(key) != variants["cascade"][qid].get(key):
                raise ValueError(f"mismatch {qid} {key}")
    summary = {"all_counts": {k: len(v) for k, v in variants.items()}, "paired_n": len(common),
               "category_counts": dict(Counter(variants["robust"][q]["tipo"] for q in common))}
    # Reuse the existing conversation-clustered bootstrap and token accounting.
    spec = importlib.util.spec_from_file_location("paired_report", ROOT / "scripts/paired-report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    summary["paired_difference"] = {}
    for category in ["all", "open-domain"]:
        ids = sorted(q for q in common if category == "all" or variants["robust"][q]["tipo"] == category)
        if not ids:
            continue
        ref, candidate = variants["robust"], variants["cascade"]
        interval = report.paired_interval(candidate, ref, ids, "f1_locomo")
        summary["paired_difference"][category] = {
            "n": len(ids), "f1_delta": 100 * statistics.mean(candidate[q]["f1_locomo"] - ref[q]["f1_locomo"] for q in ids),
            "ci95": interval[:2], "bootstrap_unit": interval[2],
            "wins": sum(candidate[q]["f1_locomo"] > ref[q]["f1_locomo"] for q in ids),
            "losses": sum(candidate[q]["f1_locomo"] < ref[q]["f1_locomo"] for q in ids),
            "ties": sum(candidate[q]["f1_locomo"] == ref[q]["f1_locomo"] for q in ids),
            "tokens": {name: {"reader": statistics.mean(report.tokens(rows[q], "qa") for q in ids),
                              "total": statistics.mean(report.tokens(rows[q], None) for q in ids)}
                       for name, rows in variants.items()}}
    for name, rows in variants.items():
        paired = [rows[q] for q in sorted(common)]
        od = [r for r in paired if r["tipo"] == "open-domain"]
        summary[name] = {"all": stats(paired), "open_domain_paired": stats(od),
                         "open_domain_full": stats([r for r in rows.values() if r["tipo"] == "open-domain"])}
    summary["cascade_routes_open"] = {}
    for route in ["PLAN", "DIRECT"]:
        ids = [q for q in common if variants["cascade"][q]["tipo"] == "open-domain"
               and (variants["cascade"][q].get("diagnosticos", {}).get("roteador") or {}).get("rota") == route]
        summary["cascade_routes_open"][route] = {k: stats([v[q] for q in ids]) for k, v in variants.items()}
    ablation = ROOT / "runs/ablation-qwen-c2048-k5"
    summary["ablation"] = {name: stats([r for r in load(ablation / name).values() if r["tipo"] == "open-domain"])
                           for name in ["full", "no-plan", "no-proof", "no-verify", "no-temporal-score"]}
    records = [{"qid": q, **{k: v[q] for k, v in variants.items()}}
               for q in sorted(common) if variants["robust"][q]["tipo"] == "open-domain"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "cases.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["qid\tquestion\tgold\trobust\tcascade\tF1 robust\tF1 cascade\troute"]
    for item in records:
        r, c = item["robust"], item["cascade"]
        lines.append("\t".join(map(str, [item["qid"], r["pergunta"], r["respostas_ouro"][0],
                    r["resposta"], c["resposta"], round(r["f1_locomo"] * 100, 2), round(c["f1_locomo"] * 100, 2),
                    (c.get("diagnosticos", {}).get("roteador") or {}).get("rota", "missing")])))
    (args.output / "questions.tsv").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "paired_n": len(common),
                      "open_domain_n": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
