"""Offline audit of reflection ablations; never calls a model or edits predictions.

Usage: python scripts/audit-reflection-runs.py RUN_ROOT --output AUDIT_ROOT
The comma replay is applied to every variant, without using gold answers or
categories. It diagnoses serialization losses; it is not a new benchmark run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paired_report", ROOT / "scripts/paired-report.py")
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
NAMES = ("cascade", "summary-reflection", "reader-reflection", "both")
METRICS = ("f1_locomo", "bleu1_locomo", "em_locomo")


def extra(row, title):
    return next((e.get("text", "") for e in row.get("diagnosticos", {}).get("trechos_extras", [])
                 if e.get("title") == title), "")


def comparison(candidate, reference, ids):
    result = {}
    for category in ("all", *paired.CATS):
        subset = [q for q in ids if category == "all" or reference[q]["tipo"] == category]
        if not subset:
            continue
        values = {}
        for metric in METRICS:
            low, high, unit = paired.paired_interval(candidate, reference, subset, metric)
            values[metric] = {
                "mean": 100 * st.mean(candidate[q][metric] for q in subset),
                "reference_mean": 100 * st.mean(reference[q][metric] for q in subset),
                "delta": 100 * st.mean(candidate[q][metric] - reference[q][metric] for q in subset),
                "ci95": [low, high], "bootstrap_unit": unit,
            }
        values.update(n=len(subset),
                      wins=sum(candidate[q]["f1_locomo"] > reference[q]["f1_locomo"] + 1e-9 for q in subset),
                      losses=sum(candidate[q]["f1_locomo"] < reference[q]["f1_locomo"] - 1e-9 for q in subset))
        result[category] = values
    return result


def context_comparison(a, b, ids):
    counts, examples = Counter(), []
    for q in ids:
        left, right = a[q].get("diagnosticos", {}), b[q].get("diagnosticos", {})
        checks = {
            "route": (left.get("roteador", {}).get("rota"), right.get("roteador", {}).get("rota")),
            "facts": (extra(a[q], "Facts from the memory"), extra(b[q], "Facts from the memory")),
            "summary_selection": (left.get("fatos_entregues", {}).get("resumos_selecionados"),
                                  right.get("fatos_entregues", {}).get("resumos_selecionados")),
            "summaries": (extra(a[q], "Chunk summaries"), extra(b[q], "Chunk summaries")),
        }
        different = [key for key, (x, y) in checks.items() if x != y]
        counts.update(different)
        if different:
            counts["questions_with_any_difference"] += 1
            if len(examples) < 10:
                examples.append({"qid": q, "fields": different})
    return {"n": len(ids), "counts": dict(counts), "examples": examples}


def behavior(rows, ids):
    reflective = [rows[q].get("reflexao_leitor") for q in ids if rows[q].get("reflexao_leitor")]
    durations = [q for q in ids if re.search(r"\bhow long\b", rows[q]["pergunta"], re.I)]
    bare_durations = [q for q in durations if re.fullmatch(r"\d+(?:\.\d+)?", rows[q]["resposta"].strip())]
    counts, unique = [], {}
    for q in ids:
        memories = rows[q].get("diagnosticos", {}).get("fatos_entregues", {}).get("summary_reflection", [])
        counts.append(sum(m.get("accepted", 0) for m in memories))
        for memory in memories:
            unique[(q.split(":")[1], memory["pid"])] = memory
    return {
        "answer_kind": dict(Counter(r.get("answer_kind", "") for r in reflective)),
        "invalid_schema": sum(not r.get("schema_valid", False) for r in reflective),
        "insufficient_information_answers": sum(rows[q]["resposta"].strip().casefold() == "insufficient information" for q in ids),
        "empty_answers": sum(not rows[q]["resposta"].strip() for q in ids),
        "mean_whitespace_words": st.mean(len(rows[q]["resposta"].split()) for q in ids),
        "mean_reader_tokens": st.mean(paired.tokens(rows[q], "qa") for q in ids),
        "mean_inference_tokens": st.mean(paired.tokens(rows[q], None) for q in ids),
        "how_long_questions": len(durations), "bare_numeric_how_long": len(bare_durations),
        "bare_numeric_how_long_examples": bare_durations,
        "questions_with_inferred_memories": sum(c > 0 for c in counts),
        "mean_memories_delivered": st.mean(counts),
        "unique_selected_chunks": len(unique),
        "unique_selected_memories_accepted": sum(m.get("accepted", 0) for m in unique.values()),
        "unique_selected_memories_rejected": sum(m.get("rejected", 0) for m in unique.values()),
        "unique_selected_chunk_status": dict(Counter(m.get("status") for m in unique.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {name: paired.load(args.run_root / name) for name in NAMES}
    ids = sorted(rows["cascade"])
    for name in NAMES[1:]:
        paired.paired_ids(rows[name], rows["cascade"])
    spaced = {}
    sources = {name: {} for name in NAMES}
    input_files, manifests = {}, {}
    for name in NAMES:
        spaced[name] = {}
        for q, row in rows[name].items():
            # Only letters after commas; numeric separators such as 1,000 stay intact.
            prediction = re.sub(r",(?=[^\W\d_])", ", ", row["resposta"])
            item = {**row, "resposta": prediction}
            item.update(paired.score_record(item))
            spaced[name][q] = item
        for path in sorted((args.run_root / name).glob("**/locomo/*.jsonl")):
            input_files[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if text.strip():
                    sources[name][json.loads(text)["qid"]] = {"path": path.as_posix(), "line": line}
        manifests[name] = [
            {"path": path.as_posix(), "llm": data.get("llm"), "codigo_hash": data.get("codigo_hash")}
            for path in sorted((args.run_root / name).glob("**/run.json"))
            for data in [json.loads(path.read_text(encoding="utf-8"))]
        ]
    report = {
        "protocol": "reflection-offline-audit-v1", "questions": len(ids),
        "note": "No LLM calls. Comma replay changes prediction whitespace only, equally for all variants; original runs stay untouched.",
        "inputs_sha256": input_files, "run_manifests": manifests,
        "raw": {name: comparison(rows[name], rows["cascade"], ids) for name in NAMES},
        "comma_replay": {name: comparison(spaced[name], spaced["cascade"], ids) for name in NAMES},
        "format_effect": {name: comparison(spaced[name], rows[name], ids) for name in NAMES},
        "behavior": {name: behavior(rows[name], ids) for name in NAMES},
        "context_pairs": {}, "reader_conditional": {}, "serialization": {},
    }
    for base, candidate in (("cascade", "reader-reflection"), ("summary-reflection", "both"),
                            ("cascade", "summary-reflection"), ("reader-reflection", "both")):
        label = base + " -> " + candidate
        report["context_pairs"][label] = context_comparison(rows[base], rows[candidate], ids)
        if candidate in {"reader-reflection", "both"} and base in {"cascade", "summary-reflection"}:
            equal_ids = [q for q in ids if
                         all(extra(rows[base][q], title) == extra(rows[candidate][q], title)
                             for title in ("Facts from the memory", "Chunk summaries")) and
                         rows[base][q]["diagnosticos"].get("roteador", {}).get("rota") ==
                         rows[candidate][q]["diagnosticos"].get("roteador", {}).get("rota")]
            report["reader_conditional"][label] = {
                "raw": comparison(rows[candidate], rows[base], ids),
                "comma_replay": comparison(spaced[candidate], spaced[base], ids),
                "exact_context_n": len(equal_ids),
                "exact_context_raw": comparison(rows[candidate], rows[base], equal_ids),
                "exact_context_comma_replay": comparison(spaced[candidate], spaced[base], equal_ids),
            }
    for name in NAMES:
        changed = [q for q in ids if rows[name][q]["resposta"] != spaced[name][q]["resposta"]]
        report["serialization"][name] = {
            "answers_changed": len(changed),
            "f1_increased": sum(spaced[name][q]["f1_locomo"] > rows[name][q]["f1_locomo"] + 1e-9 for q in changed),
            "f1_decreased": sum(spaced[name][q]["f1_locomo"] < rows[name][q]["f1_locomo"] - 1e-9 for q in changed),
            "examples": changed[:15],
        }
    selected = set()
    for gold_term in ("pomodoro", "uno", "canada", "c.s. lewis"):
        selected.update(q for q in ids if rows["cascade"][q]["tipo"] == "open-domain"
                        and gold_term in rows["cascade"][q]["respostas_ouro"][0].casefold())
    selected.update(("locomo:conv-48:qa16", "locomo:conv-47:qa141", "locomo:conv-43:qa81",
                     "locomo:conv-43:qa148", "locomo:conv-48:qa98", "locomo:conv-42:qa93"))
    for base, candidate in (("cascade", "reader-reflection"), ("summary-reflection", "both"),
                            ("cascade", "summary-reflection")):
        delta = lambda q: rows[candidate][q]["f1_locomo"] - rows[base][q]["f1_locomo"]
        for category in ("open-domain", "temporal", "multi-hop"):
            subset = [q for q in ids if rows[base][q]["tipo"] == category]
            selected.update(sorted(subset, key=lambda q: (delta(q), q))[:3])
            selected.update(sorted(subset, key=lambda q: (-delta(q), q))[:3])
    cases = []
    for q in sorted(selected):
        if q not in rows["cascade"]:
            continue
        base = rows["cascade"][q]
        cases.append({"qid": q, "question": base["pergunta"], "gold": base["respostas_ouro"],
                      "category": base["tipo"], "variants": {
                          name: {"answer": rows[name][q]["resposta"],
                                 "answer_spaced": spaced[name][q]["resposta"],
                                 "f1": rows[name][q]["f1_locomo"],
                                 "f1_spaced": spaced[name][q]["f1_locomo"],
                                 "reader_reflection": rows[name][q].get("reflexao_leitor"),
                                 "source": sources[name][q],
                                 "facts": extra(rows[name][q], "Facts from the memory"),
                                 "summaries": extra(rows[name][q], "Chunk summaries")}
                          for name in NAMES}})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output / "questions.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["qid", "category", "question", "gold"] +
                        [name + suffix for name in NAMES for suffix in ("_answer", "_f1", "_f1_spaced")])
        for q in ids:
            base = rows["cascade"][q]
            writer.writerow([q, base["tipo"], base["pergunta"], base["respostas_ouro"][0]] +
                            [value for name in NAMES for value in
                             (rows[name][q]["resposta"], rows[name][q]["f1_locomo"], spaced[name][q]["f1_locomo"])])
    print(json.dumps({"questions": len(ids), "cases": len(cases), "output": args.output.as_posix(),
                      "context_pairs": report["context_pairs"], "serialization": report["serialization"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
