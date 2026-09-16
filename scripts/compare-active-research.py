#!/usr/bin/env python3
"""Strict paired report for the active proof-research LoCoMo matrix.

Usage: python scripts/compare-active-research.py runs/locomo-proof-ablation-...
The launcher writes one directory per variant under the root.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from statistics import mean


VARIANTS = ("control", "frontier", "obligations", "frontier-obligations",
            "evidence", "operators", "verified", "no-acquisition", "one-plan")
WITNESS_SWITCHES = ("active_frontier", "active_obligations", "active_context",
                    "active_operators", "verify_witnesses", "enable_acquisition",
                    "query_plans", "max_query_plans")
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def numeric_value(value: str) -> int | None:
    tokens = re.findall(r"\b(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
                        value.casefold())
    if len(tokens) != 1:
        return None
    return int(tokens[0]) if tokens[0].isdigit() else WORDS[tokens[0]]


def read_variant(path: Path) -> tuple[dict, dict[str, dict]]:
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"{path}: status {status.get('status')!r}; retome a rodada")
    rows: dict[str, dict] = {}
    identity = None
    conversations = sorted((path / "conversations").glob("conv*/benchmark/*"))
    if not conversations:
        raise ValueError(f"{path}: nenhuma conversa encontrada")
    pilot = json.loads((path / "pilot.json").read_text(encoding="utf-8"))
    requested = pilot.get("settings", {}).get("locomo_conversation")
    expected_conversations = 10 if requested == "all" else 1
    if len(conversations) != expected_conversations:
        raise ValueError(f"{path}: {len(conversations)}/{expected_conversations} conversas")
    for run in conversations:
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
        corpus = json.loads((run / "locomo" / "corpus.json").read_text(encoding="utf-8"))
        data = [json.loads(line) for line in (run / "locomo" / "witnessrag.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = set(corpus.get("question_ids", []))
        actual = {item["qid"] for item in data}
        if len(actual) != len(data) or actual != expected:
            raise ValueError(f"{run}: perguntas ausentes, excedentes ou duplicadas")
        if rows.keys() & actual:
            raise ValueError(f"{path}: qids duplicados entre conversas")
        rows.update({item["qid"]: item for item in data})
        config = copy.deepcopy(manifest["config"])
        config.pop("n_questions", None)  # each conversation has its own denominator
        for name in WITNESS_SWITCHES:
            config["witness"].pop(name, None)
        config["qa"].pop("operator_reader", None)
        record = {"conversation": run.parent.parent.name,
                  "corpus_hash": corpus.get("corpus_hash"),
                  "questions_hash": corpus.get("questions_hash"),
                  "code": manifest.get("codigo_hash"),
                  "llm": manifest.get("llm", {}).get("deployment"),
                  "embedder": manifest.get("embedder", {}).get("chave"),
                  "config": config}
        if identity is None:
            identity = {"code": record["code"], "llm": record["llm"],
                        "embedder": record["embedder"], "config": config,
                        "corpora": []}
        elif any(record[k] != identity[k] for k in ("code", "llm", "embedder", "config")):
            raise ValueError(f"{path}: configuração mudou entre conversas")
        identity["corpora"].append((record["conversation"], record["corpus_hash"],
                                     record["questions_hash"]))
    identity["corpora"].sort()
    return identity, rows


def score(rows: list[dict], field: str) -> float:
    return mean(float(row.get(field, 0.0)) for row in rows) if rows else 0.0


def summarize(rows: list[dict]) -> dict:
    multi = [row for row in rows if row.get("tipo") == "multi-hop"]
    counts = [row for row in multi if re.search(r"^\s*how many\b",
              row.get("pergunta", ""), re.I)]
    numeric = [(numeric_value(row.get("resposta", "")),
                numeric_value(row["respostas_ouro"][0])) for row in counts]
    evaluable = [(a, b) for a, b in numeric if a is not None and b is not None]
    return {"n": len(rows), "multi_n": len(multi),
            "f1_oficial": score(rows, "f1_locomo"),
            "f1_multi": score(multi, "f1_locomo"),
            "ar5_multi": score(multi, "all_recall@5"),
            "f1_single": score([r for r in rows if r.get("tipo") == "single-hop"],
                               "f1_locomo"),
            "disparo_multi": mean("fallback" not in r.get("diagnosticos", {})
                                  for r in multi) if multi else 0.0,
            "contagens_n": len(counts),
            "f1_contagens": score(counts, "f1_locomo"),
            "acerto_numerico_avaliavel": sum(a == b for a, b in evaluable) / len(evaluable)
                                        if evaluable else None,
            "contagens_numericas_avaliaveis": len(evaluable),
            "latencia_recuperacao_multi_s": score(multi, "latencia_recuperacao_s"),
            "latencia_leitura_multi_s": score(multi, "latencia_leitura_s"),
            "buscas_dirigidas_multi": mean(r.get("diagnosticos", {})
                .get("pesquisa_provas", {}).get("buscas_dirigidas", 0) for r in multi)
                if multi else 0.0}


def compare(root: Path, variants: tuple[str, ...] = VARIANTS) -> dict:
    loaded = {name: read_variant(root / name) for name in variants}
    baseline_id, baseline_rows = loaded[variants[0]]
    for name, (identity, rows) in loaded.items():
        if identity != baseline_id:
            raise ValueError(f"{name}: código, corpus, modelo ou configuração não comparável")
        if set(rows) != set(baseline_rows):
            raise ValueError(f"{name}: conjunto de perguntas diferente")
    report = {"root": str(root.resolve()), "variants": {}, "paired": {}}
    for name, (_, rows) in loaded.items():
        report["variants"][name] = summarize(list(rows.values()))
        if name == variants[0]:
            continue
        qids = [qid for qid, row in baseline_rows.items() if row.get("tipo") == "multi-hop"]
        delta = [rows[qid]["f1_locomo"] - baseline_rows[qid]["f1_locomo"] for qid in qids]
        report["paired"][name] = {"delta_f1_multi": mean(delta),
                                   "wins": sum(x > 1e-9 for x in delta),
                                   "losses": sum(x < -1e-9 for x in delta),
                                   "ties": sum(abs(x) <= 1e-9 for x in delta)}
    return report


def markdown(report: dict) -> str:
    lines = ["# Ablação da pesquisa de provas", "",
             f"Rodadas: `{report['root']}`", "",
             "| Condição | n | F1 oficial | F1 multi | Δ multi | AR@5 multi | "
             "F1 contagem | Acerto numérico¹ | Disparo multi | Busca multi (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["variants"].items():
        paired = report["paired"].get(name, {})
        numeric = item["acerto_numerico_avaliavel"]
        lines.append(f"| {name} | {item['n']} | {item['f1_oficial']:.4f} | "
                     f"{item['f1_multi']:.4f} | {paired.get('delta_f1_multi', 0):+.4f} | "
                     f"{item['ar5_multi']:.4f} | {item['f1_contagens']:.4f} | "
                     f"{numeric:.4f} ({item['contagens_numericas_avaliaveis']}) | "
                     f"{item['disparo_multi']:.4f} | "
                     f"{item['latencia_recuperacao_multi_s']:.2f} |" if numeric is not None else
                     f"| {name} | {item['n']} | {item['f1_oficial']:.4f} | "
                     f"{item['f1_multi']:.4f} | {paired.get('delta_f1_multi', 0):+.4f} | "
                     f"{item['ar5_multi']:.4f} | {item['f1_contagens']:.4f} | — | "
                     f"{item['disparo_multi']:.4f} | "
                     f"{item['latencia_recuperacao_multi_s']:.2f} |")
    lines.extend(["", "¹ Diagnóstico adicional: só conta respostas/gabaritos com um número "
                  "único reconhecível; não substitui a métrica oficial.", "",
                  "Comparação pareada (multi-hop, contra control):", ""])
    for name, item in report["paired"].items():
        lines.append(f"- `{name}`: {item['wins']} ganhos, {item['losses']} perdas, "
                     f"{item['ties']} empates.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    report = compare(args.root, tuple(args.variants))
    root = args.root
    (root / "ablation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    rendered = markdown(report)
    (root / "ablation.md").write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
