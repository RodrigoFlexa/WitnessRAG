#!/usr/bin/env python3
"""Compare the single-plan and multi-plan WitnessRAG runs on identical questions.

The default is deliberately strict: both runs must be complete, use the same
code and configuration (apart from the planning switch), and contain exactly
the question IDs declared by the corpus manifest.

    python scripts/compare-planning.py runs/locomo-plan-ablation-...-simple \
        runs/locomo-plan-ablation-...-multi

Use ``--allow-partial`` only for diagnosis.  Partial results are explicitly
labelled and are compared on the intersection of completed question IDs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


METRICS = (
    ("f1_locomo", "F1ofic"),
    ("em_locomo", "EMofic"),
    ("f1", "F1"),
    ("em", "EM"),
    ("recall@5", "R@5"),
    ("all_recall@5", "AR@5"),
)


def benchmark_dir(path: Path) -> Path:
    path = path.resolve()
    if (path / "run.json").is_file():
        return path
    candidates = [p for p in (path / "benchmark").glob("*") if (p / "run.json").is_file()]
    if not candidates:
        raise ValueError(f"{path}: nenhuma rodada de benchmark encontrada")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "locomo" / "witnessrag.jsonl"
    if not path.is_file():
        raise ValueError(f"{path}: resultados do WitnessRAG ausentes")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    rows = {record["qid"]: record for record in records}
    if len(rows) != len(records):
        raise ValueError(f"{path}: há qids duplicados")
    return rows


def planning_enabled(manifest: dict[str, Any]) -> bool:
    return bool(manifest.get("config", {}).get("witness", {}).get("query_plans", False))


def comparable_config(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove only the fields that define the planning ablation."""
    config = copy.deepcopy(manifest.get("config", {}))
    witness = config.setdefault("witness", {})
    witness.pop("query_plans", None)
    witness.pop("max_query_plans", None)
    return config


def validate(simple: Path, multi: Path, allow_partial: bool) -> tuple[set[str], list[str]]:
    manifests = [read_json(simple / "run.json"), read_json(multi / "run.json")]
    if planning_enabled(manifests[0]) == planning_enabled(manifests[1]):
        raise ValueError("as rodadas devem usar modos diferentes: uma simples e uma com --query-plans")
    if planning_enabled(manifests[0]):
        raise ValueError("informe primeiro a rodada simples e depois a rodada com múltiplos planos")

    differences: list[str] = []
    if comparable_config(manifests[0]) != comparable_config(manifests[1]):
        differences.append("configuração diferente além do modo de planejamento")
    for field in ("backend", "deployment"):
        if manifests[0].get("llm", {}).get(field) != manifests[1].get("llm", {}).get(field):
            differences.append(f"LLM.{field} diferente")
    if manifests[0].get("embedder", {}).get("chave") != manifests[1].get("embedder", {}).get("chave"):
        differences.append("embedder diferente")
    if manifests[0].get("codigo_hash") != manifests[1].get("codigo_hash"):
        differences.append("versão do código diferente")

    corpora = [read_json(run / "locomo" / "corpus.json") for run in (simple, multi)]
    for field in ("corpus_hash", "questions_hash"):
        if not corpora[0].get(field) or corpora[0].get(field) != corpora[1].get(field):
            differences.append(f"{field} diferente")
    if differences:
        raise ValueError("comparação inválida: " + "; ".join(differences))

    rows = [load_rows(simple), load_rows(multi)]
    expected = set(corpora[0].get("question_ids", []))
    if not expected:
        raise ValueError("manifesto não contém a lista esperada de perguntas")
    missing = [expected - set(block) for block in rows]
    extras = [set(block) - expected for block in rows]
    notices = []
    if any(missing) or any(extras):
        message = (f"rodada incompleta: simples={len(rows[0])}/{len(expected)}, "
                   f"múltiplos={len(rows[1])}/{len(expected)}")
        if not allow_partial:
            raise ValueError(message + "; retome as execuções ou use --allow-partial para diagnóstico")
        notices.append("PARCIAL — " + message)
    paired = set(rows[0]) & set(rows[1]) & expected
    if not paired:
        raise ValueError("nenhuma pergunta concluída em ambas as rodadas")
    return paired, notices


def mean(rows: dict[str, dict[str, Any]], qids: set[str], field: str) -> float:
    return 100.0 * sum(float(rows[qid].get(field, 0.0)) for qid in qids) / len(qids)


def fired(record: dict[str, Any]) -> bool:
    return "fallback" not in record.get("diagnosticos", {})


def changed(record: dict[str, Any]) -> bool:
    return bool(record.get("diagnosticos", {}).get("contexto_alterado_pelo_witness", False))


def table(simple_rows: dict[str, dict[str, Any]], multi_rows: dict[str, dict[str, Any]],
          qids: set[str], category: str | None = None) -> None:
    if category is not None:
        qids = {qid for qid in qids
                if simple_rows[qid].get("tipo") == category
                and multi_rows[qid].get("tipo") == category}
    if not qids:
        return
    label = "todas" if category is None else category
    print(f"\n{label} — {len(qids)} perguntas pareadas")
    headers = [name for _, name in METRICS] + ["disparo", "mudou"]
    print(f"  {'planejamento':<12} {'n':>4} " + " ".join(f"{name:>8}" for name in headers))
    print("  " + "-" * (18 + 9 * len(headers)))
    for name, rows in (("simples", simple_rows), ("múltiplos", multi_rows)):
        values = [mean(rows, qids, field) for field, _ in METRICS]
        values += [100 * sum(fired(rows[qid]) for qid in qids) / len(qids),
                   100 * sum(changed(rows[qid]) for qid in qids) / len(qids)]
        print(f"  {name:<12} {len(qids):>4} " + " ".join(f"{value:>8.2f}" for value in values))

    deltas = [mean(multi_rows, qids, field) - mean(simple_rows, qids, field)
              for field, _ in METRICS]
    print(f"  {'delta':<12} {'':>4} " + " ".join(f"{value:>+8.2f}" for value in deltas))
    differences = [multi_rows[qid].get("f1_locomo", 0.0)
                   - simple_rows[qid].get("f1_locomo", 0.0) for qid in qids]
    wins = sum(value > 1e-12 for value in differences)
    losses = sum(value < -1e-12 for value in differences)
    print(f"  F1 oficial por pergunta: ganha {wins}, perde {losses}, "
          f"empata {len(differences) - wins - losses}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("simple", type=Path, help="rodada sem --query-plans")
    parser.add_argument("multi", type=Path, help="rodada com --query-plans")
    parser.add_argument("--allow-partial", action="store_true",
                        help="compara a interseção incompleta apenas para diagnóstico")
    args = parser.parse_args()
    try:
        simple, multi = benchmark_dir(args.simple), benchmark_dir(args.multi)
        paired, notices = validate(simple, multi, args.allow_partial)
        simple_rows, multi_rows = load_rows(simple), load_rows(multi)
    except ValueError as exc:
        parser.error(str(exc))
    for notice in notices:
        print(notice)
    table(simple_rows, multi_rows, paired)
    for category in ("single-hop", "multi-hop"):
        table(simple_rows, multi_rows, paired, category)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
