"""Tabela da ablação do WitnessRAG (docs/ablacao.md).

    python scripts/ablation-report.py runs/ablation-openai-c512-k20

Lê <raiz>/<variante>/**/locomo/witnessrag.jsonl, pareia as perguntas com o
completo (<raiz>/full) e escreve <raiz>/ablation.md e ablation.json: F1 oficial
do LoCoMo por categoria, diferença para o completo com IC 95% por bootstrap de
conversas, e tokens por pergunta (leitor e total, sem a construção da memória).
"""
from __future__ import annotations

import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrag.eval.locomo_official import score_record  # noqa: E402

ORDER = ["full", "no-plan", "no-proof", "no-verify", "no-temporal-score"]
NAMES = {"full": "WitnessRAG (completo)", "no-plan": "sem plano", "no-proof": "sem prova",
         "no-verify": "sem verificação", "no-temporal-score": "sem nota temporal"}
CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
BUILD = {"memory.summary", "index.openie", "index.ner"}


def load(folder: Path) -> dict[str, dict]:
    rows = {}
    for path in glob.glob(str(folder / "**" / "locomo" / "witnessrag.jsonl"), recursive=True):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if "f1_locomo" not in r:
                    r.update(score_record(r))
                rows[r["qid"]] = r
    return rows


def tokens(r: dict, stage: str | None) -> float:
    stages = (r.get("uso_llm") or {}).get("por_estagio") or {}
    if stage:
        s = stages.get(stage) or {}
        return s.get("tokens_prompt", 0) + s.get("tokens_resposta", 0)
    return sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
               for k, v in stages.items() if k not in BUILD)


def main() -> None:
    root = Path(sys.argv[1])
    data = {v: load(root / v) for v in ORDER if (root / v).exists()}
    if "full" not in data:
        sys.exit("falta a variante 'full' (o WitnessRAG completo)")
    full = data["full"]
    conv = lambda q: q.split(":")[1] if ":" in q else q
    random.seed(0)
    lines = ["| variante | n | F1 | Δ F1 [IC 95%] | " + " | ".join(CATS)
             + " | tokens leitor/perg. | tokens total/perg. |",
             "|---|---:|---:|---|" + "---:|" * len(CATS) + "---:|---:|"]
    out = {}
    for variant, rows in data.items():
        ids = sorted(set(rows) & set(full))
        if not ids:
            continue
        f1 = lambda R, I: 100 * st.mean(R[q]["f1_locomo"] for q in I) if I else float("nan")
        by = defaultdict(list)
        for q in ids:
            by[conv(q)].append(q)
        deltas = sorted(
            f1(rows, s) - f1(full, s)
            for s in ([q for c in random.choices(list(by), k=len(by)) for q in by[c]]
                      for _ in range(2000)))
        delta = f1(rows, ids) - f1(full, ids)
        ci = "" if variant == "full" else f"{delta:+.2f} [{deltas[50]:+.2f}; {deltas[1950]:+.2f}]"
        cats = [f1(rows, [q for q in ids if rows[q].get("tipo") == c]) for c in CATS]
        reader = st.mean(tokens(rows[q], "qa") for q in ids)
        total = st.mean(tokens(rows[q], None) for q in ids)
        lines.append(f"| {NAMES.get(variant, variant)} | {len(ids)} | {f1(rows, ids):.2f} | {ci} | "
                     + " | ".join(f"{c:.1f}" for c in cats) + f" | {reader:,.0f} | {total:,.0f} |")
        out[variant] = {"n": len(ids), "f1": f1(rows, ids), "delta": delta,
                        "ic95": [deltas[50], deltas[1950]],
                        "por_categoria": dict(zip(CATS, cats)),
                        "tokens_leitor": reader, "tokens_total": total,
                        "conversas": len(by)}
    table = "\n".join(lines)
    print(table)
    (root / "ablation.md").write_text(
        f"# Ablação do WitnessRAG ({root.name})\n\nF1 oficial do LoCoMo; Δ contra o completo nas "
        "mesmas perguntas; IC 95% por bootstrap de conversas; tokens por pergunta sem a "
        "construção da memória.\n\n" + table + "\n", encoding="utf-8")
    (root / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
