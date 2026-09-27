"""Paired comparison of WitnessRAG runs on the same LoCoMo questions.

    python scripts/paired-report.py runs/conv05 witnessrag wr-plan wr-fill ...

The first name is the reference. Reads <root>/<name>/**/locomo/witnessrag.jsonl,
pairs the questions, and writes <root>/compare.md and compare.json: official
LoCoMo F1 by category, the paired difference to the reference with a 95%
bootstrap interval over QUESTIONS (one conversation: there is nothing else to
resample), wins/losses, tokens per question (reader and total, without the
memory construction) and planning statistics (cycles, proofs, readings used,
relation alternatives written).
"""
from __future__ import annotations

import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrag.eval.locomo_official import score_record  # noqa: E402

CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
BUILD = {"memory.summary", "index.openie", "index.ner"}


def load(folder: Path) -> dict[str, dict]:
    rows = {}
    for path in glob.glob(str(folder / "**" / "locomo" / "*.jsonl"), recursive=True):
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


def plan_stats(rows: dict[str, dict], ids: list[str]) -> dict[str, float]:
    cycles = proofs = with_alt = with_readings = reading_used = plans = 0
    for q in ids:
        d = rows[q].get("diagnosticos") or {}
        p = d.get("planejamento") or {}
        cycles += p.get("chamadas_plano", 0)
        proofs += d.get("motivo_parada") in {"prova_confirmada", "prova_ja_no_contexto",
                                             "prova_sem_verificacao"}
        final = d.get("plano_final") or {}
        if final.get("leitura"):
            reading_used += 1
        for c in d.get("ciclos") or []:
            plan = c.get("plano") or {}
            atoms = ((plan.get("consulta") or {}).get("atoms")) or []
            if not atoms:
                continue
            plans += 1
            with_alt += any(a.get("alternatives") for a in atoms)
            with_readings += bool(plan.get("outras_leituras"))
    n = max(1, len(ids))
    return {"planos_por_pergunta": round(cycles / n, 2),
            "perguntas_com_prova_%": round(100 * proofs / n, 1),
            "planos_com_alternativas_%": round(100 * with_alt / max(1, plans), 1),
            "planos_com_outras_leituras_%": round(100 * with_readings / max(1, plans), 1),
            "prova_por_outra_leitura": reading_used}


def main() -> None:
    root = Path(sys.argv[1])
    names = sys.argv[2:]
    data = {n: load(root / n) for n in names if (root / n).exists()}
    ref_name = names[0]
    ref = data[ref_name]
    random.seed(0)
    f1 = lambda R, I: 100 * st.mean(R[q]["f1_locomo"] for q in I) if I else float("nan")
    lines = ["| variante | n | F1 | Δ F1 [IC 95%] | ganhos/perdas | " + " | ".join(CATS)
             + " | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |",
             "|---|---:|---:|---|---|" + "---:|" * len(CATS) + "---:|---:|---:|---:|---:|"]
    out = {}
    for name, rows in data.items():
        ids = sorted(set(rows) & set(ref))
        if not ids:
            continue
        delta = f1(rows, ids) - f1(ref, ids)
        boots = sorted(f1(rows, s) - f1(ref, s)
                       for s in (random.choices(ids, k=len(ids)) for _ in range(4000)))
        lo, hi = boots[100], boots[3899]
        wins = sum(rows[q]["f1_locomo"] > ref[q]["f1_locomo"] + 1e-9 for q in ids)
        losses = sum(rows[q]["f1_locomo"] < ref[q]["f1_locomo"] - 1e-9 for q in ids)
        cats = {c: f1(rows, [q for q in ids if rows[q].get("tipo") == c]) for c in CATS}
        reader = st.mean(tokens(rows[q], "qa") for q in ids)
        total = st.mean(tokens(rows[q], None) for q in ids)
        stats = plan_stats(rows, ids)
        ci = "" if name == ref_name else f"{delta:+.2f} [{lo:+.2f}; {hi:+.2f}]"
        wl = "" if name == ref_name else f"{wins}/{losses}"
        lines.append(f"| {name} | {len(ids)} | {f1(rows, ids):.2f} | {ci} | {wl} | "
                     + " | ".join(f"{cats[c]:.1f}" for c in CATS)
                     + f" | {reader:,.0f} | {total:,.0f} | {stats['planos_por_pergunta']}"
                     + f" | {stats['perguntas_com_prova_%']}% | {stats['prova_por_outra_leitura']} |")
        out[name] = {"n": len(ids), "f1": f1(rows, ids), "delta": delta, "ic95": [lo, hi],
                     "ganhos": wins, "perdas": losses, "por_categoria": cats,
                     "tokens_leitor": reader, "tokens_total": total, **stats}
    counts = {c: sum(ref[q].get("tipo") == c for q in ref) for c in CATS}
    header = (f"# Comparação pareada ({root.name})\n\nF1 oficial do LoCoMo; Δ contra "
              f"`{ref_name}` nas mesmas perguntas; IC 95% por bootstrap de perguntas "
              f"(4000 reamostragens). Perguntas por categoria: "
              + ", ".join(f"{c} {n}" for c, n in counts.items())
              + ". Tokens por pergunta sem a construção da memória.\n\n")
    (root / "compare.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    (root / "compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(header + "\n".join(lines))


if __name__ == "__main__":
    main()
