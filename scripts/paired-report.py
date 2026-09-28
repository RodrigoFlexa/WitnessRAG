"""Paired comparison of WitnessRAG runs on the same LoCoMo questions.

    python scripts/paired-report.py runs/conv05 witnessrag wr-plan wr-fill ...

The first name is the reference. Reads <root>/<name>/**/locomo/*.jsonl or
<root>/<name>.jsonl exports,
pairs the questions, and writes <root>/compare.md and compare.json: official
LoCoMo F1 by category, the paired difference to the reference with a 95%
bootstrap interval over conversations (questions if only one conversation),
local BLEU-1, wins/losses, tokens per question (reader and total, without the
memory construction) and planning statistics (cycles, proofs, readings used,
relation alternatives written).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrag.eval.locomo_official import score_record  # noqa: E402

CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
BUILD = {"memory.summary", "memory.reflection", "index.openie", "index.ner"}


def load(folder: Path) -> dict[str, dict]:
    """Accept a single exported JSONL or a run tree; never silently overwrite IDs."""
    rows = {}
    paths = [folder] if folder.is_file() else sorted(folder.glob("**/locomo/*.jsonl"))
    if not paths:
        raise ValueError(f"No LoCoMo records found: {folder}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["qid"] in rows:
                    raise ValueError(f"Duplicate qid {r['qid']} in {folder}")
                # Recompute from predictions, even if an older metric is saved.
                r.update(score_record(r))
                rows[r["qid"]] = r
    if not rows:
        raise ValueError(f"No LoCoMo records found: {folder}")
    return rows


def paired_ids(rows: dict, ref: dict) -> list[str]:
    if rows.keys() != ref.keys():
        raise ValueError("Question sets differ; refusing a silent intersection")
    for qid in ref:
        for key in ("pergunta", "respostas_ouro", "tipo", "categoria_locomo"):
            if rows[qid].get(key) != ref[qid].get(key):
                raise ValueError(f"Mismatched {key} for {qid}")
    return sorted(ref)


def paired_interval(rows: dict, ref: dict, ids: list[str], metric: str,
                    samples: int = 4000) -> tuple[float, float, str]:
    """Micro-average delta; resample conversations together when >1 is available."""
    groups = {}
    for qid in ids:
        parts = qid.split(":")
        conversation = parts[1] if len(parts) >= 3 else "single-conversation"
        groups.setdefault(conversation, []).append(100 * (rows[qid][metric] - ref[qid][metric]))
    clustered = len(groups) > 1
    units = ([(sum(values), len(values)) for values in groups.values()] if clustered else
             [(value, 1) for values in groups.values() for value in values])
    rng = random.Random(0)
    boots = []
    for _ in range(samples):
        selected = rng.choices(units, k=len(units))
        boots.append(sum(s for s, n in selected) / sum(n for s, n in selected))
    boots.sort()
    return (boots[int(samples * .025)], boots[int(samples * .975) - 1],
            "conversations" if clustered else "questions")


def tokens(r: dict, stage: str | None) -> float:
    stages = (r.get("uso_llm") or {}).get("por_estagio") or {}
    if stage:
        s = stages.get(stage) or {}
        return s.get("tokens_prompt", 0) + s.get("tokens_resposta", 0)
    shared = (r.get("diagnosticos") or {}).get("reflection_study")
    if shared:
        retrieval = (shared.get("shared_retrieval_usage") or {}).get("por_estagio") or {}
        # Logical per-cell budget: one shared retrieval plus this cell's reader.
        # Actual construction is recorded in shared snapshots, never duplicated.
        return (sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
                    for k, v in retrieval.items() if k not in BUILD)
                + sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
                      for k, v in stages.items() if k == "qa" or k.startswith("qa.")))
    return sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
               for k, v in stages.items() if k not in BUILD)


def plan_stats(rows: dict[str, dict], ids: list[str]) -> dict[str, float]:
    cycles = proofs = with_alt = with_readings = reading_used = plans = 0
    distinct = verifications = cache_hits = delivered = conflicts = 0
    for q in ids:
        d = rows[q].get("diagnosticos") or {}
        p = d.get("planejamento") or {}
        distinct += p.get("planos_distintos", 0)
        verifications += p.get("chamadas_verificacao", 0)
        cache_hits += p.get("cache_atomos_hits", 0)
        delivered += d.get("n_testemunhas_no_contexto", 0)
        conflicts += str((d.get("operacao") or {}).get("operator_status", "")).startswith("ambiguous_or_conflicting")
        cycles += p.get("chamadas_plano", 0)
        proofs += (d.get("classe_prova") == "full" if d.get("controlador") == "contract-portfolio-v1"
                   else d.get("motivo_parada") in {"prova_confirmada", "prova_ja_no_contexto", "prova_sem_verificacao"})
        final = d.get("plano_final") or {}
        if final.get("leitura") and d.get("motivo_parada") in {
                "prova_confirmada", "prova_ja_no_contexto", "prova_sem_verificacao"}:
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
            "prova_por_outra_leitura": reading_used,
            "planos_distintos_por_pergunta": round(distinct / n, 2),
            "verificacoes_por_pergunta": round(verifications / n, 2),
            "cache_atomos_hits": cache_hits, "testemunhas_entregues": delivered,
            "conflitos_nao_resolvidos": conflicts}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, help="Separate report directory")
    args = parser.parse_args()
    root, names = args.root, args.names
    data = {n: load(root / n if (root / n).exists() else root / (n + ".jsonl"))
            for n in names}
    ref_name = names[0]
    ref = data[ref_name]
    f1 = lambda R, I: 100 * st.mean(R[q]["f1_locomo"] for q in I) if I else float("nan")
    lines = ["| variante | n | F1 | Δ F1 [IC 95%] | BLEU-1 local | Δ BLEU-1 [IC 95%] | ganhos/perdas | " + " | ".join(CATS)
             + " | tokens leitor | tokens total | planos/perg. | com prova | prova por outra leitura |",
             "|---|---:|---:|---|---:|---|---|" + "---:|" * len(CATS) + "---:|---:|---:|---:|---:|"]
    out = {}
    for name, rows in data.items():
        ids = paired_ids(rows, ref)
        delta = f1(rows, ids) - f1(ref, ids)
        lo, hi, unit = paired_interval(rows, ref, ids, "f1_locomo")
        bleu = 100 * st.mean(rows[q]["bleu1_locomo"] for q in ids)
        bleu_delta = bleu - 100 * st.mean(ref[q]["bleu1_locomo"] for q in ids)
        blo, bhi, _ = paired_interval(rows, ref, ids, "bleu1_locomo")
        wins = sum(rows[q]["f1_locomo"] > ref[q]["f1_locomo"] + 1e-9 for q in ids)
        losses = sum(rows[q]["f1_locomo"] < ref[q]["f1_locomo"] - 1e-9 for q in ids)
        cats = {c: f1(rows, [q for q in ids if rows[q].get("tipo") == c])
                if any(rows[q].get("tipo") == c for q in ids) else None for c in CATS}
        bleu_cats = {c: 100 * st.mean(rows[q]["bleu1_locomo"] for q in ids if rows[q].get("tipo") == c)
                     if any(rows[q].get("tipo") == c for q in ids) else None for c in CATS}
        reader = st.mean(tokens(rows[q], "qa") for q in ids)
        total = st.mean(tokens(rows[q], None) for q in ids)
        stats = plan_stats(rows, ids)
        ci = "" if name == ref_name else f"{delta:+.2f} [{lo:+.2f}; {hi:+.2f}]"
        bci = "" if name == ref_name else f"{bleu_delta:+.2f} [{blo:+.2f}; {bhi:+.2f}]"
        wl = "" if name == ref_name else f"{wins}/{losses}"
        lines.append(f"| {name} | {len(ids)} | {f1(rows, ids):.2f} | {ci} | {bleu:.2f} | {bci} | {wl} | "
                     + " | ".join(f"{cats[c]:.1f}" if cats[c] is not None else "—" for c in CATS)
                     + f" | {reader:,.0f} | {total:,.0f} | {stats['planos_por_pergunta']}"
                     + f" | {stats['perguntas_com_prova_%']}% | {stats['prova_por_outra_leitura']} |")
        out[name] = {"n": len(ids), "f1": f1(rows, ids), "delta": delta, "ic95": [lo, hi],
                     "bleu1_local": bleu, "bleu1_delta": bleu_delta, "bleu1_ic95": [blo, bhi],
                     "bootstrap_unit": unit,
                     "ganhos": wins, "perdas": losses, "por_categoria": cats,
                     "bleu1_local_por_categoria": bleu_cats,
                     "tokens_leitor": reader, "tokens_total": total, **stats}
    counts = {c: sum(ref[q].get("tipo") == c for q in ref) for c in CATS}
    header = (f"# Comparação pareada ({root.name})\n\nF1 oficial do LoCoMo; Δ contra "
              f"`{ref_name}` nas mesmas perguntas; IC 95% por bootstrap de "
              f"{'conversas' if unit == 'conversations' else 'perguntas'} "
              f"(4000 reamostragens; semente 0). BLEU-1 local: precisão de unigramas "
              f"com penalidade de brevidade, normalização e stemming; não é uma "
              f"métrica oficial do LoCoMo nem equivalência confirmada com Zero-Mem. Perguntas por categoria: "
              + ", ".join(f"{c} {n}" for c, n in counts.items())
              + ". Tokens por pergunta sem a construção da memória. Em estudos de reflexão v2, "
              "o total lógico usa a recuperação compartilhada mais o reader de cada célula; "
              "não somar quatro vezes o custo real da recuperação.\n\n")
    output = args.output or root
    output.mkdir(parents=True, exist_ok=True)
    lines.extend(["", "BLEU-1 local por categoria:", "",
                  "| variante | " + " | ".join(CATS) + " |",
                  "|---|" + "---:|" * len(CATS)])
    for name, values in out.items():
        lines.append("| " + name + " | " + " | ".join(
            f"{values['bleu1_local_por_categoria'][c]:.2f}"
            if values['bleu1_local_por_categoria'][c] is not None else "—" for c in CATS) + " |")
    (output / "compare.md").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    (output / "compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print(header + "\n".join(lines))


if __name__ == "__main__":
    main()
