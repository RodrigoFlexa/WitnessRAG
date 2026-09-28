"""Strict paired 2x2 report, including retrieval controls and interaction."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("paired", ROOT / "scripts/paired-report.py")
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)
NAMES = ("cascade", "summary-reflection", "reader-reflection", "both")
LABELS = ("Cascade atual", "Inferências nos resumos", "Reflexão no reader", "Ambos")


def extra(row, title):
    return next((e.get("text", "") for e in row.get("diagnosticos", {}).get("trechos_extras", [])
                 if e.get("title") == title), "")


def retrieval_control(rows, ids):
    """Factor changes must not quietly change route, facts or summary selection."""
    errors = []
    for q in ids:
        baseline = rows["cascade"][q]
        route = (baseline.get("diagnosticos", {}).get("roteador") or {}).get("rota")
        if route not in {"PLAN", "DIRECT"}:
            errors.append({"qid": q, "field": "missing_cascade_route"})
        for name in NAMES[1:]:
            item = rows[name][q]
            other_route = (item.get("diagnosticos", {}).get("roteador") or {}).get("rota")
            if route != other_route:
                errors.append({"qid": q, "variant": name, "field": "route"})
            if not extra(baseline, "Facts from the memory") or extra(baseline, "Facts from the memory") != extra(item, "Facts from the memory"):
                errors.append({"qid": q, "variant": name, "field": "facts"})
            selected = baseline.get("diagnosticos", {}).get("fatos_entregues", {}).get("resumos_selecionados")
            other = item.get("diagnosticos", {}).get("fatos_entregues", {}).get("resumos_selecionados")
            if selected is None or selected != other:
                errors.append({"qid": q, "variant": name, "field": "summary_selection"})
        for a, b in [("cascade", "reader-reflection"), ("summary-reflection", "both")]:
            if extra(rows[a][q], "Chunk summaries") != extra(rows[b][q], "Chunk summaries"):
                errors.append({"qid": q, "variant": b, "field": "summaries"})
    return {"questions": len(ids), "mismatches": len(errors), "examples": errors[:12]}


def effect(rows, ids, positive, negative):
    # Synthetic scores permit the same conversation bootstrap to evaluate a
    # linear contrast, including the factorial interaction, without new metrics.
    ref, candidate = {}, {}
    for q in ids:
        ref[q] = {"f1_locomo": sum(rows[n][q]["f1_locomo"] for n in negative)}
        candidate[q] = {"f1_locomo": sum(rows[n][q]["f1_locomo"] for n in positive)}
    delta = 100 * st.mean(candidate[q]["f1_locomo"] - ref[q]["f1_locomo"] for q in ids)
    low, high, unit = paired.paired_interval(candidate, ref, ids, "f1_locomo")
    return {"delta_f1": delta, "ci95": [low, high], "bootstrap_unit": unit}


def report(root):
    rows = {name: paired.load(root / name) for name in NAMES}
    ids = sorted(rows["cascade"])
    for name in NAMES[1:]:
        paired.paired_ids(rows[name], rows["cascade"])
    control = retrieval_control(rows, ids)
    (root / "reflection-retrieval-control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
    if control["mismatches"]:
        raise ValueError("A recuperação mudou entre células. Consulte reflection-retrieval-control.json; "
                         "o relatório causal foi recusado.")
    result = {"retrieval_control": control, "categories": {}}
    lines = ["# Ablação de reflexão — cascade, Qwen", "",
             "Quatro células: resumos inferenciais × reader reflexivo. Perguntas pareadas; "
             "rotas, fatos e resumos compartilhados conferidos. F1 oficial e BLEU-1 local, em %. "
             "IC95% por bootstrap de conversas (4.000 amostras).", ""]
    contrasts = {
        "Resumos com reader atual": (["summary-reflection"], ["cascade"]),
        "Reader com resumos atuais": (["reader-reflection"], ["cascade"]),
        "Reader com resumos inferenciais": (["both"], ["summary-reflection"]),
        "Resumos com reader reflexivo": (["both"], ["reader-reflection"]),
        "Ambos contra baseline": (["both"], ["cascade"]),
        "Interação (ambos − resumos − reader + baseline)": (["both", "cascade"], ["summary-reflection", "reader-reflection"])}
    for category in ["all", *paired.CATS]:
        subset = [q for q in ids if category == "all" or rows["cascade"][q]["tipo"] == category]
        if not subset:
            continue
        stats = {}
        lines += [f"## {'Todas' if category == 'all' else category} (n={len(subset)})", "",
                  "| Variante | F1 | BLEU-1 | EM | Δ F1 | IC95% Δ | tokens reader | tokens inferência |",
                  "|---|---:|---:|---:|---:|---|---:|---:|"]
        for name, label in zip(NAMES, LABELS):
            data = rows[name]
            metric = {key: 100 * st.mean(data[q][key] for q in subset)
                      for key in ("f1_locomo", "bleu1_locomo", "em_locomo")}
            change = effect(rows, subset, [name], ["cascade"])
            metric.update(change)
            metric["reader_tokens"] = st.mean(paired.tokens(data[q], "qa") for q in subset)
            metric["inference_tokens"] = st.mean(paired.tokens(data[q], None) for q in subset)
            stats[name] = metric
            ci = change["ci95"]
            lines.append(f"| {label} | {metric['f1_locomo']:.2f} | {metric['bleu1_locomo']:.2f} | "
                         f"{metric['em_locomo']:.2f} | {change['delta_f1']:+.2f} | "
                         f"[{ci[0]:+.2f}; {ci[1]:+.2f}] | {metric['reader_tokens']:.0f} | {metric['inference_tokens']:.0f} |")
        effects = {label: effect(rows, subset, *terms) for label, terms in contrasts.items()}
        lines += ["", "| Contraste | Δ F1 | IC95% |", "|---|---:|---|"]
        for label, item in effects.items():
            a, b = item["ci95"]
            lines.append(f"| {label} | {item['delta_f1']:+.2f} | [{a:+.2f}; {b:+.2f}] |")
        lines.append("")
        result["categories"][category] = {"n": len(subset), "variants": stats, "contrasts": effects}
    costs = {}
    lines += ["## Construção das memórias", "",
              "Tokens lógicos registrados nas respostas, separados dos tokens de inferência. "
              "Chamadas atendidas pelo cache não equivalem a novo gasto físico. A extração/indexação "
              "anterior às perguntas deve ser consultada também nos logs da rodada.", "",
              "| Variante | tokens resumos | tokens reflexão nos resumos |", "|---|---:|---:|"]
    for name, label in zip(NAMES, LABELS):
        costs[name] = {stage: sum(paired.tokens(rows[name][q], stage) for q in ids)
                       for stage in ("memory.summary", "memory.reflection")}
        lines.append(f"| {label} | {costs[name]['memory.summary']:.0f} | {costs[name]['memory.reflection']:.0f} |")
    result["memory_tokens"] = costs
    delivery = {}
    lines += ["", "## Entrega e formato", "",
              "| Variante | perguntas com memória inferida | memórias inferidas/pergunta | readers com schema válido |",
              "|---|---:|---:|---:|"]
    for name, label in zip(NAMES, LABELS):
        counts = [sum(item.get("accepted", 0) for item in rows[name][q].get("diagnosticos", {})
                      .get("fatos_entregues", {}).get("summary_reflection", [])) for q in ids]
        reflective = [rows[name][q]["reflexao_leitor"] for q in ids if rows[name][q].get("reflexao_leitor")]
        delivery[name] = {"questions_with_inferences": sum(n > 0 for n in counts),
                          "mean_inferences_delivered": st.mean(counts),
                          "reflective_readers": len(reflective),
                          "valid_reader_schema": sum(bool(item.get("schema_valid")) for item in reflective)}
        item = delivery[name]
        lines.append(f"| {label} | {item['questions_with_inferences']}/{len(ids)} | "
                     f"{item['mean_inferences_delivered']:.2f} | "
                     f"{item['valid_reader_schema']}/{item['reflective_readers']} |")
    result["delivery"] = delivery
    lines += ["", "A variante de resumos adiciona contexto: consultar tokens do reader antes de atribuir "
              "um ganho exclusivamente à qualidade da inferência. Todas mantêm 40 fatos por padrão, "
              "seis resumos e o mesmo teto de 128 tokens de saída do reader. "
              "Inferências armazenadas não são fatos certificados. Casos usados para ajustar prompts "
              "são desenvolvimento; a promoção do método requer validação reservada.", ""]
    (root / "reflection-ablation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "reflection-ablation.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python scripts/reflection-ablation-report.py ROOT")
    report(Path(sys.argv[1]))
