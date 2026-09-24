"""
Relatórios no formato da Tabela 1(b) do GAM.

Por recorte: n, respondidas, bloqueadas, erros; a métrica do GAM com a
semântica de falha do GAM (F1: média sobre as respondidas; RULER: falha conta
como erro) e a métrica estrita (toda falha vale 0); diagnósticos (EM, a
pontuação oficial da RULER), páginas, tokens e custo.

Por motor: a linha da Tabela 1(b) (HotpotQA 56K/224K/448K, RULER Retri./MT/
AGG./QA, NarrativeQA), as 13 tarefas da RULER e as linhas do artigo para
comparação. Uma célula de recorte incompleto sai com asterisco.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from wrag.gambench import protocol as P
from wrag.util import read_json, read_jsonl, write_json


def latest_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Uma linha por amostra: a última gravada (uma retomada pode refazer erros)."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["sid"]] = row
    return list(latest.values())


def _pct(values: list[float]) -> float | None:
    return round(100.0 * mean(values), 2) if values else None


def split_report(rows: list[dict[str, Any]], benchmark: str, split: str, expected: int,
                 manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = latest_rows(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    filtered = [r for r in rows if r.get("status") == "filtered"]
    errors = [r for r in rows if r.get("status") == "error"]
    report: dict[str, Any] = {
        "benchmark": benchmark, "split": split,
        "engine": (manifest or {}).get("engine") or (rows[0].get("engine") if rows else ""),
        "coluna": P.column_of(benchmark, split),
        "esperadas": expected, "registradas": len(rows), "respondidas": len(ok),
        "bloqueadas": len(filtered), "erros": len(errors),
        "completo": len(ok) + len(filtered) >= expected and not errors,
    }
    if benchmark == "ruler":
        report["metrica"] = "acuracia"
        # GAM: falha (erro ou bloqueio) vale 0 e fica no denominador.
        report["valor"] = _pct([r.get("accuracy", 0.0) for r in rows])
        report["valor_estrito"] = report["valor"]
        report["ruler_oficial"] = _pct([r.get("ruler_oficial", 0.0) for r in rows])
    else:
        report["metrica"] = "f1"
        # GAM: média só sobre quem tem F1 (respondidas).
        report["valor"] = _pct([r["f1"] for r in ok if "f1" in r])
        report["valor_estrito"] = _pct([r.get("f1", 0.0) if r.get("status") == "ok" else 0.0
                                        for r in rows])
        report["em"] = _pct([r["em"] for r in ok if "em" in r])
    answered = [r for r in rows if r.get("status") in {"ok", "filtered"}]
    report["paginas_media"] = round(mean(r["n_paginas"] for r in answered), 1) \
        if answered and all("n_paginas" in r for r in answered) else None
    report["tokens_contexto_media"] = round(mean(r["tokens_contexto"] for r in answered)) \
        if answered and all("tokens_contexto" in r for r in answered) else None
    reader = [r.get("leitor", {}) for r in ok]
    report["leitor"] = {
        "tokens_prompt_media": round(mean(x.get("prompt_tokens", 0) for x in reader)) if reader else None,
        "tokens_resposta_media": round(mean(x.get("completion_tokens", 0) for x in reader), 1) if reader else None,
        "cortadas_por_limite": sum(1 for x in reader if x.get("finish_reason") == "length"),
        "em_cache": sum(1 for x in reader if x.get("cached")),
    }
    engine_calls = sum(r.get("uso_motor", {}).get("chamadas", 0) for r in rows)
    engine_prompt = sum(r.get("uso_motor", {}).get("tokens_prompt_sem_cache", 0) for r in rows)
    engine_completion = sum(r.get("uso_motor", {}).get("tokens_resposta_sem_cache", 0) for r in rows)
    report["motor"] = {"chamadas_llm": engine_calls, "tokens_prompt_sem_cache": engine_prompt,
                       "tokens_resposta_sem_cache": engine_completion,
                       "alterou_contexto": sum(1 for r in rows if (r.get("diagnostico") or {})
                                               .get("contexto_alterado_pelo_witness"))}
    times = [r.get("tempo", {}) for r in rows if r.get("tempo")]
    report["tempo_medio_s"] = {k: round(mean(t.get(k, 0.0) for t in times), 2)
                               for k in ("paginas_s", "indice_s", "busca_s")} if times else {}
    if errors:
        report["exemplos_de_erro"] = [{"sid": r["sid"], "erro": r.get("error", "")[:300]}
                                      for r in errors[:5]]
    return report


def write_split_report(output: Path, report: dict[str, Any]) -> None:
    write_json(output / "report.json", report)
    value = "—" if report["valor"] is None else f"{report['valor']:.2f}"
    strict = "—" if report["valor_estrito"] is None else f"{report['valor_estrito']:.2f}"
    lines = [f"# {report['engine']} — {report['benchmark']} {report['split']}", "",
             f"{report['metrica'].upper()} (semântica do GAM): **{value}**  ",
             f"estrito (falha = 0): {strict}  ",
             f"amostras: {report['registradas']}/{report['esperadas']} registradas, "
             f"{report['respondidas']} respondidas, {report['bloqueadas']} bloqueadas, "
             f"{report['erros']} com erro" + ("" if report["completo"] else " — INCOMPLETO"), ""]
    if report.get("ruler_oficial") is not None:
        lines.append(f"pontuação oficial da RULER (diagnóstico): {report['ruler_oficial']:.2f}")
    if report.get("em") is not None:
        lines.append(f"EM (diagnóstico): {report['em']:.2f}")
    lines += ["", f"páginas por contexto: {report['paginas_media']}; "
                  f"tokens de contexto: {report['tokens_contexto_media']}",
              f"leitor: {report['leitor']}", f"motor: {report['motor']}",
              f"tempo médio: {report['tempo_medio_s']}"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# tabela por motor
# ---------------------------------------------------------------------------

def collect(root: Path, engine: str) -> dict[tuple[str, str], dict[str, Any]]:
    found = {}
    base = root / engine
    for benchmark in P.BENCHMARKS:
        for path in sorted((base / benchmark).glob("*/predictions.jsonl")):
            split = path.parent.name
            manifest = read_json(path.parent / "manifest.json", {}) or {}
            expected = P.EXPECTED.get((benchmark, split), 0)
            selection = manifest.get("selecao") or {}
            if selection.get("amostras"):
                expected = selection["amostras"]
            report = split_report(read_jsonl(path), benchmark, split, expected, manifest)
            report["protocolo_completo"] = expected == P.EXPECTED.get((benchmark, split))
            found[(benchmark, split)] = report
    return found


def table_row(reports: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Valor de cada coluna da Tabela 1(b). RULER: média das tarefas do grupo."""
    row: dict[str, dict[str, Any]] = {}
    for split, column in (("56k", "HotpotQA 56K"), ("224k", "HotpotQA 224K"),
                          ("448k", "HotpotQA 448K")):
        report = reports.get(("hotpotqa", split))
        row[column] = _cell([report] if report else [], 1)
    for group, tasks in P.RULER_GROUPS.items():
        parts = [reports.get(("ruler", task)) for task in tasks]
        row[f"RULER {group}"] = _cell([p for p in parts if p], len(tasks))
    report = reports.get(("narrativeqa", "test"))
    row["NarrativeQA"] = _cell([report] if report else [], 1)
    return row


def _cell(parts: list[dict[str, Any]], needed: int) -> dict[str, Any]:
    values = [p["valor"] for p in parts if p.get("valor") is not None]
    if not values:
        return {"valor": None, "completo": False}
    complete = (len(values) == needed and
                all(p.get("completo") and p.get("protocolo_completo", True) for p in parts))
    return {"valor": round(mean(values), 2), "completo": complete,
            "tarefas": len(values), "tarefas_esperadas": needed,
            "n": sum(p["registradas"] for p in parts)}


def suite_report(root: Path, engines: Iterable[str], model_label: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"protocolo": P.GAM_PAPER, "gam_commit": P.GAM_COMMIT,
                           "modelo": model_label, "motores": {}}
    for engine in engines:
        reports = collect(root, engine)
        if not reports:
            continue
        out["motores"][engine] = {
            "tabela": table_row(reports),
            "recortes": {f"{b}/{s}": r for (b, s), r in sorted(reports.items())},
        }
    return out


def _fmt(cell: dict[str, Any]) -> str:
    if cell.get("valor") is None:
        return "—"
    return f"{cell['valor']:.2f}" + ("" if cell.get("completo") else "*")


def render_suite(report: dict[str, Any]) -> str:
    cols = P.TABLE_COLUMNS
    header = "| linha | " + " | ".join(cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = ["# Protocolo do GAM (Tabela 1b): resultados", "",
             f"Modelo: {report.get('modelo') or '(não informado)'}. "
             "HotpotQA e NarrativeQA em F1; RULER em acurácia. "
             "`*` = recorte incompleto ou com erros.", "", header, sep]
    for engine, block in report["motores"].items():
        table = block["tabela"]
        lines.append(f"| **{engine}** (este trabalho) | " +
                     " | ".join(_fmt(table[c]) for c in cols) + " |")
    for model, rows in P.PAPER_TABLE_1B.items():
        for name in ("RAG", "GAM"):
            values = rows[name]
            lines.append(f"| {name}, {model} (artigo) | " +
                         " | ".join(f"{v:.2f}" for v in values) + " |")
    for engine, block in report["motores"].items():
        lines += ["", f"## {engine}: recortes", "",
                  "| recorte | métrica | valor (GAM) | estrito | n | respondidas | bloqueadas | erros | "
                  "oficial RULER | páginas | tokens leitor |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in block["recortes"].items():
            value = "—" if r["valor"] is None else f"{r['valor']:.2f}"
            strict = "—" if r["valor_estrito"] is None else f"{r['valor_estrito']:.2f}"
            official = "" if r.get("ruler_oficial") is None else f"{r['ruler_oficial']:.2f}"
            lines.append(f"| {name}{'' if r['completo'] else '*'} | {r['metrica']} | {value} | "
                         f"{strict} | {r['registradas']}/{r['esperadas']} | {r['respondidas']} | "
                         f"{r['bloqueadas']} | {r['erros']} | {official} | {r['paginas_media']} | "
                         f"{r['leitor'].get('tokens_prompt_media')} |")
    lines += ["", "Linhas do artigo: GPT-4o-mini e Qwen2.5-14B-Instruct (Tabela 1b, "
              "arXiv:2511.18423). Comparação direta só vale para o mesmo modelo."]
    return "\n".join(lines) + "\n"


def write_suite(root: Path, engines: Iterable[str], model_label: str = "") -> dict[str, Any]:
    report = suite_report(root, engines, model_label)
    write_json(root / "gam_table.json", report)
    (root / "gam_table.md").write_text(render_suite(report), encoding="utf-8")
    return report
