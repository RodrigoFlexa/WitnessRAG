"""
Relatório de uma rodada.

A decisão metodológica que mais afeta os números está aqui: **o conjunto de
perguntas excluídas é a UNIÃO das perguntas que qualquer método perdeu para o
filtro de conteúdo do Azure**. Se cada método fosse avaliado só nas perguntas que
ele conseguiu processar, o método que teve mais bloqueios competiria num
subconjunto diferente — e provavelmente mais fácil, já que o filtro tende a
recusar conteúdo sensível, que também tende a ser mais difícil. A tabela
principal usa o mesmo denominador para todos, e a coluna `filtradas` mostra
quantas cada um perdeu.

O relatório sai em Markdown (para colar no artigo) e em JSON (para plotar).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from wrag.eval import metrics as M
from wrag.util import get_logger, read_json, read_jsonl, write_json

log = get_logger("wrag.eval.report")

MAIN_COLUMNS = [
    ("recall@2", "R@2"),
    ("recall@5", "R@5"),
    ("all_recall@5", "AR@5"),
    ("em", "EM"),
    ("f1", "F1"),
]


def build_report(run_dir: Path) -> Path:
    manifest = read_json(run_dir / "run.json", {})
    datasets = manifest.get("datasets", [])
    methods = manifest.get("metodos", [])

    payload: dict[str, Any] = {"run": str(run_dir), "manifesto": manifest, "datasets": {}}
    lines: list[str] = []
    lines.append(f"# Resultados — {run_dir.name}\n")
    lines.append(_setup_block(manifest))

    for dataset in datasets:
        records = {m: read_jsonl(run_dir / dataset / f"{m}.jsonl") for m in methods}
        if not any(records.values()):
            continue
        excluded = _excluded_union(records)
        summary = read_json(run_dir / dataset / "summary.json", {})
        block, data = _dataset_block(dataset, records, excluded, summary)
        lines.append(block)
        payload["datasets"][dataset] = data

    lines.append(_caveats())
    text = "\n".join(lines)
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    write_json(run_dir / "report.json", payload)
    log.info("relatório em %s", run_dir / "report.md")
    return run_dir / "report.md"


def _setup_block(manifest: dict) -> str:
    llm = manifest.get("llm", {})
    embedder = manifest.get("embedder", {})
    cfg = manifest.get("config", {})
    filtro = manifest.get("filtro_conteudo", {})
    rows = [
        "## Configuração\n",
        f"- **LLM**: `{llm.get('backend')}` / deployment `{llm.get('deployment') or '—'}`",
        f"- **Embeddings**: `{embedder.get('backend')}` (`{embedder.get('chave')}`)",
        f"- **top-k**: {cfg.get('top_k')} · **perguntas por dataset**: {cfg.get('n_questions')}"
        f" · **seed**: {cfg.get('seed')}",
        f"- **Feixe da junção**: {cfg.get('witness', {}).get('beam_width')}"
        f" · **rodadas de aquisição**: {cfg.get('witness', {}).get('acquisition_rounds')}"
        f" · **orçamento de memória**: {cfg.get('witness', {}).get('budget_fraction')}",
        f"- **Bloqueios por filtro de conteúdo**: {filtro.get('total', 0)}"
        f" {json.dumps(filtro.get('por_estagio', {}), ensure_ascii=False)}",
    ]
    if embedder.get("backend") == "tfidf":
        rows.append(
            "\n> **Atenção**: a rodada usou o embedder TF-IDF de fallback. Ele existe para "
            "verificar o encanamento offline e **não** é um retriever de artigo. Nenhum número "
            "desta rodada deve ser comparado com valores publicados."
        )
    return "\n".join(rows) + "\n"


def _excluded_union(records: dict[str, list[dict]]) -> set[str]:
    excluded: set[str] = set()
    for rows in records.values():
        excluded |= {r["qid"] for r in rows if r.get("filtrada")}
    return excluded


def _dataset_block(dataset: str, records: dict[str, list[dict]], excluded: set[str],
                   summary: dict) -> tuple[str, dict]:
    corpus = summary.get("corpus", {})
    lines = [f"\n## {dataset}\n",
             f"{corpus.get('n_questions', '?')} perguntas · {corpus.get('n_passages', '?')} passagens · "
             f"{corpus.get('hops_medio', '?')} hops em média · "
             f"{len(excluded)} pergunta(s) excluída(s) por filtro de conteúdo em algum método\n"]
    reduced = summary.get('subset_corpus', False) or summary.get('corpus_scope', 'provided').startswith('pilot_')
    lines.append(f"Corpus reduzido/piloto: {reduced} · "
                 f"embedding ajustado: `{summary.get('embedder', {}).get('chave', '?')}`.\n")

    data: dict[str, Any] = {"corpus": corpus, "excluidas": sorted(excluded), "metodos": {}}
    by_id = {}
    for method, rows in records.items():
        by_id[method] = {r["qid"]: r for r in rows}
        if len(by_id[method]) != len(rows):
            raise ValueError(f"resultados duplicados para {dataset}/{method}")
    all_ids = set(corpus.get("question_ids", [])) or set().union(*(set(r) for r in by_id.values()))
    common = all_ids.intersection(*(set(r) for r in by_id.values()))
    data["ausentes_por_metodo"] = {m: sorted(all_ids - set(r)) for m, r in by_id.items()}
    data["ids_comparados"] = sorted(common - excluded)
    lines.append(f"Comparação pareada: {len(common - excluded)} perguntas presentes em todos os métodos.\n")
    if all_ids - common:
        lines.append(f"> Rodada incompleta: {len(all_ids - common)} perguntas sem resultado em algum método.\n")
    compared = {m: [r[qid] for qid in sorted(common)] for m, r in by_id.items()}

    header = "| método | " + " | ".join(label for _k, label in MAIN_COLUMNS) + " | abst. | filtradas | lat. (s) |"
    sep = "|" + "---|" * (len(MAIN_COLUMNS) + 4)
    lines += ["### Tabela principal (mesmo denominador para todos)\n", header, sep]

    for method, rows in records.items():
        kept = [r for r in compared[method] if r["qid"] not in excluded]
        values = {key: M.aggregate(r[key] for r in kept) for key, _label in MAIN_COLUMNS}
        abstention = M.aggregate(float(r.get("abstencao", False)) for r in kept)
        latency = M.aggregate(r.get("latencia_recuperacao_s", 0.0) for r in kept)
        n_filtered = sum(1 for r in rows if r.get("filtrada"))
        cells = " | ".join(_pct(values[key]) for key, _l in MAIN_COLUMNS)
        lines.append(f"| {method} | {cells} | {_pct(abstention)} | {n_filtered} | {latency:.2f} |")

        ci = {key: M.bootstrap_ci([r[key] for r in kept if r[key] == r[key]])
              for key, _l in MAIN_COLUMNS}
        data["metodos"][method] = {
            "n_avaliadas": len(kept), "n_filtradas": n_filtered,
            "metricas": values, "ic95": ci, "abstencao": abstention, "latencia_s": latency,
        }

    lines.append("\nIC95% por bootstrap (1000 reamostragens):\n")
    lines.append("| método | " + " | ".join(f"{label} IC95%" for _k, label in MAIN_COLUMNS) + " |")
    lines.append("|" + "---|" * (len(MAIN_COLUMNS) + 1))
    for method in records:
        ci = data["metodos"][method]["ic95"]
        cells = " | ".join(f"[{_pct(ci[key][0])}, {_pct(ci[key][1])}]" for key, _l in MAIN_COLUMNS)
        lines.append(f"| {method} | {cells} |")

    lines += _paired_block(compared, excluded, data)
    lines += _by_shape(compared, excluded, data)
    lines += _structural_block(compared, excluded, data)
    lines += _witness_block(compared, excluded, data)
    lines += _risk_block(compared, excluded, data)
    lines += _cost_block(summary, data)
    return "\n".join(lines) + "\n", data


def _paired_block(records, excluded, data):
    if "witnessrag" not in records:
        return []
    reference = {r["qid"]: r for r in records["witnessrag"] if r["qid"] not in excluded}
    lines = ["\n### Diferenças pareadas: WITNESS-RAG menos comparador\n",
             "IC95% exploratório da diferença, sem correção por múltiplas comparações.\n",
             "| comparador | ΔF1 IC95% | ΔAR@5 IC95% |", "|---|---|---|"]
    block = {}
    for method, rows in records.items():
        if method == "witnessrag":
            continue
        paired = [(reference[r["qid"]], r) for r in rows if r["qid"] in reference]
        block[method] = {}
        cells = []
        for key in ("f1", "all_recall@5"):
            ci = M.paired_bootstrap_ci([a[key] for a, b in paired], [b[key] for a, b in paired])
            block[method][key] = ci
            cells.append(f"[{_pct(ci[0])}, {_pct(ci[1])}]")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    data["diferencas_pareadas_ic95"] = block
    return lines


def _structural_block(records, excluded, data):
    block = {}
    lines = ["\n### Resposta do executor antes do leitor\n",
             "| método | n | EM estrutural | contexto com testemunha completa |",
             "|---|---|---|---|"]
    for method, rows in records.items():
        kept = [r for r in rows if r["qid"] not in excluded and r.get("em_estrutural") is not None]
        if not kept:
            continue
        em = M.aggregate(r["em_estrutural"] for r in kept)
        coverage = M.aggregate(float(r.get("testemunha_no_contexto", False)) for r in kept)
        block[method] = {"n": len(kept), "em": em, "cobertura": coverage}
        lines.append(f"| {method} | {len(kept)} | {_pct(em)} | {_pct(coverage)} |")
    data["resposta_estrutural"] = block
    return lines if block else []


def _by_shape(records: dict[str, list[dict]], excluded: set[str], data: dict) -> list[str]:
    """Quebra por forma da consulta e por número de hops.

    A média global esconde o que a proposta prevê: o ganho da busca por
    testemunhas deveria concentrar-se em cadeia e interseção, não em single-hop.
    Se a diferença estiver espalhada uniformemente, a explicação provavelmente não
    é a estrutura lógica.
    """
    lines = ["\n### Por número de hops (F1 / all-recall@5)\n"]
    hops = sorted({r.get("n_hops", 1) for rows in records.values() for r in rows})
    lines.append("| método | " + " | ".join(f"{h} hop(s)" for h in hops) + " |")
    lines.append("|" + "---|" * (len(hops) + 1))
    per_method: dict[str, dict] = {}
    for method, rows in records.items():
        kept = [r for r in rows if r["qid"] not in excluded]
        cells = []
        per_method[method] = {}
        for h in hops:
            subset = [r for r in kept if r.get("n_hops") == h]
            f1 = M.aggregate(r["f1"] for r in subset)
            ar = M.aggregate(r["all_recall@5"] for r in subset)
            per_method[method][str(h)] = {"n": len(subset), "f1": f1, "all_recall@5": ar}
            cells.append(f"{_pct(f1)} / {_pct(ar)} (n={len(subset)})" if subset else "—")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    data["por_hops"] = per_method

    shapes = sorted({r.get("forma_consulta") for rows in records.values() for r in rows
                     if r.get("forma_consulta")})
    if shapes:
        lines.append("\n### Por forma da consulta compilada (só métodos que compilam)\n")
        lines.append("| método | " + " | ".join(shapes) + " |")
        lines.append("|" + "---|" * (len(shapes) + 1))
        by_shape: dict[str, dict] = {}
        for method, rows in records.items():
            kept = [r for r in rows if r["qid"] not in excluded and r.get("forma_consulta")]
            if not kept:
                continue
            by_shape[method] = {}
            cells = []
            for shape in shapes:
                subset = [r for r in kept if r.get("forma_consulta") == shape]
                f1 = M.aggregate(r["f1"] for r in subset)
                by_shape[method][shape] = {"n": len(subset), "f1": f1}
                cells.append(f"{_pct(f1)} (n={len(subset)})" if subset else "—")
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        data["por_forma"] = by_shape
    return lines


def _witness_block(records: dict[str, list[dict]], excluded: set[str], data: dict) -> list[str]:
    """Cobertura de testemunha, quando o dataset tem evidência anotada (2Wiki)."""
    rows_with = {m: [r for r in rows if r["qid"] not in excluded
                     and r.get("cobertura_testemunha") == r.get("cobertura_testemunha")
                     and r.get("cobertura_testemunha") is not None]
                 for m, rows in records.items()}
    rows_with = {m: r for m, r in rows_with.items() if r}
    if not rows_with:
        return []
    lines = ["\n### Evidência e resposta estrutural\n",
             "Cobertura lexical de triplas dirigidas, sensível ao vocabulário. "
             "Não certifica fidelidade semântica ao texto. Só contam provas completas no contexto entregue.\n",
             "| método | cobertura de triplas | cobertura = 100% | EM estrutural |", "|---|---|---|---|"]
    block = {}
    for method, rows in rows_with.items():
        coverage = M.aggregate(r["cobertura_testemunha"] for r in rows)
        complete = M.aggregate(float(r["cobertura_testemunha"] >= 1.0) for r in rows)
        em = M.aggregate(r.get("em_estrutural", float("nan")) for r in rows)
        block[method] = {"n": len(rows), "cobertura": coverage, "completa": complete, "em_estrutural": em}
        lines.append(f"| {method} | {_pct(coverage)} | {_pct(complete)} | {_pct(em)} |")
    data["testemunhas"] = block
    return lines


def _risk_block(records: dict[str, list[dict]], excluded: set[str], data: dict) -> list[str]:
    """Seletividade empírica por score não calibrado e acerto estrutural."""
    lines: list[str] = []
    block = {}
    for method, rows in records.items():
        kept = [r for r in rows if r["qid"] not in excluded and r.get("risco") is not None
                and r.get("em_estrutural") is not None]
        if len(kept) < 10:
            continue
        # Ordena pelo limite SEM recorte: recortado em 1.0, quase tudo empata e a
        # curva viraria uma reta por artefato de apresentação.
        scores = [r.get("risco_bruto", r["risco"]) for r in kept]
        curve = M.risk_coverage_curve(scores, [r["em_estrutural"] for r in kept])
        block[method] = [{"cobertura": p.coverage, "em_estrutural": p.accuracy, "limiar": p.threshold}
                         for p in curve]
    if not block:
        return lines
    lines += ["\n### Seletividade da resposta estrutural (score não calibrado)\n",
              "EM estrutural, incluindo falhas sem testemunha. Empates são aceitos em bloco; "
              "a cobertura efetiva aparece entre parênteses. Esta curva não certifica risco probabilístico.\n",
              "| método | 20% | 40% | 60% | 80% | 100% |", "|---|---|---|---|---|---|"]
    for method, curve in block.items():
        picks = []
        for target in (0.2, 0.4, 0.6, 0.8, 1.0):
            point = min(curve, key=lambda p: abs(p["cobertura"] - target))
            picks.append(f"{_pct(point['em_estrutural'])} ({_pct(point['cobertura'])}%)")
        lines.append(f"| {method} | " + " | ".join(picks) + " |")
    data["risco_cobertura"] = block
    return lines


def _cost_block(summary: dict, data: dict) -> list[str]:
    shared = summary.get("indexacao_compartilhada", {})
    lines = ["\n### Custo\n",
             f"Indexação compartilhada: {shared.get('seconds', '?')} s · "
             f"{json.dumps(shared.get('usage', {}).get('total', {}), ensure_ascii=False)}\n",
             "Custos de LLM: tokens lógicos e tokens sem cache local. Embeddings não estão "
             "contabilizados em tokens; estes números não representam custo financeiro total.\n",
             "| método | chamadas (consulta) | tokens prompt | tokens resposta | tempo (s) |",
             "|---|---|---|---|---|"]
    cost = {}
    for method, info in (summary.get("metodos") or {}).items():
        usage = (info.get("uso_consulta") or {}).get("total", {})
        cost[method] = usage
        lines.append(f"| {method} | {usage.get('chamadas', 0)} | {usage.get('tokens_prompt', 0)} | "
                     f"{usage.get('tokens_resposta', 0)} | {info.get('tempo_consulta_s', 0)} |")
    lines += ["\n| método | indexação própria (s) | seleção de memória (s) | tokens LLM offline adicionais |",
              "|---|---|---|---|"]
    for method, info in (summary.get("metodos") or {}).items():
        indexing, selection = info.get("indexacao", {}), info.get("selecao", {})
        tokens = sum(x.get("usage", {}).get("total", {}).get("tokens_prompt", 0)
                     + x.get("usage", {}).get("total", {}).get("tokens_resposta", 0) for x in (indexing, selection))
        lines.append(f"| {method} | {indexing.get('seconds', 0):.3f} | {selection.get('seconds', 0):.3f} | {tokens} |")
    data["custo"] = cost
    attempts = summary.get("tentativas_indexacao", [])
    if len(attempts) > 1:
        lines.append(f"\nRetomadas: {len(attempts) - 1} reindexações adicionais, "
                     f"{sum(a['seconds'] for a in attempts[1:]):.3f} s. "
                     "Custos detalhados em index_attempts.json; a tabela preserva a indexação inicial.\n")
    return lines


def _caveats() -> str:
    return """
## Como ler estes números

1. **Escopo do corpus.** O padrão mantém todas as passagens disponíveis no arquivo
   de corpus. `--subset-corpus` reduz esse universo; a condição é registrada.
2. **Denominador pareado.** Só entram perguntas presentes em todos os métodos,
   excluindo a união dos bloqueios. A tabela estima desempenho nesse subconjunto;
   não mede disponibilidade operacional sobre todas as solicitações.
3. **Comparadores locais.** GraphRAG, HippoRAG e HippoRAG2 são adaptações com extração
   compartilhada, não reproduções fiéis dos sistemas publicados. Aquisição dirigida
   adiciona informação e custo ao WITNESS-RAG; desligue-a para isolar o executor.
   O comparador `relational` executa SQL exato sobre a mesma consulta compilada.
4. **Anotações privilegiadas.** `witnessrag-annotated` e o nome legado
   `witnessrag-oracle` usam traduções heurísticas de anotações, às vezes a resposta
   ouro. São diagnósticos privilegiados, não um teto garantido nem uma ablação pura.
5. **Garantias condicionais.** Exact sem cortes é completo para a consulta e os
   fatos fornecidos. Não certifica a extração ou a compilação do texto. Similaridade
   e scores de risco não calibrados não são probabilidades.
6. **Incerteza experimental.** Use diferenças pareadas, tamanho de efeito e várias
   sementes. Sobreposição de ICs individuais não é um teste de igualdade. Hops por
   número de passagens são apenas um proxy; a fonte está nos registros.
7. **Orçamento.** O ILP otimiza testemunhas enumeradas das demandas disponíveis.
   A máscara de fatos não mede redução física de RAM. Demandas sintetizadas usam
   um prior estrutural amostrado, não uma distribuição validada de perguntas reais.
"""


def _pct(value: float) -> str:
    if value != value:
        return "—"
    return f"{100 * value:.1f}"
