"""Gráficos estáticos do benchmark, sempre sobre os mesmos IDs pareados.

python -m wrag.eval.plots runs/ID
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math

from wrag.eval.report import build_report
from wrag.util import read_json, read_jsonl, write_json


def plot_run(run_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    build_report(run_dir)
    report = read_json(run_dir / "report.json")
    destination = run_dir / "figures"
    destination.mkdir(exist_ok=True)
    manifest = report["manifesto"]
    pilot = read_json(run_dir.parent.parent / "data_selection.json", {})
    is_reduced = pilot.get("corpus_reduced", manifest.get("config", {}).get("subset_corpus", False))
    figure_data = {}
    links = ["# Gráficos do benchmark\n",
             "Comparadores são adaptações locais dos artigos. IC95% por bootstrap de perguntas; "
             "não incorpora variação entre sementes, modelos ou corpora.\n"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 170})
    for dataset, block in report["datasets"].items():
        ids = set(block["ids_comparados"])
        names = list(block["metodos"])
        if not ids:
            links.append(f"## {dataset}\n\nSem perguntas concluídas por todos os métodos; não há comparação válida.\n")
            continue
        rows = {name: [r for r in read_jsonl(run_dir / dataset / f"{name}.jsonl") if r["qid"] in ids]
                for name in names}
        incomplete = any(block.get("ausentes_por_metodo", {}).values())
        description = (f"{dataset} | n pareado = {len(ids)} | "
                       f"{manifest.get('llm', {}).get('deployment') or manifest.get('llm', {}).get('backend')}\n"
                       f"{'corpus reduzido' if is_reduced else 'corpus dos arquivos fornecidos'} | "
                       f"{'rodada parcial' if incomplete else 'rodada concluída'}")
        if manifest.get("llm", {}).get("backend") == "stub":
            description += " | STUB: teste funcional"
        x = np.arange(len(names))

        def save(fig, suffix):
            base = f"{dataset}-{suffix}"
            fig.suptitle(description, fontsize=11)
            fig.tight_layout(rect=[0, 0.03, 1, .88])
            for extension in ("png", "svg", "pdf"):
                fig.savefig(destination / f"{base}.{extension}", bbox_inches="tight")
            plt.close(fig)
            links.append(f"![{suffix}]({base}.png)\n")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key, label in zip(axes.flat, ["em", "f1", "recall@5", "all_recall@5"],
                                  ["EM do leitor", "F1 do leitor", "Recall@5", "Todas as passagens de apoio @5"]):
            values = [block["metodos"][n]["metricas"][key] * 100 for n in names]
            errors = []
            for n, value in zip(names, values):
                low, high = block["metodos"][n]["ic95"][key]
                errors.append([max(0, value - low * 100), max(0, high * 100 - value)]
                              if all(math.isfinite(v) for v in (value, low, high)) else [0, 0])
            ax.bar(x, values, color=["#d67831" if n == "witnessrag" else "#367b9e" for n in names],
                   yerr=np.asarray(errors).T, capsize=3)
            ax.set(title=label, ylabel="%", ylim=(0, 108))
            ax.set_xticks(x, names, rotation=30, ha="right")
            ax.grid(axis="y", alpha=.15)
        save(fig, "qualidade")

        latency = [sum(r.get("latencia_recuperacao_s", 0) + r.get("latencia_leitura_s", 0)
                       for r in rows[n]) / len(ids) for n in names]
        tokens = [sum(r.get("uso_llm", {}).get("total", {}).get("tokens_prompt", 0)
                      + r.get("uso_llm", {}).get("total", {}).get("tokens_resposta", 0)
                      for r in rows[n]) / len(ids) for n in names]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, values, label in zip(axes, (latency, tokens),
                                    ("Recuperação + leitor: segundos/pergunta", "Tokens LLM lógicos/pergunta (incluem cache)")):
            ax.bar(x, values, color="#367b9e")
            ax.set_title(label)
            ax.set_xticks(x, names, rotation=30, ha="right")
            ax.grid(axis="y", alpha=.15)
        save(fig, "custo-consulta")

        structural = [(n, block["resposta_estrutural"][n]) for n in names
                      if n in block.get("resposta_estrutural", {})]
        if structural:
            fig, ax = plt.subplots(figsize=(9, 5))
            sx = np.arange(len(structural))
            ax.bar(sx - .18, [b["em"] * 100 for _, b in structural], .36, label="EM estrutural")
            ax.bar(sx + .18, [b["cobertura"] * 100 for _, b in structural], .36,
                   label="Contexto com testemunha completa")
            ax.set_xticks(sx, [n for n, _ in structural])
            ax.set(ylabel="%", ylim=(0, 110))
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
            save(fig, "testemunhas")

        summary = read_json(run_dir / dataset / "summary.json", {})
        info = summary.get("metodos", {})
        if info:
            shared = summary.get("indexacao_compartilhada", {}).get("seconds", 0)
            own = [info.get(n, {}).get("indexacao", {}).get("seconds", 0)
                   + info.get(n, {}).get("selecao", {}).get("seconds", 0) for n in names]
            shared_values = [0 if n in {"dense", "bm25"} else shared for n in names]
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar(x, shared_values, label="Índice compartilhado (mesmo custo atribuído por método)")
            ax.bar(x, own, bottom=shared_values, label="Indexação própria + seleção")
            ax.set_xticks(x, names, rotation=30, ha="right")
            ax.set_ylabel("Tempo de indexação (s)")
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), fontsize=8)
            save(fig, "indexacao")
        figure_data[dataset] = {"question_ids": sorted(ids), "methods": names,
                                "latency_seconds_per_question": latency, "logical_tokens_per_question": tokens}
    write_json(destination / "plot_data.json", figure_data)
    (destination / "README.md").write_text("\n".join(links), encoding="utf-8")
    print(f"Gráficos: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_run(parser.parse_args().run_dir.resolve())


if __name__ == "__main__":
    main()
