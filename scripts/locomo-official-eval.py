"""Recalcula uma rodada do LoCoMo com as métricas do avaliador oficial.

Não chama modelo: usa as respostas já salvas em `<rodada>/locomo/<metodo>.jsonl`.
Além do F1/EM oficiais por categoria, calcula o recall por evidência em nível de
fala, que é o que o oficial mede; o recall por bloco do harness continua no
relatório. Escreve `locomo_oficial.json` na pasta da rodada.

    python scripts/locomo-official-eval.py runs/<piloto>/benchmark/<rodada> \
        --data-selection runs/<piloto>/data_selection.json \
        --corpus runs/<piloto>/data/locomo_corpus.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrag.eval import locomo_official as L
from wrag.util import sha, write_json

DIALOG_ID = re.compile(r"^\[([^\]]+)\]")


def passage_dialog_ids(corpus_path: Path) -> dict[str, list[str]]:
    """pid -> falas contidas no bloco, com o mesmo pid usado pelo harness."""
    mapping: dict[str, list[str]] = {}
    for passage in json.loads(corpus_path.read_text(encoding="utf-8")):
        title, text = passage["title"].strip(), passage["text"].strip()
        pid = "p-" + sha(title, text)[:24]
        ids: list[str] = []
        for line in text.splitlines():
            match = DIALOG_ID.match(line.strip())
            if match and match.group(1) not in ids:
                ids.append(match.group(1))
        mapping[pid] = ids
    return mapping


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="pasta da rodada (contém locomo/)")
    parser.add_argument("--data-selection", type=Path,
                        help="data_selection.json: categorias e evidências anotadas")
    parser.add_argument("--corpus", type=Path,
                        help="locomo_corpus.json: expande blocos recuperados em falas")
    parser.add_argument("--k", type=int, default=5, help="passagens consideradas no recall")
    args = parser.parse_args()

    if not L.available():
        raise SystemExit("instale `nltk` para reproduzir o avaliador oficial")

    selection = json.loads(args.data_selection.read_text(encoding="utf-8")) if args.data_selection else {}
    mapping = selection.get("question_mapping", {})
    dialogs = passage_dialog_ids(args.corpus) if args.corpus else {}

    report: dict[str, dict] = {"fonte": L.SOURCE, "k": args.k, "metodos": {}}
    for path in sorted((args.run_dir / "locomo").glob("*.jsonl")):
        method = path.stem
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        per_category: dict[int, list[dict]] = {}
        for row in rows:
            annotation = mapping.get(row["qid"], {})
            if annotation.get("category"):
                row["categoria_locomo"] = annotation["category"]
            scores = L.score_record(row)
            row.update(scores)
            evidence = annotation.get("evidence_dialog_ids", [])
            if evidence and dialogs:
                retrieved = [d for pid in row["recuperadas"][: args.k] for d in dialogs.get(pid, [])]
                row["recall_evidencia"] = L.evidence_recall(retrieved, evidence)
            per_category.setdefault(scores["categoria_locomo"], []).append(row)

        entry = {"n": len(rows), "por_categoria": {}}
        entry["f1_locomo"] = mean([r["f1_locomo"] for r in rows])
        entry["bleu1_locomo"] = mean([r["bleu1_locomo"] for r in rows])
        entry["em_locomo"] = mean([r["em_locomo"] for r in rows])
        entry["f1_harness"] = mean([r["f1"] for r in rows])
        entry["em_harness"] = mean([r["em"] for r in rows])
        recalls = [r["recall_evidencia"] for r in rows if "recall_evidencia" in r]
        if recalls:
            entry["recall_evidencia"] = mean(recalls)
        for category, subset in sorted(per_category.items()):
            entry["por_categoria"][category] = {
                "nome": L.CATEGORY_NAME.get(category, str(category)),
                "n": len(subset),
                "f1_locomo": mean([r["f1_locomo"] for r in subset]),
                "bleu1_locomo": mean([r["bleu1_locomo"] for r in subset]),
                "em_locomo": mean([r["em_locomo"] for r in subset]),
                "f1_harness": mean([r["f1"] for r in subset]),
                "em_harness": mean([r["em"] for r in subset]),
                "recall_evidencia": mean([r["recall_evidencia"] for r in subset
                                          if "recall_evidencia" in r]),
                "recall@5_bloco": mean([r["recall@5"] for r in subset]),
            }
        report["metodos"][method] = entry

    write_json(args.run_dir / "locomo_oficial.json", report)

    print(f"avaliador oficial: {L.SOURCE}\n")
    header = f"{'método / categoria':<34}{'n':>5}{'F1 ofic':>10}{'EM ofic':>10}{'F1 harn':>10}{'R evid':>9}"
    print(header)
    for method, entry in report["metodos"].items():
        print(f"{method:<34}{entry['n']:>5}{100 * entry['f1_locomo']:>10.2f}"
              f"{100 * entry['em_locomo']:>10.2f}{100 * entry['f1_harness']:>10.2f}"
              f"{100 * entry.get('recall_evidencia', float('nan')):>9.2f}")
        for category, values in entry["por_categoria"].items():
            label = f"  {values['nome']} (cat {category})"
            print(f"{label:<34}{values['n']:>5}{100 * values['f1_locomo']:>10.2f}"
                  f"{100 * values['em_locomo']:>10.2f}{100 * values['f1_harness']:>10.2f}"
                  f"{100 * values['recall_evidencia']:>9.2f}")
    print(f"\nescrito: {args.run_dir / 'locomo_oficial.json'}")


if __name__ == "__main__":
    main()
