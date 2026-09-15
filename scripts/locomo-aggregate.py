"""Agrega as rodadas por conversa de um piloto LoCoMo numa tabela só.

Serve para reagregar depois de uma execução parcial, ou para juntar rodadas
feitas em momentos diferentes. Não chama modelo: lê os JSONL já salvos.

    python scripts/locomo-aggregate.py runs/<piloto>
    python scripts/locomo-aggregate.py runs/<piloto>/conversations/conv00/benchmark/<rodada> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrag.eval.locomo_official import aggregate_runs
from wrag.util import write_json


def discover(paths: list[Path]) -> list[Path]:
    """Aceita a pasta do piloto, a de uma conversa, ou rodadas já apontadas."""
    found: list[Path] = []
    for path in paths:
        if (path / "locomo").is_dir():
            found.append(path)
            continue
        nested = sorted(path.glob("conversations/*/benchmark/*/locomo")) or \
            sorted(path.glob("benchmark/*/locomo")) or sorted(path.glob("*/locomo"))
        found.extend(p.parent for p in nested)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="pasta do piloto ou rodadas")
    parser.add_argument("--output", type=Path, help="onde escrever o JSON (padrão: primeira pasta)")
    args = parser.parse_args()

    runs = discover(args.paths)
    if not runs:
        raise SystemExit("nenhuma rodada com locomo/*.jsonl encontrada nos caminhos dados")
    summary = aggregate_runs(runs)
    destination = (args.output or args.paths[0]) / "locomo_agregado.json"
    write_json(destination, summary)

    official = summary.get("oficial_disponivel")
    keys = (("f1_locomo", "em_locomo") if official else ()) + ("f1", "em", "recall@5", "all_recall@5")
    labels = (("F1ofic", "EMofic") if official else ()) + ("F1", "EM", "R@5", "AR@5")
    print(f"{len(runs)} rodada(s); média micro\n")
    print(f"{'método / categoria':<32} {'n':>5} " + " ".join(f"{x:>8}" for x in labels))
    for method, values in summary["metodos"].items():
        rows = [(method, values)] + [(f"  {k}", v) for k, v in values["por_categoria"].items()]
        for label, block in rows:
            cells = " ".join(f"{100 * block[k]:>8.2f}" if block.get(k) == block.get(k) else f"{'—':>8}"
                             for k in keys)
            print(f"{label:<32} {block['n']:>5} {cells}")
    print(f"\nescrito: {destination}")


if __name__ == "__main__":
    main()
