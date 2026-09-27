"""How the cascade router split the questions, and whether it split them well.

    python scripts/cascade-report.py runs/qwen-cascade

Reads <root>/witnessrag-cascade and, when present, <root>/witnessrag-robust
(always plans) and <root>/robust-no-plan (never plans). For the questions the
router sent to PLAN and to DIRECT, prints the F1 of the three runs on exactly
those questions. A good router sends to PLAN the questions where the plan
beats no plan, and to DIRECT the ones where it does not. Categories appear
only in this analysis; the router never sees them. Writes cascade.md.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

CATS = ["single-hop", "multi-hop", "temporal", "open-domain"]
BUILD = {"memory.summary", "index.openie", "index.ner"}


def load(folder: Path) -> dict[str, dict]:
    rows = {}
    for path in glob.glob(str(folder / "**" / "locomo" / "*.jsonl"), recursive=True):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                rows[r["qid"]] = r
    return rows


def tokens(r: dict) -> float:
    stages = (r.get("uso_llm") or {}).get("por_estagio") or {}
    return sum(v.get("tokens_prompt", 0) + v.get("tokens_resposta", 0)
               for k, v in stages.items() if k not in BUILD)


def main() -> None:
    root = Path(sys.argv[1])
    cas = load(root / "witnessrag-cascade")
    others = {n: load(root / n) for n in ("witnessrag-robust", "robust-no-plan") if (root / n).exists()}
    if not cas:
        sys.exit(f"sem registros em {root / 'witnessrag-cascade'}")
    route = {q: ((r.get("diagnosticos") or {}).get("roteador") or {}).get("rota", "?")
             for q, r in cas.items()}
    ids = sorted(set(cas).intersection(*[set(v) for v in others.values()]) if others else set(cas))
    f1 = lambda R, I: 100 * sum(R[q]["f1_locomo"] for q in I) / len(I) if I else float("nan")
    lines = [f"# Cascata ({root.name}), {len(ids)} perguntas\n",
             "| rota | n | " + " | ".join(CATS) + " | F1 cascata | " + " | ".join(f"F1 {n}" for n in others)
             + " | tokens cascata |", "|---|---:|" + "---:|" * len(CATS) + "---:|" + "---:|" * len(others) + "---:|"]
    for name in ("PLAN", "DIRECT", None):
        subset = [q for q in ids if name is None or route.get(q) == name]
        counts = [sum(cas[q].get("tipo") == c for q in subset) for c in CATS]
        label = name or "todas"
        lines.append(f"| {label} | {len(subset)} | " + " | ".join(map(str, counts))
                     + f" | {f1(cas, subset):.2f} | " + " | ".join(f"{f1(R, subset):.2f}" for R in others.values())
                     + f" | {sum(tokens(cas[q]) for q in subset) / max(1, len(subset)):,.0f} |")
    failures = sum(bool(((cas[q].get("diagnosticos") or {}).get("roteador") or {}).get("padrao_por")) for q in ids)
    lines.append(f"\nFalhas do roteador (foram para PLAN por padrão): {failures}")
    text = "\n".join(lines)
    (root / "cascade.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
