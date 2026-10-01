"""Read-only live task scores from a running MemoryAgentBench checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

LABELS = {"ruler_qa1_197K": "SH-QA", "ruler_qa2_421K": "MH-QA",
          "longmemeval_s*": "LME(S*)", "eventqa_full": "EventQA"}


def snapshot(output: Path, log: Path | None) -> str:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected = defaultdict(int)
    for row in manifest["identity"]["selection"]:
        expected[row["source"]] += row["questions"]
    path = output / "results.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    # The writer can be between write and fsync: only consume complete lines.
    lines = raw.split(b"\n")[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    ids = [r["qid"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados no checkpoint; pontuação não calculada")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"]].append(row)
    screen = [datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S %Z"),
              f"Respostas: {len(rows)}/{sum(expected.values())}", "",
              f"{'Tarefa':<12} {'Respondidas':>12} {'Avaliadas':>10}  Score parcial"]
    for source, total in expected.items():
        records = groups[source]
        scores = [r["metrics"][r["primary_metric"]] for r in records
                  if r["primary_metric"] in r["metrics"]]
        value = f"{100 * sum(scores) / len(scores):.2f}%" if scores else (
            "juiz pendente" if records else "aguardando respostas")
        screen.append(f"{LABELS.get(source, source):<12} {len(records):>6}/{total:<5} "
                      f"{len(scores):>10}  {value}")
    screen += ["", "Score parcial = média apenas das respostas já avaliadas.",
               "LME(S*) depende do julgamento oficial; métricas proxy não são usadas."]
    if log and log.exists():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 32768))
            tail = handle.read().decode("utf-8", errors="replace").splitlines()
        stages = [line for line in tail if "Registering " in line or "memory chunks " in line]
        screen += ["", *stages[-2:]]
    return "\n".join(screen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        while True:
            screen = snapshot(args.output, args.log)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(screen, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
