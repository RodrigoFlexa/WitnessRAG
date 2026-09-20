"""Paired reader experiment over frozen evidence/common LoCoMo retrievals.

Run: python -m wrag.reading_experiment runs/locomo-reading-01
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

from wrag import config as C
from wrag.controlled import free_port
from wrag.data import Corpus, Passage, Question
from wrag.eval.controlled import atomic_json, load
from wrag.eval.counts import numeric_diagnostic
from wrag.eval.diagnostics import stop_reason
from wrag.eval.focused_context import focus_passage
from wrag.eval.locomo_official import available, score_record
from wrag.eval.reader import read
from wrag.pilot import stop_owned, wait_ready
from wrag.util import sha

REPO = Path(__file__).resolve().parent.parent
ARMS = ("common", "focused", "focused-count")


def source_rows(source: Path) -> list[dict]:
    folders = sorted((source / "controlled").glob("*/memory.json"))
    if len(folders) != 10:
        raise ValueError(f"Esperadas 10 conversas completas no controle; encontradas {len(folders)}")
    rows = []
    seen = set()
    for memory_file in folders:
        folder = memory_file.parent
        expected = set(load(memory_file)["identity"]["corpus"]["question_ids"])
        answers = sorted((folder / "answers").glob("*.json"))
        if len(answers) != len(expected):
            raise ValueError(f"{folder}: {len(answers)}/{len(expected)} respostas")
        actual = set()
        for path in answers:
            answer = load(path)
            qid = answer["qid"]
            base = answer["cells"]["evidence/common"]
            snapshot = load(folder / "retrieval" / "evidence" / (sha(qid) + ".json"))["retrieval"]
            digest = sha({k: v for k, v in snapshot.items() if k != "hash"})
            if digest != snapshot["hash"] or base["retrieval_hash"] != digest:
                raise ValueError(f"Recuperação alterada: {qid}")
            pids = snapshot["pids"]
            if pids != base["recuperadas"] or pids != [p["pid"] for p in snapshot["passages"]]:
                raise ValueError(f"Passagens divergentes: {qid}")
            if qid in seen or qid in actual:
                raise ValueError(f"Pergunta duplicada: {qid}")
            seen.add(qid)
            actual.add(qid)
            rows.append({"qid": qid, "base": base, "snapshot": snapshot})
        if actual != expected:
            raise ValueError(f"Perguntas ausentes em {folder}")
    if len(rows) != 1123:
        raise ValueError(f"Esperadas 1123 perguntas; encontradas {len(rows)}")
    return rows


def run(output: Path, source: Path, existing_server: bool) -> None:
    if not available():
        raise RuntimeError("Instale nltk no ambiente benchmark")
    rows = source_rows(source)
    source_pilot = load(source / "pilot" / "pilot.json")
    model = source_pilot["settings"]["model"]
    code = sha([(str(p.relative_to(REPO)), p.read_text(encoding="utf-8"))
               for p in sorted((REPO / "wrag").rglob("*.py"))])
    identity = {"protocol": "frozen evidence/common retrieval; common vs focused vs focused-count",
                "source": str(source.resolve()), "model": model, "model_revision": source_pilot["settings"].get("model_revision", ""),
                "code": code, "question_hashes": sha([(r["qid"], r["snapshot"]["hash"]) for r in rows]),
                "focus_chars": 3000, "neighbors": 1, "proof_reader": False,
                "reader": {"top_k": 5, "max_tokens": 512, "temperature": 0, "answer_set": True}}
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "reading-manifest.json"
    if manifest_file.exists() and load(manifest_file) != identity:
        raise ValueError("Fonte/código/configuração alterados; use nova pasta de saída")
    atomic_json(manifest_file, identity)
    if len(list((output / "records").glob("*.json"))) == len(rows):
        report(output, rows)
        return
    port = int(os.getenv("PORT", "8089")) if existing_server else free_port(int(os.getenv("PORT", "8089")))
    env = {**os.environ, "WRAG_LLM_BACKEND": "vllm", "OPENAI_MODEL": model,
           "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1", "OPENAI_API_KEY": "local-pilot",
           "WRAG_LLM_CACHE": "0", "WRAG_CACHE_DIR": str(output / "cache"),
           "WRAG_SEED": "42", "PYTHONHASHSEED": "42"}
    command = list(source_pilot["server_command"])
    command[0] = os.getenv("VLLM_PYTHON", str(REPO / ".venv-vllm/bin/python"))
    command[command.index("--port") + 1] = str(port)
    if "--max-num-seqs" in command:
        command[command.index("--max-num-seqs") + 1] = "4"
    server = None
    server_log = None
    try:
        if not existing_server:
            server_log = (output / "vllm.log").open("a", encoding="utf-8")
            server = subprocess.Popen(command, env={**source_pilot.get("env", {}), **env,
                                                    "CUDA_VISIBLE_DEVICES": os.getenv("GPU", "4"),
                                                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                                      cwd=REPO, stdout=server_log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        wait_ready(server, port, model, time.monotonic() + 900)
        worker(output, rows, env)
        report(output, rows)
    finally:
        stop_owned(server)
        if server_log:
            server_log.close()


def worker(output: Path, rows: list[dict], env: dict) -> None:
    os.environ.update(env)
    from wrag.llm import get_llm
    llm = get_llm("vllm", use_cache=False)
    cfg = C.QAConfig(top_k=5, max_tokens=512, temperature=0, answer_set=True, proof_reader=False)
    for index, item in enumerate(rows, 1):
        qid, base, snap = item["qid"], item["base"], item["snapshot"]
        path = output / "records" / (sha(qid) + ".json")
        if path.exists():
            saved = load(path)
            if saved["retrieval_hash"] != snap["hash"] or set(saved["arms"]) != set(ARMS):
                raise ValueError(f"Checkpoint incompatível: {qid}")
            continue
        passages = [Passage(**p) for p in snap["passages"]]
        corpus = Corpus("locomo", passages, [])
        question = Question(qid, base["pergunta"], base["respostas_ouro"],
                            base["passagens_ouro"], dataset="locomo", qtype=base["tipo"])
        focused = []
        selection = []
        for passage in passages:
            text, info = focus_passage(passage.text, question.question)
            focused.append((passage.title, text))
            selection.append({"pid": passage.pid, **info, "text": text})
        arms = {}
        for arm in ARMS:
            if arm == "focused-count" and not re.search(r"\bhow many\b", question.question, re.I):
                arms[arm] = {**arms["focused"], "aliased_to": "focused"}
                continue
            reading = read(llm, corpus, question, snap["pids"], cfg,
                           passages_override=focused if arm != "common" else None,
                           count_mode=arm == "focused-count")
            record = {**base, "resposta": reading.answer}
            scores = score_record(record)
            arms[arm] = {"resposta": reading.answer, **scores, "filtrada": reading.filtered,
                         "acerto_numerico": numeric_diagnostic(question.question, reading.answer, question.answers[0]),
                         "prompt_tokens": reading.prompt_tokens, "completion_tokens": reading.completion_tokens,
                         "latency_s": reading.latency_s}
        atomic_json(path, {"qid": qid, "tipo": base["tipo"], "pergunta": base["pergunta"],
                           "ouro": base["respostas_ouro"], "n_apoios_anotados": len(base["passagens_ouro"]),
                           "pids": snap["pids"], "retrieval_hash": snap["hash"],
                           "all_recall@5": base["all_recall@5"],
                           "motivo_parada": stop_reason(snap["diagnostics"]),
                           "historical_common_f1": base["f1_locomo"],
                           "focused_selection": selection, "arms": arms})
        if index % 25 == 0:
            print(f"Leitura: {index}/{len(rows)}", flush=True)


def report(output: Path, source: list[dict]) -> None:
    records = [load(output / "records" / (sha(item["qid"]) + ".json")) for item in source]
    if len(records) != len(source) or any(set(r["arms"]) != set(ARMS) for r in records):
        raise ValueError("Leituras incompletas")
    summary = {"complete": True, "n": len(records), "source": str(source[0]["qid"]).split(":")[0],
               "arms": {}, "paired": {}}
    lines = ["# Leitura sobre recuperação congelada", "", "Mesmos pids nas três condições; leitor de provas desligado.\n",
             "| Condição | F1 geral | F1 single | F1 multi | F1 contagem | acerto numérico | tokens prompt |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        def sub(kind):
            return [r for r in records if r["tipo"] == kind]
        counts = [r for r in records if r["arms"][arm]["acerto_numerico"]["aplicavel"]]
        numeric = [r["arms"][arm]["acerto_numerico"] for r in counts
                   if r["arms"][arm]["acerto_numerico"]["avaliavel"]]
        data = {"f1": mean(r["arms"][arm]["f1_locomo"] for r in records),
                "single": mean(r["arms"][arm]["f1_locomo"] for r in sub("single-hop")),
                "multi": mean(r["arms"][arm]["f1_locomo"] for r in sub("multi-hop")),
                "count_f1": mean(r["arms"][arm]["f1_locomo"] for r in counts),
                "numeric_accuracy": mean(x["correto"] for x in numeric) if numeric else None,
                "numeric_n": len(numeric), "count_n": len(counts),
                "prompt_tokens": sum(r["arms"][arm]["prompt_tokens"] for r in records),
                "completion_tokens": sum(r["arms"][arm]["completion_tokens"] for r in records),
                "reader_seconds": sum(r["arms"][arm]["latency_s"] for r in records)}
        summary["arms"][arm] = data
        lines.append(f"| {arm} | {data['f1']:.4f} | {data['single']:.4f} | {data['multi']:.4f} | "
                     f"{data['count_f1']:.4f} | {data['numeric_accuracy'] if data['numeric_accuracy'] is not None else 0:.4f} ({len(numeric)}) | {data['prompt_tokens']} |")
    for arm in ARMS[1:]:
        multi = [r for r in records if r["tipo"] == "multi-hop"]
        delta = [r["arms"][arm]["f1_locomo"] - r["arms"]["common"]["f1_locomo"] for r in multi]
        groups = {}
        for r, d in zip(multi, delta):
            groups.setdefault(r["qid"].split(":")[1], []).append(d)
        rng = random.Random(42)
        values = list(groups.values())
        draws = sorted(sum(map(sum, sample)) / sum(map(len, sample))
                       for sample in (rng.choices(values, k=len(values)) for _ in range(10000)))
        summary["paired"][arm] = {"multi_delta": mean(delta), "multi_ci95_by_conversation": [draws[249], draws[9749]],
                                   "wins": sum(d > 0 for d in delta), "losses": sum(d < 0 for d in delta),
                                   "ties": sum(d == 0 for d in delta)}
        lines.append(f"\n{arm} − common, multi: {mean(delta):+.4f}; IC95% por conversa "
                     f"[{draws[249]:+.4f}, {draws[9749]:+.4f}].")
    lines.append("\nF1 oficial sem alteração; acerto numérico usa somente gabaritos numéricos inequívocos e conta respostas não numéricas como erro.")
    lines.append("A resposta common histórica é registrada por pergunta para medir deriva entre execuções, sem servir de controle pareado.")
    atomic_json(output / "comparison.json", summary)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("runs/locomo-controlled-01"))
    parser.add_argument("--existing-server", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    rows = source_rows(source)
    if args.check:
        print(f"Fonte íntegra: {len(rows)} perguntas; 10 conversas; pids congelados")
        return
    if os.name != "posix":
        raise RuntimeError("Execute no servidor Linux; --check funciona localmente")
    import fcntl
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".reading.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(output, source, args.existing_server)
    print(output / "comparison.md")


if __name__ == "__main__":
    main()
