"""Launcher and report for the controlled LoCoMo experiment (no legacy cache).

python -m wrag.controlled runs/locomo-controlled-01
python -m wrag.controlled runs/locomo-controlled-01 --report
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from statistics import mean

from wrag.eval.controlled import VERSION, atomic_json, load
from wrag.util import sha

REPO = Path(__file__).resolve().parent.parent
CELLS = ("evidence/common", "evidence/proof", "soft-v2/common", "soft-v2/proof")


class IncompleteExperiment(ValueError):
    pass


def configuration():
    return {"version": VERSION, "gpu": os.getenv("GPU", "4"),
            "conversation": os.getenv("LOCOMO_CONVERSATION", "all"),
            "model": "Qwen/Qwen2.5-14B-Instruct", "embed_model": "BAAI/bge-m3",
            "model_revision": os.getenv("MODEL_REVISION", ""),
            "concurrency": 8, "seed": 42,
            "code": sha([(str(p.relative_to(REPO)), p.read_text(encoding="utf-8"))
                         for p in sorted((REPO / "wrag").rglob("*.py"))]),
            "packages": {name: package_version(name) for name in
                         ("numpy", "scipy", "torch", "sentence-transformers", "transformers", "openai", "nltk")}}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def prepare(output, config):
    path = output / "controlled-manifest.json"
    if path.exists():
        if load(path)["config"] != config:
            raise ValueError("Código/configuração/ambiente mudou. Use outro diretório de saída.")
        return load(path)
    if any(p.name != ".controlled.lock" for p in output.iterdir()):
        raise ValueError("Diretório não vazio sem manifesto controlado; escolha uma nova saída.")
    manifest = {"config": config, "cache_id": sha(str(output.resolve()), config),
                "historical_caches_used": False,
                "protocol": "shared frozen memory and new LLM cache; two retrievals; four frozen-context reader cells"}
    atomic_json(path, manifest)
    return manifest


def free_port(base):
    for port in range(base, min(base + 201, 65536)):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("Nenhuma porta disponível; informe PORT com outra faixa")


def pilot_command(output, config, port):
    args = [sys.executable, "-m", "wrag.pilot", "--gpu", config["gpu"],
            "--dataset", "locomo", "--locomo-conversation", config["conversation"],
            "--methods", "witnessrag", "--model", config["model"],
            "--embed-model", config["embed_model"], "--locomo-chunk-tokens", "2048",
            "--locomo-ie-window-tokens", "512", "--top-k", "5",
            "--witness-candidate-pool", "20", "--binding-aware-grounding", "--answer-set",
            "--vocab-compile", "--hybrid-fallback", "--dialogue-ie", "--query-plans",
            "--max-query-plans", "5", "--active-frontier", "--active-obligations",
            "--active-context", "--proof-reader", "--port", str(port),
            "--seed", "42", "--concurrency", str(config["concurrency"]),
            "--vllm-python", os.getenv("VLLM_PYTHON", str(REPO / ".venv-vllm/bin/python")),
            "--cache-dir", str(output / "cache"), "--output", str(output / "pilot"),
            "--hours", os.getenv("HOURS", "18")]
    if config["model_revision"]:
        args += ["--model-revision", config["model_revision"]]
    if os.getenv("LOCOMO_FILE"):
        args += ["--locomo-file", os.environ["LOCOMO_FILE"]]
    if (output / "pilot/pilot.json").exists():
        args += ["--resume"]
    return args


def bootstrap(rows, base):
    groups = {}
    for qid, row in rows.items():
        if row["tipo"] == "multi-hop":
            groups.setdefault(qid.split(":")[1], []).append(row["f1_locomo"] - base[qid]["f1_locomo"])
    if len(groups) < 2:
        return None
    rng = random.Random(42)
    values = list(groups.values())
    draws = []
    for _ in range(10000):
        selected = rng.choices(values, k=len(values))
        draws.append(sum(map(sum, selected)) / sum(map(len, selected)))
    draws.sort()
    return [draws[249], draws[9749]]


def report(output):
    manifest = load(output / "controlled-manifest.json")
    folders = sorted((output / "controlled").glob("*/memory.json"))
    expected_n = 10 if manifest["config"]["conversation"] == "all" else 1
    if len(folders) != expected_n:
        raise IncompleteExperiment(f"Experimento incompleto: {len(folders)}/{expected_n} memórias")
    rows = {cell: {} for cell in CELLS}
    audits = []
    for memory_path in folders:
        folder = memory_path.parent
        memory = load(memory_path)
        expected = set(memory["identity"]["corpus"]["question_ids"])
        actual = set()
        audit = load(folder / "audit.json")
        if not audit["passed"] or audit["identity"]["memory"] != memory["sha256"]:
            raise ValueError("Auditoria ausente ou incompatível")
        audits.extend(audit["sample"])
        for path in sorted((folder / "answers").glob("*.json")):
            item = load(path)
            qid = item["qid"]
            if qid in actual or qid in rows[CELLS[0]]:
                raise ValueError("Pergunta duplicada")
            actual.add(qid)
            for cell in CELLS:
                row = item["cells"][cell]
                if row["qid"] != qid or row["memory_hash"] != memory["sha256"]:
                    raise ValueError("Identidade de célula divergente")
                if "f1_locomo" not in row:
                    raise ValueError("F1 oficial indisponível; instale nltk antes da execução")
                rows[cell][qid] = row
            for label in ("evidence", "soft-v2"):
                a, b = (item["cells"][label + "/" + mode] for mode in ("common", "proof"))
                if a["retrieval_hash"] != b["retrieval_hash"] or a["recuperadas"] != b["recuperadas"]:
                    raise ValueError("Leitores não receberam a mesma recuperação")
                retrieval = load(folder / "retrieval" / label / (sha(qid) + ".json"))["retrieval"]
                digest = sha({k: v for k, v in retrieval.items() if k != "hash"})
                if digest != retrieval["hash"] or digest != a["retrieval_hash"]:
                    raise ValueError("Snapshot de recuperação alterado")
        if actual != expected:
            raise IncompleteExperiment(f"{folder.name}: {len(actual)}/{len(expected)} perguntas; retome a execução")
    base = rows["evidence/common"]
    summaries = {}
    for cell, data in rows.items():
        summary = {"n": len(data), "f1": mean(r["f1_locomo"] for r in data.values()),
                   "multi_delta_ci95_by_conversation": bootstrap(data, base)}
        for kind in ("single-hop", "multi-hop"):
            subset = [r for r in data.values() if r["tipo"] == kind]
            summary[kind] = {"n": len(subset),
                             "f1": mean(r["f1_locomo"] for r in subset) if subset else None,
                             "ar5": mean(r["all_recall@5"] for r in subset) if subset else None}
        summary["retrieval_seconds_mean"] = mean(r["latencia_recuperacao_s"] for r in data.values())
        summary["reader_seconds_mean"] = mean(r["latencia_leitura_s"] for r in data.values())
        for label, key in (("retrieval", "uso_recuperacao"), ("reader", "uso_leitor")):
            summary[label + "_usage"] = {metric: sum(r[key]["total"].get(metric, 0) for r in data.values())
                                        for metric in ("chamadas", "em_cache", "tokens_prompt_sem_cache", "tokens_resposta_sem_cache")}
        summary["by_conversation"] = {}
        for conv in sorted({qid.split(":")[1] for qid in data}):
            subset = [r for qid, r in data.items() if qid.split(":")[1] == conv]
            summary["by_conversation"][conv] = {"n": len(subset), "f1": mean(r["f1_locomo"] for r in subset)}
        summaries[cell] = summary
        destination = output / "cells" / (cell.replace("/", "-") + ".jsonl")
        destination.parent.mkdir(exist_ok=True)
        destination.write_text("".join(json.dumps({k: v for k, v in row.items() if k != "reader_calls"},
                                                    ensure_ascii=False) + "\n" for row in data.values()), encoding="utf-8")
    result = {"complete": True, "cells": summaries,
              "audit": {"retrievals_checked": len(audits), "replay_identical": all(a["replay_identical"] for a in audits),
                        "fresh_identical": sum(a["fresh_identical"] for a in audits)},
              "cost_note": "Main retrieval uses a shared NEW experiment cache; report actual hits. Fresh and warm inference are measured separately on the audit sample. Retrieval shared by readers: do not sum twice. Indexing is recorded in pilot index_attempts.json."}
    result["cost_sample"] = {}
    for label in ("evidence", "soft-v2"):
        selected = [a for a in audits if a["condition"] == label]
        result["cost_sample"][label] = {
            "n": len(selected),
            "fresh_seconds_mean": mean(a["fresh_retrieval"]["latency_s"] for a in selected) if selected else None,
            "warm_seconds_mean": mean(a["warm_seconds"] for a in selected) if selected else None,
            "fresh_calls": sum(a["fresh_usage"]["total"]["chamadas"] for a in selected),
            "warm_calls": sum(a["warm_usage"]["total"]["chamadas"] for a in selected),
            "warm_hits": sum(a["warm_usage"]["total"]["em_cache"] for a in selected)}
    result["paired"] = {}
    for base_cell, changed_cell in (("evidence/common", "soft-v2/common"),
                                    ("evidence/common", "evidence/proof"),
                                    ("soft-v2/common", "soft-v2/proof")):
        pair = {}
        for kind in ("single-hop", "multi-hop"):
            deltas = [row["f1_locomo"] - rows[base_cell][qid]["f1_locomo"]
                      for qid, row in rows[changed_cell].items() if row["tipo"] == kind]
            pair[kind] = {"n": len(deltas), "delta": mean(deltas) if deltas else None,
                          "wins": sum(d > 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
                          "ties": sum(d == 0 for d in deltas)}
        pair["multi_ci95_by_conversation"] = bootstrap(rows[changed_cell], rows[base_cell])
        result["paired"][base_cell + " -> " + changed_cell] = pair
    atomic_json(output / "comparison.json", result)
    lines = ["# LoCoMo controlado", "", "| Condição | n | F1 geral | F1 single | F1 multi | Busca média (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for cell, s in summaries.items():
        def number(value):
            return f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {cell} | {s['n']} | {s['f1']:.4f} | {number(s['single-hop']['f1'])} | {number(s['multi-hop']['f1'])} | {s['retrieval_seconds_mean']:.2f} |")
    lines += ["", f"Auditoria: {len(audits)} recuperações com replay idêntico; "
              f"{result['audit']['fresh_identical']} também idênticas na repetição com inferência nova.",
              "", "Custos de busca são compartilhados pelos dois leitores; não somar duas vezes.",
              "Busca usa um cache novo compartilhado. Leitura usa chamadas novas; hits são registrados.",
              "Custo frio e quente é medido separadamente na amostra da auditoria (comparison.json).",
              "Intervalos por conversa, tokens e custos por célula estão em comparison.json.",
              "Os resultados são de desenvolvimento; estas conversas já orientaram a proposta."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report:
        report(output)
        print(output / "comparison.md")
        return
    config = configuration()
    if args.dry_run:
        print(json.dumps({"config": config, "command": pilot_command(output, config, int(os.getenv("PORT", "8089"))),
                          "output": str(output), "two_retrievals_four_readers": True}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("Execute o launcher no servidor Linux; --dry-run e --report funcionam localmente")
    import fcntl
    from wrag.eval.locomo_official import available
    if not available():
        raise RuntimeError("Instale nltk no ambiente do benchmark antes de executar")
    vllm_python = os.getenv("VLLM_PYTHON", str(REPO / ".venv-vllm/bin/python"))
    version = subprocess.run([vllm_python, "-c", "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
                             capture_output=True, text=True, check=True, timeout=30)
    config["vllm_version"] = version.stdout.strip()
    if config["conversation"] != "all" and not config["conversation"].isdigit():
        raise ValueError("LOCOMO_CONVERSATION deve ser all ou um índice inteiro")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".controlled.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(output, config)
        env = {**os.environ, "PYTHONHASHSEED": "42", "WRAG_CONTROLLED_ROOT": str(output),
               "WRAG_EXPERIMENT_CACHE_ID": manifest["cache_id"],
               "WRAG_LLM_CACHE": "1", "WRAG_EMBED_CACHE": "1"}
        for attempt in range(int(os.getenv("MAX_RESUMES", "8"))):
            try:
                report(output)
                print(f"Concluído: {output / 'comparison.md'}")
                return
            except (FileNotFoundError, IncompleteExperiment):
                pass
            port = free_port(int(os.getenv("PORT", "8089")))
            print(f"Execução controlada: tentativa {attempt + 1}, GPU {config['gpu']}, porta {port}", flush=True)
            completed = subprocess.run(pilot_command(output, config, port), env=env, cwd=REPO)
            status_path = output / "pilot/status.json"
            status = load(status_path).get("status") if status_path.exists() else "unknown"
            if status not in {"complete", "time_limit", "interrupted"}:
                raise RuntimeError(f"Execução falhou ({completed.returncode}); veja {output / 'pilot/benchmark.log'} e controlled/*/audit-error.json")
            if status == "complete":
                try:
                    report(output)
                except IncompleteExperiment:
                    continue
                print(f"Concluído: {output / 'comparison.md'}")
                return
        raise RuntimeError("Prazo de tentativas atingido. Retome com o mesmo comando e diretório.")


if __name__ == "__main__":
    main()
