"""Paired LoCoMo experiment: obligation-guided search on frozen Witness memory."""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import random
import re
import subprocess
import time
from pathlib import Path
from statistics import mean

from wrag import config as C
from wrag.controlled import free_port
from wrag.data import Question
from wrag.eval.controlled import atomic_json, load
from wrag.eval.counts import numeric_diagnostic
from wrag.eval.gap_search import prior_excerpts, propose, replace_tail, verify
from wrag.eval.locomo_official import available, score_record
from wrag.eval.reader import read
from wrag.pilot import stop_owned, wait_ready
from wrag.reading_experiment import source_rows
from wrag.util import sha
from wrag.witness.query import looks_like_answer_set

REPO = Path(__file__).resolve().parent.parent
ARMS = ("common", "gap-lexical", "gap-verified", "gap-two", "count-full")


def memories(source: Path) -> dict[str, Path]:
    answer = {}
    for path in sorted((source / "controlled").glob("*/memory.json")):
        for qid in load(path)["identity"]["corpus"]["question_ids"]:
            if qid in answer:
                raise ValueError(f"Duplicate memory question: {qid}")
            answer[qid] = path
    return answer


def corpus_from(path: Path):
    manifest = load(path)
    raw = path.with_suffix(".pkl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError(f"Frozen memory checksum changed: {path}")
    # Controlled checkpoints generated locally by this repository only.
    graph, _ = pickle.loads(raw)
    if graph.corpus.stats()["corpus_hash"] != manifest["identity"]["corpus"]["corpus_hash"]:
        raise ValueError(f"Frozen corpus changed: {path}")
    return graph.corpus


def identity(source: Path, rows: list[dict], pilot: dict) -> dict:
    try:
        source_id = source.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        source_id = str(source.resolve())
    return {
        "version": 1, "source": source_id,
        "model": pilot["settings"]["model"],
        "model_revision": pilot["settings"].get("model_revision", ""),
        "code": sha([(str(p.relative_to(REPO)), p.read_text(encoding="utf-8"))
                    for p in sorted((REPO / "wrag").rglob("*.py"))]),
        "retrieval_hashes": sha([(r["qid"], r["snapshot"]["hash"]) for r in rows]),
        "protocol": "frozen evidence/common; full pages; 5 pids; common reader",
        "arms": list(ARMS), "top_k": 5, "max_added_primary": 1,
        "max_added_exploratory": 2, "max_verify": 4,
        "proof_reader": False, "temperature": 0, "answer_set": True,
    }


def prepare(output: Path, expected: dict):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "gap-manifest.json"
    if path.exists():
        if load(path) != expected:
            raise ValueError("Code/source/config changed; use a new output directory")
        return
    atomic_json(path, expected)


def item_diagnostic(question: str, answer: str, source_text: str) -> dict:
    if not looks_like_answer_set(question):
        return {"applicable": False}
    items = [x.strip() for x in answer.split(",") if x.strip()]
    text = re.sub(r"\W+", " ", source_text.casefold())
    missing = [x for x in items if re.sub(r"\W+", " ", x.casefold()).strip() not in text]
    return {"applicable": True, "items": items, "unmatched_literal": missing,
            "note": "Literal mismatch does not imply absence of semantic support."}


def answer(llm, corpus, question, pids, cfg, *, count=False):
    reading = read(llm, corpus, question, pids, cfg, method="witnessrag", count_mode=count)
    official = score_record({"resposta": reading.answer, "respostas_ouro": question.answers,
                             "tipo": question.qtype, "qid": question.qid})
    pages = [corpus.get(pid).text for pid in pids]
    return {
        "resposta": reading.answer, **official, "filtered": reading.filtered,
        "pids": list(pids), "all_recall@5": float(
            bool(question.gold_pids) and set(question.gold_pids) <= set(pids)),
        "numeric": numeric_diagnostic(question.question, reading.answer, question.answers[0]),
        "items": item_diagnostic(question.question, reading.answer, "\n".join(pages)),
        "prompt_tokens": reading.prompt_tokens, "completion_tokens": reading.completion_tokens,
        "reader_seconds": reading.latency_s,
    }


def worker(output: Path, source: Path, rows: list[dict], gpu: str, port_base: int,
           existing_server: bool, shard: int, shards: int):
    if not available():
        raise RuntimeError("nltk unavailable in benchmark environment")
    pilot = load(source / "pilot" / "pilot.json")
    model = pilot["settings"]["model"]
    port = port_base if existing_server else free_port(port_base)
    env = {**os.environ, "WRAG_LLM_BACKEND": "vllm", "OPENAI_MODEL": model,
           "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
           "OPENAI_API_KEY": "local-pilot", "WRAG_LLM_CACHE": "0",
           "WRAG_SEED": "42", "PYTHONHASHSEED": "42"}
    command = list(pilot["server_command"])
    command[0] = os.getenv("VLLM_PYTHON", str(REPO / ".venv-vllm/bin/python"))
    command[command.index("--port") + 1] = str(port)
    server = None
    log = None
    try:
        if not existing_server:
            log = (output / f"vllm-{shard}.log").open("a", encoding="utf-8")
            server = subprocess.Popen(
                command, env={**pilot.get("env", {}), **env,
                              "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_ready(server, port, model, time.monotonic() + 900)
        os.environ.update(env)
        from wrag.llm import get_llm
        llm = get_llm("vllm", use_cache=False)
        cfg = C.QAConfig(top_k=5, max_tokens=512, temperature=0,
                         answer_set=True, proof_reader=False)
        memory_files = memories(source)
        loaded = {}
        selected = [r for r in rows if int(sha(r["qid"])[:8], 16) % shards == shard]
        for index, item in enumerate(selected, 1):
            qid, base, snap = item["qid"], item["base"], item["snapshot"]
            target = output / "records" / (sha(qid) + ".json")
            if target.exists():
                saved = load(target)
                if saved["source_hash"] != snap["hash"] or set(saved["arms"]) != set(ARMS):
                    raise ValueError(f"Incompatible checkpoint: {qid}")
                continue
            path = memory_files[qid]
            if path not in loaded:
                loaded[path] = corpus_from(path)
            corpus = loaded[path]
            for page in snap["passages"]:
                actual = corpus.get(page["pid"])
                if (actual.title, actual.text) != (page["title"], page["text"]):
                    raise ValueError(f"Source page changed: {qid}/{page['pid']}")
            question = Question(qid, base["pergunta"], base["respostas_ouro"],
                                base["passagens_ouro"], dataset="locomo", qtype=base["tipo"])
            pids = snap["pids"]
            eligible = (base["tipo"] == "multi-hop" and
                        snap["diagnostics"].get("classe_prova") != "full")
            candidates = propose(question.question, snap["diagnostics"], corpus, pids) if eligible else []
            lexical = replace_tail(pids, [c.pid for c in candidates[:1]])
            aggressive = replace_tail(pids, [c.pid for c in candidates[:2]], max_added=2)
            checked, accepted = [], []
            for candidate in candidates[:4]:
                prior = prior_excerpts(question.question, candidate.obligation, corpus, pids)
                result = verify(llm, question.question, candidate,
                                corpus.get(candidate.pid).text, prior)
                checked.append({"pid": candidate.pid, "obligation": candidate.obligation,
                                "probe": candidate.probe, "score": candidate.score,
                                "candidate_line": candidate.quote, **result})
                if result["accepted"]:
                    accepted.append(candidate.pid)
                if accepted:
                    break
            verified = replace_tail(pids, accepted)
            contexts = {"common": pids, "gap-lexical": lexical,
                        "gap-verified": verified, "gap-two": aggressive}
            answers = {}
            first_arm = {}
            arms = {}
            for arm, context in contexts.items():
                key = tuple(context)
                if key not in answers:
                    answers[key] = answer(llm, corpus, question, context, cfg)
                    first_arm[key] = arm
                arms[arm] = {**answers[key]}
                if arm != first_arm[key]:
                    arms[arm]["aliased_to"] = first_arm[key]
            if re.search(r"\bhow many\b", question.question, re.I):
                arms["count-full"] = answer(llm, corpus, question, pids, cfg, count=True)
            else:
                arms["count-full"] = {**arms["common"], "aliased_to": "common"}
            atomic_json(target, {
                "qid": qid, "tipo": base["tipo"], "pergunta": question.question,
                "ouro": question.answers, "n_apoios_anotados": len(question.gold_pids),
                "source_hash": snap["hash"], "baseline_pids": pids,
                "eligible": eligible, "obligations": sorted({c.obligation for c in candidates}),
                "candidates": [c.__dict__ for c in candidates],
                "verification": checked, "arms": arms,
            })
            if index % 25 == 0:
                print(f"GPU {gpu}, shard {shard}: {index}/{len(selected)}", flush=True)
    finally:
        stop_owned(server)
        if log:
            log.close()


def report(output: Path, rows: list[dict]) -> dict:
    records = []
    for item in rows:
        path = output / "records" / (sha(item["qid"]) + ".json")
        if not path.exists():
            raise ValueError(f"Incomplete experiment: {path}")
        record = load(path)
        if record["source_hash"] != item["snapshot"]["hash"] or set(record["arms"]) != set(ARMS):
            raise ValueError(f"Incompatible result: {record['qid']}")
        records.append(record)
    result = {"complete": True, "n": len(records), "arms": {}, "paired": {},
              "eligible": sum(r["eligible"] for r in records),
              "lexical_changed": sum(r["arms"]["gap-lexical"]["pids"] != r["baseline_pids"] for r in records),
              "verified_changed": sum(r["arms"]["gap-verified"]["pids"] != r["baseline_pids"] for r in records),
              "two_changed": sum(r["arms"]["gap-two"]["pids"] != r["baseline_pids"] for r in records)}
    lines = ["# Busca dirigida na memória congelada", "",
             "Mesmas cinco passagens integrais no leitor comum; apenas a recuperação varia.", "",
             "| Condição | F1 geral | F1 single | F1 multi | AR@5 multi | F1 contagem | acerto numérico | tokens leitor |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    multi = [r for r in records if r["tipo"] == "multi-hop"]
    single = [r for r in records if r["tipo"] == "single-hop"]
    for arm in ARMS:
        counts = [r for r in records if r["arms"][arm]["numeric"]["aplicavel"]]
        numeric = [r["arms"][arm]["numeric"] for r in counts if r["arms"][arm]["numeric"]["avaliavel"]]
        s = {
            "f1": mean(r["arms"][arm]["f1_locomo"] for r in records),
            "single": mean(r["arms"][arm]["f1_locomo"] for r in single),
            "multi": mean(r["arms"][arm]["f1_locomo"] for r in multi),
            "multi_ar5": mean(r["arms"][arm]["all_recall@5"] for r in multi),
            "count_f1": mean(r["arms"][arm]["f1_locomo"] for r in counts),
            "count_n": len(counts),
            "numeric_accuracy": mean(x["correto"] for x in numeric) if numeric else None,
            "numeric_n": len(numeric),
            "reader_prompt_tokens": sum(r["arms"][arm]["prompt_tokens"] for r in records),
            "reader_completion_tokens": sum(r["arms"][arm]["completion_tokens"] for r in records),
            "reader_seconds": sum(r["arms"][arm]["reader_seconds"] for r in records),
            "unmatched_literal_items": sum(len(r["arms"][arm]["items"].get("unmatched_literal", []))
                                           for r in records),
        }
        result["arms"][arm] = s
        lines.append(f"| {arm} | {s['f1']:.4f} | {s['single']:.4f} | {s['multi']:.4f} | "
                     f"{s['multi_ar5']:.4f} | {s['count_f1']:.4f} | "
                     f"{s['numeric_accuracy'] or 0:.4f} ({len(numeric)}) | {s['reader_prompt_tokens']} |")
    for arm in ARMS[1:]:
        subset = multi if arm != "count-full" else [
            r for r in records if re.search(r"\bhow many\b", r["pergunta"], re.I)]
        delta = [r["arms"][arm]["f1_locomo"] - r["arms"]["common"]["f1_locomo"] for r in subset]
        groups = {}
        for row, d in zip(subset, delta):
            groups.setdefault(row["qid"].split(":")[1], []).append(d)
        rng = random.Random(42)
        values = list(groups.values())
        draws = sorted(sum(map(sum, sample)) / sum(map(len, sample))
                       for sample in (rng.choices(values, k=len(values)) for _ in range(10000)))
        result["paired"][arm] = {"n": len(subset), "delta": mean(delta),
                                 "ci95_by_conversation": [draws[249], draws[9749]],
                                 "wins": sum(d > 0 for d in delta), "losses": sum(d < 0 for d in delta),
                                 "ties": sum(d == 0 for d in delta)}
        lines.append(f"\n{arm} − common: {mean(delta):+.4f}; IC95% por conversa "
                     f"[{draws[249]:+.4f}, {draws[9749]:+.4f}] (n={len(subset)}).")
    checks = [c for r in records for c in r["verification"]]
    result["verification"] = {"calls": len(checks), "accepted": sum(c["accepted"] for c in checks),
                              "prompt_tokens": sum(c["prompt_tokens"] for c in checks),
                              "completion_tokens": sum(c["completion_tokens"] for c in checks),
                              "seconds": sum(c["latency_s"] for c in checks)}
    measured = [cell for row in records for cell in row["arms"].values()
                if "aliased_to" not in cell]
    result["actual_reader_cost"] = {
        "calls": len(measured),
        "prompt_tokens": sum(c["prompt_tokens"] for c in measured),
        "completion_tokens": sum(c["completion_tokens"] for c in measured),
        "seconds": sum(c["reader_seconds"] for c in measured),
    }
    lines += ["", f"Elegíveis: {result['eligible']}; contexto alterado: lexical "
              f"{result['lexical_changed']}, verificado {result['verified_changed']}, "
              f"dois novos {result['two_changed']}.",
              "Custos reais de chamadas novas e do verificador estão em comparison.json; "
              "tokens por condição incluem as respostas reutilizadas.",
              "Correspondência literal ausente não equivale a item sem suporte semântico.",
              "F1 oficial permanece a métrica principal; acerto numérico é secundário."]
    atomic_json(output / "comparison.json", result)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=Path("runs/locomo-controlled-01"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--existing-server", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    source, output = args.source.resolve(), args.output.resolve()
    rows = source_rows(source)
    if args.check:
        paths = memories(source)
        if len(paths) != len(rows):
            raise ValueError("Frozen memories do not cover every question")
        for path in set(paths.values()):
            corpus_from(path)
        print(f"Fonte íntegra: {len(rows)} perguntas, 10 memórias congeladas")
        return
    prepare(output, identity(source, rows, load(source / "pilot" / "pilot.json")))
    if args.prepare:
        print(output / "gap-manifest.json")
        return
    if args.report:
        report(output, rows)
        print(output / "comparison.md")
        return
    if os.name != "posix":
        raise RuntimeError("Run on Linux; --check and --report work locally")
    import fcntl
    with (output / f".gap-{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker(output, source, rows, os.getenv("GPU", "4"), int(os.getenv("PORT", "8091")),
               args.existing_server, args.shard, args.shards)
    if args.shards == 1:
        report(output, rows)
        print(output / "comparison.md")


if __name__ == "__main__":
    main()
