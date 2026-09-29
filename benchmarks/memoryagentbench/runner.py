"""Checkpointed evaluation. Resume rejects changed data, code and settings."""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from wrag.llm.base import usage_delta
from . import protocol as P
from . import reader
from .data import chunks_for, dataset_root, tokenizer_identity, verify_data
from .metrics import MovieScorer, score


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                # Never discard a broken checkpoint: require an explicit fix.
                raise ValueError(f"Damaged checkpoint {path}, line {index + 1}") from exc
    return rows


def append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def lock(output: Path):
    path = output / "run.lock"
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": time.time()}, handle)
    except FileExistsError as exc:
        raise RuntimeError(f"Run locked: {path}. Check its process before removing a stale lock.") from exc
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def run(samples: list, engine, llm, cache: Path, output: Path, *, protocol: str = "official",
        reflection: bool = True, resume: bool = False, max_questions: int = 0,
        max_contexts: int = 0, seed: int | None = None, identity_extra: dict | None = None) -> dict:
    if not samples:
        raise ValueError("No samples selected")
    output.mkdir(parents=True, exist_ok=True)
    selection = [{"key": s.key, "source": s.source, "split": s.split,
                  "questions": len(s.questions),
                  "qa_hash": P.digest([s.questions, s.answers, s.metadata])} for s in samples]
    settings = {s.source: P.task_settings(s.source, protocol) for s in samples}
    identity = {"schema_version": 1, "dataset_revision": P.DATASET_REVISION,
                "upstream_commit": P.UPSTREAM_COMMIT, "files": verify_data(cache),
                "tokenizer": tokenizer_identity(cache), "selection": selection,
                "settings": settings, "engine": engine.cfg.to_dict(),
                "task_adaptations": engine.protocol_identity() if hasattr(engine, "protocol_identity") else {},
                "reflection": reflection, "reader_seed": seed,
                "model": getattr(llm, "deployment", llm.name),
                "provider_identity": llm.cache_identity() if hasattr(llm, "cache_identity") else llm.name,
                "python_hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
                **(identity_extra or {})}
    path = output / "manifest.json"
    with lock(output):
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not resume:
                raise ValueError("Output already exists; use --resume or a new output folder")
            if P.digest(previous["identity"]) != P.digest(identity):
                raise ValueError("Resume identity mismatch (code/data/model/settings). Use a new output folder.")
        else:
            previous = {"identity": identity, "started": time.time()}
            atomic_json(path, previous)
        rows = read_rows(output / "results.jsonl")
        ids = [r["qid"] for r in rows]
        intended = {s.qid(i) for s in samples for i in range(len(s.questions))}
        if len(ids) != len(set(ids)) or not set(ids) <= intended:
            raise ValueError("Duplicate or foreign questions in checkpoint")
        done = set(ids)
        movies = None
        if any(s.source.startswith("recsys_") for s in samples):
            if os.environ.get("PYTHONHASHSEED") != "0":
                raise ValueError("Recommendation metric requires PYTHONHASHSEED=0; use the PowerShell entrypoint.")
            movies = MovieScorer(json.loads((dataset_root(cache) / "entity2id.json").read_text(encoding="utf-8")))
        visited = 0
        for sample in samples:
            if all(sample.qid(i) in done for i in range(len(sample.questions))):
                continue
            if (max_questions and len(rows) >= max_questions) or (max_contexts and visited >= max_contexts):
                break
            visited += 1
            chunks = chunks_for(sample, cache, settings[sample.source])
            print(f"Registering {sample.source} context={sample.row_index} chunks={len(chunks)}", flush=True)
            memory_before, memory_started = llm.usage.snapshot(), time.perf_counter()
            try:
                cost = engine.prepare(sample.key, sample.source, chunks, sample.context)
            except Exception as exc:
                append(output / "index.jsonl", {"context_key": sample.key, "source": sample.source,
                       "timestamp": time.time(), "failed": True, "error_type": type(exc).__name__,
                       "cost": {"seconds": time.perf_counter() - memory_started,
                                "usage": usage_delta(llm.usage.snapshot(), memory_before)}})
                raise
            append(output / "index.jsonl", {"context_key": sample.key, "source": sample.source,
                   "timestamp": time.time(), "cost": cost})
            for index, question in enumerate(sample.questions):
                qid = sample.qid(index)
                if qid in done:
                    continue
                if max_questions and len(rows) >= max_questions:
                    break
                before, start = llm.usage.snapshot(), time.perf_counter()
                query = sample.query(index)
                try:
                    selected = engine.select(qid, question)
                    result = (None if selected.filtered else reader.answer(llm, selected.context, query,
                              settings[sample.source], reflection=reflection, seed=seed))
                    if result is not None and result.error:
                        raise RuntimeError(f"Reader failed: {result.error}")
                    # Protocol-critical parameters cannot silently disappear in
                    # a compatible endpoint's parameter negotiation.
                    if set(getattr(llm, "_unsupported", ())) & {"max_tokens", "temperature"}:
                        raise RuntimeError("Endpoint rejected the official output budget/temperature")
                except Exception as exc:
                    append(output / "query-failures.jsonl", {"qid": qid, "error_type": type(exc).__name__,
                           "seconds": time.perf_counter() - start,
                           "usage": usage_delta(llm.usage.snapshot(), before)})
                    raise
                text = result.text if result is not None and not result.filtered else ""
                metrics, parsed = score(sample.source, text, sample.evaluation_answers(index), movies)
                filtered = selected.filtered or bool(result and result.filtered)
                if filtered:
                    metrics = {key: 0.0 for key in metrics}
                    metrics[P.primary_metric(sample.source)] = 0.0
                row = {"qid": qid, "context_key": sample.key, "source": sample.source, "split": sample.split,
                       "question_index": index, "qa_pair_id": sample.official_id(index),
                       "question": question, "query": query, "answer": sample.evaluation_answers(index),
                       "output": text, "parsed_output": parsed, "metrics": metrics,
                       "primary_metric": P.primary_metric(sample.source),
                       "judge_pending": P.primary_metric(sample.source) not in metrics,
                       "filtered": filtered,
                       "finish_reason": result.finish_reason if result else "retrieval_filter",
                       "usage": usage_delta(llm.usage.snapshot(), before),
                       "query_time_len": time.perf_counter() - start,
                       "retrieval_seconds": selected.retrieve_s, "diagnostics": selected.diagnostics,
                       "reader_context": selected.context, "reader_context_hash": P.digest(selected.context),
                       "reader_output_budget": settings[sample.source]["generation_max_length"],
                       "completion_tokens": result.completion_tokens if result else 0}
                append(output / "results.jsonl", row)
                rows.append(row)
                done.add(qid)
                print(f"[{len(rows)}/{len(intended)}] {sample.source} {row['primary_metric']}="
                      f"{metrics.get(row['primary_metric'], 'judge pending')}", flush=True)
                atomic_json(output / "status.json", {"answered": len(rows), "intended": len(intended),
                            "generation_complete": len(rows) == len(intended), "updated": time.time()})
        from .report import report
        return report(output)


def judge_run(samples: list, output: Path, clients: dict, *, code_hash: str = "") -> dict:
    from .judge import evaluate
    if not (output / "manifest.json").exists():
        raise FileNotFoundError("No benchmark manifest")
    with lock(output):
        rows = read_rows(output / "results.jsonl")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"]["dataset_revision"] != P.DATASET_REVISION:
            raise ValueError("Judge dataset revision does not match this run")
        lookup = {s.qid(i): (s, i) for s in samples for i in range(len(s.questions))}
        identity = {"code_hash": code_hash, "models": {
            kind: {"model": getattr(client, "deployment", client.name),
                   "provider": client.cache_identity() if hasattr(client, "cache_identity") else client.name}
            for kind, client in clients.items()}}
        identity_path = output / "judge-manifest.json"
        if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Judge identity changed; do not combine scores from different judges")
        atomic_json(identity_path, identity)
        judgments = read_rows(output / "judgments.jsonl")
        done = {j["qid"]: j for j in judgments}
        if len(done) != len(judgments):
            raise ValueError("Duplicate judge records")
        for row in rows:
            if not row["judge_pending"] or row["qid"] in done:
                continue
            if row["qid"] not in lookup:
                raise ValueError("Reference question absent from the pinned dataset")
            sample, index = lookup[row["qid"]]
            if sample.evaluation_answers(index) != row["answer"] or sample.questions[index] != row["question"]:
                raise ValueError("Judge/reference mismatch")
            kind = "longmemeval" if sample.source.startswith("longmemeval_") else "summary"
            llm = clients[kind]
            if getattr(llm, "reasoning", False):
                raise ValueError("Official judges require non-reasoning output budgets")
            before = llm.usage.snapshot()
            # Content-filtered agent answers are scored zero rather than
            # silently excluding them or asking a judge to excuse them.
            try:
                outcome = ({"metrics": {row["primary_metric"]: 0.0}, "responses": [],
                            "reason": "agent_filtered"} if row["filtered"] else evaluate(sample, index, row["output"], llm))
                if set(getattr(llm, "_unsupported", ())) & {"max_tokens", "temperature", "top_p", "seed"}:
                    raise RuntimeError("Endpoint rejected official judge generation parameters")
            except Exception as exc:
                append(output / "judge-failures.jsonl", {"qid": row["qid"], "error_type": type(exc).__name__,
                       "usage": usage_delta(llm.usage.snapshot(), before)})
                raise
            append(output / "judgments.jsonl", {"qid": row["qid"], "output_hash": P.digest(row["output"]),
                   **outcome, "usage": usage_delta(llm.usage.snapshot(), before)})
            print(f"Judged {row['source']} q{index}: {outcome['metrics'][row['primary_metric']]:.4f}", flush=True)
        from .report import report
        return report(output)
