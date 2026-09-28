"""Shared, locked inputs for the four reflection cells, including split GPU runs.

Only enabled by the reflection launcher. A query's cascade retrieval is frozen
once; summary interpretations are appended afterward without changing its facts,
plans, route or selected chunks. QA stays on the worker's own model server.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

from wrag.eval import controlled
from wrag.llm.base import usage_delta
from wrag.util import sha


def root():
    value = os.environ.get("WRAG_REFLECTION_STUDY_ROOT")
    return Path(value) if value else None


@contextmanager
def file_lock(path):
    """OS-owned lock, released on crashes; protects processes AND threads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if not handle.tell():
                handle.write(b"0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_graph(ctx, builder):
    folder = root() / "shared"
    with file_lock(folder / "locks" / (ctx.corpus.stats()["corpus_hash"] + ".graph.lock")):
        controlled.frozen_graph(ctx, builder, destination_root=folder)


def saved(path, identity, corpus, build):
    """Never serve a partially written snapshot or silently rebuild a bad one."""
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if sha(data["identity"]) != sha(identity):
                raise ValueError("Reflection study inputs changed; use a new output directory")
            return controlled.thaw(corpus, data["retrieval"]), data, True
        before = build.llm.usage.snapshot() if hasattr(build, "llm") else None
        result = build()
        if result.filtered:
            # A transport/filter failure must not become shared permanent evidence.
            raise RuntimeError("Filtered reflection-study retrieval; resume after checking the model log")
        data = {"identity": identity, "retrieval": controlled.snapshot(corpus, result)}
        if before is not None:
            data["construction_usage"] = usage_delta(build.llm.usage.snapshot(), before)
        atomic_json(path, data)
        return controlled.thaw(corpus, data["retrieval"]), data, False


def retrieve(retriever, corpus, question, cfg):
    if root() is None:
        return retriever.retrieve(question, cfg.top_k)
    if not hasattr(retriever, "_enrich_summary_result"):
        raise ValueError("Reflection study requires WitnessRAG facts+summary")
    # A private configuration avoids toggling a shared flag while concurrent
    # questions are in flight. The indexed memory is read from the frozen graph.
    base = copy.copy(retriever)
    base.ctx = replace(retriever.ctx, run=replace(cfg,
        witness=replace(cfg.witness, summary_reflection=False)))
    identity = {
        "version": 2, "qid": question.qid, "question": question.question,
        "memory": getattr(retriever.ctx, "controlled_memory_hash", ""),
        "corpus": corpus.stats()["corpus_hash"], "top_k": cfg.top_k,
        "witness": asdict(base.ctx.run.witness), "seed": cfg.seed,
    }
    # Generation/delivery limit belongs solely to the summary factor; it must
    # not invalidate the otherwise identical literal cascade retrieval.
    identity["witness"].pop("summary_reflection_limit", None)
    folder = root() / "shared" / "retrieval" / identity["corpus"][:24]

    def build_base():
        from wrag.llm.filters import LEDGER
        result = base.retrieve(question, cfg.top_k)
        result.filtered = result.filtered or LEDGER.question_blocked(
            corpus.name, retriever.name, question.qid)
        return result

    build_base.llm = retriever.ctx.llm
    result, snapshot, hit = saved(folder / (sha(question.qid) + ".json"), identity, corpus, build_base)
    logical_usage = snapshot.get("construction_usage", {})
    base_hash = snapshot["retrieval"]["hash"]
    if cfg.witness.summary_reflection:
        enhanced_identity = {**identity, "base_hash": base_hash,
                             "reflection_limit": cfg.witness.summary_reflection_limit}

        def build_enhanced():
            return retriever._enrich_summary_result(copy.deepcopy(result), question)

        build_enhanced.llm = retriever.ctx.llm
        result, enhanced, enhanced_hit = saved(
            folder / (sha(question.qid) + ".summary.json"), enhanced_identity, corpus, build_enhanced)
        result.diagnostics["reflection_summary_snapshot_hit"] = enhanced_hit
    result.diagnostics["reflection_study"] = {
        "base_snapshot_hash": base_hash, "memory_hash": identity["memory"],
        "retrieval_replayed": hit, "shared_retrieval_usage": logical_usage,
        "cost_note": "Retrieval is built once for four cells; shared cost must not be summed four times.",
    }
    return result
