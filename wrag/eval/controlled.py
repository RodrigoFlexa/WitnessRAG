"""Controlled LoCoMo: one frozen memory, two retrievals, four reader cells.

Enabled only by WRAG_CONTROLLED_ROOT. Pickles are private, locally generated
checkpoints; never load a memory file supplied by an untrusted source.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from wrag.llm.base import LLMResult, usage_delta, sum_usage
from wrag.util import sha

VERSION = 1


def root():
    value = os.environ.get("WRAG_CONTROLLED_ROOT")
    return Path(value) if value else None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def conversation_dir(corpus):
    return root() / "controlled" / corpus.stats()["corpus_hash"][:24]


def frozen_graph(ctx, builder, source_root=None):
    """Freeze graph + vectors + extraction, and refuse incompatible resumes."""
    folder = ((Path(source_root) / "controlled" /
               ctx.corpus.stats()["corpus_hash"][:24]) if source_root else
              conversation_dir(ctx.corpus))
    if source_root and not folder.exists():
        raise FileNotFoundError(f"Frozen memory missing for this corpus: {folder}")
    if not source_root:
        folder.mkdir(parents=True, exist_ok=True)
    identity = {"version": VERSION, "corpus": ctx.corpus.stats(),
                "ie": asdict(ctx.run.ie), "graph": asdict(ctx.run.graph),
                "embedder": ctx.embedder.cache_key(),
                "model": getattr(ctx.llm, "deployment", ctx.llm.name)}
    manifest = folder / "memory.json"
    payload = folder / "memory.pkl"
    if manifest.exists():
        saved = load(manifest)
        if sha(saved["identity"]) != sha(identity):
            raise ValueError("Controlled memory identity changed; use a new output directory")
        raw = payload.read_bytes()
        if hashlib.sha256(raw).hexdigest() != saved["sha256"]:
            raise ValueError("Controlled memory checksum mismatch")
        ctx.kg, ctx.extraction = pickle.loads(raw)
        ctx.kg.corpus = ctx.corpus
    else:
        if source_root:
            raise FileNotFoundError(f"Frozen memory manifest missing: {manifest}")
        builder(ctx, with_passage_nodes=True)
        raw = pickle.dumps((ctx.kg, ctx.extraction), protocol=pickle.HIGHEST_PROTOCOL)
        temporary = payload.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(payload)
        saved = {"identity": identity, "sha256": hashlib.sha256(raw).hexdigest()}
        atomic_json(manifest, saved)
    ctx.controlled_memory_hash = saved["sha256"]


@contextmanager
def tape(llm, recorded=None):
    """Capture exact calls; replay never accesses the server or persistent cache.

    Request-key queues support concurrent independent extraction calls. A changed
    request, missing call or extra call fails instead of silently using the LLM.
    """
    original = llm._complete
    entries = []
    lock = threading.Lock()
    queues = defaultdict(deque)
    for item in recorded or []:
        queues[item["key"]].append(item)

    def complete(messages, params, stage="misc"):
        request = {"messages": messages, "params": asdict(params), "stage": stage}
        key = sha(request)
        if recorded is not None:
            with lock:
                if not queues[key]:
                    raise ValueError(f"Replay diverged at {stage}: request {key}")
                item = queues[key].popleft()
            return replace(LLMResult(**item["result"]), cached=True, latency_s=0.0)
        result = original(messages, params, stage)
        item = {**request, "key": key, "result": asdict(result)}
        if hasattr(llm, "build_kwargs"):
            kwargs = llm.build_kwargs(messages, params)
            item["effective_request"] = kwargs
            item["cache_path"] = str(llm._cache_path(kwargs))
        with lock:
            entries.append(item)
        return result

    llm._complete = complete
    try:
        yield entries
        if recorded is not None and any(queues.values()):
            raise ValueError("Replay made fewer LLM calls than recorded")
    finally:
        llm._complete = original


@contextmanager
def no_cache(llm):
    previous = getattr(llm, "use_cache", None)
    if previous is not None:
        llm.use_cache = False
    try:
        yield
    finally:
        if previous is not None:
            llm.use_cache = previous


@contextmanager
def condition(retriever, soft):
    cfg = retriever.ctx.run.witness
    previous = cfg.soft_obligations, cfg.proof_reader
    cfg.soft_obligations, cfg.proof_reader = soft, True
    try:
        yield
    finally:
        cfg.soft_obligations, cfg.proof_reader = previous


def retrieval_signature(result):
    # Latency is observational; plans, scores and all diagnostic decisions count.
    return sha(result.pids, result.scores, result.diagnostics, result.filtered)


def snapshot(corpus, result):
    data = asdict(result)
    data["passages"] = [asdict(corpus.get(pid)) for pid in result.pids]
    data["hash"] = sha(data)
    return data


def thaw(corpus, data):
    from wrag.methods.base import RetrievalResult
    checked = {k: v for k, v in data.items() if k != "hash"}
    if sha(checked) != data["hash"]:
        raise ValueError("Retrieval snapshot checksum mismatch")
    if data["passages"] != [asdict(corpus.get(pid)) for pid in data["pids"]]:
        raise ValueError("Reader source passages differ from frozen retrieval")
    return RetrievalResult(**{k: data[k] for k in
                              ("pids", "scores", "diagnostics", "filtered", "latency_s")})


def sample_questions(corpus):
    selected = []
    for kind in ("single-hop", "multi-hop"):
        selected.extend(sorted((q for q in corpus.questions if q.qtype == kind),
                               key=lambda q: sha(q.qid))[:2])
    return selected


def retrieve_saved(retriever, corpus, question, cfg, label):
    folder = conversation_dir(corpus) / "retrieval" / label
    path = folder / (sha(question.qid) + ".json")
    identity = {"qid": question.qid, "question": question.question,
                "memory": retriever.ctx.controlled_memory_hash,
                "config": cfg.to_dict(), "condition": label}
    if path.exists():
        item = load(path)
        if sha(item["identity"]) != sha(identity):
            raise ValueError("Retrieval checkpoint identity changed")
        thaw(corpus, item["retrieval"])
        return item
    llm = retriever.ctx.llm
    before = llm.usage.snapshot()
    # Shared new cache couples identical requests across experimental conditions.
    with condition(retriever, label == "soft-v2"), tape(llm) as calls:
        result = retriever.retrieve(question, cfg.top_k)
    item = {"identity": identity, "retrieval": snapshot(corpus, result),
            "calls": calls, "usage": usage_delta(llm.usage.snapshot(), before),
            "cost_mode": "experiment_cache_actual_hits_recorded"}
    atomic_json(path, item)
    return item


def preflight(retriever, corpus, cfg):
    """Run before scoring: exact replay + independent fresh repeat on 4 questions."""
    folder = conversation_dir(corpus)
    path = folder / "audit.json"
    identity = {"config": cfg.to_dict(), "memory": retriever.ctx.controlled_memory_hash}
    if path.exists():
        saved = load(path)
        if sha(saved["identity"]) != sha(identity) or not saved["passed"]:
            raise ValueError("Audit checkpoint incompatible")
        return
    results = []
    try:
        for question in sample_questions(corpus):
            for label in ("evidence", "soft-v2"):
                item = retrieve_saved(retriever, corpus, question, cfg, label)
                expected = thaw(corpus, item["retrieval"])
                with condition(retriever, label == "soft-v2"), tape(retriever.ctx.llm, item["calls"]):
                    started = time.perf_counter()
                    repeated = retriever.retrieve(question, cfg.top_k)
                    replay_seconds = time.perf_counter() - started
                if retrieval_signature(expected) != retrieval_signature(repeated):
                    atomic_json(folder / "audit-divergence.json", {
                        "qid": question.qid, "condition": label,
                        "expected": asdict(expected), "actual": asdict(repeated)})
                    raise ValueError(f"Retrieval replay differs for {question.qid}/{label}")
                before = retriever.ctx.llm.usage.snapshot()
                with condition(retriever, label == "soft-v2"), tape(retriever.ctx.llm) as warm_calls:
                    warm = retriever.retrieve(question, cfg.top_k)
                warm_usage = usage_delta(retriever.ctx.llm.usage.snapshot(), before)
                before = retriever.ctx.llm.usage.snapshot()
                with condition(retriever, label == "soft-v2"), no_cache(retriever.ctx.llm), tape(retriever.ctx.llm) as fresh_calls:
                    fresh = retriever.retrieve(question, cfg.top_k)
                results.append({"qid": question.qid, "condition": label,
                                "replay_identical": True, "replay_seconds": replay_seconds,
                                "fresh_identical": retrieval_signature(expected) == retrieval_signature(fresh),
                                "fresh_retrieval": snapshot(corpus, fresh), "fresh_calls": fresh_calls,
                                "fresh_usage": usage_delta(retriever.ctx.llm.usage.snapshot(), before),
                                "warm_seconds": warm.latency_s, "warm_usage": warm_usage,
                                "warm_calls": warm_calls,
                                "warm_identical": retrieval_signature(expected) == retrieval_signature(warm)})
        atomic_json(path, {"identity": identity, "passed": True, "sample": results,
                           "note": "Replay cost is in-memory tape, not disk cache or fresh inference."})
    except Exception as exc:
        atomic_json(folder / "audit-error.json", {"error": str(exc), "completed": results})
        raise


def answer(name, retriever, corpus, question, cfg, standard_answer):
    folder = conversation_dir(corpus)
    path = folder / "answers" / (sha(question.qid) + ".json")
    if path.exists():
        saved = load(path)
        if sha(saved["config"]) != sha(cfg.to_dict()) or saved["memory"] != retriever.ctx.controlled_memory_hash:
            raise ValueError("Answer checkpoint identity changed")
        return saved["cells"]["evidence/common"]
    cells = {}
    # Alternate order by qid, without consulting answers or observed quality.
    labels = ["evidence", "soft-v2"]
    if int(sha(question.qid)[0], 16) % 2:
        labels.reverse()
    for label in labels:
        item = retrieve_saved(retriever, corpus, question, cfg, label)
        modes = ["common", "proof"]
        if int(sha(question.qid, label)[0], 16) % 2:
            modes.reverse()
        for mode in modes:
            cell_path = folder / "readers" / label / mode / (sha(question.qid) + ".json")
            if cell_path.exists():
                record = load(cell_path)
                if record["retrieval_hash"] != item["retrieval"]["hash"]:
                    raise ValueError("Reader checkpoint belongs to another retrieval")
            else:
                result = thaw(corpus, item["retrieval"])
                frozen = SimpleNamespace(ctx=retriever.ctx, retrieve=lambda *_: copy.deepcopy(result))
                reader_cfg = replace(cfg, qa=replace(cfg.qa, proof_reader=mode == "proof"))
                with no_cache(retriever.ctx.llm), tape(retriever.ctx.llm) as reader_calls:
                    record = standard_answer(name, frozen, corpus, question, reader_cfg)
                record["uso_leitor"] = record["uso_llm"]
                record["uso_recuperacao"] = item["usage"]
                record["uso_llm"] = sum_usage([item["usage"], record["uso_leitor"]])
                record["reader_calls"] = reader_calls
                record["retrieval_hash"] = item["retrieval"]["hash"]
                record["memory_hash"] = retriever.ctx.controlled_memory_hash
                record["condition"] = label
                record["reader"] = mode
                atomic_json(cell_path, record)
            cells[label + "/" + mode] = record
    atomic_json(path, {"qid": question.qid, "config": cfg.to_dict(),
                       "memory": retriever.ctx.controlled_memory_hash, "cells": cells})
    return cells["evidence/common"]
