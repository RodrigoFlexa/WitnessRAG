"""Isolated-context runner for RULER-128k and NarrativeQA.

Each example is indexed independently.  This is essential for RULER: contexts
are synthetic test instances, not documents in one shared retrieval corpus.
Input is JSON or JSONL and may use common Hugging Face field variants.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.eval.metrics import exact_match, token_f1
from wrag.eval.reader import read
from wrag.methods import build_context, build_methods
from wrag.witness.context_selection import select_complement
from wrag.util import append_jsonl, read_jsonl, setup_logging, write_json


def _load(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            value = value.get("data") or value.get("examples") or [value]
        return list(value)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _field(item: dict, *names, default=""):
    for name in names:
        value = item.get(name)
        if value not in (None, ""):
            return value
    return default


def _answers(item: dict) -> list[str]:
    raw = _field(item, "answers", "answer", "outputs", "output", default=[])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        raw = raw.get("text") or raw.get("answers") or []
    out = []
    for value in raw or []:
        if isinstance(value, dict):
            value = value.get("text") or value.get("answer")
        if value not in (None, ""):
            out.append(str(value))
    return out


def _normalize(item: dict, index: int, benchmark: str) -> tuple[str, str, str, list[str], str]:
    document = item.get("document")
    if isinstance(document, dict):
        context = _field(document, "text", "summary")
    else:
        context = _field(item, "context", "input", "document", "story", "text")
    question = item.get("question")
    if isinstance(question, dict):
        question = _field(question, "text", "question")
    question = str(question or _field(item, "query", "prompt"))
    # Official RULER JSONL stores the generated context and final instruction
    # together in ``input``. Recover the last non-empty paragraph as the query
    # rather than embedding all 128k tokens as a retrieval query.
    if benchmark == "ruler" and not question and context:
        blocks = [part.strip() for part in str(context).split("\n\n") if part.strip()]
        question = blocks[-1] if blocks else str(context).strip().splitlines()[-1]
        if str(context).rstrip().endswith(question):
            context = str(context).rstrip()[:-len(question)].rstrip()
    qid = str(_field(item, "id", "_id", "qid", default=f"{benchmark}-{index}"))
    task = str(_field(item, "task", "type", "subset", default=benchmark)).lower()
    answers = _answers(item)
    if not context or not question or not answers:
        raise ValueError(f"example {qid} lacks context, question, or answer")
    return qid, str(context), question, answers, task


def _ruler_score(answer: str, references: list[str], task: str) -> float:
    """NVIDIA RULER synthetic metric on a 0--1 scale."""
    prediction = answer.strip().lower()
    refs = [str(x).strip().lower() for x in references if str(x).strip()]
    if not refs:
        return 0.0
    if task.lower() == "qa":
        return float(any(ref in prediction for ref in refs))
    return sum(ref in prediction for ref in refs) / len(refs)


def _chunks(tokenizer, text: str, size: int, overlap: int, limit: int) -> list[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if limit:
        tokens = tokens[:limit]
    step = size - overlap
    return [tokenizer.decode(tokens[start:start + size], skip_special_tokens=True)
            for start in range(0, len(tokens), step) if tokens[start:start + size]]


def _config(profile: str, top_k: int, task: str = "") -> C.RunConfig:
    cfg = C.RunConfig(n_questions=1, top_k=top_k, methods=("witnessrag",),
                      corpus_scope="isolated_context_per_example")
    cfg.witness.hybrid_fallback = True
    cfg.witness.binding_aware_grounding = True
    cfg.witness.vocabulary_aware_compile = True
    cfg.witness.query_plans = True
    cfg.witness.max_query_plans = 3
    cfg.witness.active_frontier = True
    cfg.witness.active_obligations = True
    cfg.witness.active_context = True
    cfg.witness.enable_acquisition = False
    if profile in {"soft", "temporal", "full"}:
        cfg.witness.soft_obligations = True
    if profile in {"temporal", "full"}:
        cfg.witness.temporal_memory = True
        cfg.qa.operator_reader = True
    if profile == "full":
        cfg.witness.complementary_context = True
    if task.lower() in {"mt", "agg", "aggregation", "multi-target"}:
        cfg.witness.answer_set = True
        cfg.qa.answer_set = True
    return cfg


def run(args) -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision or None)
    raw = _load(args.input)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    done = {row["qid"] for row in read_jsonl(records_path)} if args.resume else set()
    cfg = _config(args.profile, args.top_k, args.task)
    manifest = {"benchmark": args.benchmark, "input": str(args.input.resolve()),
                "model": args.model, "model_revision": args.model_revision,
                "profile": args.profile, "context_tokens": args.context_tokens,
                "chunk_tokens": args.chunk_tokens, "chunk_overlap": args.chunk_overlap,
                "top_k": args.top_k, "engine": args.engine, "config": cfg.to_dict()}
    write_json(output / "manifest.json", manifest)
    for index, item in enumerate(raw):
        qid, context, question_text, answers, task = _normalize(item, index, args.benchmark)
        task = args.task or task
        if qid in done:
            continue
        texts = _chunks(tokenizer, context, args.chunk_tokens, args.chunk_overlap,
                        args.context_tokens)
        passages = [Passage(f"{qid}:chunk:{i}", f"chunk {i + 1}", text,
                            sequence=i, source_ids=(qid,)) for i, text in enumerate(texts)]
        question = Question(qid, question_text, answers, dataset=args.benchmark, qtype=task)
        corpus = Corpus(args.benchmark, passages, [question])
        started = time.perf_counter()
        ctx = build_context(corpus, cfg)
        retriever = build_methods(ctx, [args.engine])[args.engine]
        retrieval = retriever.retrieve(question, args.top_k)
        if args.engine == "hybrid" and args.profile in {"temporal", "full"}:
            selected = select_complement(corpus, question.question, retrieval.pids, {}, args.top_k,
                                         temporal=True, complementary=args.profile == "full")
            retrieval.pids = selected.pids
            retrieval.diagnostics["selecao_contexto"] = asdict(selected)
        result = read(ctx.llm, corpus, question, retrieval.pids, cfg.qa, method=args.engine)
        row = {"qid": qid, "task": task, "question": question_text, "answers": answers,
               "answer": result.answer, "f1": token_f1(result.answer, answers),
               "em": exact_match(result.answer, answers), "retrieved": retrieval.pids,
               "context_tokens": min(args.context_tokens, len(tokenizer.encode(context, add_special_tokens=False)))
                   if args.context_tokens else len(tokenizer.encode(context, add_special_tokens=False)),
               "chunks": len(passages), "latency_s": time.perf_counter() - started,
               "diagnostics": retrieval.diagnostics}
        if args.benchmark == "ruler":
            row["ruler_score"] = _ruler_score(result.answer, answers, task)
        append_jsonl(records_path, row)
        _report(output, manifest, read_jsonl(records_path), len(raw))
    return _report(output, manifest, read_jsonl(records_path), len(raw))


def _report(output: Path, manifest: dict, rows: list[dict], expected: int) -> dict:
    groups = {task: [r for r in rows if r["task"] == task]
              for task in sorted({r["task"] for r in rows})}
    block = lambda values: {"n": len(values), "f1": mean(r["f1"] for r in values) if values else None,
                            "em": mean(r["em"] for r in values) if values else None,
                            "ruler_score": mean(r["ruler_score"] for r in values)
                                if values and "ruler_score" in values[0] else None,
                            "latency_s": mean(r["latency_s"] for r in values) if values else None}
    report = {"benchmark": manifest["benchmark"], "profile": manifest["profile"],
              "completed": len(rows), "expected": expected, "overall": block(rows),
              "by_task": {name: block(values) for name, values in groups.items()}}
    write_json(output / "report.json", report)
    lines = [f"# {manifest['benchmark']} — {manifest['profile']}", "",
             f"Concluídos: {len(rows)}/{expected}", "",
             "| tarefa | n | score RULER | F1 | EM | latência (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, values in {"overall": rows, **groups}.items():
        b = block(values)
        official = "—" if b["ruler_score"] is None else f"{b['ruler_score']:.4f}"
        lines.append(f"| {name} | {b['n']} | {official} | {b['f1'] or 0:.4f} | "
                     f"{b['em'] or 0:.4f} | {b['latency_s'] or 0:.2f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["ruler", "narrativeqa"], required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=["base", "soft", "temporal", "full"], default="full")
    p.add_argument("--task", default="", help="fixed task label when the input file omits it")
    p.add_argument("--engine", choices=["hybrid", "witnessrag-lite", "witnessrag"],
                   default="witnessrag-lite",
                   help="lite scales to 128k; witnessrag also runs OpenIE/joins per example")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--model-revision", default="")
    p.add_argument("--context-tokens", type=int, default=131072)
    p.add_argument("--chunk-tokens", type=int, default=1024)
    p.add_argument("--chunk-overlap", type=int, default=128)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    setup_logging()
    args = parser().parse_args(argv)
    if args.chunk_tokens <= args.chunk_overlap or args.top_k < 1:
        raise ValueError("chunk-tokens must exceed overlap and top-k must be positive")
    run(args)


if __name__ == "__main__":
    main()
