#!/usr/bin/env python3
"""Development pilot on preselected zero-F1 errors and perfect-F1 controls.

Reuse the exact historical extraction and initial reader evidence. Only the
retry executes fresh retrieval. Gold is used ONLY for selection and scoring,
after generation; no gold-driven hints, answers or plans enter the controller.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "runs/local-plans-v2-gpt4omini-all")
    parser.add_argument("--conversation", type=int, default=0)
    parser.add_argument("--errors", type=int, default=6)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--all-questions", action="store_true", help="Evaluate every source question without selecting by score")
    parser.add_argument("--expected-fact-budget", type=int, help="Refuse a source run with a different budget")
    parser.add_argument("--include-qids", nargs="*", default=[], help="Additional diagnostic cases; never passed to the controller")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reflection-replan-sufficiency-v2-final")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reader-control", action="store_true", help="On completed retry rows, reread original evidence without another search")
    args = parser.parse_args()
    if args.errors < 1 or args.controls < 1:
        parser.error("Use at least one error and one control")
    os.environ["WRAG_EMBED_DEVICE"] = args.embed_device
    for key in ("WRAG_REFLECTION_STUDY_ROOT", "WRAG_CONTROLLED_ROOT", "WRAG_FROZEN_MEMORY_SOURCE"):
        os.environ.pop(key, None)
    from wrag import config as C
    for key in ("WRAG_REFLECTION_STUDY_ROOT", "WRAG_CONTROLLED_ROOT", "WRAG_FROZEN_MEMORY_SOURCE"):
        os.environ.pop(key, None)
    from wrag.eval.memory_variant import load_source
    from wrag.eval.reflection_study import atomic_json
    from wrag.eval.locomo_official import score_record
    from wrag.eval import metrics as M
    from wrag.embed import SentenceTransformerEmbedder
    from wrag.graph import build_graph
    from wrag.ie import ExtractionResult
    from wrag.llm import get_llm
    from wrag.llm.base import usage_delta
    from wrag.methods.base import IndexContext, RetrievalResult
    from wrag.methods.witnessrag import WitnessRAGRetriever
    from wrag.witness.reflection_replan import VERSION, adaptive_read
    from benchmarks.memoryagentbench.protocol import code_hash
    import torch
    torch.set_num_threads(4)
    pilot = json.loads((args.source_run / "pilot.json").read_text(encoding="utf-8"))
    if pilot["settings"].get("model") != args.model:
        raise ValueError("Use the same model as the source run for the paired replan comparison")
    C.CACHE_DIR = args.cache_dir or Path(pilot["settings"]["cache_dir"])
    C.EMBED_STRICT_DEVICE = args.embed_device.startswith("cuda")
    C.EMBED_DEVICE, C.EMBED_BATCH_SIZE = args.embed_device, 32
    corpus, facts, dated, baseline, provenance = load_source(args.source_run, args.conversation, C.CACHE_DIR)
    errors = [row for row in baseline.values() if row.get("f1_locomo") == 0][:args.errors]
    controls = [row for row in baseline.values() if row.get("f1_locomo") == 1][:args.controls]
    selected = list(baseline.values()) if args.all_questions else errors + controls
    if not args.all_questions and (len(errors) != args.errors or len(controls) != args.controls or not selected):
        raise ValueError("Not enough zero-F1 errors / perfect-F1 controls")
    for qid in args.include_qids:
        if qid not in baseline:
            raise ValueError(f"Unknown diagnostic qid: {qid}")
        if qid not in {r["qid"] for r in selected}:
            selected.append(baseline[qid])
    settings = provenance["source_configuration"]
    cfg = C.RunConfig(dataset="locomo", top_k=settings["top_k"], methods=("witnessrag",))
    cfg.graph = C.GraphConfig(**settings["graph"])
    cfg.ie = C.IEConfig(**settings["ie"])
    cfg.witness = C.WitnessConfig(**settings["witness"])
    if args.expected_fact_budget is not None and cfg.witness.fact_budget != args.expected_fact_budget:
        raise ValueError(f"Expected {args.expected_fact_budget} facts, source has {cfg.witness.fact_budget}")
    cfg.qa = replace(C.QAConfig(**settings["qa"]), reflection_replan=True)
    cfg.n_questions = len(selected)
    if args.reader_control:
        from wrag.eval.reader import read
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("controller_version") != VERSION or manifest["model"] != args.model
                or manifest["source"] != json.loads(json.dumps(provenance))):
            raise ValueError("Reader control must use the same model and verified source as the pilot")
        rows = [json.loads(x) for x in (args.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        llm = get_llm("openai", deployment=args.model)
        by_id = {q.qid: q for q in corpus.questions}
        control_path = args.output / "reader-control.jsonl"
        controls = [json.loads(x) for x in control_path.read_text(encoding="utf-8").splitlines()] if control_path.exists() else []
        done = {r["qid"] for r in controls}
        for row in rows:
            if not row["trace"]["performed"] or row["qid"] in done:
                continue
            source = baseline[row["qid"]]
            blocks = copy.deepcopy(source["diagnosticos"]["trechos_extras"])
            before = llm.usage.snapshot()
            answer = read(llm, corpus, by_id[row["qid"]], [], replace(cfg.qa, reflection_replan=False),
                          extra_passages=blocks, facts_mode=source["diagnosticos"]["leitura_fatos"])
            if answer.filtered:
                raise RuntimeError("Filtered reader control")
            score = score_record({"dataset": corpus.name, "tipo": row["tipo"], "resposta": answer.answer,
                                  "respostas_ouro": row["respostas_ouro"]})["f1_locomo"]
            control = {"qid": row["qid"], "question": row["pergunta"], "answer": answer.answer,
                       "f1": score, "adaptive_f1": row["f1_locomo"],
                       "new_retrieval": False, "uso_llm": usage_delta(llm.usage.snapshot(), before)}
            with control_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(control, ensure_ascii=False) + "\n")
            print(json.dumps(control, ensure_ascii=False), flush=True)
        return
    identity = {"code_hash": code_hash(ROOT), "controller_version": VERSION,
                "model": args.model, "source": provenance,
                "selected_ids": [r["qid"] for r in selected], "configuration": cfg.to_dict(),
                "selection_rule": ("all source questions" if args.all_questions else
                                   f"first {args.errors} historical zero-F1 errors and {args.controls} perfect-F1 controls plus declared diagnostic cases"),
                "embed_device": args.embed_device,
                "endpoint": os.environ.get("OPENAI_BASE_URL", ""),
                "initial_retrieval": "exact_saved_reader_evidence", "retry_retrieval": "real_full_graph",
                "memory_cost": "historical extraction reused; not charged as new inference"}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError("Pilot identity changed or --resume omitted; choose another output")
    else:
        atomic_json(manifest, identity)
    path = args.output / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {r["qid"] for r in rows}
    if len(done) != len(rows) or not done <= set(identity["selected_ids"]):
        raise ValueError("Foreign or duplicate pilot records")
    llm = get_llm("openai", deployment=args.model)
    original_chat = llm.chat
    def reader_only(*values, **kwargs):
        if kwargs.get("stage") not in {"qa", "memory.sufficiency", "memory.gap_coverage"}:
            raise AssertionError(f"Unexpected generative stage: {kwargs.get('stage')}")
        return original_chat(*values, **kwargs)
    llm.chat = reader_only
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device=args.embed_device)
    extraction = ExtractionResult(facts=facts)
    kg = build_graph(corpus, extraction, embedder, cfg.graph, with_passage_nodes=True)
    retriever = WitnessRAGRetriever(IndexContext(corpus, llm, embedder, cfg, extraction, kg))
    retriever.index()
    by_id = {q.qid: q for q in corpus.questions}
    for source in selected:
        if source["qid"] in done:
            continue
        q = by_id[source["qid"]]
        initial = RetrievalResult(pids=source["recuperadas"], scores=source["scores"],
                                   diagnostics=copy.deepcopy(source["diagnosticos"]))
        before = llm.usage.snapshot()
        retrieval, answer = adaptive_read(retriever, corpus, q, initial, cfg.qa, "witnessrag-replan2")
        if answer.filtered:
            raise RuntimeError("Filtered pilot answer; the question remains pending")
        trace = retrieval.diagnostics["reflection_replan"]
        row = {"qid": q.qid, "pergunta": q.question, "tipo": q.qtype, "dataset": corpus.name,
               "respostas_ouro": q.answers, "resposta": answer.answer,
               "historical_answer": source["resposta"], "historical_f1": source["f1_locomo"],
               "group": ("zero_f1" if source["f1_locomo"] == 0 else
                         "control" if source["f1_locomo"] == 1 else "partial_f1"),
               "historical_usage": source["uso_llm"], "trace": trace,
               "f1": M.token_f1(answer.answer, q.answers), "uso_llm": usage_delta(llm.usage.snapshot(), before),
               "total_prompt_tokens": answer.prompt_tokens, "total_completion_tokens": answer.completion_tokens,
               "final_fact_indices": retrieval.diagnostics["fatos_entregues"]["indices"],
               "reader_context": retrieval.diagnostics["trechos_extras"]}
        row.update(score_record(row))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(f"[{len(rows)}/{len(selected)}] {q.qid} {row['group']} "
              f"F1 {row['historical_f1']:.3f} -> final {row['f1_locomo']:.3f}; "
              f"decision={trace['gate']['decision']}; retry={trace['performed']}; "
              f"new_facts={len(trace['new_fact_indices'])}", flush=True)
    groups = {}
    for group in ("zero_f1", "control", "partial_f1"):
        items = [r for r in rows if r["group"] == group]
        if not items:
            continue
        groups[group] = {"n": len(items), "historical_f1": statistics.mean(r["historical_f1"] for r in items),
                        "final_f1": statistics.mean(r["f1_locomo"] for r in items),
                        "retries": sum(r["trace"]["performed"] for r in items),
                        "improved_after_retry": sum(r["trace"]["performed"] and
                                                    r["f1_locomo"] > r["historical_f1"] for r in items)}
    costs = {}
    for key, usage_key in (("historical", "historical_usage"), ("experimental", "uso_llm")):
        costs[key] = {field: sum(r[usage_key]["total"].get(field, 0) for r in rows)
                      for field in ("chamadas", "tokens_prompt", "tokens_resposta", "em_cache")}
    summary = {"groups": groups, "questions": len(rows), "complete": len(rows) == len(selected),
               "controller_version": VERSION, "costs": costs,
               "mean_prompt_tokens": statistics.mean(r["total_prompt_tokens"] for r in rows),
               "mean_completion_tokens": statistics.mean(r["total_completion_tokens"] for r in rows),
               "mean_checker_prompt_tokens": statistics.mean(r["trace"]["verifier_usage"]["prompt_tokens"] for r in rows),
               "development_only": not args.all_questions, "model": args.model}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
