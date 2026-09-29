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
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reflection-replan-development")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reader-control", action="store_true", help="On completed retry rows, reread original evidence without another search")
    args = parser.parse_args()
    if args.errors < 1 or args.controls < 1:
        parser.error("Use at least one error and one control")
    os.environ["WRAG_EMBED_DEVICE"] = "cpu"
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
    from wrag.witness.reflection_replan import adaptive_read
    from benchmarks.memoryagentbench.protocol import code_hash
    import torch
    torch.set_num_threads(4)
    pilot = json.loads((args.source_run / "pilot.json").read_text(encoding="utf-8"))
    C.CACHE_DIR = Path(pilot["settings"]["cache_dir"])
    C.EMBED_STRICT_DEVICE = False
    C.EMBED_DEVICE, C.EMBED_BATCH_SIZE = "cpu", 32
    corpus, facts, dated, baseline, provenance = load_source(args.source_run, args.conversation, C.CACHE_DIR)
    errors = [row for row in baseline.values() if row.get("f1_locomo") == 0][:args.errors]
    controls = [row for row in baseline.values() if row.get("f1_locomo") == 1][:args.controls]
    selected = errors + controls
    if len(errors) != args.errors or len(controls) != args.controls or not selected:
        raise ValueError("Not enough zero-F1 errors / perfect-F1 controls")
    settings = provenance["source_configuration"]
    cfg = C.RunConfig(dataset="locomo", top_k=settings["top_k"], methods=("witnessrag",))
    cfg.graph = C.GraphConfig(**settings["graph"])
    cfg.ie = C.IEConfig(**settings["ie"])
    cfg.witness = C.WitnessConfig(**settings["witness"])
    cfg.qa = replace(C.QAConfig(**settings["qa"]), reflection_replan=True)
    cfg.n_questions = len(selected)
    if args.reader_control:
        from wrag.eval.reader import read
        manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != args.model or manifest["source"] != json.loads(json.dumps(provenance)):
            raise ValueError("Reader control must use the same model and verified source as the pilot")
        rows = [json.loads(x) for x in (args.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        llm = get_llm("openai", deployment=args.model)
        by_id = {q.qid: q for q in corpus.questions}
        dates = {t.turn_id: str(t.when) for ts in dated.turns.values() for t in ts if t.when and t.turn_id}
        control_path = args.output / "reader-control.jsonl"
        controls = [json.loads(x) for x in control_path.read_text(encoding="utf-8").splitlines()] if control_path.exists() else []
        done = {r["qid"] for r in controls}
        for row in rows:
            if not row["trace"]["performed"] or row["qid"] in done:
                continue
            source = baseline[row["qid"]]
            blocks = copy.deepcopy(source["diagnosticos"]["trechos_extras"])
            for block in blocks:
                for turn, date in dates.items():
                    block["text"] = block["text"].replace(f"[{turn}]", f"[{turn}; session={date}]")
            final_contract = [b for b in row["reader_context"] if b["title"] == "Final retry reading contract"]
            blocks.extend(copy.deepcopy(final_contract))
            before = llm.usage.snapshot()
            answer = read(llm, corpus, by_id[row["qid"]], [], replace(cfg.qa, reflection_replan=False),
                          extra_passages=blocks, facts_mode=source["diagnosticos"]["leitura_fatos"])
            if answer.filtered:
                raise RuntimeError("Filtered reader control")
            score = score_record({"dataset": corpus.name, "tipo": row["tipo"], "resposta": answer.answer,
                                  "respostas_ouro": row["respostas_ouro"]})["f1_locomo"]
            control = {"qid": row["qid"], "question": row["pergunta"], "answer": answer.answer,
                       "f1": score, "adaptive_f1": row["f1_locomo"], "initial_gate_f1": row["initial_gate_f1"],
                       "new_retrieval": False, "uso_llm": usage_delta(llm.usage.snapshot(), before)}
            with control_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(control, ensure_ascii=False) + "\n")
            print(json.dumps(control, ensure_ascii=False), flush=True)
        return
    identity = {"code_hash": code_hash(ROOT), "model": args.model, "source": provenance,
                "selected_ids": [r["qid"] for r in selected], "configuration": cfg.to_dict(),
                "selection_rule": f"first {args.errors} historical zero-F1 errors and {args.controls} perfect-F1 controls",
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
        if kwargs.get("stage") not in {"qa", "qa.replan_gate"}:
            raise AssertionError(f"Unexpected generative stage: {kwargs.get('stage')}")
        return original_chat(*values, **kwargs)
    llm.chat = reader_only
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")
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
        retrieval, answer = adaptive_read(retriever, corpus, q, initial, cfg.qa, "witnessrag-replan1")
        if answer.filtered:
            raise RuntimeError("Filtered pilot answer; the question remains pending")
        trace = retrieval.diagnostics["reflection_replan"]
        row = {"qid": q.qid, "pergunta": q.question, "tipo": q.qtype, "dataset": corpus.name,
               "respostas_ouro": q.answers, "resposta": answer.answer,
               "historical_answer": source["resposta"], "historical_f1": source["f1_locomo"],
               "group": "zero_f1" if source["f1_locomo"] == 0 else "control",
               "initial_gate_answer": trace["initial_answer"], "trace": trace,
               "f1": M.token_f1(answer.answer, q.answers), "uso_llm": usage_delta(llm.usage.snapshot(), before),
               "reader_prompt_tokens": answer.prompt_tokens, "reader_completion_tokens": answer.completion_tokens,
               "final_fact_indices": retrieval.diagnostics["fatos_entregues"]["indices"],
               "reader_context": retrieval.diagnostics["trechos_extras"]}
        row.update(score_record(row))
        row["initial_gate_f1"] = score_record({**row, "resposta": trace["initial_answer"]})["f1_locomo"]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        print(f"[{len(rows)}/{len(selected)}] {q.qid} {row['group']} "
              f"F1 {row['historical_f1']:.3f} -> gate {row['initial_gate_f1']:.3f} -> "
              f"final {row['f1_locomo']:.3f}; retry={trace['performed']}; "
              f"new_facts={len(trace['new_fact_indices'])}", flush=True)
    groups = {}
    for group in ("zero_f1", "control"):
        items = [r for r in rows if r["group"] == group]
        groups[group] = {"n": len(items), "historical_f1": statistics.mean(r["historical_f1"] for r in items),
                        "gate_f1": statistics.mean(r["initial_gate_f1"] for r in items),
                        "final_f1": statistics.mean(r["f1_locomo"] for r in items),
                        "retries": sum(r["trace"]["performed"] for r in items),
                        "improved_by_retry": sum(r["f1_locomo"] > r["initial_gate_f1"] for r in items)}
    summary = {"groups": groups, "questions": len(rows), "complete": len(rows) == len(selected),
               "mean_prompt_tokens": statistics.mean(r["reader_prompt_tokens"] for r in rows),
               "mean_completion_tokens": statistics.mean(r["reader_completion_tokens"] for r in rows),
               "development_only": True, "model": args.model}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
