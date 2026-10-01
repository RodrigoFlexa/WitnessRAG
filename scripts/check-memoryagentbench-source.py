#!/usr/bin/env python
"""Small source-adapter smoke test on synthetic evidence, using the run backend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8095/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    args = p.parse_args()
    args.command, args.backend, args.model_revision = "run", "vllm", ""
    from benchmarks.memoryagentbench.__main__ import configure_runtime
    C = configure_runtime(args)
    C.EMBED_DEVICE, C.EMBED_MODEL, C.EMBED_STRICT_DEVICE = "cuda", "BAAI/bge-m3", True
    os.environ["WRAG_EMBED_DEVICE"] = "cuda"
    from wrag.embed import get_embedder
    from wrag.llm import get_llm
    from benchmarks.memoryagentbench.engine import WitnessEngine, standard_config
    from benchmarks.memoryagentbench import reader, protocol
    llm = get_llm("vllm", deployment=args.model)
    embedder = get_embedder("st", force_new=True)
    engine = WitnessEngine(llm, embedder, standard_config(), adaptation="ar-source-v2")
    cases = [
        ("ruler_qa1_197K", "Document 1:\nMira, sister of Omar, married Niko.",
         "Who did Mira marry?", "Niko"),
        ("ruler_qa2_421K", "Document 1:\nMira works at Celadon Labs. Celadon Labs is located in Riverport.",
         "In which city does Mira work?", "Riverport"),
        ("longmemeval_s*", "['Chat Time: 2022/11/17 (Thu) 12:04', "
         "[{'role': 'user', 'content': 'I read The Cloud Garden. I moved to Harborview last week.'}]]",
         "What book did the user read?", "The Cloud Garden"),
    ]
    results = []
    for index, (source, text, question, expected) in enumerate(cases):
        acquired = engine.prepare(f"synthetic-source-smoke-{index}", source, [text], text)
        selected = engine.select(f"synthetic:{index}", question)
        result = reader.answer(llm, selected.context, question + " Only give the answer.",
                               protocol.task_settings(source, "paper"))
        provenance = selected.diagnostics["source_adaptation"]
        checks = {"expected_answer": expected.casefold() in result.text.casefold(),
                  "grounded_facts": bool(engine.retriever.memory.facts),
                  "literal_source_delivered": bool(provenance["excerpts"]),
                  "bounded_sources": provenance["excerpt_chars"] <= provenance["excerpt_budget"],
                  "local_planning": selected.diagnostics["planejamento"]["chamadas"] == 0,
                  "official_output_budget": result.completion_tokens <= protocol.task_settings(source, "paper")["generation_max_length"]}
        results.append({"source": source, "question": question, "output": result.text,
                        "checks": checks, "acquisition": acquired,
                        "facts": [f.to_dict() for f in engine.retriever.memory.facts],
                        "diagnostics": selected.diagnostics})
        print(f"Synthetic {source}: {result.text!r}; checks={checks}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not all(all(r["checks"].values()) for r in results):
        raise SystemExit("Source adapter smoke test failed; inspect saved diagnostics before launch")


if __name__ == "__main__":
    main()
