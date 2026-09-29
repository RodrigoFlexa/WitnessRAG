#!/usr/bin/env python3
"""Run 10/20/40 compact-memory arms on a shared, saved v2 executor result."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, default=ROOT/"runs/local-plans-v2-gpt4omini-all")
    p.add_argument("--conversation", type=int, choices=range(10), default=0)
    p.add_argument("--facts", type=int, choices=(10,20,40), nargs="+", default=[10,20,40])
    p.add_argument("--questions", type=int, default=0, help="0=all; positive=balanced smoke sample")
    p.add_argument("--output", type=Path, default=ROOT/"runs/memory-reflector-conv00")
    p.add_argument("--cache-dir", type=Path, default=ROOT/"runs/.cache/witness-openai")
    p.add_argument("--extraction-cache", type=Path)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--base-url", default=None, help="Optional local OpenAI-compatible model server")
    p.add_argument("--body", choices=("statement","triple"), default="statement")
    p.add_argument("--target-max-tokens", type=int, default=384)
    p.add_argument("--reflector-max-tokens", type=int, default=1024)
    p.add_argument("--reader-max-tokens", type=int, default=128)
    p.add_argument("--memory-only", action="store_true", help="Stop after reflector; do not call the benchmark reader")
    p.add_argument("--dry-run", action="store_true", help="Validate all source inputs and fact packets; zero API calls")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-code-update", action="store_true", help="Archive previous code identity and preserve results during an authorized bug-fix continuation")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.questions < 0 or len(set(args.facts)) != len(args.facts) or min(args.target_max_tokens, args.reflector_max_tokens, args.reader_max_tokens) < 1:
        p.error("Questions must be nonnegative, budgets unique, output token limits positive")
    os.environ["WRAG_CACHE_DIR"] = str(args.cache_dir.resolve())
    os.environ["WRAG_LLM_CACHE"] = "1"
    for key in ("WRAG_REFLECTION_STUDY_ROOT", "WRAG_CONTROLLED_ROOT", "WRAG_FROZEN_MEMORY_ROOT"):
        os.environ.pop(key, None)
    if args.base_url is not None:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    from wrag.eval.memory_variant import choose_questions, load_source, run_experiment
    from wrag.witness.query_memory import compact_packet
    corpus, facts, dated, baseline, provenance = load_source(
        args.source_run, args.conversation, args.cache_dir, args.extraction_cache)
    if args.dry_run:
        chosen = choose_questions([q for q in corpus.questions if q.qid in baseline], args.questions)
        counts = {}
        for n in args.facts:
            packets = [compact_packet(facts, dated, baseline[q.qid]["diagnosticos"], n, args.body) for q in chosen]
            counts[str(n)] = {"questions": len(packets), "max_delivered": max(len(v.records) for v in packets),
                              "dropped_packages": sum(len(v.dropped_packages) for v in packets)}
        print(json.dumps({"dry_run": True, "api_calls": 0, "source": {
                            key:provenance[key] for key in ("conversation", "source_predictions", "extraction_cache", "fact_sources_verified", "executor_mode")},
                          "budgets": counts, "body": args.body}, ensure_ascii=False, indent=2))
        return
    from wrag.eval import locomo_official as LO
    if not args.memory_only and not LO.available():
        raise RuntimeError("LoCoMo scorer unavailable; install the existing project requirements")
    from wrag.llm import get_llm
    llm = get_llm("openai", deployment=args.model)
    original_chat = llm.chat
    def scoped_chat(*values, **kwargs):
        allowed = {"memory.target", "memory.target.repair", "memory.reflect", "memory.reflect.repair"} | (set() if args.memory_only else {"qa"})
        if kwargs.get("stage") not in allowed:
            raise AssertionError("Unexpected generative stage: " + str(kwargs.get("stage")))
        return original_chat(*values, **kwargs)
    llm.chat = scoped_chat
    code_files = [ROOT/"scripts/run-memory-variants.py", *sorted((ROOT/"wrag").rglob("*.py"))]
    code_hash = hashlib.sha256(b"".join(str(f.relative_to(ROOT)).encode()+f.read_bytes() for f in code_files)).hexdigest()
    try:
        result = run_experiment(llm, corpus, facts, dated, baseline, provenance, args.output,
            budgets=tuple(args.facts), questions=args.questions, body=args.body, resume=args.resume,
            memory_only=args.memory_only, target_max_tokens=args.target_max_tokens,
            reflector_max_tokens=args.reflector_max_tokens, reader_max_tokens=args.reader_max_tokens,
            model=args.model, endpoint=os.environ.get("OPENAI_BASE_URL", ""), code_hash=code_hash,
            allow_code_update=args.allow_code_update)
        print(json.dumps({"complete": result["complete"], "paired_n": result["paired_n"],
                          "report": str((args.output/"comparison.md").resolve())}, ensure_ascii=False))
    finally:
        llm.chat = original_chat
        llm.close()


if __name__ == "__main__":
    main()
