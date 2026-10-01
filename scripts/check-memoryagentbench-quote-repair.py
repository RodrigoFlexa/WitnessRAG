"""Reproduce the real extraction failure and validate literal ID recovery."""
from __future__ import annotations

import argparse
import json
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
    configure_runtime(args)
    from benchmarks.memoryagentbench.data import iter_samples, chunks_for
    from benchmarks.memoryagentbench.protocol import task_settings, digest, code_hash
    from benchmarks.memoryagentbench.source_memory import source_spans, extract_source
    from benchmarks.memoryagentbench.engine import standard_config
    from wrag.data import Passage
    from wrag.llm import get_llm
    from wrag.llm.base import usage_delta
    sample = next(iter_samples(args.cache, ("Accurate_Retrieval",), ("ruler_qa1_197K",), "paper"))
    text = chunks_for(sample, args.cache, task_settings(sample.source, "paper"))[180]
    passage = Passage("mab-" + digest(sample.key)[:20] + "-c180", "", text)
    spans = source_spans(text, 180)
    original = next(s for s in spans if s.sid == "S180.12")
    llm = get_llm("vllm", deployment=args.model)
    cfg = standard_config().ie
    before = llm.usage.snapshot()
    result, rejected = extract_source(passage, llm, cfg, spans, conversation=False)
    observed = usage_delta(llm.usage.snapshot(), before)
    restored = [f for f in result.facts if f.turn_id == original.sid]
    before = llm.usage.snapshot()
    second, cached_rejections = extract_source(passage, llm, cfg, spans, conversation=False)
    cached_usage = usage_delta(llm.usage.snapshot(), before)
    checks = {"failed_record_recovered": bool(restored),
              "literal_quoted_source": bool(restored) and all(f.statement == original.text for f in restored),
              "unfccc_relation": any("unfccc" in f.subject.lower() for f in restored),
              "valid_source_ids": all(f.turn_id in {s.sid for s in spans} for f in result.facts),
              "cache_replay_identical": second.facts == result.facts,
              "cache_replay_no_new_tokens": cached_usage["total"].get("tokens_prompt_sem_cache", 0) == 0
                    and cached_usage["total"].get("tokens_resposta_sem_cache", 0) == 0}
    row = {"code_hash": code_hash(ROOT), "model": args.model, "source": sample.source,
           "chunk_index": 180, "source_id": original.sid, "source_text": original.text,
           "checks": checks, "facts": [f.to_dict() for f in result.facts],
           "rejections": rejected, "observed_usage": observed,
           "cache_rejections": cached_rejections, "cache_usage": cached_usage}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Real quote repair: facts={len(result.facts)}, checks={checks}", flush=True)
    if not all(checks.values()):
        raise SystemExit("Real quote-repair check failed")


if __name__ == "__main__":
    main()
