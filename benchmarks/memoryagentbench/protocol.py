"""Pinned official settings. Paper settings are explicit, never mixed silently."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

UPSTREAM_COMMIT = "538026089d1a8a8eff05121d0db89b388f360eba"
DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
DATASET = "ai-hyz/MemoryAgentBench"
UPSTREAM_URL = "https://github.com/HUST-AI-HYZ/MemoryAgentBench"
SPLITS = ("Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution")
VENDOR = Path(__file__).parent / "vendor"
CONFIGS = json.loads((VENDOR / "task_configs.json").read_text(encoding="utf-8"))
DATA_FILES = json.loads((VENDOR / "data_files.json").read_text(encoding="utf-8"))
TOP_K = 10
TEMPERATURE = 0.7  # released GPT-4o-mini RAG agent configs
# Table 3 has ten tasks. Other sources are length/task ablations, not extra
# equally weighted columns of that table.
PAPER_SOURCES = (
    "ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*", "eventqa_full",
    "icl_banking77_5900shot_balance", "recsys_redial_full",
    "infbench_sum_eng_shots2", "detective_qa",
    "factconsolidation_sh_262k", "factconsolidation_mh_262k",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_settings(source: str, protocol: str = "official") -> dict:
    if protocol not in ("official", "paper"):
        raise ValueError(f"Unknown protocol: {protocol}")
    conf = dict(CONFIGS[source])
    conf.update(protocol=protocol, temperature=TEMPERATURE, top_k=TOP_K,
                tokenizer="o200k_base", sentence_tokenizer="nltk-punkt-english",
                context_truncation=False)
    if protocol == "paper":
        # Tables 14/15 of the supplied ICLR 2026 paper differ from release YAML.
        if source.startswith(("ruler_", "longmemeval_", "factconsolidation_")):
            conf["chunk_size"] = 512
        if source.startswith("longmemeval_"):
            conf["generation_max_length"] = 100
        if source == "detective_qa":
            conf["generation_max_length"] = 500
    return conf


def primary_metric(source: str) -> str:
    if source.startswith("longmemeval_"):
        return "longmemeval_judge_accuracy"
    if source.startswith("infbench_"):
        return "summary_judge_f1"
    if source.startswith("recsys_"):
        return "recsys_recall@5"
    if source.startswith("icl_") or source == "detective_qa":
        return "exact_match"
    return "substring_exact_match"


def code_hash(root: Path) -> str:
    paths = sorted((root / "wrag").rglob("*.py"))
    paths += sorted((root / "benchmarks" / "memoryagentbench").rglob("*"))
    return digest([(p.relative_to(root).as_posix(), file_digest(p))
                   for p in paths if p.is_file() and "__pycache__" not in p.parts])
