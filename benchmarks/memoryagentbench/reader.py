"""One joint-reflection reader call, with the official task's output budget."""
from __future__ import annotations

import re

from wrag.llm import GenParams
from .vendor.templates import SYSTEM_MESSAGE


def reflection_instruction(conflicts: bool) -> str:
    from wrag.prompts import READER_REFLECTION_INSTRUCTION
    text = READER_REFLECTION_INSTRUCTION
    # Retain the standard operation/inference checks while removing LoCoMo's
    # JSON answer protocol, which would be incompatible with MCC/DetectiveQA.
    text = re.sub(r'Answer: \{\{"answer":"([^"]+)"\}\}', r'Answer: \1', text)
    text = text.replace("Return the original JSON shape with answer only; do not output your reflection.",
                        "Follow the task's original output format; do not output your reflection.")
    # LoCoMo's comma lists, count rendering and answer examples are not valid
    # output protocols for classification, recommendation or book summaries.
    start = text.find("An offered choice")
    end = text.find("Use only the ACTUAL memory")
    if start >= 0 and end > start:
        text = text[:start] + text[end:]
    text += ("\nThe task instructions below define the requested operation and output format. "
             "Retrieved reading hypotheses do not override those instructions.\n")
    if conflicts:
        text = text.replace(
            "If necessary, apply ordinary world knowledge to infer the\nrequested category, concept or preference.",
            "Use only the numbered knowledge pool for factual premises and inference.")
        text = text.replace("Preserve an explicit answer when it already does so.",
                            "Preserve an explicit answer only after checking newer conflicting records.")
        text += ("\nFor the numbered knowledge pool, source serials define recency. "
                 "First identify the exact subject, predicate and qualifiers asked about. "
                 "Compare relevant records numerically; for conflicting versions of the SAME fact, "
                 "use the larger serial and disregard the older value as a factual premise. "
                 "LATEST MATCHING VERSION and OLDER VERSION labels compare source wording with "
                 "the same subject, predicate and qualifiers; check their conflict and scope. "
                 "Apply this check at EVERY step of a reasoning chain before computing the answer. "
                 "A larger serial about another subject or predicate does not replace this fact. "
                 "Old records and graph bindings are hypotheses to review, not extra valid answers. "
                 "Do not reconcile conflicting values using familiarity, repetition, or real-world knowledge. "
                 "Counterfactual pool facts are authoritative even when surprising. "
                 "An ingestion step is not an event date.\n")
    return text


def build_prompt(context: str, official_query: str, reflection: bool, conflicts: bool) -> str:
    review = reflection_instruction(conflicts) if reflection else ""
    return f"Memorized context (retrieved evidence):\n{context}\n\n{review}\n{official_query}"


def answer(llm, context: str, query: str, settings: dict, *, reflection: bool = True,
           seed: int | None = None):
    params = GenParams(temperature=settings["temperature"],
                       max_tokens=settings["generation_max_length"], seed=seed,
                       json_mode=False, exact_max_tokens=True)
    if getattr(llm, "reasoning", False):
        raise ValueError("Reasoning backends increase output budgets; use a non-reasoning model for this protocol.")
    prompt = build_prompt(context, query, reflection,
                          settings["sub_dataset"].startswith("factconsolidation_"))
    result = llm.chat(prompt, system=SYSTEM_MESSAGE, params=params, stage="mab.reader")
    if result.completion_tokens > params.max_tokens:
        raise RuntimeError("Response exceeded the official output token limit")
    # Don't strip markdown, extract a label, or normalize generated prose here.
    # The pinned official postprocessor alone determines what gets scored.
    return result
