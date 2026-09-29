"""Official LongMemEval and HELMET book evaluation, separate from the agent."""
from __future__ import annotations

import json
import re

from wrag.llm import GenParams
from . import protocol as P
from .vendor.longmem_judge import get_anscheck_prompt

SUMMARY_PROMPTS = json.loads((P.VENDOR / "summary_prompts.json").read_text(encoding="utf-8"))
LONGMEM_MODEL = "gpt-4o"
SUMMARY_MODEL = "gpt-4o-2024-05-13"


def parse_summary_json(text: str):
    # Same last-object parser as the published summarization evaluator.
    matches = re.findall(r"\{.*?\}", text, re.S)
    if matches:
        try:
            return json.loads(matches[-1])
        except ValueError:
            matches = re.findall(r"(?:```json)(.+)(?:```)", text, re.S)
            try:
                return json.loads(matches[-1])
            except (ValueError, IndexError):
                return None
    return None


def evaluate(sample, index: int, output: str, llm) -> dict:
    if sample.source.startswith("longmemeval_"):
        meta = sample.metadata
        types, ids = meta.get("question_types") or [], meta.get("question_ids") or []
        if index >= len(types) or index >= len(ids):
            raise ValueError("Missing official LongMemEval question type/ID")
        prompt = get_anscheck_prompt(types[index], sample.questions[index],
                                    sample.answers[index], output, abstention="_abs" in ids[index])
        response = llm.chat(prompt, params=GenParams(temperature=0, seed=None,
                            max_tokens=10, exact_max_tokens=True), stage="mab.judge.longmemeval")
        if response.filtered or response.error or not response.text:
            raise RuntimeError("LongMemEval judge did not produce a valid response")
        # Author evaluator uses substring 'yes', not lexical QA F1 or a substitute.
        return {"metrics": {"longmemeval_judge_accuracy": float("yes" in response.text.lower())},
                "model": getattr(llm, "deployment", llm.name), "responses": [response.text]}
    if sample.source.startswith("infbench_"):
        keypoints = sample.metadata.get("keypoints") or []
        text = output.strip()
        prompts = [SUMMARY_PROMPTS["fluency_prompt_book"].format(text=text),
                   SUMMARY_PROMPTS["recall_prompt_book"].format(
                       keypoints="\n".join(f"{i + 1}. {v}" for i, v in enumerate(keypoints)), summary=text),
                   SUMMARY_PROMPTS["precision_prompt_book"].format(
                       expert_summary=sample.evaluation_answers(index)[0], summary=text)]
        responses = [llm.chat(p, params=GenParams(temperature=0.1, top_p=0.9, seed=42,
                     max_tokens=4096, exact_max_tokens=True), stage=f"mab.judge.summary.{stage}")
                     for p, stage in zip(prompts, ("fluency", "recall", "precision"))]
        if any(r.filtered or r.error or not r.text for r in responses):
            raise RuntimeError("Summary judge did not produce all three responses")
        parsed = [parse_summary_json(r.text) for r in responses]
        if any(not isinstance(d, dict) for d in parsed):
            raise ValueError("Malformed summary judge JSON; no score was committed")
        flu, rec, prec = parsed
        # Validate types/ranges; malformed judges remain pending instead of
        # being silently excluded from a favorable average.
        values = [flu.get("fluency"), rec.get("recall"), prec.get("precision"), prec.get("sentence_count")]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
            raise ValueError("Invalid numeric judge scores")
        if not (0 <= values[0] <= 1 and 0 <= values[1] <= len(keypoints)
                and 0 <= values[2] <= values[3]):
            raise ValueError("Judge scores out of rubric range")
        recall = values[1] / len(keypoints) if keypoints else 0
        precision = values[2] / values[3] if values[3] else 0
        score = values[0] * 2 * recall * precision / (recall + precision) if recall + precision else 0
        return {"metrics": {"summary_judge_f1": score, "summary_judge_recall": recall,
                            "summary_judge_precision": precision, "summary_judge_fluency": values[0]},
                "model": getattr(llm, "deployment", llm.name), "responses": [r.text for r in responses],
                "parsed": parsed}
    raise ValueError(f"No LLM judge for {sample.source}")
