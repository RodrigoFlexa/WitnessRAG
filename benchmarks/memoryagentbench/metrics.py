"""Primary metrics ported from official eval_other_utils.py (MIT license).

No permissive label extraction: 'label: 43' is NOT normalized to '43'.
ROUGE is deliberately omitted: it is not a primary metric of this benchmark.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache


def normalize_answer(text: str) -> str:
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def parse_output(text: str, answer_prefix: str = "Answer:") -> str | None:
    for pattern in (re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", re.I),
                    re.compile(r"(?:^)(.*)(?:\n|$)")):
        match = pattern.search(text)
        if match:
            return re.sub(f"^{re.escape(answer_prefix)}", "", match[1].strip(), flags=re.I).strip()
    return None


def flatten_answers(answers) -> list[str]:
    if isinstance(answers, str):
        return [answers]
    if answers and isinstance(answers[0], list):
        return [a for group in answers for a in group]
    return list(answers)


def f1(prediction: str, answer: str) -> float:
    p, a = normalize_answer(prediction), normalize_answer(answer)
    if (p in {"yes", "no", "noanswer"} or a in {"yes", "no", "noanswer"}) and p != a:
        return 0.0
    pt, at = p.split(), a.split()
    common = sum((Counter(pt) & Counter(at)).values())
    return 2 * common / (len(pt) + len(at)) if common else 0.0


def basic_metrics(prediction: str, answers) -> dict[str, float]:
    gold = flatten_answers(answers)
    if not gold:
        raise ValueError("No reference answers")
    p = normalize_answer(prediction)
    return {"exact_match": float(any(p == normalize_answer(a) for a in gold)),
            "substring_exact_match": float(any(normalize_answer(a) in p for a in gold)),
            "f1": max(f1(prediction, a) for a in gold)}


def movie_name(text: str) -> str:
    text = text.split("/")[-1].replace("_", " ").replace("-", " ").replace(">", " ")
    return re.sub(r"\s+", " ", re.sub(r"\([^()]*\)", "", text)).strip()


def edit_distance(a: str, b: str) -> int:
    """Myers bit-vector Levenshtein; same distance as upstream editdistance.eval.

    Pure Python fallback avoids installing packages in a running experiment's
    environment and remains fast enough for the full movie candidate catalogue.
    """
    if not a:
        return len(b)
    masks = {}
    for i, char in enumerate(a):
        masks[char] = masks.get(char, 0) | (1 << i)
    pv, mv, score, high = (1 << len(a)) - 1, 0, len(a), 1 << (len(a) - 1)
    for char in b:
        eq = masks.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph, mh = mv | ~(xh | pv), pv & xh
        score += bool(ph & high) - bool(mh & high)
        ph, mh = (ph << 1) | 1, mh << 1
        pv, mv = mh | ~(xv | ph), ph & xv
    return score


class MovieScorer:
    def __init__(self, name_to_id: dict[str, int]):
        self.id_to_name = {value: movie_name(name) for name, value in name_to_id.items()}
        # Upstream uses list(set(...)) for ties. Entrypoint pins PYTHONHASHSEED=0;
        # no threshold, exact-name-only or fuzzy bonus is added here.
        self.candidates = list(set(self.id_to_name.values()))
        if not self.candidates:
            raise ValueError("Empty movie catalogue")

    @lru_cache(maxsize=8192)
    def nearest(self, text: str) -> str:
        return min(self.candidates, key=lambda c: edit_distance(text.lower(), c.lower()))

    def score(self, output: str, answers) -> tuple[dict[str, float], list[str]]:
        if "1." in output:
            text = output.split("1.", 1)[1]
        else:
            text = output.replace(",", "\n")
        names = []
        for item in text.split("\n"):
            item = re.sub(r"\([^()]*\)", "", item.strip())
            item = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", item)
            item = re.sub(r"\s+", " ", item).strip()
            names.append(self.nearest(item))  # also retain blank lines, as upstream
        gold = [self.id_to_name[int(x.strip())] for x in flatten_answers(answers)]
        return ({f"recsys_recall@{k}": sum(x in names[:k] for x in gold) / len(gold)
                 for k in (1, 5, 10)}, names)


def score(source: str, output: str, answers, movies: MovieScorer | None = None) -> tuple[dict, object]:
    if source.startswith("recsys_"):
        if movies is None:
            raise ValueError("Recommendation scoring requires entity2id.json")
        return movies.score(output, answers)
    parsed = parse_output(output)
    if source.startswith(("icl_", "eventqa_")):
        metrics = basic_metrics(parsed or "", answers)
    else:
        metrics = basic_metrics(output, answers)
        if parsed is not None:
            other = basic_metrics(parsed, answers)
            metrics = {key: max(value, other[key]) for key, value in metrics.items()}
    if source.startswith("eventqa_"):
        gold = flatten_answers(answers)
        metrics["eventqa_recall"] = float(all(a.lower() in output.lower() for a in gold))
    return metrics, parsed
