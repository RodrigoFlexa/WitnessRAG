"""Secondary, strict numeric diagnostic. Never changes official LoCoMo F1."""
from __future__ import annotations

import re

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19, "twenty": 20,
         "once": 1, "twice": 2, "thrice": 3}


def exact_number(value: str) -> int | None:
    """Accept a definite isolated integer, excluding dates and lower bounds."""
    s = (value or "").strip().casefold().strip('"\' .!?')
    s = re.sub(r"\s+(?:times?|visits?|children|kids|games|events?)$", "", s)
    if s in WORDS:
        return WORDS[s]
    if re.fullmatch(r"\d{1,4}", s):
        return int(s)
    return None


def numeric_diagnostic(question: str, prediction: str, gold: str) -> dict:
    if not re.search(r"\bhow many\b", question, re.I):
        return {"aplicavel": False}
    expected, actual = exact_number(gold), exact_number(prediction)
    return {"aplicavel": True, "ouro": expected, "resposta": actual,
            "avaliavel": expected is not None,
            "correto": expected is not None and actual is not None and expected == actual}
