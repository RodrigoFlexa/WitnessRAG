"""Extract verbatim dialogue windows from frozen LoCoMo chunks."""
from __future__ import annotations

import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
TURN = re.compile(r"^\[D\d+:\d+\] (?!Image caption)", re.I)
STOP = set("what when where which who whom whose how many did does do was were is are the a an to of for in on with and or from by at it they she he them their her his have has had that this".split())


def terms(value: str) -> set[str]:
    return {word for word in TOKEN.findall(value.casefold()) if len(word) > 2 and word not in STOP}


def focus_passage(text: str, question: str, max_chars: int = 3000, neighbors: int = 1) -> tuple[str, dict]:
    """Select original speaker turns; retain each selected turn's original session date."""
    lines = text.splitlines()
    date = ""
    turns = []
    wants_images = bool(re.search(r"\b(image|photo|picture|pic)\b", question, re.I))
    for line in lines:
        if line.startswith("Session date:"):
            date = line
        elif TURN.match(line) or (wants_images and re.match(r"^\[D\d+:\d+\] Image caption", line)):
            turns.append((date, line))
    if not turns:
        return text[:max_chars], {"turns": [], "fallback": True}
    query = terms(question)
    df = Counter(word for _, line in turns for word in terms(line))
    scored = []
    for i, (_, line) in enumerate(turns):
        overlap = query & terms(line)
        score = sum(1 / (1 + df[word] ** .5) for word in overlap)
        scored.append((score, i))
    chosen = set()
    for score, i in sorted(scored, reverse=True):
        if score <= 0 and chosen:
            break
        trial = chosen | set(range(max(0, i-neighbors), min(len(turns), i+neighbors+1)))
        rendered = render(turns, trial)
        if len(rendered) <= max_chars:
            chosen = trial
        if len(chosen) >= 12:
            break
    if not chosen:
        best = max(scored)[1]
        chosen = {best}
    output = render(turns, chosen)
    if len(output) > max_chars:
        output = output[:max_chars]
    return output, {"turns": [i for i in sorted(chosen)], "fallback": False,
                    "original_chars": len(text), "selected_chars": len(output)}


def render(turns: list[tuple[str, str]], indices: set[int]) -> str:
    result = []
    previous_date = None
    previous_index = None
    for i in sorted(indices):
        date, line = turns[i]
        if previous_index is not None and i != previous_index + 1:
            result.append("[...]")
        if date != previous_date:
            result.append(date)
        result.append(line)
        previous_date, previous_index = date, i
    return "\n".join(result)
