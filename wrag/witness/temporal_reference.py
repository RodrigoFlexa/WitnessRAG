"""Temporal reference: declared by the planner, grounded by the executor,
checked by the reflection.

Planning. One short LLM call reads ONLY the question and the memory's present
date (the agent's "now"). It declares how time enters the question: whether
the answer is a time, which event it concerns, an explicit window, a
restriction relative to another event, an order (first/last) and the precision
the answer needs. No memory content, gold answer, supporting passage or
benchmark category is shown to this call, so its output can be cached and
audited on its own. An invalid or blocked reply leaves the rule-based contract
unchanged.

Execution. The declared reference becomes a concrete interval: an explicit
window is parsed by the deterministic calendar parser; an event-relative
reference is grounded in the memory by matching the reference event to dated
facts of the named people. The executor also projects the requested event's
time from the witnesses it found (candidate times), at the precision of the
fact that states it.

Reflection. The reflection receives the declared reference and the candidate
times and checks that a date answer is the requested event's own time,
complete and consistent with them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wrag.llm.base import GenParams
from wrag.witness.timeline import Interval, anchored_phrase, format_interval, natural_interval, parse_anchor

VERSION = "temporal-reference-v1"

TARGETS = ("date", "duration", "other")
SCOPES = ("none", "window", "event")
RELATIONS = ("within", "before", "after")
ORDERS = ("none", "first", "last")
GRANULARITIES = ("day", "week", "month", "year", "season", "any")

SYSTEM = ("You describe the temporal needs of a question before any memory search. "
          "You never answer the question. Reply with one JSON object.")

TEMPLATE = """Describe how time enters the question below. Do NOT answer it.
The memory is a dated record of conversations. Its latest record is {now}.

Fields:
- target: "date" when the question asks for a time (when, what date, which
  month/year/day, in what season); "duration" when it asks how long or how much
  time passed; otherwise "other".
- event: the event or state the question is about, in a few words of the
  question (for example "Lina adopted a puppy").
- scope: "window" when the answer is restricted to an explicit period stated in
  the question; "event" when it is restricted by another event (before, after,
  during or since that event); otherwise "none".
- window: for scope "window", the period copied from the question (for example
  "July 2022", "the summer of 2023", "last year"); otherwise "".
- relation: "within", "before" or "after" the window or reference event; "" if
  scope is "none". "During" and "while" are "within"; "since" is "after".
- anchor_event: for scope "event", the reference event in a few words (for
  example "returning from Chicago"); otherwise "".
- order: "first" for the first or earliest occurrence; "last" for the latest,
  most recent or current one; otherwise "none".
- granularity: the precision the answer needs: "day", "week", "month",
  "year", "season" or "any".

Return exactly: {{"target":"...","event":"...","scope":"...","window":"","relation":"","anchor_event":"","order":"none","granularity":"any"}}

Synthetic examples (do not copy their names):
Question: "When did Lina adopt her puppy?"
{{"target":"date","event":"Lina adopted her puppy","scope":"none","window":"","relation":"","anchor_event":"","order":"none","granularity":"day"}}
Question: "Where did Omar travel in March 2021?"
{{"target":"other","event":"Omar travelled","scope":"window","window":"March 2021","relation":"within","anchor_event":"","order":"none","granularity":"any"}}
Question: "What did Rui buy after moving to Porto?"
{{"target":"other","event":"Rui bought something","scope":"event","window":"","relation":"after","anchor_event":"Rui moved to Porto","order":"none","granularity":"any"}}
Question: "In which year did Ada start her first job?"
{{"target":"date","event":"Ada started her first job","scope":"none","window":"","relation":"","anchor_event":"","order":"first","granularity":"year"}}
Question: "What is Ben's current hobby?"
{{"target":"other","event":"Ben's hobby","scope":"none","window":"","relation":"","anchor_event":"","order":"last","granularity":"any"}}
Question: "How long has Mei been learning piano?"
{{"target":"duration","event":"Mei has been learning piano","scope":"none","window":"","relation":"","anchor_event":"","order":"none","granularity":"any"}}

QUESTION: {question}"""


@dataclass
class TemporalReference:
    target: str = "other"
    event: str = ""
    scope: str = "none"
    window: str = ""
    relation: str = ""
    anchor_event: str = ""
    order: str = "none"
    granularity: str = "any"
    valid: bool = False
    source: str = "none"
    repairs: list[str] = field(default_factory=list)

    def active(self) -> bool:
        return self.valid and (self.target in {"date", "duration"} or self.scope != "none"
                               or self.order != "none")

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "target": self.target, "event": self.event,
                "scope": self.scope, "window": self.window, "relation": self.relation,
                "anchor_event": self.anchor_event, "order": self.order,
                "granularity": self.granularity, "valid": self.valid, "source": self.source,
                "repairs": self.repairs, "status": "question_only_hypothesis"}

    def describe(self) -> str:
        """One line for the reader and the reflection; empty when inactive."""
        if not self.active():
            return ""
        parts = []
        event = f' "{self.event}"' if self.event else ""
        if self.target == "date":
            parts.append(f"asks the time of{event or ' the event'}"
                         + (f" at {self.granularity} precision" if self.granularity != "any" else ""))
        elif self.target == "duration":
            parts.append(f"asks a duration{(' for' + event) if event else ''}")
        elif event:
            parts.append(f"concerns{event}")
        if self.scope == "window" and self.window:
            parts.append(f"restricted {self.relation or 'within'} {self.window}")
        if self.scope == "event" and self.anchor_event:
            parts.append(f'restricted {self.relation or "relative to"} the event "{self.anchor_event}"')
        if self.order == "first":
            parts.append("the first or earliest occurrence")
        if self.order == "last":
            parts.append("the latest or current occurrence")
        return "; ".join(parts)


def _pick(value: Any, allowed: tuple[str, ...], default: str, name: str,
          repairs: list[str], aliases: dict[str, str] | None = None) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in allowed:
        return text
    if aliases and text in aliases and aliases[text] in allowed:
        repairs.append(f"{name}:{text}->{aliases[text]}")
        return aliases[text]
    if text:
        repairs.append(f"{name}:{text}->{default}")
    return default


def parse_reference(data: Any) -> TemporalReference:
    """Strict vocabulary with a few fixed synonyms; nothing is invented."""
    if not isinstance(data, dict):
        return TemporalReference(repairs=["no_json_object"])
    repairs: list[str] = []
    target = _pick(data.get("target"), TARGETS, "other", "target", repairs,
                   {"time": "date", "when": "date", "how_long": "duration", "none": "other"})
    scope = _pick(data.get("scope"), SCOPES, "none", "scope", repairs,
                  {"period": "window", "relative": "event", "anchor": "event"})
    relation = _pick(data.get("relation"), RELATIONS + ("",), "", "relation", repairs,
                     {"during": "within", "while": "within", "in": "within", "since": "after",
                      "prior_to": "before", "until": "before"})
    order = _pick(data.get("order"), ORDERS, "none", "order", repairs,
                  {"earliest": "first", "latest": "last", "recent": "last", "current": "last",
                   "most_recent": "last", "": "none"})
    granularity = _pick(data.get("granularity"), GRANULARITIES, "any", "granularity", repairs,
                        {"date": "day", "exact": "day", "": "any"})
    text = lambda key, n: " ".join(str(data.get(key) or "").split())[:n]
    window, anchor = text("window", 80), text("anchor_event", 120)
    if scope == "window" and not window:
        repairs.append("window_without_period->none")
        scope = "none"
    if scope == "event" and not anchor:
        repairs.append("event_scope_without_anchor->none")
        scope = "none"
    if scope == "none":
        relation, window, anchor = "", "", ""
    elif not relation:
        relation = "within"
    return TemporalReference(target, text("event", 120), scope, window, relation, anchor,
                             order, granularity, valid=True, source="llm", repairs=repairs)


_FIRST_CUE = re.compile(r"\b(first|earliest|initially|originally|start(?:ed)? out)\b", re.I)
_LAST_CUE = re.compile(r"\b(last|latest|most recent|recently|current|currently|now|still|these days)\b", re.I)


def ground_order(reference: TemporalReference, question_text: str) -> TemporalReference:
    """An order needs a literal cue: "finish" or "end" is not "the latest"."""
    cue = _FIRST_CUE if reference.order == "first" else _LAST_CUE if reference.order == "last" else None
    if cue is not None and not cue.search(question_text):
        reference.repairs.append(f"order:{reference.order}->none(no_cue)")
        reference.order = "none"
    return reference


def plan_reference(retriever, question_text: str) -> TemporalReference:
    """Question-only call, cached per question text on the retriever."""
    cache = retriever.__dict__.setdefault("_temporal_reference_cache", {})
    if question_text in cache:
        return cache[question_text]
    dated = retriever.dated
    now = format_interval(Interval(dated.last, dated.last)) if dated is not None and dated.last else "unknown"
    result = retriever.ctx.llm.chat(TEMPLATE.format(now=now, question=question_text), system=SYSTEM,
                                     params=GenParams(temperature=0.0, max_tokens=160, json_mode=True,
                                                      exact_max_tokens=True),
                                     stage="plan.temporal_reference")
    if result.filtered or not result.ok:
        reference = TemporalReference(repairs=["filtered" if result.filtered else "empty_reply"])
    else:
        reference = ground_order(parse_reference(result.json()), question_text)
    cache[question_text] = reference
    return reference


# -- execution -----------------------------------------------------------------

def session_date(dated, index: int):
    pid, position = dated.fact_turn[index]
    turns = dated.turns.get(pid) or []
    if 0 <= position < len(turns) and turns[position].when is not None:
        return turns[position].when
    interval = dated.passage_interval.get(pid)
    return interval.start if interval is not None and interval.start == interval.end else None


def fact_time_label(retriever, index: int) -> str:
    """The time of a fact exactly as the reader's notes state it."""
    dated, fact = retriever.dated, retriever.memory.facts[index]
    said = session_date(dated, index)
    interval = dated.fact_interval[index] if index < len(dated.fact_interval) else None
    if index < len(dated.fact_time_source) and dated.fact_time_source[index] == "expressao":
        expression = " ".join(str(getattr(fact, "time", "") or "").split())[:40]
        anchored = anchored_phrase(expression, said)
        natural = natural_interval(interval)
        if anchored and natural and anchored != natural:
            return f"{anchored} ({natural})"
        return anchored or natural
    when = format_interval(Interval(said, said)) if said else format_interval(interval)
    return f"stated in the session of {when}" if when else ""


def resolve_anchor_event(planner, reference: TemporalReference) -> dict[str, Any]:
    """Ground the reference event in the memory without generating anything.

    Candidates are dated facts of the named people (all facts when no person is
    named) that share a content word with the reference event, ranked by the
    embedding similarity of the reference event to the fact. The reference is
    resolved only when the best candidate is not tied with a candidate of a
    different date; otherwise every close date is reported to the reader.
    """
    from wrag.witness.local_contract import stems
    r, memory, dated = planner.r, planner.memory, planner.dated
    words = stems(reference.anchor_event) - stems(" ".join(name for name, _ in planner.mentions))
    if not words or not memory.fact_vectors.size:
        return {"status": "unresolved", "reason": "no_content_words"}
    owners = planner.owner_nodes
    pool = [i for i in planner.available
            if (not owners or planner.node(memory.facts[i].subj_id) in owners
                or planner.node(memory.facts[i].obj_id) in owners)
            and words & planner._fact_terms.get(i, frozenset())]
    if not pool:
        return {"status": "unresolved", "reason": "no_matching_fact"}
    names = " ".join(name for name, _ in planner.mentions)
    vector = r.ctx.embedder.encode([(names + ": " if names else "") + reference.anchor_event])[0]
    sims = memory.fact_vectors[pool] @ vector
    order = sorted(range(len(pool)), key=lambda n: (-float(sims[n]), pool[n]))
    best = pool[order[0]]
    best_interval = dated.fact_interval[best]
    rows = []
    for n in order[:6]:
        i = pool[n]
        rows.append({"fact": i, "similarity": round(float(sims[n]), 4),
                     "statement": (memory.facts[i].statement or memory.facts[i].verbalize())[:160],
                     "time": fact_time_label(r, i),
                     "explicit": dated.fact_time_source[i] == "expressao"})
    rivals = [row for row in rows[1:]
              if row["similarity"] >= rows[0]["similarity"] - 0.03
              and dated.fact_interval[row["fact"]] is not None and best_interval is not None
              and abs((dated.fact_interval[row["fact"]].start - best_interval.start).days) > 3]
    if best_interval is None:
        return {"status": "unresolved", "reason": "undated", "candidates": rows}
    if rivals:
        return {"status": "ambiguous", "candidates": [rows[0]] + rivals[:3]}
    return {"status": "resolved", "fact": best, "interval": best_interval,
            "explicit": rows[0]["explicit"], "candidates": rows[:1]}


_TIME_QUESTION = re.compile(
    r"^\s*(?:when\b|what (?:date|day|month|year|time|week|season)\b|which (?:date|day|month|year|week|season)\b|"
    r"(?:in|on|during) (?:what|which) (?:date|day|month|year|week|season)\b|how long ago\b)", re.I)


def asks_time(reference: TemporalReference | None, question_text: str, operation: str) -> bool:
    """A time is asked when the grammar says so, or when the declared target is
    a date AND the question opens with a time interrogative. A date stated
    inside the question ("on 25 February 2022") is a scope, not the target."""
    if reference is None or not reference.valid:
        return False
    if operation == "date":
        return True
    return reference.target == "date" and bool(_TIME_QUESTION.search(question_text))


def candidate_times(planner, retained, delivered=(), limit: int = 4) -> list[dict[str, Any]]:
    """Times of the requested event, projected by the executor (hypotheses).

    Only the event facts of the selected witnesses, and only times the fact
    itself states: a session date dates the mention, never the event.
    """
    from wrag.witness.local_contract import stems
    r, memory, dated = planner.r, planner.memory, planner.dated
    rows, seen = [], set()

    def add(i, origin):
        if i in seen or i >= len(dated.fact_time_source) or dated.fact_time_source[i] != "expressao":
            return
        seen.add(i)
        interval = dated.fact_interval[i]
        rows.append({"fact": i, "statement": (memory.facts[i].statement or memory.facts[i].verbalize())[:160],
                     "time": fact_time_label(r, i), "origin": origin,
                     "start": interval.start.isoformat() if interval else "",
                     "end": interval.end.isoformat() if interval else "", "explicit": True})

    for _score, c, w, _sem, _temp in retained:
        for i in planner.event_fact_ids(c, w):
            add(i, "witness")
    # No fallback to merely relevant delivered facts: on conv07 their times
    # were other events' times (-39.5 F1 on the 5 questions they reached).
    return rows[:limit]


def reader_lines(reference: TemporalReference | None, resolution: dict[str, Any] | None,
                 times: list[dict[str, Any]], order: str = "none") -> list[str]:
    """Only a projected time reaches the reader; a bare description of the
    reference did not help it (v3, conv03: -1.5 on 46 questions)."""
    if reference is None or not times:
        return []
    rows = sorted(times, key=lambda row: row["start"] or "9999") if order in {"first", "last"} else list(times)
    if order == "last":
        rows = rows[::-1]
    head = ("Times of the requested event" + (f' ("{reference.event}")' if reference.event else "")
            + " stated in the memory (hypotheses; the event must match the question"
            + ("; earliest first" if order == "first" else "; latest first" if order == "last" else "") + "):")
    return [head] + [f'- {row["statement"]} -> {row["time"]}' for row in rows] + [
        "Answer with the matching event's own time, in the form stated there; never a bare day such as \"the 17th\"."]


_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"


_FRAGMENT = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)\b|\b(?:last|next|this|ago|yesterday|today|tomorrow|tonight|weekend|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)


def incomplete_date(answer: str) -> bool:
    """A calendar FRAGMENT: a bare day ("the 17th", "15th after his trip") or a
    relative phrase without its anchor ("last Friday", "two days ago").

    Anchored forms ("the week before 9 June 2023") carry their date. An answer
    that places the event by an age or another event ("when she was 10",
    "after graduating") is a legitimate non-calendar time, not a fragment."""
    low = answer.casefold()
    if (re.search(r"\b(" + _MONTHS + r")\b", low) or re.search(r"\b(19|20)\d{2}\b", low)
            or re.search(r"\b(spring|summer|fall|autumn|winter)\b", low)):
        return False
    return bool(_FRAGMENT.search(low))


def answer_interval(answer: str, now) -> Interval | None:
    """Calendar period an answer denotes; an explicit range keeps both ends."""
    from wrag.witness.timeline import parse_dates
    if ";" in answer:
        return None
    dates = parse_dates(answer)
    if len(dates) >= 2 and re.search(r"\s(?:-|–|to|until)\s", answer):
        return Interval(min(dates), max(dates), answer)
    return parse_anchor(answer, now=now)


def same_period(answers: list[str], now) -> bool:
    """Every answer denotes a calendar period and all of them overlap."""
    values = [answer_interval(a, now) for a in answers]
    if len(values) < 2 or any(v is None for v in values):
        return False
    return max(v.start for v in values) <= min(v.end for v in values)


def overlaps(value: Interval, rows: list[dict[str, Any]], slack_days: int = 3) -> bool:
    from datetime import date, timedelta
    for row in rows:
        if not row.get("start"):
            continue
        start = date.fromisoformat(row["start"]) - timedelta(days=slack_days)
        end = date.fromisoformat(row["end"]) + timedelta(days=slack_days)
        if value.start <= end and value.end >= start:
            return True
    return False
