"""
Cascade router (docs/cascata.md): one short LLM call decides whether a
question needs the proof controller (PLAN) or can be answered from the facts
closest to the question (DIRECT).

The router reads only the question text. It never sees the benchmark category,
the answer, annotated evidence or the memory. Any failure (blocked call,
unparsable output) routes to PLAN, the full method, so the router can only
save work, never skip the plan by accident.

Off by default (``WitnessConfig.plan_router = ""``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from wrag.data import Question
from wrag.llm import LLM, GenParams
from wrag.llm.filters import LEDGER

ROUTE_PLAN = "PLAN"
ROUTE_DIRECT = "DIRECT"

ROUTER_SYSTEM = ("You decide how much search a question over a long conversation "
                 "memory needs. You answer with one JSON object and nothing else.")

# Synthetic names only. The criteria describe what the search must do (time
# reasoning, collecting, linking facts), never a benchmark category.
ROUTER_TEMPLATE = """A question will be answered from the memory of a long chat between two
people that went on for months. There are two ways to search the memory:

DIRECT: take the remembered statements most similar to the question and answer.
  Fast. Enough when one statement answers the question.
PLAN: write a search plan over a graph of dated facts and prove the answer from
  them. Slower. Needed when the answer must be built from several facts.

Choose PLAN when the question needs any of these:
- time: when something happened, a date, how long, how long ago, the order of
  events, the first or the latest time, what happened before or after something;
- several items or a count spread over the conversations (all the places, every
  hobby, how many times);
- linking facts: the answer is found through another fact (the city where the
  friend of Ana moved; the gift from the person who visited).
Choose DIRECT for everything else: what someone likes, does, owns, said, feels
or plans, a reason, an opinion, or whether something is likely about a person.

Examples:
"What instrument does Nira play?" -> {{"route": "DIRECT"}}
"Why did Omar quit his job?" -> {{"route": "DIRECT"}}
"Would Lia enjoy a jazz festival?" -> {{"route": "DIRECT"}}
"When did Nira start playing the violin?" -> {{"route": "PLAN"}}
"How many trips has Omar taken this year?" -> {{"route": "PLAN"}}
"What sports does Lia play?" -> {{"route": "PLAN"}}
"Where did the brother of Nira move?" -> {{"route": "PLAN"}}

Question: "{question}"
Answer with {{"route": "PLAN"}} or {{"route": "DIRECT"}}."""


@dataclass
class RouteDecision:
    route: str = ROUTE_PLAN
    called: bool = False
    raw: str = ""
    fallback: str = ""            # why PLAN was forced ("", "filtrado", "invalido")

    @property
    def plan(self) -> bool:
        return self.route == ROUTE_PLAN

    def to_dict(self) -> dict[str, Any]:
        out = {"rota": self.route, "chamou": self.called, "saida": self.raw[:60]}
        if self.fallback:
            out["padrao_por"] = self.fallback
        return out


def parse_route(text: str, data: Any) -> str:
    """PLAN or DIRECT from the model output; "" when neither is clear."""
    if isinstance(data, dict):
        value = str(data.get("route") or data.get("decision") or "").strip().upper()
        if value in {ROUTE_PLAN, ROUTE_DIRECT}:
            return value
    found = set(re.findall(r"\b(PLAN|DIRECT)\b", (text or "").upper()))
    return found.pop() if len(found) == 1 else ""


def route_question(llm: LLM, question: Question, *, dataset: str = "",
                   method: str = "witnessrag") -> RouteDecision:
    result = llm.chat(ROUTER_TEMPLATE.format(question=question.question.replace('"', "'")),
                      system=ROUTER_SYSTEM,
                      params=GenParams(temperature=0.0, max_tokens=16, json_mode=True),
                      stage="witness.route")
    if result.filtered:
        LEDGER.add("route", dataset, method, question.qid, "roteador bloqueado")
        return RouteDecision(ROUTE_PLAN, True, "", "filtrado")
    route = parse_route(result.text or "", result.json())
    if not route:
        return RouteDecision(ROUTE_PLAN, True, result.text or "", "invalido")
    return RouteDecision(route, True, result.text or "")
