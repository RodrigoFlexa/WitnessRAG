"""Question-only hypotheses and literal-source retrieval for local plans.

This is a bounded English grammar, not a semantic parser with a completeness
guarantee. Uncertain clauses remain reader obligations. No benchmark metadata,
generated memories, inferred facts, or answer-specific rules are used.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from wrag.witness.local_plans import contract as temporal_contract, terms

_IRREGULAR = {"went": "go", "gone": "go", "seen": "see", "saw": "see",
              "bought": "buy", "made": "make", "done": "do", "did": "do",
              "taken": "take", "took": "take", "children": "child", "kids": "child",
              "suggestion": "recommend", "suggested": "recommend", "suggest": "recommend"}
_HEADS = {"book", "film", "movie", "song", "instrument", "subject", "topic", "symbol",
          "activity", "event", "item", "place", "city", "country", "person", "name",
          "hobby", "artist", "band", "job", "career", "pet", "food", "type", "kind"}
_FILLER = {"both", "each", "recent", "ago", "year", "month", "day", "week", "time",
           "many", "much", "do", "get", "kind", "type", "some", "same", "other"}


@lru_cache(maxsize=30000)
def stems(text):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return frozenset(stemmer.stem(_IRREGULAR.get(w, w)) for w in terms(text))


@dataclass
class ReadingContract:
    temporal: object
    head: str = ""
    predicate_terms: set[str] = field(default_factory=set)
    owner_names: list[str] = field(default_factory=list)
    conjunction: bool = False
    answer_role: str = "either"
    count_unit: str = ""
    predicate_text: str = ""
    subject_scope: str = ""

    def to_dict(self):
        return {**self.temporal.to_dict(), "answer_head": self.head,
                "predicate_terms": sorted(self.predicate_terms), "owners": self.owner_names,
                "conjunction": self.conjunction, "answer_role": self.answer_role,
                "count_unit": self.count_unit, "subject_scope":self.subject_scope,
                "status": "question_only_hypothesis"}

    def instructions(self):
        lines = ["Reading task (question-derived hypothesis, not evidence):",
                 f"Requested operation: {self.temporal.operation}; answer type: {self.head or self.temporal.expected_type}."]
        if self.conjunction:
            lines.append("The same answer must satisfy the requested relation for every named participant: " + ", ".join(self.owner_names) + ".")
        if self.subject_scope:
            lines.append("The requested relation concerns the named owner's " + self.subject_scope + ", rather than automatically the owner personally.")
        if self.count_unit == "occurrence":
            lines.append("Count supported event occurrences, not fact rows, people, plans, photographs or repeated mentions. Group mentions of the same occurrence; retrieval does not establish global completeness.")
        if self.temporal.operation == "set":
            lines.append("Collect the supported members across sources; exclude related activities that do not satisfy the requested condition.")
        if self.temporal.operation == 'premises':
            lines.append("For an inferred or hypothetical answer, distinguish the observed facts from the scenario in the question. Check how a changed premise affects the inference; an observed intention alone does not settle a counterfactual.")
        period=self.temporal.period
        if period.interval and self.temporal.temporal_side=='near':
            # Grounded only by the date the reference event was mentioned:
            # a ranking hint, not a constraint on the answer.
            lines.append(f"Temporal reference (hypothesis): the reference event \"{period.text}\" was mentioned around {period.interval.start} to {period.interval.end}; verify its actual time in the sources.")
        elif period.interval and (period.kind!='now' or self.temporal.temporal_side=='recent'):
            lines.append(f"Temporal reference (hypothesis): {period.interval.start} to {period.interval.end}; relation: {self.temporal.temporal_side or period.kind}; reference: {period.text}.")
            lines.append("The requested event's answer date must satisfy that relation to the reference. Session dates and dates of other events are not substitutes. Verify event-relative anchors in the quoted sources.")
        lines += ["Return the requested value rather than its owner or a generic description. Resolve descriptive references using corroborating source turns when supported; mark ambiguity when multiple antecedents remain.",
                  "Check roles, qualifiers, event time and modality against the original question. A graph join and its score do not certify an answer."]
        return "\n".join(lines)


def reading_contract(text, dated, names):
    base = temporal_contract(text, dated)
    low = text.casefold()
    head = ""
    match = re.search(r"^(?:what|which)\s+(?:(?:kind|type)s?\s+of\s+)?([a-z]+)", low)
    if match:
        token = match[1]
        head = next((h for h in _HEADS if stems(h) == stems(token)), "")
        if not head:
            phrase=re.split(r'\b(?:is|are|was|were|has|have|had|did|does|do|will|would|could|should)\b',low[match.start(1):],maxsplit=1)[0].strip()
            if 0<len(phrase.split())<=4 and '?' not in phrase:
                head=phrase
    if re.search(r"\b(?:activities|events|items|books|symbols|instruments|artists|bands)\b", low) and base.operation == "value":
        base.operation = "set"
        base.pending.append("retrieval_does_not_prove_global_completeness")
    if re.search(r"\bwhat do\b.*\b(?:like|enjoy)\b", low):
        base.operation = "set"
    conjunction = bool(re.search(r"\bboth\b|\bin common\b|\beach\b", low)) and len(names) >= 2
    # Remove the event-reference clause: its verb must not replace the event
    # whose date is asked for. The temporal compiler still sees the whole q.
    main = re.split(r"\b(?:before|after|since|prior to)\b", low, maxsplit=1)[0]
    for name in sorted(names, key=len, reverse=True):
        if ' ' not in name:
            main = re.sub(r"(?<!\w)" + re.escape(name.casefold()) + r"(?:'s)?(?!\w)", " ", main)
    predicate = stems(main) - stems(" ".join(_FILLER | _HEADS))
    scope = re.search(r"\b\w+'s\s+(children|kids|pets|family)\b",low)
    scope_text=scope[1] if scope else ""
    if scope_text:predicate=predicate-stems(scope_text)
    role = "subject" if re.match(r"who\s+(?!did\b|does\b|do\b|has\b|have\b)", low) else "object"
    if base.operation in {"premises", "compare", "duration"}:
        role = "either"
    unit = "occurrence" if re.search(r"\bhow many times\b|\bhow many (?:visits|trips|events|occurrences)\b", low) else "entity"
    return ReadingContract(base, head, predicate, names, conjunction, role,
                           unit if base.operation == "count" else "", main.strip(),scope_text)


class SourceIndex:
    """Small BM25 index of original turns; all output is literal source text."""
    def __init__(self, dated):
        self.dated=dated
        self.turns = [(pid, i, turn) for pid, turns in dated.turns.items()
                      for i, turn in enumerate(turns) if turn.speaker and turn.text]
        self.bags = [Counter(stems(turn.text + " " + turn.speaker)) for _, _, turn in self.turns]
        df = Counter(term for bag in self.bags for term in bag)
        self.idf = {term: math.log(1 + (len(self.bags)-n+.5)/(n+.5)) for term,n in df.items()}
        self.average = sum(sum(bag.values()) for bag in self.bags)/max(1,len(self.bags))

    def rank(self, terms_, owner="", before=None, temporal=None):
        results = []
        for index, ((pid, pos, turn), bag) in enumerate(zip(self.turns, self.bags)):
            if before and turn.when and turn.when > before:
                continue
            size = sum(bag.values())
            score = sum(self.idf.get(t,0)*bag[t]*2.2/(bag[t]+1.2*(.25+.75*size/max(1,self.average)))
                        for t in terms_ if bag[t])
            if owner:
                if turn.speaker.casefold() == owner.casefold():
                    score *= 1.5
                elif owner.casefold() not in turn.text.casefold():
                    score *= .35
            if score > 0:
                if temporal is not None and temporal.time_weight and turn.when:
                    from wrag.witness.scoring import proximity,proximity_scale
                    from wrag.witness.timeline import Interval
                    prior=proximity(Interval(turn.when,turn.when),temporal.period,
                                    proximity_scale(temporal.period,self.dated))
                    score*=1-temporal.time_weight+temporal.time_weight*prior
                results.append((score, index))
        return sorted(results, key=lambda x:(-x[0],x[1]))

    def support(self, text, contract, limit=8):
        selected = []
        def add(index, reason):
            for position,(i,old_reason) in enumerate(selected):
                if i==index:
                    if reason=='candidate_antecedent_not_resolved':
                        selected[position]=(i,reason)
                    return
            selected.append((index,reason))
        # Separate origins keep one participant's contextual chatter from
        # consuming the entire source budget of an intersection question.
        query = stems(text)
        predicate = contract.predicate_terms or query
        for owner in contract.owner_names:
            for _, index in self.rank(predicate, owner=owner,temporal=contract.temporal)[:2]:
                add(index,"participant_predicate")
        for _, index in self.rank(query,temporal=contract.temporal)[:3]:
            add(index,"question_relevance")
        # A descriptive reference requires candidate antecedents, not an
        # invented edge. Deliver earlier explicit quotations of the same type.
        if contract.head in {"book","film","movie","song"}:
            refs = [(i, self.turns[i][2]) for i,_ in selected
                    if re.search(r"\b(?:that|this|the)\s+(?:book|film|movie|song)\b", self.turns[i][2].text, re.I)]
            for _, referring in refs[:2]:
                ranked = self.rank(stems(contract.head + " read recommend"), before=referring.when,temporal=contract.temporal)
                quoted = [(score,i) for score,i in ranked if re.search(r'["“][^"”]+["”]',self.turns[i][2].text)]
                for _, index in quoted[:2]:
                    add(index,"candidate_antecedent_not_resolved")
        # Keep antecedents in the budget even when the ordinary probes fill it.
        selected.sort(key=lambda item:(item[1] != "candidate_antecedent_not_resolved",))
        return [{"pid":self.turns[i][0], "position":self.turns[i][1],
                 "turn_id":self.turns[i][2].turn_id, "speaker":self.turns[i][2].speaker,
                 "when":str(self.turns[i][2].when or ""), "text":self.turns[i][2].text,
                 "reason":reason} for i,reason in selected[:limit]]
