"""Answer-blind temporal and complementary selection under a fixed top-k budget."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from wrag.util import normalize

_STOP = {"the", "a", "an", "is", "was", "were", "did", "does", "do", "to", "of",
         "in", "on", "at", "for", "and", "or", "what", "which", "who", "when", "where",
         "how", "many", "much", "has", "have", "had", "their", "his", "her"}
_TEMPORAL = {"when", "before", "after", "first", "last", "latest", "recent", "recently",
             "earlier", "later", "oldest", "newest", "during", "since", "until", "ago",
             "yesterday", "tomorrow", "week", "month", "year", "date", "time", "long"}


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text)) if len(w) > 1 and w not in _STOP}


def is_temporal_question(question: str) -> bool:
    words = set(re.findall(r"[a-z]+", normalize(question)))
    return bool(words & _TEMPORAL or re.search(r"\b(?:19|20)\d{2}\b", question))


def _requirements(diagnostics: dict) -> list[str]:
    out: list[str] = []
    for plan in diagnostics.get("planos_compilados") or []:
        query = plan.get("consulta") or {}
        for value in query.get("source_conditions") or []:
            if str(value).strip():
                out.append(str(value))
        check = plan.get("obrigacoes") or {}
        out.extend(str(x) for x in check.get("faltas") or [] if str(x).strip())
        for atom in query.get("atoms") or []:
            phrase = " ".join(str(atom.get(k, "")) for k in ("subject", "relation", "object")
                              if not str(atom.get(k, "")).startswith("?"))
            if phrase.strip():
                out.append(phrase)
    return list(dict.fromkeys(out))[:6]


@dataclass(frozen=True)
class Selection:
    pids: list[str]
    changed: bool
    added: str = ""
    removed: str = ""
    reason: str = ""
    score: float = 0.0


def select_complement(corpus, question: str, baseline: list[str], diagnostics: dict,
                      k: int = 5, *, temporal: bool = True,
                      complementary: bool = True,
                      candidate_pids: list[str] | None = None) -> Selection:
    """Replace at most the last passage using only question/plan/corpus data.

    The first four retrieved passages are protected at k=5.  A candidate must
    cover a question or plan term absent from the protected context.  Temporal
    questions additionally prefer an appropriate chronological endpoint.  No
    answer, question category, or annotated support is consulted.
    """
    base = list(dict.fromkeys(baseline))[:k]
    if not base or (not complementary and not (temporal and is_temporal_question(question))):
        return Selection(base, False)
    protected = base[:max(1, k - 1)]
    protected_terms = set().union(*(_terms(corpus.get(pid).full) for pid in protected))
    required_text = " ".join([question] + _requirements(diagnostics))
    wanted = _terms(required_text)
    missing = wanted - protected_terms
    temporal_query = temporal and is_temporal_question(question)
    if not missing and not temporal_query:
        return Selection(base, False, reason="no_missing_facet")

    allowed = set(candidate_pids) if candidate_pids is not None else None
    docs = [(p, _terms(p.full)) for p in corpus.passages
            if p.pid not in base and (allowed is None or p.pid in allowed)]
    df = Counter(term for _p, words in docs for term in words)
    n = max(1, len(docs))
    latest = any(x in normalize(question) for x in ("last", "latest", "recent", "newest", "after"))
    earliest = any(x in normalize(question) for x in ("first", "earliest", "oldest", "before"))
    max_sequence = max((p.sequence for p in corpus.passages), default=0)
    best = (0.0, "", "")
    for passage, words in docs:
        # Temporal endpoint selection may legitimately repeat the event/entity
        # terms already seen in an earlier passage; chronology is the novelty.
        direct = words & (wanted if temporal_query else (missing or wanted))
        if not direct:
            continue
        lexical = sum(math.log((n + 1) / (df[t] + 1)) + 1.0 for t in direct)
        chronology = 0.0
        reason = "complementary_facet"
        if temporal_query and passage.sequence >= 0 and max_sequence > 0:
            position = passage.sequence / max_sequence
            chronology = (position if latest else (1.0 - position if earliest else 0.25)) * 1.5
            if passage.session_time:
                chronology += 0.5
            reason = "temporal_endpoint" if (latest or earliest) else "temporal_evidence"
        score = lexical + chronology
        candidate = (score, passage.pid, reason)
        if candidate > best:
            best = candidate
    if not best[1]:
        return Selection(base, False, reason="no_complement")
    out = protected + [best[1]]
    for pid in base:
        if len(out) >= k:
            break
        if pid not in out:
            out.append(pid)
    removed = next((pid for pid in base if pid not in out), "")
    return Selection(out[:k], out[:k] != base, best[1], removed, best[2], best[0])
