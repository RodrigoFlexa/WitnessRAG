"""Núcleo teórico do WITNESS-RAG."""

from wrag.witness.query import Atom, ConjunctiveQuery, compile_query
from wrag.witness.search import Gap, SearchResult, Witness, WitnessSearcher
from wrag.witness.provenance import AnswerCandidate, rank_passages, score_answers
from wrag.witness.memory import MemoryView
from wrag.witness import budget

__all__ = [
    "Atom", "ConjunctiveQuery", "compile_query",
    "Witness", "WitnessSearcher", "SearchResult", "Gap",
    "AnswerCandidate", "score_answers", "rank_passages",
    "MemoryView", "budget",
]
