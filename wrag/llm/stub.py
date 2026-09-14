"""
Backend determinístico para testes offline do pipeline.

Extração e respostas são regras simplificadas, sem valor como resultados
científicos. O filtro simulado permite exercitar bloqueios de conteúdo.
Estes testes não substituem a validação dos provedores reais.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from wrag import config as C
from wrag.llm.base import LLM, GenParams, LLMResult

INPUT_MARKER = "### INPUT"

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "they", "this", "to", "was", "were", "which", "who",
    "with", "what", "when", "where", "whose", "did", "does", "do",
}

# Cuidado com o padrão: uma versão com quantificador aninhado — (?:[A-Z]\w*(?:\s+of\s+)?)+
# — trava por backtracking catastrófico em passagens longas que não casam. Cada
# ramo aqui consome pelo menos um token capitalizado, então o casamento é linear.
_TOKEN = r"[A-Z][\w'’\-]*"
_ENTITY_RE = re.compile(
    rf"\b{_TOKEN}(?:\s+(?:of|de|da|do|the|von|van|del)\s+{_TOKEN}|\s+{_TOKEN})*\b"
    r"|\b\d{1,2}\s+[A-Za-z]+\s+\d{3,4}\b"
    r"|\b\d{3,4}\b"
)


def payload_of(messages: list[dict[str, str]]) -> str:
    text = messages[-1]["content"]
    if INPUT_MARKER in text:
        return text.split(INPUT_MARKER, 1)[1].strip()
    return text.strip()


def _entities(text: str, limit: int = 24) -> list[str]:
    seen: list[str] = []
    for match in _ENTITY_RE.finditer(text):
        span = match.group(0).strip(" ,.;:'\"")
        if len(span) < 2 or span.lower() in _STOPWORDS:
            continue
        if span not in seen:
            seen.append(span)
        if len(seen) >= limit:
            break
    return seen


def _triples(text: str, limit: int = 20) -> list[list[str]]:
    out: list[list[str]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        spans = [(m.start(), m.end(), m.group(0).strip(" ,.;:'\"")) for m in _ENTITY_RE.finditer(sentence)]
        spans = [s for s in spans if len(s[2]) > 1]
        for i in range(len(spans) - 1):
            subject, obj = spans[i][2], spans[i + 1][2]
            if subject == obj:
                continue
            middle = sentence[spans[i][1]:spans[i + 1][0]]
            words = [w for w in re.findall(r"[a-zA-Z]+", middle) if w.lower() not in _STOPWORDS]
            relation = " ".join(words[:5]).lower() or "related to"
            out.append([subject, relation, obj])
            if len(out) >= limit:
                return out
    return out


class StubLLM(LLM):
    name = "stub"

    def __init__(self, filter_rate: float | None = None) -> None:
        super().__init__()
        if filter_rate is None:
            filter_rate = float(C._env_float("WRAG_STUB_FILTER_RATE", 0.0))
        self.filter_rate = max(0.0, min(1.0, filter_rate))

    def _should_filter(self, payload: str) -> bool:
        if self.filter_rate <= 0:
            return False
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return (digest[0] / 255.0) < self.filter_rate

    def _complete(self, messages: list[dict[str, str]], params: GenParams,
                  stage: str = "misc") -> LLMResult:
        payload = payload_of(messages)
        if self._should_filter(payload):
            return LLMResult(text="", finish_reason="content_filter", filtered=True)

        handler = {
            "index.ner": self._ner,
            "retrieve.ner": self._ner,
            "index.openie": self._openie,
            "witness.acquire": self._openie,
            "retrieve.filter": self._filter_triples,
            "witness.compile": self._compile_query,
            "graphrag.community": self._community,
            "qa": self._answer,
        }.get(stage, self._echo)

        text = handler(payload, messages)
        return LLMResult(
            text=text,
            prompt_tokens=max(1, len(payload) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_s=0.0,
            finish_reason="stop",
        )

    # -- tarefas -------------------------------------------------------------

    def _ner(self, payload: str, _messages: Any) -> str:
        return json.dumps({"named_entities": _entities(payload)}, ensure_ascii=False)

    def _openie(self, payload: str, _messages: Any) -> str:
        return json.dumps({"triples": _triples(payload)}, ensure_ascii=False)

    def _filter_triples(self, payload: str, _messages: Any) -> str:
        """Recognition memory de brinquedo: mantém as triplas que compartilham
        alguma palavra de conteúdo com a pergunta."""
        question = ""
        if "PERGUNTA:" in payload:
            question = payload.split("PERGUNTA:", 1)[1].split("\n", 1)[0]
        qwords = {w.lower() for w in re.findall(r"\w+", question) if w.lower() not in _STOPWORDS}
        kept: list[list[str]] = []
        for line in payload.splitlines():
            match = re.match(r"\s*\d+\.\s*\((.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\)\s*$", line)
            if not match:
                continue
            triple = [match.group(1), match.group(2), match.group(3)]
            words = {w.lower() for w in re.findall(r"\w+", " ".join(triple))}
            if words & qwords:
                kept.append(triple)
        return json.dumps({"fact": kept[:5]}, ensure_ascii=False)

    def _compile_query(self, payload: str, _messages: Any) -> str:
        """Compilação de brinquedo: um átomo por entidade encontrada, ligados
        por uma variável intermediária quando há duas ou mais."""
        question = payload.split("\n", 1)[0]
        ents = _entities(question, limit=3)
        content = [w for w in re.findall(r"[a-zA-Z]+", question) if w.lower() not in _STOPWORDS]
        relation = " ".join(content[:3]).lower() or "related to"
        if not ents:
            return json.dumps({"answer_var": "x", "atoms": [], "fallback": question}, ensure_ascii=False)
        atoms = [{"relation": relation, "subject": ents[0], "object": "?m"}]
        atoms.append({"relation": relation, "subject": "?m", "object": "?x"})
        return json.dumps({"answer_var": "x", "atoms": atoms, "expected_type": "entity",
                           "fallback": question}, ensure_ascii=False)

    def _community(self, payload: str, _messages: Any) -> str:
        ents = _entities(payload, limit=8)
        return json.dumps({
            "title": " / ".join(ents[:3]) or "comunidade",
            "summary": "Comunidade envolvendo " + ", ".join(ents[:8]) + ".",
        }, ensure_ascii=False)

    def _answer(self, payload: str, _messages: Any) -> str:
        """Devolve a entidade mais frequente do contexto que não está na pergunta."""
        question = ""
        if "PERGUNTA:" in payload:
            question = payload.rsplit("PERGUNTA:", 1)[1]
        qents = {e.lower() for e in _entities(question)}
        counts: dict[str, int] = {}
        for ent in _entities(payload, limit=400):
            if ent.lower() in qents:
                continue
            counts[ent] = counts.get(ent, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"
        return json.dumps({"answer": best}, ensure_ascii=False)

    def _echo(self, payload: str, _messages: Any) -> str:
        return json.dumps({"text": payload[:200]}, ensure_ascii=False)
