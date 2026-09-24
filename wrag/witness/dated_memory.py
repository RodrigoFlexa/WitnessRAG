"""
Memória datada: a etapa REGISTRAR do desenho v3, sem LLM.

A extração de fatos (OpenIE de diálogo, feita uma vez por corpus) produz
triplas com um campo de tempo livre. Este módulo completa a memória com o que
o desenho exige de TODO item:

* um intervalo de datas: para um trecho, as datas das sessões que ele cobre;
  para um fato, a data em que o evento aconteceu, resolvida a partir da data da
  sessão em que foi dito ("last week" dito em 27/6/2023 é 19-25/6/2023). Um
  fato sem expressão de tempo recebe a data da sessão em que foi relatado;
* uma importância em [0, 1): a intensidade emocional da fala de origem, um
  proxy intrínseco da afetividade (EMAS 2026), calculado uma vez e independente
  da pergunta;
* a fala de origem de cada fato, usada pela verificação para mostrar o texto
  original (com falante e vizinhança) em vez da tripla resumida.

Tudo é determinístico. Em corpora sem datas as funções devolvem None e a
proximidade temporal vira neutra.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

import numpy as np

from wrag.util import canonical_symbol, normalize
from wrag.witness.timeline import (Interval, format_interval, parse_date,
                                   resolve_expression)

# Léxico compacto de ativação emocional (arousal). Pesos 1.0 = emoção comum,
# 1.5 = emoção intensa. Pequeno e auditável; um classificador treinado é
# trabalho futuro.
_AROUSAL = {w: 1.0 for w in (
    "love loved loving happy glad joy joyful excited exciting fun proud pride grateful "
    "thankful blessed hope hopeful inspired inspiring passion passionate fulfilling "
    "meaningful special amazing awesome wonderful beautiful sad upset angry mad "
    "afraid scared fear worried worry anxious nervous stressed stress lonely hurt "
    "tough hard difficult struggle struggling miss missed cry cried tears hug "
    "relieved relief calm peace peaceful comfort comforting support supportive "
    "motivated motivation dream dreams feel feeling felt feelings emotional "
    "overwhelmed frustrated disappointed embarrassed nostalgic grief grieving "
    "cherish cherished treasure healing heal therapy therapeutic empowering "
    "empowered accepted acceptance belonging bond bonding powerful touching "
    "moving heartwarming".split())}
_AROUSAL.update({w: 1.5 for w in (
    "thrilled ecstatic devastated heartbroken terrified furious elated overjoyed "
    "incredible unbelievable life-changing traumatic trauma panic euphoric "
    "adore adored hate hated shocked".split())})
_INTENSIFIERS = frozenset("so very really truly totally super incredibly extremely".split())

_TURN = re.compile(r"^\[([^\]\s]+)(?:\s+date=([^\]]+))?\]\s+([^:\]]{1,60}):\s*(.*)$")
_SESSION = re.compile(r"^Session date:\s*(.+)$")
_STOP = frozenset(
    "the a an is was were did does do to of in on at for and or what which who when "
    "where how many much has have had their his her he she they it its i my me you "
    "your be been this that with about as from by would could should".split())


def arousal(text: str, kappa: float = 2.0) -> float:
    """Excitação emocional em [0, 1): 1 - exp(-E/κ)."""
    words = re.findall(r"[a-z][a-z\-']*", (text or "").lower())
    energy = sum(_AROUSAL.get(word, 0.0) for word in words)
    energy += 0.3 * sum(1 for word in words if word in _INTENSIFIERS)
    energy += 0.5 * min(3, (text or "").count("!"))
    return 1.0 - math.exp(-energy / max(1e-6, kappa))


@dataclass(frozen=True)
class Turn:
    """Uma fala de um trecho, com a data da sessão em que foi dita."""

    turn_id: str
    speaker: str
    text: str
    when: date | None = None
    line: int = -1


def split_turns(text: str, session_time: str = "") -> list[Turn]:
    """Falas de um trecho conversacional; frases, em corpus sem falas."""
    turns: list[Turn] = []
    current = parse_date(session_time)
    for number, line in enumerate((text or "").splitlines()):
        session = _SESSION.match(line)
        if session:
            current = parse_date(session.group(1)) or current
            continue
        match = _TURN.match(line.strip())
        if match:
            when = parse_date(match.group(2) or "") or current
            turns.append(Turn(match.group(1), match.group(3).strip(),
                              match.group(4).strip(), when, number))
    if turns:
        return turns
    return [Turn("", "", sentence.strip(), current, -1)
            for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text))
            if len(w) > 2 and w not in _STOP}


class DatedMemory:
    """Datas, importância e fala de origem de cada trecho e de cada fato."""

    def __init__(self, corpus: Any, facts: Sequence[Any]) -> None:
        self.turns: dict[str, list[Turn]] = {}
        self.passage_interval: dict[str, Interval | None] = {}
        self.passage_importance: dict[str, float] = {}
        dates: list[date] = []
        for passage in corpus.passages:
            turns = split_turns(passage.text, getattr(passage, "session_time", ""))
            self.turns[passage.pid] = turns
            when = sorted({t.when for t in turns if t.when is not None})
            self.passage_interval[passage.pid] = (Interval(when[0], when[-1])
                                                  if when else None)
            dates.extend(when)
            values = [arousal(t.text) for t in turns if t.speaker]
            self.passage_importance[passage.pid] = float(np.mean(values)) if values else 0.0
        self.first = min(dates) if dates else None
        self.last = max(dates) if dates else None
        # Importância dos trechos normalizada na própria memória: a média de
        # ativação varia pouco entre trechos longos, e sem normalizar o sinal
        # não teria contraste.
        if self.passage_importance:
            low, high = (min(self.passage_importance.values()),
                         max(self.passage_importance.values()))
            span = high - low
            self.passage_importance = {
                pid: ((value - low) / span if span > 1e-9 else 0.0)
                for pid, value in self.passage_importance.items()}

        n = len(facts)
        self.fact_interval: list[Interval | None] = [None] * n
        self.fact_importance = np.zeros(n, dtype=np.float32)
        self.fact_turn: list[tuple[str, int]] = [("", -1)] * n
        self.fact_time_source: list[str] = [""] * n
        self._turn_terms: dict[str, list[set[str]]] = {
            pid: [_terms(f"{t.speaker} {t.text}") for t in turns]
            for pid, turns in self.turns.items()}
        for index, fact in enumerate(facts):
            self._register(index, fact)

    # -- registro de um fato -------------------------------------------------

    def _best_turn(self, fact: Any) -> int:
        """A fala que mais provavelmente originou o fato: sobreposição de termos
        do objeto (peso 2), do sujeito e da relação. Desempate pela primeira."""
        options = self._turn_terms.get(fact.pid) or []
        if not options:
            return -1
        obj, subj, rel = _terms(fact.object), _terms(fact.subject), _terms(fact.relation)
        turns = self.turns.get(fact.pid) or []
        best, best_score = -1, 0.0
        for index, terms in enumerate(options):
            score = 2.0 * len(obj & terms) + len(subj & terms) + 0.5 * len(rel & terms)
            speaker = canonical_symbol(turns[index].speaker) if index < len(turns) else ""
            if speaker and speaker == canonical_symbol(fact.subject):
                score += 0.5
            if score > best_score:
                best, best_score = index, score
        return best

    def _register(self, index: int, fact: Any) -> None:
        turn_index = self._best_turn(fact)
        turns = self.turns.get(fact.pid) or []
        turn = turns[turn_index] if 0 <= turn_index < len(turns) else None
        self.fact_turn[index] = (fact.pid, turn_index)
        reference = turn.when if turn else None
        if reference is None:
            interval = self.passage_interval.get(fact.pid)
            reference = interval.start if interval else None
        resolved = resolve_expression(getattr(fact, "time", "") or "", reference)
        if resolved is not None:
            self.fact_time_source[index] = "expressao"
        elif reference is not None:
            resolved = Interval(reference, reference, "session")
            self.fact_time_source[index] = "sessao"
        self.fact_interval[index] = resolved
        self.fact_importance[index] = arousal(turn.text) if turn else 0.0

    def extend(self, facts: Sequence[Any], start: int) -> None:
        """Registra fatos acrescentados depois da construção (aquisição)."""
        need = start + len(facts)
        if need > len(self.fact_interval):
            extra = need - len(self.fact_interval)
            self.fact_interval.extend([None] * extra)
            self.fact_turn.extend([("", -1)] * extra)
            self.fact_time_source.extend([""] * extra)
            self.fact_importance = np.concatenate(
                [self.fact_importance, np.zeros(extra, dtype=np.float32)])
        for offset, fact in enumerate(facts):
            self._register(start + offset, fact)

    # -- leitura ---------------------------------------------------------------

    @property
    def dated(self) -> bool:
        return self.first is not None and self.last is not None

    @property
    def span_days(self) -> int:
        if not self.dated:
            return 0
        return max(1, (self.last - self.first).days + 1)

    def fact_time_text(self, index: int) -> str:
        if not 0 <= index < len(self.fact_interval):
            return ""
        return format_interval(self.fact_interval[index])

    def excerpt(self, index: int, window: int = 1, max_chars: int = 900) -> str:
        """A fala de origem de um fato, com as vizinhas e a data da sessão."""
        if not 0 <= index < len(self.fact_turn):
            return ""
        pid, turn_index = self.fact_turn[index]
        turns = self.turns.get(pid) or []
        if not 0 <= turn_index < len(turns):
            return ""
        lo, hi = max(0, turn_index - window), min(len(turns), turn_index + window + 1)
        when = turns[turn_index].when
        header = f"(session date: {format_interval(Interval(when, when))})" if when else ""
        lines = [f"[{t.turn_id}] {t.speaker}: {t.text}" for t in turns[lo:hi]]
        text = "\n".join(([header] if header else []) + lines)
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def stats(self) -> dict[str, Any]:
        sources = {"expressao": 0, "sessao": 0, "": 0}
        for value in self.fact_time_source:
            sources[value] = sources.get(value, 0) + 1
        return {"t_inicio": self.first.isoformat() if self.first else "",
                "t_ultimo": self.last.isoformat() if self.last else "",
                "trechos": len(self.passage_interval),
                "fatos": len(self.fact_interval),
                "fatos_tempo_da_expressao": sources["expressao"],
                "fatos_tempo_da_sessao": sources["sessao"],
                "fatos_sem_tempo": sources[""],
                "importancia_media_fatos": round(float(self.fact_importance.mean())
                                                 if len(self.fact_importance) else 0.0, 4)}
