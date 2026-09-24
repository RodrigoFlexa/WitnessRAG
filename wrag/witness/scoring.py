"""
Uma pontuação para toda a busca (desenho v3, Seção 4.1).

    pontuação(x) = w_sim · sim(x) + w_imp · imp(x) + w_tempo · prox(x)

* sim: para um trecho, a fusão RRF da busca densa com a lexical, normalizada
  para [0, 1] sobre a memória; para um fato, o grau de casamento com o passo do
  plano (relação e nomes), calculado pelo aterramento.
* imp: a importância gravada no Registrar (dated_memory.py).
* prox: proximidade da data do item ao período de referência do plano,
  exp(-dias/λ). Recência é o caso "período = hoje"; "primeira vez" é o caso
  "período = início"; uma data citada é o caso "período = janela".

Os pesos vêm do plano, em níveis (none, normal, strong), e são sempre
renormalizados para somar 1 com a similaridade com pelo menos metade do peso.
Com todo o peso na similaridade a pontuação é exatamente a ordem do híbrido.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np

from wrag.witness.dated_memory import DatedMemory
from wrag.witness.timeline import Interval, format_interval, interval_distance, parse_anchor

LEVELS = ("none", "normal", "strong")
PERIOD_KINDS = ("now", "start", "window")


@dataclass(frozen=True)
class Weights:
    similarity: float = 1.0
    importance: float = 0.0
    time: float = 0.0

    def __post_init__(self) -> None:
        values = (self.similarity, self.importance, self.time)
        if any(v < -1e-9 for v in values) or abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"pesos devem ser >= 0 e somar 1: {values}")
        if self.similarity < 0.5 - 1e-9:
            raise ValueError("a similaridade deve ter pelo menos metade do peso")

    @property
    def pure_similarity(self) -> bool:
        return self.importance <= 1e-12 and self.time <= 1e-12

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class WeightLevels:
    """Quanto cada nível acrescenta, antes da renormalização.

    Calibrados na busca de trechos do LoCoMo (scripts/proof-offline-eval.py,
    memórias congeladas, sem LLM): com o período "hoje", qualquer peso de tempo
    ou de importância mudou de 12% a 72% dos contextos sem ganho de revocação,
    então o nível "normal" deixa todo o peso na similaridade. Com uma janela
    de tempo citada na pergunta, peso 0,3 no tempo elevou a revocação em todas
    as categorias; é o nível "strong". A importância só ganha peso quando o
    plano pede ("strong"), e pouco. "none" é sempre zero.
    """

    time_normal: float = 0.0
    time_strong: float = 0.3
    importance_normal: float = 0.0
    importance_strong: float = 0.1

    def weights(self, time: str = "normal", importance: str = "normal") -> Weights:
        time = time if time in LEVELS else "normal"
        importance = importance if importance in LEVELS else "normal"
        w_time = {"none": 0.0, "normal": self.time_normal, "strong": self.time_strong}[time]
        w_imp = {"none": 0.0, "normal": self.importance_normal,
                 "strong": self.importance_strong}[importance]
        extra = w_time + w_imp
        if extra > 0.5:
            w_time, w_imp = 0.5 * w_time / extra, 0.5 * w_imp / extra
        return Weights(1.0 - w_time - w_imp, w_imp, w_time)


@dataclass(frozen=True)
class Period:
    """Período de referência do plano: hoje, o início da memória ou uma janela."""

    kind: str = "now"
    interval: Interval | None = None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tipo": self.kind, "texto": self.text,
                "intervalo": format_interval(self.interval)}


def resolve_period(kind: str, text: str, memory: DatedMemory | None,
                   question_time: date | None = None) -> tuple[Period, list[str]]:
    """Transforma a escolha do planejador em um intervalo concreto.

    ``question_time`` é o instante da pergunta; na ausência dele, a última data
    da memória (o presente de um agente conversacional). Uma janela que não se
    consegue interpretar volta para "hoje", com o reparo registrado.
    """
    repairs: list[str] = []
    now = question_time or (memory.last if memory is not None else None)
    kind = (kind or "now").strip().lower()
    if kind not in PERIOD_KINDS:
        repairs.append(f"periodo_desconhecido:{kind}")
        kind = "now"
    if kind == "window":
        interval = parse_anchor(text, now=now)
        if interval is None:
            repairs.append("janela_nao_interpretada")
            kind = "now"
        else:
            return Period("window", interval, text), repairs
    if kind == "start":
        first = memory.first if memory is not None else None
        if first is None:
            repairs.append("memoria_sem_datas")
            return Period("now", None, ""), repairs
        return Period("start", Interval(first, first, "start"), "start"), repairs
    if now is None:
        return Period("now", None, ""), repairs
    return Period("now", Interval(now, now, "now"), "now"), repairs


def proximity_scale(period: Period, memory: DatedMemory | None, min_days: float = 7.0,
                    point_fraction: float = 0.25) -> float:
    """λ: metade da janela citada (mínimo de uma semana); para hoje ou início,
    uma fração da duração da memória."""
    if period.interval is None:
        return max(1.0, min_days)
    if period.kind == "window":
        return max(min_days, period.interval.width_days / 2.0)
    span = memory.span_days if memory is not None else 0
    return max(min_days, point_fraction * span)


def proximity(item: Interval | None, period: Period, scale: float) -> float:
    if item is None or period.interval is None:
        return 0.0
    return math.exp(-interval_distance(item, period.interval) / max(1e-6, scale))


class MemoryScorer:
    """Aplica a pontuação a trechos e prepara o termo fixo dos fatos."""

    def __init__(self, memory: DatedMemory, min_days: float = 7.0,
                 point_fraction: float = 0.25) -> None:
        self.memory = memory
        self.min_days = min_days
        self.point_fraction = point_fraction

    def scale(self, period: Period) -> float:
        return proximity_scale(period, self.memory, self.min_days, self.point_fraction)

    def passage_scores(self, fused: Mapping[str, float], pids: Sequence[str],
                       weights: Weights, period: Period) -> dict[str, float]:
        """Pontuação de cada trecho de ``pids`` sob o plano.

        ``fused`` são os escores RRF do híbrido; trechos ausentes valem zero
        antes da normalização, que é feita sobre ``pids`` (a memória inteira).
        """
        raw = [float(fused.get(pid, 0.0)) for pid in pids]
        low, high = (min(raw), max(raw)) if raw else (0.0, 0.0)
        span = high - low
        scale = self.scale(period)
        scores = {}
        for pid, value in zip(pids, raw):
            sim = (value - low) / span if span > 1e-12 else 0.0
            imp = self.memory.passage_importance.get(pid, 0.0)
            prox = proximity(self.memory.passage_interval.get(pid), period, scale)
            scores[pid] = (weights.similarity * sim + weights.importance * imp
                           + weights.time * prox)
        return scores

    def rank(self, fused: Mapping[str, float], fused_order: Sequence[str],
             pids: Sequence[str], weights: Weights, period: Period) -> list[str]:
        """Ordena ``pids`` pela pontuação; empates seguem a ordem do híbrido.

        Com todo o peso na similaridade, devolve exatamente a ordem do híbrido.
        """
        if weights.pure_similarity:
            wanted = set(pids)
            ordered = [pid for pid in fused_order if pid in wanted]
            seen = set(ordered)
            return ordered + [pid for pid in pids if pid not in seen]
        scores = self.passage_scores(fused, pids, weights, period)
        position = {pid: i for i, pid in enumerate(fused_order)}
        return sorted(pids, key=lambda pid: (-scores[pid], position.get(pid, len(position)),
                                             pid))

    def fact_prior(self, weights: Weights, period: Period, n_facts: int) -> np.ndarray:
        """Parte da pontuação de um fato que não depende do passo do plano:
        w_imp · imp + w_tempo · prox. A busca soma w_sim · casamento."""
        prior = np.zeros(n_facts, dtype=np.float32)
        if weights.pure_similarity:
            return prior
        scale = self.scale(period)
        limit = min(n_facts, len(self.memory.fact_interval))
        for index in range(limit):
            prior[index] = (weights.importance * float(self.memory.fact_importance[index])
                            + weights.time * proximity(self.memory.fact_interval[index],
                                                       period, scale))
        return prior
