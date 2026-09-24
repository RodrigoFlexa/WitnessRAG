"""
Lentes de memória: sinais que o PLANO escolhe para reordenar evidências.

Inspiração: o artigo "Bridging Reflective and Semantic Memory" (EMAS 2026)
pontua cada memória por ``Score = R · [β·S + (1-β)·A]``: relevância como
filtro multiplicativo, estabilidade temporal S (Ebbinghaus) e afetividade A.
Aqui a mesma estrutura vira um conjunto de LENTES que o planejador liga ou não
por pergunta, em vez de pesos fixos para todas as perguntas:

    s(p | q, π) = R̃(p, q) · (1 + β · Σ_ℓ 1[ℓ ∈ π] · Φ_ℓ(p | q, π))

* R̃: relevância híbrida (RRF denso+BM25) normalizada no pool de candidatos;
  é multiplicativa: uma memória irrelevante nunca sobe por ser recente ou
  emocional.
* Φ_T: lente temporal, sobre o relógio de referência [t₀, t_now]:
    recent : retenção de Ebbinghaus exp(-(t_now - τ)/s), s = s₀ + α·(n - 1),
             n = número de passagens que reafirmam o mesmo (sujeito, relação);
    early  : exp(-(τ_min - t₀)/s₀);
    anchor : exp(-d(τ, I)/w), I o intervalo da âncora da pergunta.
* Φ_A: saliência afetiva: excitação emocional das falas da entidade de foco
  que tocam os termos da pergunta. Proxy intrínseco da afetividade: o sinal de
  utilidade por reflexão do artigo EMAS exige feedback, que não existe no
  protocolo de teste (ver docs).
* Φ_C: confiança corroborada: 1-(1-c)^m de fatos do grafo sobre as entidades
  de foco, vezes a similaridade do fato com as necessidades de informação.

Integração conservadora: as lentes só competem pela cauda do contexto
(``max_swaps`` posições), nunca removem passagens de uma testemunha entregue e
só trocam quando o ganho vem de uma lente (e não de relevância pura, cuja
ordem já é a do híbrido). Sem lente ativa a função é a identidade.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

import numpy as np

from wrag.util import canonical_symbol, normalize
from wrag.witness.timeline import (Interval, ReferenceClock, parse_anchor, parse_date,
                                   passage_dates)

# Léxico compacto de excitação emocional (arousal). Pesos 1.0 = emoção comum,
# 1.5 = emoção intensa. É deliberadamente pequeno e auditável; o artigo discute
# a troca por um classificador treinado como trabalho futuro.
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
_STOP = frozenset(
    "the a an is was were did does do to of in on at for and or what which who when "
    "where how many much has have had their his her he she they it its i my me you "
    "your be been this that with about as from by would could should".split())

_TURN = re.compile(r"^\[([^\]\s]+)(?:\s+date=([^\]]+))?\]\s+([^:\]]{1,60}):\s*(.*)$")
_SESSION = re.compile(r"^Session date:\s*(.+)$")


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text))
            if len(w) > 2 and w not in _STOP}


def arousal(text: str, kappa: float = 2.0) -> float:
    """Excitação emocional em [0, 1): 1 - exp(-E/κ)."""
    words = re.findall(r"[a-z][a-z\-']*", (text or "").lower())
    energy = sum(_AROUSAL.get(word, 0.0) for word in words)
    energy += 0.3 * sum(1 for word in words if word in _INTENSIFIERS)
    energy += 0.5 * min(3, (text or "").count("!"))
    return 1.0 - math.exp(-energy / max(1e-6, kappa))


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    when: date | None = None


def split_turns(text: str, session_time: str = "") -> list[Turn]:
    """Falas de uma passagem conversacional; frases, em corpus sem falas."""
    turns: list[Turn] = []
    current = parse_date(session_time)
    for line in (text or "").splitlines():
        session = _SESSION.match(line)
        if session:
            current = parse_date(session.group(1)) or current
            continue
        match = _TURN.match(line.strip())
        if match:
            when = parse_date(match.group(2) or "") or current
            turns.append(Turn(match.group(3).strip(), match.group(4).strip(), when))
    if turns:
        return turns
    return [Turn("", sentence.strip(), current)
            for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]


@dataclass
class LensPolicy:
    """Lentes escolhidas pelo plano, depois de validadas contra a memória."""

    temporal: str = "none"            # none | recent | early | anchor
    anchor: Interval | None = None
    salience: bool = False
    confidence: bool = False
    repairs: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.temporal != "none" or self.salience or self.confidence

    @property
    def names(self) -> list[str]:
        out = [f"temporal:{self.temporal}"] if self.temporal != "none" else []
        return out + (["salience"] if self.salience else []) + \
            (["confidence"] if self.confidence else [])

    def to_dict(self) -> dict[str, Any]:
        return {"temporal": self.temporal,
                "anchor": self.anchor.to_dict() if self.anchor else None,
                "salience": self.salience, "confidence": self.confidence,
                "repairs": list(self.repairs)}


def policy_from_contract(contract, clock: ReferenceClock) -> LensPolicy:
    policy = LensPolicy(temporal=contract.temporal_lens, salience=contract.salience_lens,
                        confidence=contract.confidence_lens)
    if policy.temporal != "none" and not clock.available:
        policy.repairs.append("temporal:memoria_sem_relogio->none")
        policy.temporal = "none"
    if policy.temporal == "anchor":
        policy.anchor = parse_anchor(contract.time_anchor, clock.now)
        if policy.anchor is None:
            policy.repairs.append("temporal:ancora_nao_interpretavel->none")
            policy.temporal = "none"
    return policy


class MemorySignals:
    """Sinais offline da memória (etapa de memorização, sem LLM).

    Para cada passagem: datas de referência e falas. Para o grafo: quantas
    passagens reafirmam cada (sujeito canônico, relação), o ensaio, que dá a
    estabilidade; e quantas extrações sustentam cada tripla canônica, a
    corroboração, que dá a confiança.
    """

    def __init__(self, corpus, memory=None, *, stability_base_days: float = 30.0,
                 stability_gain_days: float = 30.0, arousal_kappa: float = 2.0) -> None:
        self.corpus = corpus
        self.memory = memory
        self.s0 = float(stability_base_days)
        self.alpha = float(stability_gain_days)
        self.kappa = float(arousal_kappa)
        self.dates: dict[str, list[date]] = {}
        self.turns: dict[str, list[Turn]] = {}
        for passage in corpus.passages:
            self.dates[passage.pid] = passage_dates(passage.text, passage.session_time)
            self.turns[passage.pid] = split_turns(passage.text, passage.session_time)
        self.clock = ReferenceClock.from_dates(d for values in self.dates.values() for d in values)

        self.facts_by_pid: dict[str, list[int]] = defaultdict(list)
        self.rehearsal: dict[tuple[int, int], int] = {}
        self.corroboration: Counter = Counter()
        self.fact_dates: dict[int, date] = {}
        if memory is not None:
            pids_by_key: dict[tuple[int, int], set[str]] = defaultdict(set)
            for index, fact in enumerate(memory.facts):
                self.facts_by_pid[fact.pid].append(index)
                subject = self._cluster(fact.subj_id, fact.subject)
                pids_by_key[(subject, fact.rel_id)].add(fact.pid)
                self.corroboration[self._triple_key(fact)] += 1
                when = parse_date(fact.time or "")
                if when:
                    self.fact_dates[index] = when
            self.rehearsal = {key: len(pids) for key, pids in pids_by_key.items()}

    # -- identidade ----------------------------------------------------------

    def _cluster(self, eid: int, surface: str) -> int:
        if self.memory is not None and eid is not None and eid >= 0:
            return self.memory.cluster(eid)
        return -1 - (zlib.crc32(canonical_symbol(surface).encode("utf-8")) & 0xFFFFFF)

    def _triple_key(self, fact) -> tuple[int, int, str]:
        return (self._cluster(fact.subj_id, fact.subject), fact.rel_id,
                canonical_symbol(fact.object))

    def focus(self, entities: Sequence[str]) -> tuple[set[int], set[str]]:
        clusters: set[int] = set()
        names = {canonical_symbol(e) for e in entities if e.strip()}
        if self.memory is not None:
            for entity in entities:
                eid = self.memory.entity_id(entity)
                if eid is not None and eid >= 0:
                    clusters.add(self.memory.cluster(eid))
        return clusters, names

    def _focus_facts(self, pid: str, clusters: set[int], names: set[str]) -> list[int]:
        if self.memory is None or (not clusters and not names):
            return []
        out = []
        for index in self.facts_by_pid.get(pid, []):
            fact = self.memory.facts[index]
            ends = {self._cluster(fact.subj_id, fact.subject), self._cluster(fact.obj_id, fact.object)}
            surfaces = {canonical_symbol(fact.subject), canonical_symbol(fact.object)}
            if ends & clusters or any(name and (name in s or s in name)
                                      for name in names for s in surfaces if s):
                out.append(index)
        return out

    # -- lentes --------------------------------------------------------------

    def temporal(self, pid: str, policy: LensPolicy, clusters: set[int],
                 names: set[str]) -> float:
        if policy.temporal == "none" or not self.clock.available:
            return 0.0
        focus_facts = self._focus_facts(pid, clusters, names)
        times = list(self.dates.get(pid, []))
        times += [self.fact_dates[i] for i in focus_facts if i in self.fact_dates]
        if not times:
            return 0.0
        if policy.temporal == "recent":
            rehearsal = max((self.rehearsal.get((self._cluster(self.memory.facts[i].subj_id,
                                                               self.memory.facts[i].subject),
                                                 self.memory.facts[i].rel_id), 1)
                             for i in focus_facts), default=1)
            stability = self.s0 + self.alpha * (rehearsal - 1)
            return max(math.exp(-max(0, (self.clock.now - t).days) / stability) for t in times)
        if policy.temporal == "early":
            return math.exp(-max(0, (min(times) - self.clock.start).days) / self.s0)
        if policy.temporal == "anchor" and policy.anchor is not None:
            width = max(7.0, policy.anchor.width_days / 2.0)
            return max(math.exp(-policy.anchor.distance_days(t) / width) for t in times)
        return 0.0

    def salience(self, pid: str, names: set[str], question_terms: set[str]) -> float:
        best = 0.0
        for turn in self.turns.get(pid, []):
            speaker = canonical_symbol(turn.speaker)
            text = canonical_symbol(turn.text)
            if names:
                about = any(name and (name == speaker or name in text) for name in names)
                if not about:
                    continue
            overlap = len(_terms(turn.text) & question_terms)
            relevance = 0.5 + 0.5 * min(1.0, overlap / 2.0)
            best = max(best, arousal(turn.text, self.kappa) * relevance)
        return best

    def confidence(self, pid: str, clusters: set[int], names: set[str],
                   need: np.ndarray | None) -> float:
        if self.memory is None or need is None or not self.memory.fact_vectors.size:
            return 0.0
        best = 0.0
        norm_need = float(np.linalg.norm(need)) or 1.0
        for index in self._focus_facts(pid, clusters, names):
            if index >= self.memory.fact_vectors.shape[0]:
                continue
            vector = self.memory.fact_vectors[index]
            similarity = float(np.dot(vector, need)) / (
                (float(np.linalg.norm(vector)) or 1.0) * norm_need)
            fact = self.memory.facts[index]
            base = min(0.999, max(0.0, float(fact.confidence or 0.9)))
            corroborated = 1.0 - (1.0 - base) ** max(1, self.corroboration[self._triple_key(fact)])
            best = max(best, corroborated * max(0.0, similarity))
        return best

    def stats(self) -> dict[str, Any]:
        return {"relogio": self.clock.to_dict(),
                "passagens_datadas": sum(1 for v in self.dates.values() if v),
                "fatos_datados": len(self.fact_dates),
                "chaves_ensaio": len(self.rehearsal)}


def lens_values(signals: MemorySignals, pids: Sequence[str], policy: LensPolicy,
                focus_entities: Sequence[str], question: str,
                need: np.ndarray | None) -> dict[str, dict[str, float]]:
    clusters, names = signals.focus(focus_entities)
    question_terms = _terms(question)
    out: dict[str, dict[str, float]] = {}
    for pid in dict.fromkeys(pids):
        values: dict[str, float] = {}
        if policy.temporal != "none":
            values["temporal"] = signals.temporal(pid, policy, clusters, names)
        if policy.salience:
            values["salience"] = signals.salience(pid, names, question_terms)
        if policy.confidence:
            values["confidence"] = signals.confidence(pid, clusters, names, need)
        out[pid] = values
    return out


def rerank_with_lenses(current: Sequence[str], pool: Sequence[str], pool_scores: Sequence[float],
                       k: int, lenses: dict[str, dict[str, float]], *, weight: float = 1.0,
                       max_swaps: int = 1, margin: float = 0.10,
                       protected: set[str] | None = None) -> tuple[list[str], dict[str, Any]]:
    """Troca no máximo ``max_swaps`` posições da cauda por evidência de lente.

    Uma troca exige (i) que o candidato tenha ganho de lente maior que o da
    passagem removida e (ii) que o score composto supere o dela por ``margin``.
    Passagens protegidas (prefixo e testemunhas) nunca saem.
    """
    out = list(dict.fromkeys(current))[:k]
    diagnostics: dict[str, Any] = {"trocas": [], "max_trocas": max_swaps}
    if not out or max_swaps <= 0 or not lenses:
        return out, diagnostics
    raw = {pid: float(score) for pid, score in zip(pool, pool_scores)}
    if raw:
        low, high = min(raw.values()), max(raw.values())
        relevance = {pid: (1.0 if high <= low else (value - low) / (high - low))
                     for pid, value in raw.items()}
    else:
        relevance = {}

    def gain(pid: str) -> float:
        return sum(lenses.get(pid, {}).values())

    def score(pid: str) -> float:
        return relevance.get(pid, 0.0) * (1.0 + weight * gain(pid))

    locked = set(out[:max(1, k - max_swaps)]) | (set(protected or ()) & set(out))
    for _ in range(max_swaps):
        tail = [pid for pid in out if pid not in locked]
        challengers = [pid for pid in pool if pid not in out]
        if not tail or not challengers:
            break
        weakest = min(tail, key=lambda pid: (score(pid), -out.index(pid)))
        best = max(challengers, key=lambda pid: (score(pid), -list(pool).index(pid)))
        if gain(best) <= gain(weakest) or score(best) <= (1.0 + margin) * score(weakest):
            break
        position = out.index(weakest)
        out[position] = best
        locked.add(best)
        diagnostics["trocas"].append({
            "entrou": best, "saiu": weakest, "posicao": position,
            "score_entrou": round(score(best), 4), "score_saiu": round(score(weakest), 4),
            "lentes_entrou": {k_: round(v, 4) for k_, v in lenses.get(best, {}).items()},
            "lentes_saiu": {k_: round(v, 4) for k_, v in lenses.get(weakest, {}).items()},
        })
    diagnostics["alterou"] = bool(diagnostics["trocas"])
    return out, diagnostics
