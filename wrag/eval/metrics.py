"""
Métricas de recuperação, resposta e seletividade.

Recall mede apoio recuperado e all-recall exige todas as passagens anotadas.
Cobertura de triplas compara relações dirigidas lexicalmente; não valida a
semântica da extração. A curva de seletividade usa scores não calibrados,
com empates aceitos em bloco, contra o acerto da resposta estrutural.
"""

from __future__ import annotations

from collections import Counter
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from wrag.util import normalize, normalize_answer, canonical_symbol


# ---------------------------------------------------------------------------
# Recuperação
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def all_recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return float("nan")
    return float(set(gold) <= set(retrieved[:k]))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(retrieved[:k])


# ---------------------------------------------------------------------------
# Resposta
# ---------------------------------------------------------------------------

def exact_match(prediction: str, answers: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: Sequence[str]) -> float:
    """F1 por token, no protocolo do MuSiQue/SQuAD, máximo sobre os aliases."""
    best = 0.0
    p_tokens = normalize_answer(prediction).split()
    for answer in answers:
        g_tokens = normalize_answer(answer).split()
        if not p_tokens or not g_tokens:
            best = max(best, float(p_tokens == g_tokens))
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(p_tokens)
        recall = overlap / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def is_abstention(prediction: str) -> bool:
    text = normalize_answer(prediction)
    return text in ("", "insufficient information", "unknown", "informacao insuficiente")


# ---------------------------------------------------------------------------
# Testemunhas
# ---------------------------------------------------------------------------

def witness_coverage(witness_facts: Sequence[Sequence[str]],
                     gold_evidences: Sequence[tuple[str, str, str]]) -> float:
    """Cobertura lexical de triplas dirigidas. Não é validação semântica do texto.

    Vocabulários diferentes exigem um mapeamento independente, definido antes da
    avaliação. Nunca ignorar o predicado para declarar uma prova correta.
    """
    if not gold_evidences:
        return float("nan")
    extracted = {tuple(canonical_symbol(x) for x in f) for f in witness_facts if len(f) == 3}
    gold = {tuple(canonical_symbol(x) for x in f) for f in gold_evidences}
    return len(extracted & gold) / len(gold)


def endpoint_coverage(witness_facts: Sequence[Sequence[str]],
                      gold_evidences: Sequence[tuple[str, str, str]]) -> float:
    """Fração das triplas de evidência anotadas cobertas pela testemunha.

    O casamento é frouxo de propósito: uma tripla extraída conta como cobrindo
    uma anotada quando sujeito e objeto batem (normalizados) — a relação fica de
    fora porque a anotação do 2Wiki usa vocabulário do Wikidata ("date of death")
    e a extração aberta usa vocabulário do texto ("died on"). Exigir a relação
    literal mediria a distância entre dois vocabulários, não a qualidade da prova.
    """
    if not gold_evidences:
        return float("nan")
    extracted = {(normalize(f[0]), normalize(f[2])) for f in witness_facts if len(f) >= 3}
    extracted |= {(o, s) for s, o in extracted}
    hits = sum(1 for s, _r, o in gold_evidences if (normalize(s), normalize(o)) in extracted)
    return hits / len(gold_evidences)


def complete_witness(witness_facts: Sequence[Sequence[str]],
                     gold_evidences: Sequence[tuple[str, str, str]]) -> float:
    coverage = witness_coverage(witness_facts, gold_evidences)
    return float("nan") if coverage != coverage else float(coverage >= 1.0)


# ---------------------------------------------------------------------------
# Risco versus cobertura
# ---------------------------------------------------------------------------

@dataclass
class RiskCoveragePoint:
    coverage: float
    accuracy: float
    threshold: float


def risk_coverage_curve(scores: Sequence[float], correct: Sequence[float],
                        n_points: int = 11) -> list[RiskCoveragePoint]:
    """Acurácia em função da fração respondida, ordenando por risco crescente.

    `scores` é o risco (menor é melhor). Uma curva que sobe à esquerda significa
    que o certificado sabe onde o sistema erra; uma curva plana significa que o
    número não carrega informação — e essa é uma refutação concreta da parte
    probabilística da proposta, não um detalhe.
    """
    if len(scores) != len(correct):
        raise ValueError("scores e resultados devem ter o mesmo tamanho")
    pairs = sorted((s, c) for s, c in zip(scores, correct) if math.isfinite(s) and math.isfinite(c))
    if not pairs:
        return []
    out: list[RiskCoveragePoint] = []
    n = len(pairs)
    for i in range(1, n_points + 1):
        take = max(1, round(n * i / n_points))
        # Um limiar não pode aceitar somente parte de um empate.
        while take < n and pairs[take][0] == pairs[take - 1][0]:
            take += 1
        if out and out[-1].coverage == take / n:
            continue
        subset = pairs[:take]
        out.append(RiskCoveragePoint(
            coverage=take / n,
            accuracy=sum(c for _s, c in subset) / take,
            threshold=subset[-1][0],
        ))
    return out


def paired_bootstrap_ci(left: Sequence[float], right: Sequence[float], **kwargs) -> tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("comparações pareadas exigem as mesmas perguntas")
    return bootstrap_ci([a - b for a, b in zip(left, right)
                         if math.isfinite(a) and math.isfinite(b)], **kwargs)


def aggregate(values: Iterable[float]) -> float:
    values = [v for v in values if v == v]  # descarta NaN
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    """Intervalo percentílico. Com 100 perguntas, a largura costuma ser de vários
    pontos — o que é exatamente o motivo de reportá-lo: um piloto raramente
    separa dois métodos que diferem por 2 pontos."""
    import random

    values = [v for v in values if v == v]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
