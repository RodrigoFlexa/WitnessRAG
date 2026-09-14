"""
Proveniência por conjuntos de testemunhas e pontuação heurística.

Fatos conjuntos formam uma conjunção; testemunhas alternativas formam uma
disjunção. O score usa a melhor testemunha e não cresce ao duplicar provas.
Os campos risk/risk_raw são scores não calibrados inspirados no limite da
união. Confianças fixas e similaridades não são probabilidades verificadas.
Uma garantia probabilística exigiria limites de erro válidos, incluindo a
compilação, o aterramento e a seleção adaptativa da testemunha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from wrag import config as C
from wrag.graph import KnowledgeGraph
from wrag.util import canonical_symbol as normalize
from wrag.witness.search import Witness


@dataclass
class AnswerCandidate:
    answer: str
    witnesses: list[Witness] = field(default_factory=list)
    score: float = 0.0
    risk: float = 1.0        # score recortado em [0,1], para leitura
    risk_raw: float = 1.0    # score sem recorte; é o que ordena a curva
    # ^ a distinção não é cosmética: com um retriever fraco o limite estoura 1
    #   em quase toda pergunta, e recortar antes de ordenar transformaria a curva
    #   risco-cobertura numa reta — um artefato no diagnóstico.

    @property
    def best(self) -> Witness | None:
        return self.witnesses[0] if self.witnesses else None

    def monomials(self, kg: KnowledgeGraph) -> list[list[str]]:
        """O polinômio de proveniência, em forma legível: uma lista por monômio."""
        return [[kg.facts[i].fid for i in w.facts] for w in self.witnesses]

    def to_dict(self, kg: KnowledgeGraph, max_witnesses: int = 3) -> dict[str, Any]:
        return {
            "resposta": self.answer,
            "score": round(self.score, 4),
            "score_tipo": "melhor_testemunha_heuristico",
            "risco_calibrado": False,
            "risco": round(self.risk, 4),
            "risco_bruto": round(self.risk_raw, 4),
            "n_testemunhas": len(self.witnesses),
            "testemunhas": [w.to_dict(kg) for w in self.witnesses[:max_witnesses]],
        }


def score_answers(
    witnesses: Sequence[Witness],
    kg: KnowledgeGraph,
    cfg: C.WitnessConfig,
) -> list[AnswerCandidate]:
    """Agrupa testemunhas por resposta e avalia o polinômio de proveniência."""
    groups: dict[str, AnswerCandidate] = {}
    for witness in witnesses:
        key = normalize(witness.answer)
        if not key:
            continue
        candidate = groups.setdefault(key, AnswerCandidate(answer=witness.answer))
        candidate.witnesses.append(witness)

    out: list[AnswerCandidate] = []
    for candidate in groups.values():
        candidate.witnesses.sort(key=lambda w: (-_witness_support(w, kg, cfg), w.cost, w.facts))
        # Duplicar uma prova ou adicionar cópias de uma fonte não aumenta o score.
        candidate.score = max(_witness_support(w, kg, cfg) for w in candidate.witnesses)
        candidate.risk_raw = _union_bound(candidate.witnesses[:1], kg, cfg)
        candidate.risk = float(max(0.0, min(1.0, candidate.risk_raw)))
        out.append(candidate)

    out.sort(key=lambda c: (-c.score, c.risk))
    return out


def _witness_support(witness: Witness, kg: KnowledgeGraph, cfg: C.WitnessConfig) -> float:
    """∏_{e∈W} p_e, com p_e a confiança do fato temperada pelo casamento do átomo.

    `witness.score` já é o produto dos scores de casamento; multiplicá-lo pelas
    confianças dá o peso do monômio. Um casamento fraco derruba a demonstração
    inteira, que é o comportamento certo: uma testemunha só vale o seu elo mais
    frouxo.
    """
    probability = max(0.0, min(1.0, witness.score))
    for index in set(witness.facts):
        probability *= max(0.0, min(1.0, kg.facts[index].confidence))
    return probability


def _union_bound(witnesses: Sequence[Witness], kg: KnowledgeGraph, cfg: C.WitnessConfig) -> float:
    """δ_interpretação + Σ_{e∈W*} δ_e + δ_verbalização, na melhor testemunha.

    Sem suposição de independência. `δ_e = 1 - p_e` é o erro admitido por fato, e
    a testemunha escolhida é a de menor limite — o que, note-se, é uma escolha
    adaptativa: a validade formal do limite exigiria uma testemunha fixada antes
    de olhar os dados, ou uma correção para a seleção. O relatório trata este
    número como um score de seletividade calibrável, não como garantia.
    """
    if not witnesses:
        return 1.0
    return float(min(
        (cfg.delta_interpretation + cfg.delta_verbalization
         + sum(1.0 - max(0.0, min(1.0, kg.facts[i].confidence)) for i in set(w.facts))
         + (1.0 - max(0.0, min(1.0, w.score))))
        for w in witnesses
    ))


def rank_passages(
    candidates: Sequence[AnswerCandidate],
    k: int,
    max_per_witness: int | None = None,
) -> tuple[list[str], list[float]]:
    """Passagens do contexto final: a proveniência das melhores demonstrações.

    A ordem é por qualidade da resposta e depois por custo da testemunha, o que
    coloca no topo exatamente as passagens que compõem a prova mais barata. É
    diferente de ranquear passagens por relevância: uma passagem que sozinha não
    é relevante entra alta se for o segundo elo da demonstração — que é o caso
    que o multi-hop denso costuma perder.
    """
    order: list[str] = []
    scores: list[float] = []
    have: set[str] = set()
    for rank, candidate in enumerate(candidates):
        for witness in candidate.witnesses:
            pids = tuple(dict.fromkeys(witness.pids))
            if max_per_witness is not None and len(pids) > max_per_witness:
                continue
            missing = [pid for pid in pids if pid not in have]
            if len(order) + len(missing) > k:
                continue
            for pid in missing:
                order.append(pid)
                have.add(pid)
                scores.append(1.0 / len(order))
    return order, scores


def explain(candidate: AnswerCandidate, kg: KnowledgeGraph) -> str:
    """Explicação em texto do polinômio: útil em análise de erro, não no prompt."""
    parts = []
    for witness in candidate.witnesses[:3]:
        conjunction = " ∧ ".join(
            f"({kg.facts[i].subject} | {kg.facts[i].relation} | {kg.facts[i].object})"
            for i in witness.facts
        )
        parts.append(conjunction)
    return f"{candidate.answer} ⇐ " + "  ∨  ".join(parts)
