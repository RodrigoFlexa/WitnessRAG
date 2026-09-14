"""
Construção do conjunto de demandas t = (q, a) que orienta a seleção de memória.

As demandas vêm de um split de TREINO, disjunto do de avaliação. A proposta é
categórica quanto a isso: "otimizar para as próprias perguntas de teste apenas
demonstra memorização do conjunto de avaliação".

Uma demanda só entra no conjunto quando a busca encontra alguma testemunha CUJA
RESPOSTA bate com a resposta anotada. Testemunhas que demonstram outra resposta
são demonstrações de algo errado, e preservá-las gastaria orçamento de memória
protegendo erro.
"""

from __future__ import annotations

from typing import Any, Sequence

from wrag import config as C
from wrag.data import Question
from wrag.embed import Embedder
from wrag.llm import LLM
from wrag.util import get_logger, normalize_answer, progress, canonical_symbol
from wrag.witness.budget import Demand
from wrag.witness.query import compile_query
from wrag.witness.search import WitnessSearcher

log = get_logger("wrag.witness.demands")


def build_demands(
    questions: Sequence[Question],
    searcher: WitnessSearcher,
    llm: LLM,
    cfg: C.WitnessConfig,
    compile_mode: str = "oracle",
    weight_by_hops: bool = True,
    dataset: str = "",
) -> list[Demand]:
    """Uma demanda por pergunta de treino cuja resposta tem testemunha.

    `compile_mode="oracle"` por padrão: na fase de construção da memória, usar a
    consulta anotada isola a decisão de armazenamento do ruído de compilação. Se
    o oráculo não existir para o dataset, cai para o LLM.
    """
    demands: list[Demand] = []
    for question in progress(questions, desc="demandas de treino"):
        query = compile_query(llm, question, mode=compile_mode, max_atoms=cfg.max_atoms,
                              dataset=dataset)
        if not query.atoms:
            continue
        result = searcher.join(query)
        if not result.complete:
            continue

        wanted = {normalize_answer(a) for a in question.answers}
        witnesses = [frozenset(w.facts) for w in result.witnesses
                     if normalize_answer(w.answer) in wanted]
        if not witnesses:
            continue

        # Peso por número de hops: preservar uma demonstração de três elos vale
        # mais que uma de um, porque é a que nenhum outro método recupera.
        weight = float(question.n_hops) if weight_by_hops else 1.0
        demands.append(Demand(tid=f"t{len(demands)}", weight=weight,
                              witnesses=_dedupe_minimal(witnesses)))

    log.info("demandas construídas: %d de %d perguntas de treino", len(demands), len(questions))
    return demands


def synthesize_demands(
    memory: Any,
    n_demands: int = 500,
    seed: int = 42,
    weight_by_hops: bool = True,
) -> list[Demand]:
    """Demandas geradas a partir da estrutura do próprio grafo.

    Perguntas de treino anotadas quase nunca existem para o corpus que se quer
    indexar — e, num piloto com corpus subamostrado, as que existem raramente têm
    todas as passagens de apoio dentro do índice. Então a alternativa realista
    (e a que um sistema em produção usaria) é instanciar padrões de consulta
    sobre o grafo e tratar cada instância como uma demanda:

    * single-hop: um fato qualquer, peso 1;
    * cadeia: (s,r₁,m) seguido de (m,r₂,o), peso 2;
    * interseção: duas relações distintas sobre a mesma entidade, peso 2.

    Não é a mesma coisa que uma distribuição real de perguntas — é um prior
    uniforme sobre o que o corpus é capaz de demonstrar, e sub-representa o que
    usuários realmente perguntam. Também não usa nenhuma resposta do conjunto de
    avaliação: o que entra aqui é a estrutura do grafo, não o gabarito.
    """
    import random

    rng = random.Random(seed)
    facts = memory.facts
    if not facts:
        return []

    by_subject_cluster: dict[int, list[int]] = {}
    for i, fact in enumerate(facts):
        if fact.subj_id >= 0:
            by_subject_cluster.setdefault(fact.subj_id, []).append(i)

    demands: list[Demand] = []
    by_query_answer: dict[tuple, Demand] = {}

    def add(key: tuple, witness: frozenset[int], weight: float) -> None:
        if not witness:
            return
        if key not in by_query_answer:
            if len(demands) >= n_demands:
                return
            demand = Demand(tid=f"syn{len(demands)}", weight=weight)
            by_query_answer[key] = demand
            demands.append(demand)
        demand = by_query_answer[key]
        demand.witnesses = _dedupe_minimal(demand.witnesses + [witness])

    indices = list(range(len(facts)))
    rng.shuffle(indices)

    for i in indices:
        fact = facts[i]
        s, r, o = (canonical_symbol(t) for t in fact.triple)
        add(("single", s, r, o), frozenset({i}), 1.0)

        # cadeia: o objeto deste fato é sujeito de outro
        if fact.obj_id >= 0:
            partners = by_subject_cluster.get(fact.obj_id, [])
            for j in rng.sample(partners, min(8, len(partners))):
                if j != i:
                    other = facts[j]
                    key = ("chain", s, r, canonical_symbol(other.relation), canonical_symbol(other.object))
                    add(key, frozenset({i, j}), 2.0 if weight_by_hops else 1.0)

        # interseção: outra relação sobre o mesmo sujeito
        if fact.subj_id >= 0:
            partners = by_subject_cluster.get(fact.subj_id, [])
            for j in rng.sample(partners, min(8, len(partners))):
                if j != i and facts[j].rel_id != fact.rel_id:
                    other = facts[j]
                    arms = tuple(sorted(((r, o), (canonical_symbol(other.relation), canonical_symbol(other.object)))))
                    add(("intersection", arms, s), frozenset({i, j}), 2.0 if weight_by_hops else 1.0)

    log.info("demandas sintetizadas a partir do grafo: %d", len(demands))
    return demands[:n_demands]


def _dedupe_minimal(witnesses: list[frozenset[int]]) -> list[frozenset[int]]:
    """Mantém só as mínimas por inclusão: um superconjunto de testemunha não
    acrescenta alternativa, só infla o ILP."""
    ordered = sorted(witnesses, key=len)
    kept: list[frozenset[int]] = []
    for witness in ordered:
        if not any(other <= witness for other in kept):
            kept.append(witness)
    return kept
