"""
Seleção de memória sob orçamento: o programa inteiro da proposta.

Dadas demandas t = (q, a) com pesos w_t ≥ 0, a utilidade de uma memória S ⊆ F é

    U(S) = Σ_t w_t · 1[∃W ∈ W_t : W ⊆ S]

e o problema é

    max_{S ⊆ F} U(S)   sujeito a   Σ_{e∈S} c_e ≤ B.

**Por que não um guloso com garantia.** U não é submodular em geral. Para uma
demanda que exige {e1, e2}, adicionar e2 à memória vazia tem ganho zero, e
adicionar e2 quando e1 já está lá tem ganho um: o ganho AUMENTA, há
complementaridade, e as garantias clássicas de cobertura gulosa não se aplicam.
`submodularity_counterexample()` executa exatamente esse par de contas.

Isso não é tecnicalidade. É a razão formal pela qual "guardar as arestas mais
relevantes individualmente" — que é o que uma poda por relevância faz — pode
eliminar justamente as combinações de que o multi-hop depende.

A formulação inteira codifica o "E" dentro de uma testemunha e o "OU" entre
testemunhas alternativas:

    y_{t,W} ≤ x_e                        ∀ e ∈ W
    y_{t,W} ≥ Σ_{e∈W} x_e − |W| + 1
    z_t     ≥ y_{t,W}                    ∀ W ∈ W_t
    z_t     ≤ Σ_{W∈W_t} y_{t,W}
    Σ_e c_e x_e ≤ B

maximizando Σ_t w_t z_t. Se todas as testemunhas pertinentes forem enumeradas e o
solucionador terminar com certificado de otimalidade, o resultado é ótimo global
daquele problema finito. Enumerando só algumas, a garantia se restringe às
alternativas enumeradas — e é isso que acontece em corpus real, onde a
enumeração completa é proibitiva. O status certifica apenas a otimização sobre as testemunhas fornecidas;
a enumeração completa precisa ser estabelecida separadamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Sequence

from wrag.util import get_logger

log = get_logger("wrag.witness.budget")


@dataclass
class Demand:
    """Uma demanda t = (q, a) e suas testemunhas conhecidas."""

    tid: str
    weight: float = 1.0
    witnesses: list[frozenset[int]] = field(default_factory=list)

    def satisfied_by(self, memory: set[int]) -> bool:
        return any(w <= memory for w in self.witnesses)


@dataclass
class SelectionResult:
    kept: set[int]
    value: float
    total_weight: float
    status: str
    cost: float
    method: str
    optimal: bool = False

    @property
    def coverage(self) -> float:
        return self.value / self.total_weight if self.total_weight else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"metodo": self.method, "status": self.status, "n_fatos_mantidos": len(self.kept),
                "custo": round(self.cost, 2), "utilidade": round(self.value, 3),
                "cobertura": round(self.coverage, 4), "otimo_certificado": self.optimal,
                "escopo_otimalidade": "testemunhas_enumeradas"}


def _validate(demands, costs, budget):
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("orçamento deve ser finito e não negativo")
    if len({d.tid for d in demands}) != len(demands):
        raise ValueError("identificadores de demandas devem ser únicos")
    if any(not math.isfinite(d.weight) or d.weight < 0 for d in demands):
        raise ValueError("pesos devem ser finitos e não negativos")
    relevant = {e for d in demands for w in d.witnesses for e in w}
    if not relevant <= costs.keys():
        raise ValueError("faltam custos de fatos presentes nas testemunhas")
    if any(not math.isfinite(c) or c < 0 for c in costs.values()):
        raise ValueError("custos devem ser finitos e não negativos")


def utility(demands: Sequence[Demand], memory: set[int]) -> float:
    return sum(d.weight for d in demands if d.satisfied_by(memory))


def select_ilp(
    demands: Sequence[Demand],
    costs: dict[int, float],
    budget: float,
    time_limit_s: int = 120,
) -> SelectionResult:
    """Resolve exatamente (ou até o limite de tempo) com CBC via PuLP."""
    _validate(demands, costs, budget)
    try:
        import pulp
    except ImportError:
        log.warning("PuLP indisponível; caindo para o guloso (sem garantia de otimalidade)")
        return select_greedy(demands, costs, budget)

    relevant = sorted({e for d in demands for w in d.witnesses for e in w})
    if not relevant:
        return SelectionResult(set(), utility(demands, set()), sum(d.weight for d in demands),
                               "vazio", 0.0, "ilp", True)

    problem = pulp.LpProblem("memoria_por_testemunhas", pulp.LpMaximize)
    x = {e: pulp.LpVariable(f"x_{e}", cat="Binary") for e in relevant}
    z: dict[str, Any] = {}
    y: dict[tuple[str, int], Any] = {}

    for di, d in enumerate(demands):
        if not d.witnesses:
            continue
        z[d.tid] = pulp.LpVariable(f"z_{di}", cat="Binary")
        for wi, witness in enumerate(d.witnesses):
            var = pulp.LpVariable(f"y_{di}_{wi}", cat="Binary")
            y[(d.tid, wi)] = var
            for e in witness:
                problem += var <= x[e]
            problem += var >= pulp.lpSum(x[e] for e in witness) - len(witness) + 1
            problem += z[d.tid] >= var
        problem += z[d.tid] <= pulp.lpSum(y[(d.tid, wi)] for wi in range(len(d.witnesses)))

    problem += pulp.lpSum(costs.get(e, 1.0) * x[e] for e in relevant) <= budget
    problem += pulp.lpSum(d.weight * z[d.tid] for d in demands if d.tid in z)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]
    solution_status = getattr(problem, "sol_status", None)
    certified = (problem.status == pulp.LpStatusOptimal and solution_status == pulp.LpSolutionOptimal)
    values = {e: x[e].value() for e in relevant}
    integral = all(v is not None and min(abs(v), abs(v - 1)) <= 1e-6 for v in values.values())
    kept = {e for e, v in values.items() if v is not None and v > 0.5}
    value = utility(demands, kept)
    cost = sum(costs.get(e, 1.0) for e in kept)
    if not integral or cost > budget + 1e-6 or solution_status not in {
            pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible}:
        fallback = select_greedy(demands, costs, budget)
        fallback.status = f"fallback_sem_incumbente_valido:{status}"
        return fallback
    status = "Optimal" if certified else f"Feasible:{status}"
    return SelectionResult(kept, value, sum(d.weight for d in demands), status, cost, "ilp", certified)


def select_greedy(
    demands: Sequence[Demand],
    costs: dict[int, float],
    budget: float,
) -> SelectionResult:
    """Guloso por testemunha inteira: razão peso/custo-incremental.

    Não é o guloso ingênuo (que escolheria fatos isolados e é exatamente o que o
    contraexemplo derruba): aqui a unidade de escolha é a TESTEMUNHA, o que
    contorna a complementaridade dentro de uma demonstração. Continua sem
    garantia — a complementaridade ENTRE demandas que compartilham fatos
    permanece — mas é o baseline honesto contra o qual medir o ILP.
    """
    _validate(demands, costs, budget)
    memory: set[int] = set()
    spent = 0.0
    pending = [d for d in demands if d.witnesses]

    while pending:
        best = None
        for d in pending:
            for witness in d.witnesses:
                missing = witness - memory
                incremental = sum(costs.get(e, 1.0) for e in missing)
                if spent + incremental > budget:
                    continue
                gain = utility(demands, memory | missing) - utility(demands, memory)
                ratio = gain / max(incremental, 1e-9)
                if best is None or ratio > best[0]:
                    best = (ratio, d, missing, incremental)
        if best is None:
            break
        _ratio, demand, missing, incremental = best
        memory |= missing
        spent += incremental
        pending = [d for d in pending if not d.satisfied_by(memory)]

    return SelectionResult(memory, utility(demands, memory), sum(d.weight for d in demands),
                           "guloso", spent, "greedy")


def submodularity_counterexample() -> dict[str, float]:
    """Executa o contraexemplo: f(S) = 1[{e1,e2} ⊆ S] tem ganho crescente.

    Devolve os dois ganhos marginais. `ganho_sozinho` é 0 e `ganho_com_e1` é 1;
    submodularidade exigiria ganho_sozinho ≥ ganho_com_e1.
    """
    def f(S: set[int]) -> float:
        return 1.0 if {1, 2} <= S else 0.0

    return {
        "ganho_sozinho": f({2}) - f(set()),
        "ganho_com_e1": f({1, 2}) - f({1}),
        "submodular": float(f({2}) - f(set()) >= f({1, 2}) - f({1})),
    }


def toy_instance() -> tuple[list[Demand], dict[int, float]]:
    """A instância de cinco fatos da proposta, para verificar a formulação.

    e1 Ana trabalha na Atlas | e2 Atlas fica em Recife | e3 Ana pesquisa Óptica
    e4 Bruno trabalha na Atlas | e5 Bruno pesquisa Óptica

    Demandas: empregador de Ana (peso 1), cidade da instituição de Ana (peso 3),
    alguém na Atlas que pesquisa Óptica (peso 4). Com orçamento de três fatos, a
    solução ótima é {e1, e2, e3} com valor 8.
    """
    demands = [
        Demand("empregador_de_ana", 1.0, [frozenset({1})]),
        Demand("cidade_da_instituicao_de_ana", 3.0, [frozenset({1, 2})]),
        Demand("pesquisador_optica_na_atlas", 4.0, [frozenset({1, 3}), frozenset({4, 5})]),
    ]
    costs = {e: 1.0 for e in range(1, 6)}
    return demands, costs
