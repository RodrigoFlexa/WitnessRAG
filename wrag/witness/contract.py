"""
Contrato de evidência: o plano agnóstico que substitui o rótulo do benchmark.

O controlador seletivo decidia a rota lendo ``Question.qtype`` (a categoria
oficial do LoCoMo). Em produção esse rótulo não existe. Aqui o próprio agente
descreve a necessidade de informação ANTES da busca, em um vocabulário que não
menciona categorias de benchmark:

    π(q) = ⟨forma da resposta, operador, escopo da evidência, foco temporal,
            entidades de foco, necessidades de informação, lentes de memória⟩

A rota é uma função DETERMINÍSTICA e auditável do contrato, derivada do
fragmento que o executor de testemunhas sabe certificar:

    COMPOSE  ⇔  operador ∈ {aggregate, join, compare}
               ∧ escopo = multiple
               ∧ forma ∈ {entity, set, count, description}
    DIRECT   caso contrário

Justificativa: o executor prova consultas conjuntivas positivas com uma
variável de resposta ligada a entidades. Uma consulta de um átomo é apenas
uma busca (a testemunha é uma passagem, e a recuperação híbrida já a
aproxima); tempo, abdução e respostas sim/não ficam fora do fragmento
executável e são servidos por outros operadores (memória temporal, leitor).

O contrato nunca vê resposta ouro, passagens de apoio nem a categoria. Um
contrato inválido ou bloqueado cai na rota DIRECT, que é exatamente a linha de
base híbrida: o erro do planejador é limitado por construção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wrag import prompts
from wrag.data import Question
from wrag.llm import LLM, GenParams
from wrag.llm.base import LLMResult
from wrag.util import get_logger

log = get_logger("wrag.witness.contract")

ANSWER_FORMS = ("entity", "set", "count", "time", "duration", "yes_no", "choice", "description")
OPERATORS = ("lookup", "aggregate", "join", "compare", "temporal", "abduce")
SCOPES = ("single", "multiple")
TIME_FOCUS = ("none", "when", "first", "last", "window", "current", "duration")
TEMPORAL_LENSES = ("none", "recent", "early", "anchor")

# Fragmento executável pelo compositor de testemunhas.
COMPOSE_OPERATORS = frozenset({"aggregate", "join", "compare"})
COMPOSE_FORMS = frozenset({"entity", "set", "count", "description"})

ROUTE_DIRECT = "direct"
ROUTE_COMPOSE = "compose"

_ALIASES = {
    # Pequenas variações observadas em modelos de instrução. Normalizar aqui
    # evita que um sinônimo trivial vire rota DIRECT por erro de formatação.
    "list": "set", "items": "set", "number": "count", "date": "time",
    "boolean": "yes_no", "yes/no": "yes_no", "yesno": "yes_no", "option": "choice",
    "explanation": "description", "reason": "description", "phrase": "entity",
    "enumerate": "aggregate", "collect": "aggregate", "aggregation": "aggregate",
    "chain": "join", "intersection": "join", "comparison": "compare",
    "inference": "abduce", "infer": "abduce", "hypothetical": "abduce",
    "time": "temporal", "multi": "multiple", "many": "multiple", "one": "single",
    "latest": "recent", "newest": "recent", "earliest": "early", "first": "early",
}


def _choice(value: Any, allowed: tuple[str, ...], default: str, repairs: list[str],
            name: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in allowed:
        return text
    alias = _ALIASES.get(text.replace("_", " "), _ALIASES.get(text))
    if alias in allowed:
        repairs.append(f"{name}:{text}->{alias}")
        return alias
    if text:
        repairs.append(f"{name}:{text}->{default}")
    return default


def _strings(value: Any, limit: int, max_len: int = 160) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip() and len(item.strip()) <= max_len:
            text = item.strip()
            if text not in out:
                out.append(text)
        if len(out) >= limit:
            break
    return out


@dataclass
class EvidenceContract:
    answer_form: str = "entity"
    operator: str = "lookup"
    evidence_scope: str = "single"
    time_focus: str = "none"
    time_anchor: str = ""
    focus_entities: list[str] = field(default_factory=list)
    info_needs: list[str] = field(default_factory=list)
    temporal_lens: str = "none"
    salience_lens: bool = False
    confidence_lens: bool = False
    valid: bool = False
    filtered: bool = False
    error: str = ""
    repairs: list[str] = field(default_factory=list)

    # -- rota -----------------------------------------------------------------

    @property
    def route(self) -> str:
        return route_of(self)

    @property
    def lenses(self) -> dict[str, Any]:
        return {"temporal": self.temporal_lens, "salience": self.salience_lens,
                "confidence": self.confidence_lens}

    @property
    def uses_lenses(self) -> bool:
        return self.temporal_lens != "none" or self.salience_lens or self.confidence_lens

    def to_dict(self) -> dict[str, Any]:
        return {"answer_form": self.answer_form, "operator": self.operator,
                "evidence_scope": self.evidence_scope,
                "time": {"focus": self.time_focus, "anchor": self.time_anchor},
                "focus_entities": list(self.focus_entities),
                "info_needs": list(self.info_needs), "lenses": self.lenses,
                "route": self.route, "valid": self.valid, "filtered": self.filtered,
                "error": self.error, "repairs": list(self.repairs)}


def route_of(contract: EvidenceContract) -> str:
    """Rota como função pura do contrato. Contrato inválido nunca compõe."""
    if not contract.valid or contract.filtered:
        return ROUTE_DIRECT
    if (contract.operator in COMPOSE_OPERATORS and contract.evidence_scope == "multiple"
            and contract.answer_form in COMPOSE_FORMS):
        return ROUTE_COMPOSE
    return ROUTE_DIRECT


def contract_from_data(data: Any) -> EvidenceContract:
    """Valida a saída do planejador; nada fora do vocabulário fechado passa."""
    if not isinstance(data, dict):
        return EvidenceContract(error="contrato_nao_json")
    repairs: list[str] = []
    time = data.get("time") if isinstance(data.get("time"), dict) else {}
    lenses = data.get("lenses") if isinstance(data.get("lenses"), dict) else {}
    contract = EvidenceContract(
        answer_form=_choice(data.get("answer_form"), ANSWER_FORMS, "entity", repairs, "form"),
        operator=_choice(data.get("operator"), OPERATORS, "lookup", repairs, "operator"),
        evidence_scope=_choice(data.get("evidence_scope"), SCOPES, "single", repairs, "scope"),
        time_focus=_choice(time.get("focus"), TIME_FOCUS, "none", repairs, "time_focus"),
        time_anchor=str(time.get("anchor") or "").strip()[:80],
        focus_entities=_strings(data.get("focus_entities"), 6, 80),
        info_needs=_strings(data.get("info_needs"), 4),
        temporal_lens=_choice(lenses.get("temporal"), TEMPORAL_LENSES, "none", repairs,
                              "temporal_lens"),
        salience_lens=lenses.get("salience") is True,
        confidence_lens=lenses.get("confidence") is True,
        valid=True,
        repairs=repairs,
    )
    # A âncora só vale como lente quando existe texto para ancorar; o parser de
    # datas decide depois se ela é interpretável (ver wrag/witness/timeline.py).
    if contract.temporal_lens == "anchor" and not contract.time_anchor:
        contract.temporal_lens = "none"
        contract.repairs.append("temporal_lens:anchor_sem_ancora->none")
    return contract


def contract_prompt(question: str) -> str:
    return prompts.CONTRACT_TEMPLATE.format(question=question)


def contract_params(temperature: float = 0.0) -> GenParams:
    return GenParams(temperature=temperature, max_tokens=700, json_mode=True)


def contract_from_result(result: LLMResult) -> EvidenceContract:
    if result.filtered:
        return EvidenceContract(filtered=True, error="contrato_bloqueado")
    if result.error:
        return EvidenceContract(error=f"erro_llm:{str(result.error)[:80]}")
    return contract_from_data(result.json())


def plan_contract(llm: LLM, question: Question, temperature: float = 0.0) -> EvidenceContract:
    """Uma chamada de planejamento; recebe somente o texto da pergunta.

    Um bloqueio de conteúdo NÃO é registrado no LEDGER: a pergunta continua na
    avaliação pela rota DIRECT (a linha de base), para que o denominador seja o
    mesmo do controlador rotulado. O bloqueio fica no diagnóstico.
    """
    result = llm.chat(contract_prompt(question.question), system=prompts.CONTRACT_SYSTEM,
                      params=contract_params(temperature), stage="witness.contract")
    contract = contract_from_result(result)
    if not contract.valid:
        log.debug("contrato inválido para %s: %s", question.qid, contract.error)
    return contract
