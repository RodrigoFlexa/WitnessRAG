"""
VERIFICAR (desenho v3): confirmar a prova antes de ela mudar o contexto.

O verificador recebe a pergunta, o plano e as respostas candidatas, cada uma
com os fatos que a sustentam e, para cada fato, a fala de origem com falante,
data da sessão e vizinhas. Ele confirma ou recusa cada candidata. Recusar exige
um problema claro (entidade, período ou tipo errados, ou falta de suporte);
na dúvida, confirma. A verificação não procura nada: quem procura é a busca.

Lições incorporadas das auditorias do repositório: exigir palavras literais
vetava paráfrases de diálogo; exigir que um único trecho contenha a cadeia
inteira vetava composições entre falas; conjuntos precisam de decisão por item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from wrag import prompts
from wrag.data import Question
from wrag.llm import LLM, GenParams
from wrag.llm.filters import LEDGER

REASONS = ("wrong_entity", "wrong_period", "wrong_type", "not_supported")


@dataclass
class Verdict:
    called: bool = False
    supported: list[int] = field(default_factory=list)
    rejected: dict[int, str] = field(default_factory=dict)
    valid: bool = False
    filtered: bool = False
    error: str = ""

    @property
    def confirmed(self) -> bool:
        return bool(self.supported)

    def to_dict(self) -> dict[str, Any]:
        return {"chamada": self.called, "valida": self.valid, "filtrada": self.filtered,
                "erro": self.error, "confirmadas": list(self.supported),
                "recusadas": {str(k): v for k, v in self.rejected.items()}}


def render_candidates(candidates: Sequence[dict[str, Any]]) -> str:
    """``candidates``: [{"answer", "facts": [{"triple", "date", "excerpt"}]}]."""
    blocks = []
    for number, candidate in enumerate(candidates, 1):
        lines = [f"A{number}: {candidate['answer']}"]
        for fact in candidate["facts"]:
            subject, relation, obj = fact["triple"]
            when = f" (event date: {fact['date']})" if fact.get("date") else ""
            lines.append(f"  fact: {subject} | {relation} | {obj}{when}")
            if fact.get("excerpt"):
                lines.extend("    " + line for line in fact["excerpt"].splitlines())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_verdict(data: Any, n_candidates: int) -> Verdict:
    verdict = Verdict(called=True)
    if not isinstance(data, dict):
        verdict.error = "json_invalido"
        return verdict

    def index(value: Any) -> int | None:
        text = str(value or "").strip().upper()
        if text.startswith("A"):
            text = text[1:]
        return int(text) - 1 if text.isdigit() and 0 < int(text) <= n_candidates else None

    def items(value: Any) -> list:
        if value is None or isinstance(value, bool):
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    for value in items(data.get("supported")):
        position = index(value.get("id") if isinstance(value, dict) else value)
        if position is not None and position not in verdict.supported:
            verdict.supported.append(position)
    for item in items(data.get("rejected")):
        position = index(item.get("id") if isinstance(item, dict) else item)
        if position is None:
            continue
        reason = str(item.get("reason") if isinstance(item, dict) else "") or "not_supported"
        verdict.rejected[position] = reason if reason in REASONS else "not_supported"
    # A candidate listed as both is rejected: the explicit problem wins.
    verdict.supported = [i for i in verdict.supported if i not in verdict.rejected]
    verdict.valid = bool(verdict.supported or verdict.rejected)
    if not verdict.valid:
        verdict.error = "sem_decisao"
    return verdict


def confirm(llm: LLM, question: Question, plan_text: str,
            candidates: Sequence[dict[str, Any]], dataset: str = "",
            method: str = "witnessrag") -> Verdict:
    if not candidates:
        return Verdict()
    result = llm.chat(
        prompts.CONFIRM_TEMPLATE.format(question=question.question, plan=plan_text,
                                        candidates=render_candidates(candidates)),
        system=prompts.CONFIRM_SYSTEM,
        params=GenParams(temperature=0.0, max_tokens=400, json_mode=True),
        stage="witness.confirm",
    )
    if result.filtered:
        LEDGER.add("confirm", dataset, method, question.qid, "verificação bloqueada")
        return Verdict(called=True, filtered=True, error="filtrada")
    return parse_verdict(result.json(), len(candidates))
