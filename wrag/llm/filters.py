"""
Reconhecimento e contabilidade de bloqueio por política de conteúdo.

Os marcadores vieram do `azure.py` da grade RMCQ, que já apanhou desse gateway.
A diferença aqui é o que fazemos depois: naquele projeto o item filtrado era
descartado; aqui ele precisa continuar existindo no relatório, porque uma
pergunta que o Azure recusa em um método e aceita em outro tornaria a comparação
entre métodos desonesta. Ver `Ledger.excluded_union`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

CONTENT_FILTER_MARKERS = (
    "responsibleaipolicyviolation",
    "content_filter",
    "content filter",
    "content management policy",
    "prompt was filtered",
    "jailbreak",
    "malicious content",
    "conteúdo malicioso",
    "conteudo malicioso",
    "self_harm",
    "hate",
    "sexual",
    "violence",
)

# Marcadores genéricos demais para valer sozinhos: "hate" aparece em textos
# comuns. Só contam quando acompanhados de sinal explícito de política.
_WEAK_MARKERS = ("hate", "sexual", "violence", "self_harm")
_STRONG_MARKERS = tuple(m for m in CONTENT_FILTER_MARKERS if m not in _WEAK_MARKERS)


def is_content_filter_error(exc: BaseException) -> bool:
    """Reconhece erro de política sem engolir outros 4xx.

    Um 400 por parâmetro inválido e um 400 por política têm o mesmo status; só o
    corpo os separa. Confundir os dois faz o benchmark descartar silenciosamente
    perguntas que na verdade expõem um bug de configuração.
    """
    parts = [str(exc)]
    for attribute in ("body", "message", "code"):
        value = getattr(exc, attribute, None)
        if value:
            try:
                parts.append(json.dumps(value, ensure_ascii=False, default=str))
            except TypeError:
                parts.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        parts.append(str(getattr(response, "text", "") or ""))
    text = " ".join(parts).casefold()
    if any(marker in text for marker in _STRONG_MARKERS):
        return True
    # Fraco só conta com sinal de política junto.
    return any(m in text for m in _WEAK_MARKERS) and ("filter" in text or "policy" in text)


@dataclass
class FilterEvent:
    stage: str
    dataset: str
    method: str
    item_id: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage, "dataset": self.dataset, "method": self.method,
            "item_id": self.item_id, "detail": self.detail[:400],
        }


class Ledger:
    """Registro do que o filtro recusou, por dataset/método/item.

    `excluded_union(dataset)` devolve as perguntas bloqueadas em QUALQUER método.
    A tabela principal do relatório usa esse conjunto para excluir as mesmas
    perguntas de todos os sistemas — do contrário, o método que teve mais
    perguntas recusadas competiria num subconjunto diferente e mais fácil.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.events: list[FilterEvent] = []
        self.path = path
        # Reentrante: `summary()` chama `by_method()`, e um Lock simples aqui
        # trava o processo inteiro no fim da rodada — silenciosamente, porque a
        # thread principal fica esperando a si mesma.
        self._lock = threading.RLock()

    def add(self, stage: str, dataset: str, method: str, item_id: str, detail: str = "") -> None:
        event = FilterEvent(stage, dataset, method, str(item_id), detail)
        with self._lock:
            self.events.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def excluded_union(self, dataset: str | None = None, stages: Iterable[str] = ("query", "qa", "compile")) -> set[str]:
        stages = set(stages)
        with self._lock:
            return {
                e.item_id for e in self.events
                if e.stage in stages and (dataset is None or e.dataset == dataset)
            }

    def by_method(self, dataset: str | None = None) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for e in self.events:
                if dataset is None or e.dataset == dataset:
                    out[e.method] = out.get(e.method, 0) + 1
        return out

    def question_blocked(self, dataset: str, method: str, item_id: str) -> bool:
        """Whether any query-time Azure call for this item hit the policy filter."""
        with self._lock:
            return any(e.dataset == dataset and e.method == method and
                       e.item_id == item_id and e.stage not in {"index", "pilot.preflight"}
                       for e in self.events)

    def indexing_blocked(self, dataset: str | None = None) -> set[str]:
        """Passagens que o filtro recusou durante a extração."""
        with self._lock:
            return {
                e.item_id for e in self.events
                if e.stage == "index" and (dataset is None or e.dataset == dataset)
            }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            per_stage: dict[str, int] = {}
            for e in self.events:
                per_stage[e.stage] = per_stage.get(e.stage, 0) + 1
            return {"total": len(self.events), "por_estagio": per_stage,
                    "por_metodo": self.by_method()}


LEDGER = Ledger()


def configure_ledger(path: Path) -> Ledger:
    LEDGER.events.clear()
    LEDGER.path = path
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                LEDGER.events.append(FilterEvent(**json.loads(line)))
            except (ValueError, TypeError):
                continue
    return LEDGER
