"""
Tipos de resposta com um modelo de NER de rótulos livres (GLiNER).

O plano pode declarar o tipo da variável de resposta (?x : "martial art") ou
só a classe grossa (expected_type: person, place, organization). O cosseno
entre a resposta e o tipo não separa o que é do tipo do que não é (medido no
v4: "kickboxing"/"martial art" = 0,69 contra "basketball"/"martial art" =
0,65). Aqui o tipo é decidido por um modelo de NER que recebe a FALA DE ORIGEM
do fato e uma lista de rótulos escolhida na hora: o rótulo pedido e alguns
rótulos de contraste. A resposta é do tipo se o trecho da fala que a contém
recebe o rótulo pedido.

Nenhum LLM é chamado e nenhum rótulo do benchmark é lido: só a fala e o tipo
escrito pelo planejador. Resultado por resposta:

* score em [0, 1]: nota do rótulo pedido no trecho que contém a resposta, 0
  se o trecho foi rotulado com outra coisa ou não foi rotulado;
* None quando a resposta não foi localizada na fala (a extração parafraseou):
  sem evidência, o chamador não veta.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Iterable, Sequence

_STOP = frozenset("a an the of my his her their our your its to in on at for and or with "
                  "some any this that these those".split())

# Rótulos de contraste: só classes grossas e disjuntas. Contrastes próximos do
# tipo pedido ("sport" contra "martial art") roubavam o rótulo de respostas
# certas; por isso a predição é multi-rótulo e a nota é a do rótulo pedido.
CONTRAST = ("person", "location", "organization", "date")

EXPECTED_TO_LABEL = {"person": "person", "place": "location", "organization": "organization"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9'\-]*", (text or "").lower())
            if t not in _STOP]


class GlinerTyper:
    """Carrega o modelo uma vez; chamadas protegidas por lock (CPU, threads)."""

    def __init__(self, model: str = "urchade/gliner_medium-v2.1", threshold: float = 0.1) -> None:
        self.model_name = model
        self.threshold = threshold
        self._model = None
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, tuple[str, ...]], list[dict]] = {}

    def _load(self):
        if self._model is None:
            from gliner import GLiNER  # import tardio: dependência opcional
            self._model = GLiNER.from_pretrained(self.model_name)
        return self._model

    def entities(self, text: str, labels: Sequence[str]) -> list[dict]:
        key = (text, tuple(labels))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            model = self._load()
            found = model.predict_entities(text[:2000], list(labels), threshold=self.threshold,
                                           flat_ner=False, multi_label=True)
            self._cache[key] = found
            return found

    @staticmethod
    def labels_for(kind: str) -> list[str]:
        low = kind.strip().lower()
        return [kind] + [c for c in CONTRAST if c != low and c not in low and low not in c]

    def score(self, answer: str, texts: Iterable[str], kind: str) -> float | None:
        """Nota do tipo ``kind`` para ``answer`` nas falas ``texts``."""
        wanted = set(_tokens(answer))
        if not wanted or not kind.strip():
            return None
        labels = self.labels_for(kind)
        located = False
        best = 0.0
        for text in texts:
            if not text or not wanted & set(_tokens(text)):
                continue
            located = True
            for entity in self.entities(text, labels):
                span = set(_tokens(entity.get("text", "")))
                if not span or not (span & wanted):
                    continue
                # o trecho rotulado precisa cobrir a maior parte da resposta
                if len(span & wanted) / len(wanted) < 0.5:
                    continue
                if entity.get("label") == labels[0]:
                    best = max(best, float(entity.get("score", 0.0)))
        return best if located else None


@lru_cache(maxsize=4)
def get_typer(model: str, threshold: float) -> GlinerTyper:
    return GlinerTyper(model, threshold)
