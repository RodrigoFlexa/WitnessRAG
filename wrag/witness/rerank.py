"""
Cross-encoder reranker for the fact selection (docs/plano-robusto.md).

The reranker replaces the bi-encoder similarity when choosing which facts the
reader gets: it reads the question and the candidate together (the fact
sentence plus its source turn) and returns a relevance in [0, 1]. It never sees
an answer or a benchmark label. Scores are cached on disk by (model, question,
text), so repeated runs over the same memory cost nothing.

Off by default (``WitnessConfig.fact_rerank = ""``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Sequence

from wrag import config as C
from wrag.util import get_logger

log = get_logger("wrag.witness.rerank")

_LOCK = threading.Lock()
_INSTANCES: dict[str, "Reranker"] = {}


def _key(model: str, question: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{question}\x00{text}".encode("utf-8")).hexdigest()


class Reranker:
    def __init__(self, model: str, device: str | None = None, max_length: int = 384,
                 batch_size: int = 16) -> None:
        self.model_name = model
        self.device = device or os.environ.get("WRAG_EMBED_DEVICE", "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, float] = {}
        folder = Path(C.CACHE_DIR) / "rerank"
        folder.mkdir(parents=True, exist_ok=True)
        self._path = folder / (model.replace("/", "_") + ".jsonl")
        if self._path.exists():
            with self._path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        self._cache[row["k"]] = float(row["s"])
                    except (ValueError, KeyError):
                        continue
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            log.info("carregando reranker %s (%s)", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, max_length=self.max_length,
                                       device=self.device)
        return self._model

    def score(self, question: str, texts: Sequence[str]) -> list[float]:
        """Relevance of each text to the question, in [0, 1] (sigmoid of the logit)."""
        keys = [_key(self.model_name, question, t) for t in texts]
        with self._lock:
            missing = [i for i, k in enumerate(keys) if k not in self._cache]
            if missing:
                model = self._load()
                pairs = [(question, texts[i]) for i in missing]
                import torch
                # Raw logits, squashed here: the same scale whatever default
                # activation the installed sentence-transformers applies.
                outputs = [float(x) for x in model.predict(
                    pairs, batch_size=self.batch_size, show_progress_bar=False,
                    activation_fn=torch.nn.Identity())]
                rows = []
                for i, out in zip(missing, outputs):
                    value = 1.0 / (1.0 + math.exp(-out))
                    self._cache[keys[i]] = value
                    rows.append(json.dumps({"k": keys[i], "s": round(value, 6)}))
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(rows) + "\n")
            return [self._cache[k] for k in keys]


def get_reranker(model: str) -> Reranker:
    with _LOCK:
        if model not in _INSTANCES:
            _INSTANCES[model] = Reranker(model)
        return _INSTANCES[model]
