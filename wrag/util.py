"""Utilidades pequenas: log, progresso, hashing, normalização de texto."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("WRAG_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def progress(iterable: Iterable[T], desc: str = "", total: int | None = None) -> Iterable[T]:
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc or None, total=total, leave=False,
                    disable=os.environ.get("WRAG_NO_PROGRESS") == "1")
    except ImportError:
        return iterable


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    holder = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["elapsed"] = time.perf_counter() - start


def sha(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def canonical_symbol(text: str) -> str:
    """Identidade lexical conservadora; C, C++ e C# permanecem distintos.

    Alias semânticos exigem resolução explícita. Similaridade não é igualdade.
    """
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "").casefold()).strip()


def normalize(text: str) -> str:
    """Forma canônica de uma menção de entidade ou relação.

    Deliberadamente agressiva (minúsculas, sem acento, sem pontuação): a
    resolução de entidades depois disso é por embedding, e um casamento exato
    barato aqui evita gastar similaridade com diferenças puramente ortográficas.
    """
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def normalize_answer(text: str) -> str:
    """EM/F1: minúsculas, remoção de pontuação ASCII, artigos e espaços extras."""
    import string
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return _WS.sub(" ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def write_json(path: Path, obj: Any) -> None:
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    # Um encerramento por prazo não deve destruir o último checkpoint completo.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}-", suffix=".tmp", delete=False) as fh:
        temporary = Path(fh.name)
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def chunked(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
