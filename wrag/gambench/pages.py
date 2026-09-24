"""
Páginas do protocolo do GAM: 2.048 tokens do tokenizador do BGE-M3, sem
sobreposição, cada uma com o cabeçalho "[Session i]".

Porte literal de `_split_with_embedding_model` e `_smart_split_by_tokens`
(research/eval/*_test.py; licença MIT, Copyright (c) 2025 VectorSpaceLab),
inclusive nas duas peculiaridades do original:

* um texto que cabe numa página vira "[Session 1]\\n" + texto sem strip;
* quando há corte, a numeração começa em 0 e cada pedaço é decodificado com
  `skip_special_tokens=True` e passa por strip; pedaços vazios são pulados
  sem consumir número.

Diferença deliberada: o GAM, se o tokenizador do BGE-M3 falha ao carregar,
cai silenciosamente para o tiktoken do GPT-4o. Aqui a falha interrompe a
execução, porque páginas diferentes seriam outro protocolo.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any

from wrag.gambench import protocol as P

_LOCK = threading.Lock()
_TOKENIZERS: dict[tuple[str, str], Any] = {}

# Frase fixa cujo resultado identifica o tokenizador na manifest da rodada.
_PROBE = ("Document 1:\nThe Mosholu Parkway is a hybrid freeway-standard parkway "
          "(1935–1937) in the Bronx; VAR QPE = 64886. 李屏賓 uuid "
          "6e3b9d69-e808-4747-bd37-edbe79049679.")


def load_tokenizer(name: str = P.PAGE_TOKENIZER, revision: str = "") -> Any:
    key = (name, revision)
    with _LOCK:
        if key not in _TOKENIZERS:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name, revision=revision or None)
            # Só silencia o aviso de sequência longa; encode continua sem truncar.
            tokenizer.model_max_length = 10 ** 12
            _TOKENIZERS[key] = tokenizer
        return _TOKENIZERS[key]


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    ids = tokenizer.encode(_PROBE, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    try:
        import transformers
        version = transformers.__version__
    except ImportError:
        version = ""
    try:
        vocab = len(tokenizer)
    except TypeError:
        vocab = getattr(tokenizer, "vocab_size", None)
    return {"classe": type(tokenizer).__name__, "vocab": vocab, "transformers": version,
            "sonda_ids_sha256": hashlib.sha256(repr(ids).encode()).hexdigest()[:16],
            "sonda_decode_sha256": hashlib.sha256(decoded.encode()).hexdigest()[:16]}


@dataclass(frozen=True)
class Pages:
    pages: list[str]
    context_tokens: int


def split_pages(text: str, tokenizer: Any, max_tokens: int = P.PAGE_TOKENS) -> Pages:
    """`build_context_chunks_for_sample` do GAM com o tokenizador do BGE-M3."""
    if not text:
        return Pages([], 0)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return Pages([f"[Session 1]\n{text}"], len(tokens))
    chunks: list[str] = []
    session_id = 0
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_text = tokenizer.decode(tokens[start_idx:end_idx], skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(f"[Session {session_id}]\n{chunk_text.strip()}")
            session_id += 1
        start_idx = end_idx
    return Pages(chunks, len(tokens))
