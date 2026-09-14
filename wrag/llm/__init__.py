"""Fábrica de backends de LLM."""

from __future__ import annotations

import logging

from wrag import config as C
from wrag.llm.base import LLM, GenParams, LLMResult, parse_json_loose

log = logging.getLogger("wrag.llm")

__all__ = ["LLM", "GenParams", "LLMResult", "parse_json_loose", "get_llm"]

_CACHE: dict[str, LLM] = {}


def get_llm(backend: str | None = None, **kwargs) -> LLM:
    """Devolve o backend configurado. Instâncias são reutilizadas dentro do
    processo para que o cache em disco e a contabilidade de uso sejam únicos."""
    backend = (backend or C.LLM_BACKEND).lower()
    key = backend + repr(sorted(kwargs.items()))
    if key in _CACHE:
        return _CACHE[key]

    if backend == "azure":
        from wrag.llm.azure import AzureLLM

        llm: LLM = AzureLLM(**kwargs)
    elif backend in ("openai", "vllm", "openai_compat"):
        from wrag.llm.openai_compat import OpenAICompatLLM

        llm = OpenAICompatLLM(**kwargs)
    elif backend == "stub":
        from wrag.llm.stub import StubLLM

        llm = StubLLM(**kwargs)
    else:
        raise ValueError(f"backend de LLM desconhecido: {backend!r} "
                         f"(use azure, openai, vllm ou stub)")

    _CACHE[key] = llm
    return llm
