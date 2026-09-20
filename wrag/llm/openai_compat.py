"""
Backend para qualquer endpoint compatível com a API da OpenAI (api.openai.com,
vLLM local, LiteLLM). Herda do AzureLLM porque o que muda é só a construção do
cliente: retry, queda de parâmetro, cache e tratamento de filtro são idênticos.
"""

from __future__ import annotations

import os
from typing import Any

from wrag import config as C
from wrag.llm.azure import AzureLLM
from wrag.util import sha


class OpenAICompatLLM(AzureLLM):
    name = "openai"

    def __init__(self, deployment: str | None = None, **kwargs):
        super().__init__(deployment=deployment or os.environ.get("OPENAI_MODEL") or C.AZURE_DEPLOYMENT,
                         **kwargs)

    def build_kwargs(self, messages, params):
        kwargs = super().build_kwargs(messages, params)
        # O piso corporativo de tokens não deve ampliar cada tarefa do vLLM.
        if not self.reasoning:
            kwargs["max_tokens"] = params.max_tokens
        return kwargs

    def cache_identity(self):
        # Explicit experiment namespace: transport ports must not redefine the
        # model or the frozen extraction. Ordinary runs keep the legacy identity.
        experiment = os.environ.get("WRAG_EXPERIMENT_CACHE_ID")
        if experiment:
            return sha(self.name, self.deployment, "controlled-v1", experiment,
                       os.environ.get("WRAG_MODEL_REVISION", ""))
        return sha(self.name, self.deployment, os.environ.get("OPENAI_BASE_URL", ""),
                   os.environ.get("WRAG_MODEL_REVISION", ""))

    def _cache_path(self, kwargs):
        parent = C.CACHE_DIR / "openai" / self.cache_identity()[:16]
        return parent / f"{sha(kwargs)}.json"

    def _make_client(self) -> Any:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "sk-no-key-required"
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        return OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                      timeout=C.AZURE_TIMEOUT_S)
