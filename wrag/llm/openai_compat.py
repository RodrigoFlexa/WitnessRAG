"""
Backend para qualquer endpoint compatível com a API da OpenAI (api.openai.com,
vLLM local, LiteLLM). Herda do AzureLLM porque o que muda é só a construção do
cliente: retry, queda de parâmetro, cache e tratamento de filtro são idênticos.

Dois modos, decididos por OPENAI_BASE_URL:

* **API pública** (OPENAI_BASE_URL vazio ou apontando para api.openai.com): exige
  OPENAI_API_KEY de verdade e segue o perfil de tokens do Azure (mesmo teto
  mínimo de resposta), para que gpt-4o-mini via API seja comparável às rodadas
  com gpt-4.1-mini no gateway.
* **Servidor compatível** (vLLM/LiteLLM em outra URL): chave opcional e teto de
  tokens exato por tarefa, como nas rodadas Qwen.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from wrag import config as C
from wrag.llm.azure import AzureLLM, _status_of
from wrag.util import sha

log = logging.getLogger("wrag.llm.openai")

OPENAI_API_HOST = "api.openai.com"


def is_public_openai(base_url: str | None) -> bool:
    """True quando o cliente vai falar com a API pública da OpenAI."""
    base_url = (base_url or "").strip()
    return not base_url or OPENAI_API_HOST in base_url.lower()


class OpenAICompatLLM(AzureLLM):
    name = "openai"

    def __init__(self, deployment: str | None = None, **kwargs):
        self.public_api = is_public_openai(os.environ.get("OPENAI_BASE_URL", ""))
        super().__init__(deployment=deployment or os.environ.get("OPENAI_MODEL") or C.AZURE_DEPLOYMENT,
                         **kwargs)

    def build_kwargs(self, messages, params):
        kwargs = super().build_kwargs(messages, params)
        # O piso corporativo de tokens não deve ampliar cada tarefa do vLLM.
        # Na API pública mantemos o perfil do Azure: é o mesmo protocolo das
        # rodadas registradas com gpt-4.1-mini.
        if not self.reasoning and not self.public_api:
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
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "o SDK da OpenAI não está instalado. Rode: pip install -r requirements.txt"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        if self.public_api and not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY ausente. Defina no .env desta máquina (veja .env.example) "
                "ou no ambiente; nunca commite o .env."
            )
        kwargs: dict[str, Any] = {"api_key": api_key or "sk-no-key-required",
                                  "base_url": base_url, "max_retries": 0,
                                  "timeout": C.AZURE_TIMEOUT_S}
        organization = os.environ.get("OPENAI_ORG_ID", "").strip()
        if organization:
            kwargs["organization"] = organization
        project = os.environ.get("OPENAI_PROJECT_ID", "").strip()
        if project:
            kwargs["project"] = project
        log.info("OpenAI %s model=%s", "API pública" if self.public_api else f"base_url={base_url}",
                 self.deployment)
        return OpenAI(**kwargs)

    def _explain_fatal(self, exc: Exception) -> None:
        if not self.public_api:
            return super()._explain_fatal(exc)
        status = _status_of(exc)
        if status == 401:
            log.error("401: OPENAI_API_KEY inválida ou revogada.")
        elif status == 403:
            log.error("403: a chave não tem acesso ao modelo %r (projeto/organização).",
                      self.deployment)
        elif status == 404:
            log.error("404: modelo %r não existe ou não está disponível para esta chave.",
                      self.deployment)
        elif status == 429:
            log.error("429 persistente: limite de taxa ou de crédito da conta. "
                      "Baixe WRAG_AZURE_CONCURRENCY ou confira o billing.")
