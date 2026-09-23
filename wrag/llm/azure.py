"""
Backend Azure OpenAI para o gateway corporativo (Petrobras).

Adaptado do `azure.py` da grade RMCQ, que já sobreviveu a este gateway. As três
defesas de lá continuam valendo, e uma quarta foi acrescentada:

1. **Modelos de reasoning gastam orçamento pensando.** gpt-5/o1/o3/o4 usam
   `max_completion_tokens`, recusam `temperature` e `seed`, e consomem parte do
   teto raciocinando ANTES de escrever. Um teto pensado para uma resposta curta
   volta como `content=""` com `finish_reason="length"`. Daí o piso
   `WRAG_AZURE_REASONING_MIN_TOKENS`.
2. **Gateways rejeitam parâmetros opcionais** que o modelo suporta. Um 400 que
   cita o nome do parâmetro derruba aquele parâmetro e repete a chamada.
3. **Resposta vazia não é abstenção.** Vazio inesperado é diagnosticado; vazio
   por `length` é registrado como item esgotado.
4. **Bloqueio de conteúdo é um resultado, não uma exceção.** O item volta com
   `filtered=True` e o lote continua. É o comportamento que o benchmark precisa:
   uma passagem recusada não pode derrubar a indexação de 2000 passagens.

O cache em disco não é só economia. Indexar um corpus de 2000 passagens custa
2000 chamadas; sem cache, qualquer queda no meio joga fora tudo que já foi pago,
e um rerun de método (que reusa a MESMA extração) pagaria de novo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from wrag import config as C
from wrag.llm.base import LLM, GenParams, LLMResult
from wrag.llm.filters import is_content_filter_error

log = logging.getLogger("wrag.llm.azure")

REASONING_MARKERS = ("gpt-5", "gpt5", "o1-", "o3-", "o4-", "-o1", "-o3", "-o4")

# Parâmetros sacrificáveis se o gateway reclamar. Lista fechada de propósito:
# `model` e `messages` nunca saem.
OPTIONAL_PARAMS = (
    "seed", "temperature", "top_p", "frequency_penalty", "presence_penalty",
    "response_format", "max_tokens", "max_completion_tokens", "reasoning_effort",
)

RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)
RETRY_NAMES = ("ratelimit", "timeout", "apiconnection", "serviceunavailable", "internalserver")


def is_reasoning_deployment(name: str) -> bool:
    """A checagem é sobre o nome do DEPLOYMENT, não do modelo oficial:
    deployments corporativos vêm com prefixo/sufixo próprio, e é o nome do
    deployment que o código tem em mãos."""
    n = (name or "").lower()
    return any(m in n for m in REASONING_MARKERS)


def token_budget(max_tokens: int, reasoning: bool, reasoning_min: int) -> int:
    """Teto de tokens de resposta. 0 significa "não mande cap nenhum"."""
    if not reasoning:
        return max_tokens
    if reasoning_min <= 0:
        return 0
    return max(max_tokens, reasoning_min)


def _status_of(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _is_transient(exc: Exception) -> bool:
    if _status_of(exc) in RETRY_STATUS:
        return True
    name = type(exc).__name__.lower()
    return any(marker in name for marker in RETRY_NAMES)


def _retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        if "retry-after-ms" in headers:
            return float(headers["retry-after-ms"]) / 1000.0
        if "retry-after" in headers:
            return float(headers["retry-after"])
    except (TypeError, ValueError):
        pass
    return 0.0


class AzureEmptyResponse(RuntimeError):
    """Resposta vazia do Azure. É falha de configuração, não abstenção."""


class AzureLLM(LLM):
    name = "azure"

    def cache_identity(self) -> str:
        """Separate deployments at different gateways/API versions, not keys."""
        scope = {"base_url": os.environ.get(C.AZURE_BASE_URL_VAR, ""),
                 "endpoint": os.environ.get(C.AZURE_ENDPOINT_VAR, ""),
                 "api_version": C.AZURE_API_VERSION,
                 "deployment": self.deployment}
        return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()

    def __init__(
        self,
        deployment: str | None = None,
        concurrency: int | None = None,
        max_tokens: int | None = None,
        use_cache: bool | None = None,
    ) -> None:
        super().__init__()
        self.deployment = deployment or C.AZURE_DEPLOYMENT
        self.reasoning = is_reasoning_deployment(self.deployment)
        self.concurrency = max(1, int(concurrency or C.AZURE_CONCURRENCY))
        self.max_tokens = int(max_tokens or C.AZURE_MAX_TOKENS)
        self.use_cache = C.LLM_CACHE if use_cache is None else use_cache

        self._unsupported: set[str] = set()
        self._lock = threading.Lock()
        self._calls = 0
        self._empties = 0
        self._filters = 0
        self._client = self._make_client()

        log.info(
            "AzureLLM deployment=%s modo=%s concorrência=%d teto=%d tokens%s",
            self.deployment, "reasoning" if self.reasoning else "chat",
            self.concurrency,
            token_budget(self.max_tokens, self.reasoning, C.AZURE_REASONING_MIN_TOKENS),
            " cache=on" if self.use_cache else "",
        )

    # -- cliente ------------------------------------------------------------

    def _make_client(self) -> Any:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "o SDK da OpenAI não está instalado. Rode: pip install -r requirements-azure.txt"
            ) from exc

        api_key = os.environ.get(C.AZURE_API_KEY_VAR, "").strip()
        base_url = os.environ.get(C.AZURE_BASE_URL_VAR, "").strip()
        endpoint = os.environ.get(C.AZURE_ENDPOINT_VAR, "").strip()

        if not api_key:
            raise RuntimeError(
                f"{C.AZURE_API_KEY_VAR} ausente. Defina no .env desta máquina "
                f"(veja .env.example) — nunca commite o .env."
            )
        if not base_url and not endpoint:
            raise RuntimeError(
                f"defina {C.AZURE_BASE_URL_VAR} ou {C.AZURE_ENDPOINT_VAR} no .env.\n"
                f"  {C.AZURE_BASE_URL_VAR}: a URL é usada como está — é o que gateway "
                f"corporativo costuma exigir.\n"
                f"  {C.AZURE_ENDPOINT_VAR}: o SDK monta /openai/deployments/<modelo>/... em cima dela."
            )

        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "api_version": C.AZURE_API_VERSION,
            # O retry é nosso: precisamos intercalar queda de parâmetro rejeitado
            # com backoff, e o SDK não sabe fazer isso.
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
            if endpoint:
                log.warning("%s e %s definidos; usando %s (mutuamente exclusivos no SDK)",
                            C.AZURE_BASE_URL_VAR, C.AZURE_ENDPOINT_VAR, C.AZURE_BASE_URL_VAR)
        else:
            kwargs["azure_endpoint"] = endpoint

        http_client = self._make_http_client()
        if http_client is not None:
            kwargs["http_client"] = http_client
        return AzureOpenAI(**kwargs)

    def _make_http_client(self) -> Any:
        """Cliente HTTP com o certificado raiz corporativo, quando houver.

        Rede com inspeção TLS apresenta certificado assinado pela CA da empresa.
        Sem esse PEM a verificação falha e a conexão nem chega ao Azure — um erro
        de SSL que não se parece nada com problema de API.
        """
        if not C.AZURE_CA_BUNDLE:
            return None
        caminho = Path(C.AZURE_CA_BUNDLE)
        if not caminho.is_absolute():
            caminho = C.ROOT / caminho
        if not caminho.exists():
            raise RuntimeError(
                f"AZURE_OPENAI_CA_BUNDLE aponta para {caminho}, que não existe.\n"
                f"Copie o PEM da CA raiz para essa máquina, ou deixe a variável vazia."
            )
        import httpx

        log.info("usando certificado raiz corporativo: %s", caminho)
        try:
            return httpx.Client(verify=str(caminho),
                                timeout=httpx.Timeout(C.AZURE_TIMEOUT_S, connect=30.0))
        except ssl.SSLError as exc:
            raise RuntimeError(
                f"{caminho} não é um certificado PEM válido ({exc}).\n"
                f"O arquivo precisa começar com '-----BEGIN CERTIFICATE-----'."
            ) from exc

    # -- montagem da chamada ------------------------------------------------

    def build_kwargs(self, messages: list[dict[str, str]], params: GenParams) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.deployment, "messages": messages}
        cap = token_budget(max(self.max_tokens, params.max_tokens), self.reasoning,
                           C.AZURE_REASONING_MIN_TOKENS)
        if cap > 0:
            kwargs["max_completion_tokens" if self.reasoning else "max_tokens"] = cap

        if not self.reasoning:
            kwargs["temperature"] = params.temperature
            if params.top_p and params.top_p < 1.0:
                kwargs["top_p"] = params.top_p
            if params.seed is not None:
                kwargs["seed"] = params.seed
        elif C.AZURE_REASONING_EFFORT:
            kwargs["reasoning_effort"] = C.AZURE_REASONING_EFFORT

        if params.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for name in self._unsupported:
            kwargs.pop(name, None)
        return kwargs

    def _maybe_drop_parameter(self, exc: Exception, kwargs: dict[str, Any]) -> bool:
        """400/422 citando o nome de um parâmetro opcional: derruba e repete."""
        if _status_of(exc) not in (400, 422):
            return False
        parts = [str(exc)]
        for attribute in ("body", "message", "code"):
            value = getattr(exc, attribute, None)
            if value:
                parts.append(json.dumps(value, ensure_ascii=False, default=str))
        response = getattr(exc, "response", None)
        if response is not None:
            parts.append(str(getattr(response, "text", "") or ""))
        message = " ".join(parts)
        # This 400 means the prompt omitted the literal word JSON. The
        # response_format parameter is supported; dropping it would silently
        # weaken every subsequent structured call in this process.
        if "must contain the word 'json'" in message.casefold():
            return False
        for name in OPTIONAL_PARAMS:
            if name in self._unsupported or name not in kwargs:
                continue
            if re.search(rf"\b{re.escape(name)}\b", message, flags=re.IGNORECASE):
                with self._lock:
                    self._unsupported.add(name)
                kwargs.pop(name, None)
                log.warning("gateway rejeitou %r; removido e não será mais enviado (%s)",
                            name, message.splitlines()[0][:160])
                return True
        return False

    # -- cache --------------------------------------------------------------

    def _cache_path(self, kwargs: dict[str, Any]) -> Path:
        key = json.dumps(
            {k: kwargs.get(k) for k in
             ("model", "messages", "temperature", "top_p", "max_tokens",
              "max_completion_tokens", "seed", "reasoning_effort", "response_format")},
            sort_keys=True, ensure_ascii=False,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return C.CACHE_DIR / "azure" / self.cache_identity()[:16] / digest[:2] / f"{digest}.json"

    def _cache_read(self, path: Path) -> LLMResult | None:
        if not self.use_cache or not path.exists():
            return None
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # cache corrompido é como cache ausente
        return LLMResult(
            text=row.get("text", ""),
            prompt_tokens=row.get("prompt_tokens", 0),
            completion_tokens=row.get("completion_tokens", 0),
            latency_s=row.get("latency_s", 0.0),
            finish_reason=row.get("finish_reason", ""),
            filtered=row.get("filtered", False),
            exhausted=row.get("exhausted", False),
            cached=True,
        )

    def _cache_write(self, path: Path, result: LLMResult) -> None:
        # Itens filtrados NÃO entram no cache de longa duração: uma mudança de
        # política ou de deployment deve poder tentá-los de novo.
        if not self.use_cache or result.filtered:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "text": result.text, "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens, "latency_s": result.latency_s,
                "finish_reason": result.finish_reason, "filtered": result.filtered,
                "exhausted": result.exhausted, "deployment": self.deployment,
            }, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            log.warning("não consegui gravar o cache em %s: %s", path, exc)

    # -- uma chamada --------------------------------------------------------

    def _complete(self, messages: list[dict[str, str]], params: GenParams,
                  stage: str = "misc") -> LLMResult:
        kwargs = self.build_kwargs(messages, params)
        cache_path = self._cache_path(kwargs)
        cached = self._cache_read(cache_path)
        if cached is not None:
            return cached

        attempt = 0
        last_exc: Exception | None = None

        while attempt <= C.AZURE_MAX_RETRIES:
            started = time.perf_counter()
            try:
                response = self._client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classificado logo abaixo
                last_exc = exc
                if is_content_filter_error(exc) and C.CONTINUE_ON_CONTENT_FILTER:
                    return self._filtered_result(str(exc), cache_path)
                if self._maybe_drop_parameter(exc, kwargs):
                    continue  # não conta como tentativa: a chamada mudou
                if not _is_transient(exc) or attempt == C.AZURE_MAX_RETRIES:
                    self._explain_fatal(exc)
                    raise
                delay = min(C.AZURE_BACKOFF_MAX, C.AZURE_BACKOFF_BASE ** attempt) * (0.5 + random.random())
                time.sleep(max(delay, _retry_after_seconds(exc)))
                attempt += 1
                continue

            result = self._to_result(response, time.perf_counter() - started)
            self._check_empty(result, response)
            self._cache_write(cache_path, result)
            return result

        raise RuntimeError(f"chamada ao Azure falhou após {C.AZURE_MAX_RETRIES} tentativas") from last_exc

    def _filtered_result(self, detail: str, cache_path: Path) -> LLMResult:
        with self._lock:
            self._calls += 1
            self._filters += 1
            calls, filters = self._calls, self._filters
        log.warning("prompt bloqueado pelo filtro de conteúdo (%d/%d); item marcado e o lote segue: %s",
                    filters, calls, detail.splitlines()[0][:200])
        if calls >= C.HEALTH_CHECK_CALLS and filters / calls > C.MAX_FILTER_RATE:
            log.warning("taxa de filtro %d/%d = %.1f%%; perguntas bloqueadas serão "
                        "registradas e excluídas da avaliação, sem parar o lote",
                        filters, calls, 100 * filters / calls)
        return LLMResult(text="", finish_reason="content_filter", filtered=True)

    def _explain_fatal(self, exc: Exception) -> None:
        status = _status_of(exc)
        if status == 404:
            log.error(
                "404 do Azure para deployment=%r.\n"
                "  'Resource Not Found' aqui tem três causas possíveis:\n"
                "   1. o nome do deployment não existe nesse recurso (mais comum);\n"
                "   2. o endpoint aponta para outro recurso, ou veio com caminho sobrando;\n"
                "   3. AZURE_OPENAI_API_VERSION é antiga demais (atual: %s; gpt-5 precisa de 2025+).",
                self.deployment, C.AZURE_API_VERSION,
            )
        elif status == 401:
            log.error("401: AZURE_OPENAI_API_KEY inválida, ou é a chave de outro recurso.")
        elif status == 403:
            log.error("403: chave válida sem permissão neste deployment, ou restrição de rede/IP.")

    def _to_result(self, response: Any, latency_s: float) -> LLMResult:
        choice = response.choices[0] if response.choices else None
        text = (getattr(getattr(choice, "message", None), "content", None) or "").strip()
        usage = getattr(response, "usage", None)
        finish = getattr(choice, "finish_reason", "") or ""
        return LLMResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency_s,
            finish_reason=finish,
            filtered=(finish == "content_filter"),
        )

    def _check_empty(self, result: LLMResult, response: Any) -> None:
        with self._lock:
            self._calls += 1
            if result.filtered:
                self._filters += 1
                return
            if result.text:
                return
            self._empties += 1
            calls, empties = self._calls, self._empties

        if result.finish_reason == "length":
            result.exhausted = True
            log.warning("resposta vazia por limite de comprimento; item marcado como esgotado. %s",
                        self._diagnose(result, response))
            return

        rate = empties / calls
        log.warning("resposta vazia (%d/%d). %s", empties, calls, self._diagnose(result, response))
        if calls >= C.HEALTH_CHECK_CALLS and rate > 0.10:
            raise AzureEmptyResponse(
                f"{self.deployment}: {empties} de {calls} respostas vazias ({rate:.0%}). "
                f"{self._diagnose(result, response)}"
            )

    def _diagnose(self, result: LLMResult, response: Any) -> str:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        if result.finish_reason == "length" and self.reasoning:
            return (
                f"O modelo gastou o orçamento inteiro raciocinando (reasoning_tokens="
                f"{reasoning_tokens}) e não sobrou nada para a resposta. Suba "
                f"WRAG_AZURE_REASONING_MIN_TOKENS (tente 12000) ou baixe WRAG_AZURE_REASONING_EFFORT. "
                f"Orçamento atual: {token_budget(self.max_tokens, True, C.AZURE_REASONING_MIN_TOKENS)}."
            )
        if result.finish_reason == "length":
            return f"finish_reason='length': suba WRAG_AZURE_MAX_TOKENS (atual: {self.max_tokens})."
        return (f"finish_reason={result.finish_reason!r}. Confira se WRAG_AZURE_DEPLOYMENT "
                f"aponta para um deployment que existe neste recurso.")

    # -- lote ---------------------------------------------------------------

    def chat_many(
        self,
        prompts: Sequence[str],
        system: str | None = None,
        params: GenParams | None = None,
        stage: str = "misc",
        desc: str = "",
    ) -> list[LLMResult]:
        from wrag.util import progress

        if not prompts:
            return []
        params = params or GenParams()
        results: list[LLMResult | None] = [None] * len(prompts)
        # Falhar rápido tem que ser rápido de verdade: sem esta bandeira, um
        # deployment mal configurado detectado no primeiro prompt ainda pagaria
        # os outros do lote, porque o executor espera todo mundo terminar.
        aborted = threading.Event()

        def work(i: int) -> int:
            if aborted.is_set():
                raise RuntimeError("lote abortado por falha anterior")
            try:
                results[i] = self.chat(prompts[i], system=system, params=params, stage=stage)
            except BaseException:
                aborted.set()
                raise
            return i

        if self.concurrency == 1:
            for i in progress(range(len(prompts)), desc=desc or stage):
                work(i)
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(work, i) for i in range(len(prompts))]
                try:
                    for future in progress(futures, desc=desc or stage):
                        future.result()
                except BaseException:
                    aborted.set()
                    for f in futures:
                        f.cancel()
                    raise

        missing = [i for i, g in enumerate(results) if g is None]
        if missing:
            raise RuntimeError(f"{len(missing)} gerações não voltaram (índices {missing[:5]}...)")
        return [r for r in results if r is not None]

    def count_tokens(self, text: str) -> int:
        encoder = _encoder()
        return len(encoder.encode(text)) if encoder is not None else max(1, len(text) // 4)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


_ENCODER: Any = None
_ENCODER_TRIED = False


def _encoder() -> Any:
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 - tiktoken é opcional
        log.debug("tiktoken indisponível; contagem de tokens fica aproximada")
        _ENCODER = None
    return _ENCODER
