"""
Interface comum dos backends de LLM.

Uma decisão de contrato governa o resto do projeto: **bloqueio por política de
conteúdo não é exceção, é um resultado**. `LLMResult.filtered=True` com
`text=""` é um valor legítimo que atravessa o pipeline até o relatório, onde a
pergunta aparece na coluna `filtered` em vez de contaminar a média. Se o bloqueio
fosse uma exceção, o lote inteiro do gateway Petrobras cairia na primeira
passagem que o filtro não gostou.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

log = logging.getLogger("wrag.llm")


@dataclass
class GenParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int | None = 42
    json_mode: bool = False
    # Use max_tokens as given instead of max(backend ceiling, max_tokens). A
    # benchmark protocol that fixes the answer budget (GAM: 256) needs it; the
    # default keeps every existing request, and so every cache key, unchanged.
    exact_max_tokens: bool = False


@dataclass
class LLMResult:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str = ""
    filtered: bool = False        # bloqueado pela política de conteúdo
    exhausted: bool = False       # finish_reason='length' sem texto útil
    cached: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.filtered

    def json(self) -> Any | None:
        """Parse tolerante: modelos embrulham JSON em ``` ou em prosa."""
        if not self.text:
            return None
        return parse_json_loose(self.text)


def parse_json_loose(text: str) -> Any | None:
    """Extrai o primeiro objeto/lista JSON bem formado de um texto qualquer.

    Necessário porque `response_format=json_object` não está disponível em todo
    gateway, e porque modelos de reasoning às vezes prefixam a resposta.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # varredura por balanceamento: o primeiro { ou [ que fecha corretamente
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_string, escape = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


@dataclass
class UsageLedger:
    """Contabilidade de custo. A proposta exige medir custo de extração e de
    consulta separadamente, então cada chamada declara a que estágio pertence."""

    calls: dict[str, int] = field(default_factory=dict)
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    completion_tokens: dict[str, int] = field(default_factory=dict)
    cached_calls: dict[str, int] = field(default_factory=dict)
    filtered_calls: dict[str, int] = field(default_factory=dict)
    uncached_prompt_tokens: dict[str, int] = field(default_factory=dict)
    uncached_completion_tokens: dict[str, int] = field(default_factory=dict)
    latency_s: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, stage: str, result: LLMResult) -> None:
        with self._lock:
            self.calls[stage] = self.calls.get(stage, 0) + 1
            self.prompt_tokens[stage] = self.prompt_tokens.get(stage, 0) + result.prompt_tokens
            self.completion_tokens[stage] = self.completion_tokens.get(stage, 0) + result.completion_tokens
            self.latency_s[stage] = self.latency_s.get(stage, 0.0) + result.latency_s
            if result.cached:
                self.cached_calls[stage] = self.cached_calls.get(stage, 0) + 1
            else:
                self.uncached_prompt_tokens[stage] = self.uncached_prompt_tokens.get(stage, 0) + result.prompt_tokens
                self.uncached_completion_tokens[stage] = self.uncached_completion_tokens.get(stage, 0) + result.completion_tokens
            if result.filtered:
                self.filtered_calls[stage] = self.filtered_calls.get(stage, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = sorted(set(self.calls))
            return {
                "por_estagio": {
                    s: {
                        "chamadas": self.calls.get(s, 0),
                        "em_cache": self.cached_calls.get(s, 0),
                        "filtradas": self.filtered_calls.get(s, 0),
                        "tokens_prompt": self.prompt_tokens.get(s, 0),
                        "tokens_resposta": self.completion_tokens.get(s, 0),
                        "tokens_prompt_sem_cache": self.uncached_prompt_tokens.get(s, 0),
                        "tokens_resposta_sem_cache": self.uncached_completion_tokens.get(s, 0),
                        "latencia_s": round(self.latency_s.get(s, 0.0), 2),
                    }
                    for s in stages
                },
                "total": {
                    "chamadas": sum(self.calls.values()),
                    "tokens_prompt": sum(self.prompt_tokens.values()),
                    "tokens_resposta": sum(self.completion_tokens.values()),
                    "filtradas": sum(self.filtered_calls.values()),
                    "em_cache": sum(self.cached_calls.values()),
                    "tokens_prompt_sem_cache": sum(self.uncached_prompt_tokens.values()),
                    "tokens_resposta_sem_cache": sum(self.uncached_completion_tokens.values()),
                },
            }

    def reset(self) -> None:
        with self._lock:
            for d in (self.calls, self.prompt_tokens, self.completion_tokens,
                      self.cached_calls, self.filtered_calls, self.latency_s,
                      self.uncached_prompt_tokens, self.uncached_completion_tokens):
                d.clear()


def usage_delta(after: dict, before: dict) -> dict:
    def subtract(a, b):
        return {k: a.get(k, 0) - b.get(k, 0) for k in a.keys() | b.keys()}
    return {"total": subtract(after.get("total", {}), before.get("total", {})),
            "por_estagio": {s: subtract(after.get("por_estagio", {}).get(s, {}),
                                         before.get("por_estagio", {}).get(s, {}))
                            for s in after.get("por_estagio", {})}}


def sum_usage(snapshots) -> dict:
    out = {"total": {}, "por_estagio": {}}
    for snapshot in snapshots:
        for k, v in snapshot.get("total", {}).items():
            out["total"][k] = out["total"].get(k, 0) + v
        for stage, values in snapshot.get("por_estagio", {}).items():
            dest = out["por_estagio"].setdefault(stage, {})
            for k, v in values.items():
                dest[k] = dest.get(k, 0) + v
    return out


class LLM(ABC):
    """Backend de geração. `chat` é uma chamada; `chat_many` é um lote paralelo."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.usage = UsageLedger()

    @abstractmethod
    def _complete(self, messages: list[dict[str, str]], params: GenParams,
                  stage: str = "misc") -> LLMResult:
        """`stage` chega até aqui porque o backend `stub` despacha por tarefa.
        Backends reais ignoram o argumento."""
        ...

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        params: GenParams | None = None,
        stage: str = "misc",
    ) -> LLMResult:
        params = params or GenParams()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._complete(messages, params, stage)
        self.usage.record(stage, result)
        return result

    def chat_many(
        self,
        prompts: Sequence[str],
        system: str | None = None,
        params: GenParams | None = None,
        stage: str = "misc",
        desc: str = "",
    ) -> list[LLMResult]:
        """Default sequencial. Backends com concorrência sobrescrevem."""
        from wrag.util import progress

        return [
            self.chat(p, system=system, params=params, stage=stage)
            for p in progress(prompts, desc=desc or stage)
        ]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def close(self) -> None:
        return None

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
