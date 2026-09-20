"""Explain where a proof attempt stopped using facts already logged per question."""
from __future__ import annotations


def stop_reason(diagnostics: dict) -> str:
    status = diagnostics.get("classe_prova")
    if status == "full" or ("fallback" not in diagnostics and status is None):
        return "junção e plano aceitos"
    if status == "provisional":
        return "hipótese provisória"
    plans = diagnostics.get("planos_compilados") or []
    if not plans:
        return "sem plano"
    if not any(p.get("executavel") for p in plans):
        return "nenhum plano executável"
    if not any(p.get("fechou") for p in plans):
        return "nenhuma junção completa"
    if not any(p.get("fechou") and (p.get("obrigacoes") or {}).get("cobre_pergunta")
               for p in plans):
        return "junção fechou; cobertura rejeitada"
    if (diagnostics.get("verificacao") or {}).get("avaliadas"):
        return "verificação textual rejeitou"
    return "outra parada"
