"""
O modelo que responde ("working generator" do GAM), com os prompts literais
do código de avaliação (research/eval/*_test.py; licença MIT, Copyright (c)
2025 VectorSpaceLab). Os espaços no fim da primeira e da segunda linha dos
prompts de HotpotQA e NarrativeQA existem no original e foram mantidos.

Chamada: uma mensagem de usuário, sem system prompt, temperatura 0,3, teto de
256 tokens; do texto devolvido fica o que vem depois do último "</think>", sem
espaços nas pontas. É o que `OpenAIGenerator.generate_single` faz.

O contexto é o que o método entrega. No GAM é o resumo da pesquisa; aqui são
as páginas escolhidas pelo método, na ordem em que ele as entrega, separadas
por uma linha em branco (cada página já começa com "[Session i]").
"""

from __future__ import annotations

from typing import Any, Sequence

from wrag.gambench import protocol as P
from wrag.llm import GenParams

# Escritos com \n explícito: os espaços no fim das duas primeiras linhas são do
# original e sumiriam num editor que apara espaços finais.
_HEADER = ("Use the given Context. \n"
           "Answer with ONLY the final answer string; no extra words.\n\n"
           "Question:\n{question}\n\nContext:\n{summary}\n\nAnswer:\n")
HOTPOT_TEMPLATE = "You are a careful multi-hop reading assistant. \n" + _HEADER
NARRATIVEQA_TEMPLATE = "You are a careful reading assistant. \n" + _HEADER

RULER_TEMPLATE = "Read the text below and answer a question. Context: {summary}\n\n{question_prompt}\n\nAnswer:"


def ruler_question_prompt(question: str, example: str) -> str:
    """`build_question_prompt`: a pergunta e, depois, o exemplo (VT e CWE)."""
    parts = []
    question = (question or "").strip()
    if question:
        parts.append("Question:\n" + question)
    example = (example or "").strip()
    if example:
        parts.append("Here is the example:\n" + example)
    return "\n\n".join(parts)


def join_context(pages: Sequence[str]) -> str:
    return "\n\n".join(pages)


def build_prompt(benchmark: str, question: str, context: str, example: str = "") -> str:
    if benchmark == "hotpotqa":
        return HOTPOT_TEMPLATE.format(summary=context, question=question)
    if benchmark == "narrativeqa":
        return NARRATIVEQA_TEMPLATE.format(summary=context, question=question)
    if benchmark == "ruler":
        return RULER_TEMPLATE.format(summary=context,
                                     question_prompt=ruler_question_prompt(question, example))
    raise ValueError(f"benchmark desconhecido: {benchmark}")


def reader_params(seed: int | None = P.READER_SEED) -> GenParams:
    return GenParams(temperature=P.READER_TEMPERATURE, max_tokens=P.READER_MAX_TOKENS,
                     seed=seed, json_mode=False, exact_max_tokens=True)


def clean_response(text: str) -> str:
    """`text.split('</think>')[-1]` do gerador, seguido do `.strip()` do chamador."""
    return (text or "").split("</think>")[-1].strip()


def reader_record(result: Any) -> dict[str, Any]:
    return {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason, "cached": result.cached,
            "latency_s": round(result.latency_s, 3)}
