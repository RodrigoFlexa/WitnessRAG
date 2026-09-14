"""
Leitura final: o mesmo leitor para os cinco sistemas.

O prompt, o teto de tokens, o formato do contexto e o número de passagens são
idênticos entre métodos. É a condição para que a coluna de F1 meça recuperação e
não engenharia de prompt — e é o que o próprio HippoRAG 2 faz ao usar o mesmo
leitor para todas as linhas da Tabela 2.

Uma nota sobre o WITNESS-RAG: ele TAMBÉM produz uma resposta estrutural (a
resposta da testemunha de menor custo), disponível em
`diagnostics["resposta_estrutural"]`. O relatório mostra as duas, mas a coluna
principal é a do leitor comum, porque é a única comparável.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from wrag import config as C
from wrag import prompts
from wrag.data import Corpus, Question
from wrag.llm import LLM, GenParams
from wrag.llm.filters import LEDGER
from wrag.util import get_logger

log = get_logger("wrag.eval.reader")


@dataclass
class ReadResult:
    answer: str = ""
    filtered: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def read(
    llm: LLM,
    corpus: Corpus,
    question: Question,
    pids: Sequence[str],
    cfg: C.QAConfig | None = None,
    method: str = "",
) -> ReadResult:
    cfg = cfg or C.QAConfig()
    passages = []
    for pid in pids[: cfg.top_k]:
        try:
            passage = corpus.get(pid)
        except KeyError:
            continue
        passages.append((passage.title, passage.text))

    result = llm.chat(
        prompts.QA_TEMPLATE.format(passages=prompts.format_passages(passages),
                                   question=question.question),
        system=prompts.QA_SYSTEM,
        params=GenParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens, json_mode=True),
        stage="qa",
    )
    if result.filtered:
        LEDGER.add("qa", question.dataset, method, question.qid, "leitura bloqueada")
        return ReadResult(filtered=True, latency_s=result.latency_s)

    data = result.json()
    answer = ""
    if isinstance(data, dict):
        answer = str(data.get("answer") or "").strip()
    elif result.text:
        # Modelo devolveu prosa apesar do pedido de JSON. A primeira linha é a
        # aposta menos ruim; contar como resposta vazia puniria o método por um
        # acidente de formatação.
        answer = result.text.strip().splitlines()[0][:200]

    return ReadResult(answer=answer, prompt_tokens=result.prompt_tokens,
                      completion_tokens=result.completion_tokens, latency_s=result.latency_s)
