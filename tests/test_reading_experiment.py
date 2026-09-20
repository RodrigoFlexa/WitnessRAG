from pathlib import Path

from wrag.eval.counts import numeric_diagnostic
from wrag.eval.diagnostics import stop_reason
from wrag.eval.focused_context import focus_passage
from wrag.eval.reader import read
from wrag.data import Corpus, Passage, Question
from wrag.config import QAConfig
from wrag.llm.base import LLMResult


def test_focused_context_preserves_original_turns_and_date():
    text = ("Session date: 10 May 2023\n[D1:1] Ana: Good morning.\n"
            "[D1:2] Ben: We bought the book.\n[D1:3] Ana: Thanks.\n"
            "Session date: 12 May 2023\n[D2:1] Ana: The concert was lovely.\n"
            "[D2:1] Image caption (automatic): a red car\n")
    result, info = focus_passage(text, "Who bought the book?", max_chars=250)
    assert "Session date: 10 May 2023" in result
    assert "[D1:2] Ben: We bought the book." in result
    assert "red car" not in result
    assert info["original_chars"] == len(text)


def test_numeric_diagnostic_requires_unambiguous_count():
    assert numeric_diagnostic("How many visits?", "three", "3")["correto"]
    assert not numeric_diagnostic("How many visits?", "at least 3", "3")["correto"]
    assert not numeric_diagnostic("When?", "3", "3")["aplicavel"]


def test_stop_reason_coverage_after_join():
    diagnostic = {"classe_prova": "nenhuma", "fallback": "denso (pesquisa sem prova suficiente)",
                  "planos_compilados": [{"executavel": True, "fechou": True,
                                        "obrigacoes": {"cobre_pergunta": False}}]}
    assert stop_reason(diagnostic) == "junção fechou; cobertura rejeitada"


def test_reader_override_keeps_same_pid_and_uses_count_prompt():
    class Stub:
        def __init__(self):
            self.prompts = []

        def chat(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return LLMResult(text='{"answer":"2"}', prompt_tokens=12, completion_tokens=2)

    llm = Stub()
    corpus = Corpus("locomo", [Passage("p1", "title", "full original text")], [])
    question = Question("q1", "How many books?", ["2"], dataset="locomo")
    result = read(llm, corpus, question, ["p1"], QAConfig(top_k=1, answer_set=True),
                  passages_override=[("title", "[D1:1] Ana: two books")], count_mode=True)
    assert result.answer == "2"
    assert "[D1:1] Ana: two books" in llm.prompts[0]
    assert "full original text" not in llm.prompts[0]
    assert "Count distinct matching events" in llm.prompts[0]
