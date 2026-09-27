"""Cascade router (docs/cascata.md): one short call on the question text
decides PLAN (the full proof controller) or DIRECT (the no-plan fact path).
Off by default; any router failure means PLAN. No test calls a model server."""
from __future__ import annotations

import json

from test_proof_controller import CHAIN, POOL, QUESTION, FakeLLM, retriever
from wrag.llm.base import LLMResult
from wrag.witness.router import ROUTER_TEMPLATE, parse_route


class RoutingLLM(FakeLLM):
    def __init__(self, plans, route):
        super().__init__(plans)
        self.route = route

    def chat(self, prompt, **kwargs):
        if kwargs.get("stage") == "witness.route":
            self.calls.append(("witness.route", prompt))
            if self.route == "filtered":
                return LLMResult(filtered=True, finish_reason="content_filter")
            return LLMResult(text=self.route)
        return super().chat(prompt, **kwargs)


FACTS = dict(fact_delivery="facts", fact_budget=6)


def test_router_is_off_by_default():
    llm = RoutingLLM([CHAIN], json.dumps({"route": "DIRECT"}))
    retriever(llm, **FACTS)._retrieve_proof(QUESTION, 3, *POOL)
    assert "witness.route" not in llm.stages() and "witness.plan" in llm.stages()


def test_direct_takes_the_no_plan_path_with_one_short_call():
    llm = RoutingLLM([CHAIN], json.dumps({"route": "DIRECT"}))
    result = retriever(llm, plan_router="llm", **FACTS)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert llm.stages() == ["witness.route"]
    assert d["motivo_parada"] == "roteador_direto" and d["roteador"]["rota"] == "DIRECT"
    assert d["planejamento"]["chamadas"] == 1 and "ablacao" not in d
    assert d["leitura_fatos"] and d["trechos_extras"][0]["title"] == "Facts from the memory"
    # Same facts as the no-plan ablation.
    ablation = retriever(FakeLLM([CHAIN]), ablation="no-plan", **FACTS)._retrieve_proof(
        QUESTION, 3, *POOL).diagnostics
    assert d["trechos_extras"] == ablation["trechos_extras"]


def test_plan_runs_the_full_controller():
    llm = RoutingLLM([CHAIN], json.dumps({"route": "PLAN"}))
    result = retriever(llm, plan_router="llm")._retrieve_proof(QUESTION, 3, *POOL)
    assert llm.stages() == ["witness.route", "witness.plan", "witness.confirm"]
    assert result.diagnostics["motivo_parada"] == "prova_confirmada"
    assert result.diagnostics["planejamento"]["chamadas"] == 3


def test_any_router_failure_means_plan():
    for output in ("filtered", "no idea", '{"route": "MAYBE"}', "PLAN or DIRECT"):
        llm = RoutingLLM([CHAIN], output)
        result = retriever(llm, plan_router="llm")._retrieve_proof(QUESTION, 3, *POOL)
        assert "witness.plan" in llm.stages(), output
        assert result.diagnostics["roteador"]["rota"] == "PLAN"
        assert result.diagnostics["roteador"].get("padrao_por") in {"filtrado", "invalido"}


def test_route_parsing():
    assert parse_route('{"route":"DIRECT"}', {"route": "DIRECT"}) == "DIRECT"
    assert parse_route("", {"route": "plan"}) == "PLAN"
    assert parse_route("Route: DIRECT.", None) == "DIRECT"
    assert parse_route("PLAN or DIRECT", None) == ""


def test_router_prompt_reads_only_the_question_and_names_no_category():
    text = ROUTER_TEMPLATE.format(question="What did Ana buy?").lower()
    for word in ("single-hop", "multi-hop", "open-domain", "adversarial", "category", "locomo"):
        assert word not in text
    assert "what did ana buy?" in text


def test_the_ablation_ignores_the_router():
    llm = RoutingLLM([CHAIN], json.dumps({"route": "PLAN"}))
    retriever(llm, plan_router="llm", ablation="no-plan", **FACTS)._retrieve_proof(QUESTION, 3, *POOL)
    assert llm.stages() == []
