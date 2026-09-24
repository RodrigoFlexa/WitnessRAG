"""Controlador agnóstico: contrato de evidência, rota, equivalência e lentes.

Nenhum teste usa servidor de modelo. O ponto central é a equivalência: quando a
rota do contrato coincide com a rota rotulada, o contexto é idêntico ao do
controlador seletivo; e o rótulo do benchmark nunca é lido.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from test_witness import build_toy
from wrag import config as C
from wrag import prompts
from wrag.data import Corpus, Passage, Question
from wrag.llm.base import LLMResult
from wrag.llm.filters import LEDGER
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.contract import (ROUTE_COMPOSE, ROUTE_DIRECT, EvidenceContract,
                                   contract_from_data, plan_contract, route_of)
from wrag.witness.lenses import (LensPolicy, MemorySignals, arousal, lens_values,
                                 policy_from_contract, rerank_with_lenses, split_turns)
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.timeline import ReferenceClock, parse_anchor, passage_dates


def _contract(**overrides):
    data = {"answer_form": "set", "operator": "aggregate", "evidence_scope": "multiple",
            "time": {"focus": "none", "anchor": ""}, "focus_entities": ["Ana"],
            "info_needs": ["Where does Ana work?"],
            "lenses": {"temporal": "none", "salience": False, "confidence": False}}
    data.update(overrides)
    return data


class Planner:
    """LLM falso: devolve um contrato fixo e registra os prompts recebidos."""

    def __init__(self, data):
        self.data = data
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs.get("stage")))
        if self.data == "filtered":
            return LLMResult(filtered=True, finish_reason="content_filter")
        return LLMResult(text=json.dumps(self.data))


def _retriever(llm, **cfg_overrides):
    memory, searcher, embedder, cfg = build_toy()
    cfg.selective_witness = True
    cfg.answer_set = True
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    run = C.RunConfig()
    run.witness = cfg
    retriever = WitnessRAGRetriever(IndexContext(memory.corpus, llm, embedder, run))
    retriever.memory = memory
    retriever.searcher = searcher
    return retriever


FULL = ConjunctiveQuery(answer_var="x", atoms=[Atom("trabalha em", "Ana", "?y"),
                                               Atom("localizada em", "?y", "?x")])


# -- contrato -----------------------------------------------------------------

def test_contract_is_closed_vocabulary_and_fails_closed():
    contract = contract_from_data(_contract(answer_form="list", operator="chain",
                                            evidence_scope="multi"))
    assert (contract.answer_form, contract.operator, contract.evidence_scope) == \
        ("set", "join", "multiple")
    assert contract.valid and contract.repairs
    assert contract_from_data("not json").route == ROUTE_DIRECT
    assert not contract_from_data(None).valid
    anchorless = contract_from_data(_contract(lenses={"temporal": "anchor"}))
    assert anchorless.temporal_lens == "none"


@pytest.mark.parametrize("form,operator,scope,route", [
    ("set", "aggregate", "multiple", ROUTE_COMPOSE),
    ("count", "aggregate", "multiple", ROUTE_COMPOSE),
    ("entity", "join", "multiple", ROUTE_COMPOSE),
    ("description", "compare", "multiple", ROUTE_COMPOSE),
    ("set", "aggregate", "single", ROUTE_DIRECT),     # uma declaração basta
    ("entity", "lookup", "multiple", ROUTE_DIRECT),   # busca de um fato
    ("time", "temporal", "single", ROUTE_DIRECT),     # fora do fragmento conjuntivo
    ("yes_no", "abduce", "multiple", ROUTE_DIRECT),   # abdução não é derivável
    ("yes_no", "join", "multiple", ROUTE_DIRECT),     # sem variável de resposta
    ("time", "aggregate", "multiple", ROUTE_DIRECT),
])
def test_route_is_fragment_membership(form, operator, scope, route):
    contract = contract_from_data(_contract(answer_form=form, operator=operator,
                                            evidence_scope=scope))
    assert route_of(contract) == route


def test_planner_sees_only_question_text_and_no_category_vocabulary():
    for word in ("multi-hop", "single-hop", "open-domain", "LoCoMo", "category", "benchmark"):
        assert word.lower() not in prompts.CONTRACT_TEMPLATE.lower()
    llm = Planner(_contract())
    question = Question("q", "Where is Ana's employer located?", ["SECRET_GOLD"],
                        gold_pids=["p1"], qtype="multi-hop")
    plan_contract(llm, question)
    prompt, stage = llm.prompts[0]
    assert stage == "witness.contract"
    assert "SECRET_GOLD" not in prompt and "multi-hop" not in prompt and "p1" not in prompt


# -- equivalência com o controlador rotulado ----------------------------------

@pytest.mark.parametrize("qtype,contract,compose", [
    ("multi-hop", _contract(), True),
    ("single-hop", _contract(operator="lookup", evidence_scope="single",
                             answer_form="entity"), False),
])
def test_agreeing_route_reproduces_labelled_context(monkeypatch, qtype, contract, compose):
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query", lambda *_a, **_k: FULL)
    question = Question("q", "Where is Ana's employer located?", [], qtype=qtype)
    pool = (["p2", "p3", "p4"], [1.0, .5, .25])
    labelled = _retriever(object())._retrieve_selective(question, 3, *pool)
    agnostic = _retriever(Planner(contract))._retrieve_agnostic(question, 3, *pool)
    assert agnostic.pids == labelled.pids
    assert agnostic.diagnostics["rota"] == labelled.diagnostics["rota"]
    assert agnostic.diagnostics["rota_plano"] == (ROUTE_COMPOSE if compose else ROUTE_DIRECT)
    assert agnostic.diagnostics["controlador"] == "agnostic-v1"
    assert agnostic.diagnostics["planejamento"]["chamadas"] == \
        labelled.diagnostics["planejamento"]["chamadas"] + 1


def test_benchmark_label_is_never_read(monkeypatch):
    calls = []
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query",
                        lambda *_a, **_k: calls.append(1) or FULL)
    pool = (["p2", "p3", "p4"], [1.0, .5, .25])
    direct = _retriever(Planner(_contract(operator="lookup", evidence_scope="single")))
    result = direct._retrieve_agnostic(
        Question("q", "Where is Ana's employer located?", [], qtype="multi-hop"), 3, *pool)
    assert not calls and result.pids == ["p2", "p3", "p4"]
    compose = _retriever(Planner(_contract()))
    result = compose._retrieve_agnostic(
        Question("q", "Where is Ana's employer located?", [], qtype="single-hop"), 3, *pool)
    assert calls == [1] and set(result.pids) == {"p2", "p0", "p1"}


def test_blocked_contract_keeps_question_on_hybrid_route():
    before = len(LEDGER.events)
    retriever = _retriever(Planner("filtered"))
    result = retriever._retrieve_agnostic(
        Question("q", "Where is Ana's employer located?", [], qtype="multi-hop"), 3,
        ["p2", "p3", "p4"], [1.0, .5, .25])
    assert result.pids == ["p2", "p3", "p4"] and not result.filtered
    assert result.diagnostics["motivo_parada"] == "contract_filtered_direct"
    assert len(LEDGER.events) == before


# -- relógio e lentes ---------------------------------------------------------

def test_anchor_and_passage_dates():
    anchor = parse_anchor("the last week of August 2023")
    assert (anchor.start, anchor.end) == (date(2023, 8, 25), date(2023, 8, 31))
    assert parse_anchor("spring of 2021").end == date(2021, 5, 31)
    assert parse_anchor("last year", now=date(2023, 10, 1)).start == date(2022, 1, 1)
    assert parse_anchor("soon") is None
    text = ("Session date: 1:56 pm on 8 May, 2023\n[D1:1 date=1:56 pm on 8 May, 2023] Ana: hi\n"
            "[D1:1 temporal] reference_time=2023-05-08; \"last week\"=... "
            "[start=2023-04-24, end=2023-04-30]")
    assert passage_dates(text) == [date(2023, 4, 24), date(2023, 4, 30), date(2023, 5, 8)]


def _dated_corpus():
    passages = [
        Passage("a", "c1", "Session date: 1 January, 2023\n[D1:1] Ana: I work at Atlas."),
        Passage("b", "c2", "Session date: 10 June, 2023\n[D2:1] Ana: I moved to Recife."),
        Passage("c", "c3", "Session date: 1 December, 2023\n[D3:1] Ana: I am so thrilled "
                           "and proud, I love my new job at Orion!"),
        Passage("d", "c4", "Session date: 5 December, 2023\n[D4:1] Bruno: I am devastated "
                           "and heartbroken!"),
    ]
    return Corpus("toy-time", passages, [])


def test_temporal_lenses_follow_reference_clock():
    signals = MemorySignals(_dated_corpus())
    assert signals.clock == ReferenceClock(date(2023, 1, 1), date(2023, 12, 5))
    recent = LensPolicy(temporal="recent")
    values = lens_values(signals, ["a", "b", "c"], recent, ["Ana"], "Ana job", None)
    assert values["c"]["temporal"] > values["b"]["temporal"] > values["a"]["temporal"]
    early = lens_values(signals, ["a", "c"], LensPolicy(temporal="early"), [], "q", None)
    assert early["a"]["temporal"] == pytest.approx(1.0)
    window = LensPolicy(temporal="anchor", anchor=parse_anchor("June 2023"))
    values = lens_values(signals, ["a", "b", "c"], window, [], "q", None)
    assert max(values, key=lambda pid: values[pid]["temporal"]) == "b"


def test_salience_is_conditioned_on_focus_entity():
    assert arousal("I am so thrilled!") > arousal("I went to the store.")
    signals = MemorySignals(_dated_corpus())
    values = lens_values(signals, ["b", "c", "d"], LensPolicy(salience=True), ["Ana"],
                         "How does Ana feel about her job?", None)
    assert values["c"]["salience"] > values["b"]["salience"]
    assert values["d"]["salience"] == 0.0          # emoção de outra pessoa
    assert split_turns("[D1:1 date=8 May, 2023] Ana: hello")[0].when == date(2023, 5, 8)


def test_rerank_is_identity_without_lens_gain_and_bounded_otherwise():
    current, pool = ["a", "b", "c"], ["a", "b", "c", "d", "e"]
    scores = [5.0, 4.0, 3.0, 2.9, 1.0]
    zero = {pid: {"temporal": 0.0} for pid in pool}
    assert rerank_with_lenses(current, pool, scores, 3, zero)[0] == current
    lenses = {pid: {"temporal": 0.0} for pid in pool}
    lenses["d"] = {"temporal": 1.0}
    lenses["e"] = {"temporal": 1.0}
    out, diag = rerank_with_lenses(current, pool, scores, 3, lenses, max_swaps=1)
    assert out == ["a", "b", "d"] and len(diag["trocas"]) == 1
    # A passagem de uma testemunha entregue nunca sai.
    out, _ = rerank_with_lenses(current, pool, scores, 3, lenses, max_swaps=1,
                                protected={"c"})
    assert out == current


def test_lens_policy_needs_clock_and_parseable_anchor():
    no_clock = ReferenceClock(None, None)
    contract = contract_from_data(_contract(lenses={"temporal": "recent"}))
    assert policy_from_contract(contract, no_clock).temporal == "none"
    clock = ReferenceClock(date(2023, 1, 1), date(2023, 12, 1))
    bad = contract_from_data(_contract(time={"focus": "window", "anchor": "soon"},
                                       lenses={"temporal": "anchor"}))
    assert policy_from_contract(bad, clock).temporal == "none"


def test_lenses_in_controller_touch_only_the_tail(monkeypatch):
    llm = Planner(_contract(operator="lookup", evidence_scope="single", answer_form="entity",
                            lenses={"temporal": "none", "salience": True,
                                    "confidence": False}))
    retriever = _retriever(llm, memory_lenses=True, lens_max_swaps=1)
    retriever.signals = MemorySignals(retriever.corpus, retriever.memory)
    monkeypatch.setattr(retriever.signals, "salience",
                        lambda pid, *_a: 1.0 if pid == "p4" else 0.0)
    result = retriever._retrieve_agnostic(
        Question("q", "How does Ana feel?", [], qtype="single-hop"), 3,
        ["p0", "p1", "p2", "p4", "p3"], [5.0, 4.0, 3.0, 2.9, 1.0])
    assert result.pids[:2] == ["p0", "p1"] and result.pids[2] == "p4"
    assert result.diagnostics["lentes"]["alterou"]
    assert result.diagnostics["contexto_alterado_por_lente"]


# -- flags ----------------------------------------------------------------------

def test_agnostic_flags_reach_config(tmp_path):
    from test_pilot import parser
    from wrag.pilot import _run_config, make_plan
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--selective-witness",
                                "--agnostic-router", "--memory-lenses", "--evidence-reader",
                                "--lens-max-swaps", "2"])
    plan = make_plan(args, tmp_path)
    cfg = _run_config(plan["settings"], 1)
    assert cfg.witness.agnostic_router and cfg.witness.memory_lenses
    assert cfg.witness.lens_max_swaps == 2
    bad = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--memory-lenses"])
    with pytest.raises(ValueError):
        make_plan(bad, tmp_path)
    # O leitor não pode voltar a escolher o template pela categoria.
    no_reader = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--agnostic-router"])
    with pytest.raises(ValueError):
        make_plan(no_reader, tmp_path)


def test_route_override_is_an_explicit_ablation(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query",
                        lambda *_a, **_k: calls.append(1) or FULL)
    retriever = _retriever(Planner(_contract(operator="lookup", evidence_scope="single")),
                           route_override="compose")
    result = retriever._retrieve_agnostic(
        Question("q", "Where is Ana's employer located?", []), 3,
        ["p2", "p3", "p4"], [1.0, .5, .25])
    assert calls == [1] and result.diagnostics["rota_forcada"] == "compose"
    from test_pilot import parser
    from wrag.pilot import make_plan
    bad = parser().parse_args(["--gpu", "3", "--dataset", "locomo",
                               "--route-override", "direct"])
    with pytest.raises(ValueError):
        make_plan(bad, tmp_path)


def test_lens_mask_supports_single_lens_ablation(monkeypatch):
    llm = Planner(_contract(operator="lookup", evidence_scope="single", answer_form="entity",
                            lenses={"temporal": "none", "salience": True,
                                    "confidence": False}))
    retriever = _retriever(llm, memory_lenses=True, lens_allow="temporal")
    retriever.signals = MemorySignals(retriever.corpus, retriever.memory)
    monkeypatch.setattr(retriever.signals, "salience", lambda *_a: 1.0)
    result = retriever._retrieve_agnostic(
        Question("q", "How does Ana feel?", []), 3,
        ["p0", "p1", "p2", "p4", "p3"], [5.0, 4.0, 3.0, 2.9, 1.0])
    assert result.pids == ["p0", "p1", "p2"]
    assert "salience:desligada_na_ablacao" in result.diagnostics["lentes"]["politica"]["repairs"]
