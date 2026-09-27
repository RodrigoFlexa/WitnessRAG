"""Robust plan and fact selection (docs/plano-robusto.md): disjunctive atoms
(relation alternatives), other readings of the question in the same planning
call, the fact fill by the question only, the cross-encoder reranker and the
bitemporal rendering of dated facts.

Every option is off by default, and a run without them sends the v4 prompts
and builds the v4 contexts. No test calls a model server.
"""
from __future__ import annotations

import json

import numpy as np

from test_proof_controller import CHAIN, POOL, QUESTION, FakeLLM, retriever
from wrag import prompts
from wrag.data import Question
from wrag.llm.base import LLMResult
from wrag.witness.plan import plan_from_data
from wrag.witness.query import _parse_atoms
from wrag.witness.scoring import WeightLevels

LEVELS = WeightLevels(0.0, 0.3, 0.0, 0.1)

# The memory says "trabalha em" / "localizada em". The main reading uses words
# the memory does not use, so only an alternative (or another reading) proves.
WRONG_CHAIN = {**CHAIN, "atoms": [
    {"relation": "emprega", "subject": "?y", "object": "Ana"},
    {"relation": "fica em", "subject": "?y", "object": "?x"}]}
ALT_CHAIN = {**CHAIN, "atoms": [
    {"relation": ["emprego em", "trabalha em"], "subject": "Ana", "object": "?y"},
    {"relation": "fica em", "alternatives": ["localizada em"], "subject": "?y",
     "object": "?x"}]}


# -- prompts ---------------------------------------------------------------------

def test_prompt_is_unchanged_without_the_options():
    assert prompts.plan_template() == prompts.PLAN_TEMPLATE
    assert prompts.plan_template(True, True) == prompts.plan_template(True, True, False, 0)
    v4 = prompts.plan_template(True, True)
    assert "alternatives" not in v4 and "other_readings" not in v4


def test_robust_prompt_describes_both_fields_and_formats():
    text = prompts.plan_template(True, True, True, 2).format(
        question="Q?", max_atoms=4, vocabulary="", evidence="", feedback="")
    assert '7. "alternatives" (REQUIRED' in text and '8. "other_readings" (REQUIRED; 1 to 2' in text
    assert '"alternatives": ["..."]' in text and '"other_readings": [' in text
    assert "Nora's brother" in text and "Kai and his wife" in text
    # Every worked example shows the robust fields (the planner copies examples).
    examples = [line for line in text.splitlines() if line.startswith('{"answer_var":"')]
    assert len(examples) == 11
    for line in examples:
        data = json.loads(line)
        assert "other_readings" in data
        assert all(a.get("alternatives") for a in data["atoms"])
    alone = prompts.plan_template(False, False, False, 1).format(
        question="Q?", max_atoms=4, vocabulary="", evidence="", feedback="")
    assert '5. "other_readings" (REQUIRED; 1 to 1' in alone
    assert '"alternatives"' not in alone and '"types":{' not in alone
    only_hyp = prompts.plan_template(False, True, True, 0)
    assert '5. "hypothesis"' in only_hyp and '6. "alternatives"' in only_hyp


# -- disjunctive atoms -------------------------------------------------------------

def test_atoms_read_alternatives_from_a_list_or_a_field_and_drop_bad_ones():
    atoms = _parse_atoms([
        {"relation": ["live in", "move to", "live in"], "subject": "A", "object": "?x"},
        {"relation": "work as", "alternatives": ["job", "f(x)", "?y", "x" * 80, "a", "b"],
         "subject": "A", "object": "?z"},
    ], 4)
    assert atoms[0].relation == "live in" and atoms[0].alternatives == ["move to"]
    assert atoms[1].alternatives == ["job", "a", "b"]
    assert atoms[1].relations == ["work as", "job", "a", "b"]
    assert atoms[0].to_dict()["alternatives"] == ["move to"]


def test_alternatives_change_the_plan_signature_and_description():
    plain = plan_from_data(dict(CHAIN), QUESTION, None, LEVELS)
    alt = plan_from_data(json.loads(json.dumps(ALT_CHAIN)), QUESTION, None, LEVELS)
    assert plain.signature() != alt.signature()
    assert "fica em|localizada em(?y, ?x)" in alt.describe()


def test_a_disjunctive_atom_proves_in_one_cycle():
    llm = FakeLLM([ALT_CHAIN])
    result = retriever(llm, relation_alternatives=True)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert d["motivo_parada"] == "prova_confirmada"
    assert {"p0", "p1"} <= set(result.pids)
    assert llm.stages() == ["witness.plan", "witness.confirm"]


def test_alternatives_are_ignored_when_the_option_is_off():
    llm = FakeLLM([ALT_CHAIN])
    result = retriever(llm, proof_partial_evidence=False)._retrieve_proof(QUESTION, 3, *POOL)
    assert result.diagnostics["motivo_parada"] != "prova_confirmada"
    assert result.pids == POOL[0][:3]


def test_the_planner_is_asked_for_alternatives_only_with_the_option():
    llm = FakeLLM([CHAIN])
    retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert '"alternatives"' not in llm.calls[0][1]
    llm = FakeLLM([CHAIN])
    retriever(llm, relation_alternatives=True)._retrieve_proof(QUESTION, 3, *POOL)
    assert '"alternatives"' in llm.calls[0][1]


# -- other readings ----------------------------------------------------------------

def test_readings_are_parsed_with_the_period_of_the_plan():
    data = dict(WRONG_CHAIN, period={"reference": "start", "text": ""}, time_weight="strong",
                other_readings=[{"answer_var": "x", "atoms": CHAIN["atoms"],
                                 "aggregation": "none"},
                                {"answer_var": "x", "atoms": WRONG_CHAIN["atoms"]},
                                "junk", {"atoms": [{"relation": "r"}]}])
    plan = plan_from_data(data, QUESTION, None, LEVELS, max_readings=4)
    assert len(plan.readings) == 1 and plan.readings[0].reading == 1
    other = plan.readings[0]
    assert other.period.kind == plan.period.kind and other.time_level == plan.time_level
    assert other.weights.to_dict() == plan.weights.to_dict()
    assert any(r.startswith("leitura_2_repetida") for r in plan.repairs)
    assert any(r.startswith("leitura_3_invalida") for r in plan.repairs)
    # Without the option the field is not read at all.
    assert plan_from_data(data, QUESTION, None, LEVELS).readings == []


def test_an_invalid_main_reading_is_replaced_by_a_valid_other_reading():
    data = dict(CHAIN, atoms=[{"relation": "r(x)", "subject": "Ana", "object": "?x"}],
                other_readings=[{"answer_var": "x", "atoms": CHAIN["atoms"]}])
    plan = plan_from_data(data, QUESTION, None, LEVELS, max_readings=2)
    assert plan.valid and plan.reading == 0 and not plan.readings
    assert "leitura_promovida" in plan.repairs


def test_another_reading_proves_without_a_new_planning_call():
    data = dict(WRONG_CHAIN, other_readings=[{"answer_var": "x", "atoms": CHAIN["atoms"],
                                              "aggregation": "none"}])
    llm = FakeLLM([data])
    result = retriever(llm, plan_readings=2)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert d["motivo_parada"] == "prova_confirmada"
    assert llm.stages() == ["witness.plan", "witness.confirm"]
    cycle = d["ciclos"][0]
    assert cycle["leitura_escolhida"] == 1
    assert [a["leitura"] for a in cycle["leituras"]] == [0, 1]
    assert d["plano_final"]["leitura"] == 1


def test_without_the_option_the_same_answer_needs_a_second_cycle():
    data = dict(WRONG_CHAIN, other_readings=[{"answer_var": "x", "atoms": CHAIN["atoms"]}])
    llm = FakeLLM([data, CHAIN])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert llm.stages() == ["witness.plan", "witness.replan_v3", "witness.confirm"]
    assert result.diagnostics["motivo_parada"] == "prova_confirmada"


def test_failed_readings_are_reported_to_the_next_plan():
    bad = dict(WRONG_CHAIN, other_readings=[{"answer_var": "x", "atoms": [
        {"relation": "mora em", "subject": "Ana", "object": "?x"}]}])
    llm = FakeLLM([bad, CHAIN])
    retriever(llm, plan_readings=2)._retrieve_proof(QUESTION, 3, *POOL)
    replan = [p for s, p in llm.calls if s == "witness.replan_v3"][0]
    assert "the other readings failed too (reading 1:" in replan


# -- fact selection and rendering -----------------------------------------------------

FACT_CFG = dict(fact_delivery="facts", fact_budget=6)


def _facts(r, plan_data, question=QUESTION):
    plan = plan_from_data(json.loads(json.dumps(plan_data)), question, r.dated, LEVELS)
    return r._fact_context(question, plan, None, None, POOL[0][:3])


def test_fill_by_question_does_not_embed_the_atoms(monkeypatch):
    r = retriever(FakeLLM([CHAIN]), **FACT_CFG)
    r._build_dated_memory()
    seen = []
    original = r.ctx.embedder.encode

    def spy(texts, *a, **k):
        seen.append(list(texts))
        return original(texts, *a, **k)

    monkeypatch.setattr(r.ctx.embedder, "encode", spy)
    _facts(r, CHAIN)
    assert len(seen[-1]) == 3            # question + two atoms (v4)
    r.ctx.run.witness.fact_fill = "question"
    _facts(r, CHAIN)
    assert seen[-1] == [QUESTION.question]


def test_bitemporal_notes_keep_the_words_the_speaker_used():
    question = Question("qt", "When did Bruno camp in the mountains?", ["x"])
    plan = {"answer_var": "t", "atoms": [{"relation": "acampou em", "subject": "Bruno",
                                          "object": "montanhas", "time": "?t"}]}
    r = retriever(FakeLLM([plan]), **FACT_CFG)
    r._build_dated_memory()
    resolved, _s, _i = _facts(r, plan, question)
    assert "(event: 19 June 2023 - 25 June 2023)" in resolved
    assert "said as" not in resolved
    r.ctx.run.witness.fact_time = "both"
    both, _s, _i = _facts(r, plan, question)
    assert "(event: the week before 27 June 2023; 19 June 2023 - 25 June 2023)" in both
    # A fact dated by its session only gets no "said as".
    assert "We camped" not in both and 'praia (said as' not in both


def test_bitemporal_reader_prompt_is_opt_in():
    assert prompts.qa_facts_template() == prompts.qa_facts_template(bitemporal=False)
    text = prompts.qa_facts_template(bitemporal=True)
    assert "the weekend before 24 October 2023" in text and "a month as a month" in text


def test_anchored_phrases_and_natural_precision():
    from datetime import date
    from wrag.witness.timeline import Interval, anchored_phrase, natural_interval
    said = date(2023, 10, 24)
    assert anchored_phrase("last weekend", said) == "the weekend before 24 October 2023"
    assert anchored_phrase("Last Friday", said) == "the Friday before 24 October 2023"
    assert anchored_phrase("two days ago", said) == "two days before 24 October 2023"
    assert anchored_phrase("yesterday", said) == "the day before 24 October 2023"
    assert anchored_phrase("in two weeks", said) == "two weeks after 24 October 2023"
    assert anchored_phrase("on a rock", said) == "" and anchored_phrase("last week", None) == ""
    assert natural_interval(Interval(date(2023, 8, 1), date(2023, 8, 31))) == "August 2023"
    assert natural_interval(Interval(date(2022, 1, 1), date(2022, 12, 31))) == "2022"
    assert natural_interval(Interval(date(2023, 8, 1), date(2023, 8, 30))) == \
        "1 August 2023 - 30 August 2023"


class FakeReranker:
    def __init__(self, favourite):
        self.favourite = favourite
        self.calls = 0

    def score(self, question, texts):
        self.calls += 1
        return [0.99 if self.favourite in t else 0.01 for t in texts]


def test_reranker_reorders_the_fill_but_keeps_the_proof_first(monkeypatch):
    import wrag.witness.rerank as rerank_mod
    fake = FakeReranker("romance")
    monkeypatch.setattr(rerank_mod, "get_reranker", lambda name: fake)
    r = retriever(FakeLLM([CHAIN]), fact_delivery="facts", fact_budget=2,
                  fact_rerank="fake", fact_rerank_pool=50)
    r._build_dated_memory()
    order = list(range(len(r.memory.facts)))
    score = np.zeros(len(order), dtype=np.float32)
    ranked, pool = r._rerank_facts(QUESTION, order, score, {0: 2.0})
    assert fake.calls == 1 and pool == len(order)
    novel = next(i for i, f in enumerate(r.memory.facts) if f.object == "romance")
    assert ranked[0] == 0 and ranked[1] == novel
    text, _s, info = _facts(r, CHAIN)
    assert info["reordenados"] > 0 and "romance" in text


class RejectFirstLLM(FakeLLM):
    """The check refuses the first proof it sees and accepts the next ones."""

    def chat(self, prompt, **kwargs):
        if kwargs.get("stage") == "witness.confirm":
            self.calls.append(("witness.confirm", prompt))
            first = sum(s == "witness.confirm" for s, _p in self.calls) == 1
            if first:
                return LLMResult(text=json.dumps(
                    {"supported": [], "rejected": [{"id": "A1", "reason": "wrong_entity"}]}))
            ids = [f"A{i}" for i in range(1, 10) if f"A{i}:" in prompt]
            return LLMResult(text=json.dumps({"supported": ids, "rejected": []}))
        return super().chat(prompt, **kwargs)


def test_a_refused_proof_tries_the_other_readings_before_replanning():
    # Main reading: the one-fact plan (proves Recife through p1 alone);
    # other reading: the chain. The check refuses the first proof.
    single = {"answer_var": "x", "aggregation": "none",
              "atoms": [{"relation": "localizada em", "subject": "Atlas", "object": "?x"}],
              "other_readings": [{"answer_var": "x", "atoms": CHAIN["atoms"],
                                  "aggregation": "none"}]}
    llm = RejectFirstLLM([single])
    result = retriever(llm, plan_readings=2)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert llm.stages() == ["witness.plan", "witness.confirm", "witness.confirm"]
    assert d["motivo_parada"] == "prova_confirmada"
    assert d["plano_final"]["leitura"] == 1
    # Without the option the refusal leads to a new planning call.
    llm = RejectFirstLLM([single, CHAIN])
    retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert llm.stages()[:3] == ["witness.plan", "witness.confirm", "witness.replan_v3"]
