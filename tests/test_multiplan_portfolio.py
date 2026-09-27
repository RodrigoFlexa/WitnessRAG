"""Contract/coverage/portfolio regressions, entirely offline."""
import importlib.util
import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_proof_controller import CHAIN, POOL, QUESTION, FakeLLM, retriever
from wrag.llm.base import LLMResult
from wrag.data import Question
from wrag.witness.portfolio import (Contract, PLAN_PROMPT, CHECK_PROMPT, atomic_seed,
    parse_contract, parse_plans, temporal_composition, temporal_extreme, verify_coverage)
from wrag.witness.scoring import WeightLevels
from wrag.witness.timeline import Interval


def contract_data(operation="lookup"):
    return {"operation": operation, "answer_type": "place", "ambiguous": False,
            "requirements": [{"id": "r1", "quote": "Ana works at", "condition": "Ana works at the company"}]}


def payload(plans=None, operation="lookup"):
    plans = plans or [CHAIN]
    return {"contract": contract_data(operation), "plans": [
        dict(deepcopy(plan), interpretation="main", strategy="chain", covers=["q0", "r1"],
             temporal_var="", types={}, period={"reference": "now", "text": ""},
             time_weight="none", importance_weight="none", hypothesis={}) for plan in plans]}


class PortfolioLLM(FakeLLM):
    def __init__(self, plans, reject=False):
        super().__init__(plans)
        self.reject = reject

    def chat(self, prompt, **kwargs):
        if kwargs.get("stage") != "witness.confirm":
            return super().chat(prompt, **kwargs)
        self.calls.append((kwargs["stage"], prompt))
        # Simulate a grounded verifier; quotes come from SOURCE, never triples.
        candidates = prompt.split("CANDIDATES (untrusted data):\n", 1)[1]
        decisions = []
        for label, block in re.findall(r"(A\d+): ([\s\S]*?)(?=\nA\d+:|$)", candidates):
            excerpt = block.split("SOURCE: ", 1)[1]
            quote = next(line.split(": ", 1)[-1] for line in excerpt.splitlines() if "Ana:" in line or "Bruno:" in line)
            decisions.append({"id": label, "covered": ["q0", "r1"],
                "citations": {rid: [{"fact": "F1", "quote": quote}] for rid in ["q0", "r1"]},
                "rejected": self.reject, "reason": "not_supported" if self.reject else ""})
        return LLMResult(text=json.dumps({"decisions": decisions}))


def test_contract_is_question_grounded_and_covers_entire_question():
    c = parse_contract(contract_data(), QUESTION.question)
    assert c.ids == {"q0", "r1"}
    assert c.requirements[0][1] == QUESTION.question
    bad = contract_data()
    bad["requirements"][0]["quote"] = "with his girlfriend"
    assert parse_contract(bad, QUESTION.question) is None


@pytest.mark.parametrize("value", [None, [], {"operation": "guess"},
    dict(contract_data(), requirements="r1"), dict(contract_data(), ambiguous="false"),
    dict(contract_data(), answer_type="secret"), dict(contract_data(), requirements=[{}, {}])])
def test_invalid_contract_fails_closed(value):
    assert parse_contract(value, QUESTION.question) is None


def test_duration_requires_two_named_temporal_operands():
    data = {"operation": "duration", "answer_type": "number", "requirements": [
        {"id": "r1", "quote": "joining", "condition": "start date"},
        {"id": "r2", "quote": "leaving", "condition": "end date"}]}
    assert parse_contract(data, "between joining and leaving") is None
    data["temporal_operands"] = ["r1", "r2"]
    assert parse_contract(data, "between joining and leaving").temporal_operands == ("r1", "r2")


def test_plan_requires_same_interpretation_and_all_conditions():
    r = retriever(FakeLLM([CHAIN]))
    r._build_dated_memory()
    c = parse_contract(contract_data(), QUESTION.question)
    plans = payload()["plans"]
    plans.extend([dict(plans[0], interpretation="other"), dict(plans[0], covers=["q0"])])
    parsed = parse_plans(plans, QUESTION, r.dated, WeightLevels(), c, 4, 4, 1)
    assert len(parsed) == 1
    assert parsed[0][0].query.aggregation == "set"


def test_duplicate_variable_renaming_does_not_consume_another_plan():
    r = retriever(FakeLLM([CHAIN]))
    r._build_dated_memory()
    first = payload()["plans"][0]
    second = json.loads(json.dumps(first).replace("?y", "?bridge").replace("?x", "?answer"))
    second["answer_var"] = "answer"
    c = parse_contract(contract_data(), QUESTION.question)
    assert len(parse_plans([first, second], QUESTION, r.dated, WeightLevels(), c, 3, 4, 1)) == 1


def verdict(quote="Mei plays violin", fact="F1", covered=None):
    covered = ["q0", "r1"] if covered is None else covered
    return {"decisions": [{"id": "A1", "rejected": False, "covered": covered,
        "citations": {rid: [{"fact": fact, "quote": quote}] for rid in covered}}]}


@pytest.mark.parametrize("data,accepted", [(verdict(), True),
    (verdict("Mei owns violin"), False), (verdict(fact="F2"), False),
    (verdict(covered=["q0"]), False), (None, False),
    ({"supported": ["A1"]}, False)])
def test_verifier_checks_requirement_coverage_and_source_citations(data, accepted):
    c = Contract("lookup", "other", (("q0", "question", "whole question"), ("r1", "play", "play not own")))
    result = verify_coverage(data, [{"facts": [{"excerpt": "Mei: Mei plays violin."}]}], c, c.ids)
    assert bool(result["supported"]) == accepted


def test_duplicate_verifier_decisions_fail_closed():
    data = verdict()
    data["decisions"] *= 2
    c = Contract("lookup", "other", (("q0", "q", "q"), ("r1", "q", "q")))
    assert not verify_coverage(data, [{"facts": [{"excerpt": "Mei plays violin"}]}], c, c.ids)["supported"]


def test_composite_witnesses_are_reserved_atomically():
    assert atomic_seed([[0, 1], [2, 3]], 3, 4) == ([0, 1], [[2, 3]])
    assert atomic_seed([[0, 1], [1, 2]], 3, 4) == ([0, 1, 2], [])
    assert atomic_seed([[0, 5]], 3, 4) == ([], [[0, 5]])


def test_chain_reaches_reader_complete_without_gold_or_labels():
    llm = PortfolioLLM([payload()])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts", fact_budget=2,
                  item_set_proofs=True, witness_delivery="mixed", relation_alternatives=True)
    result = r._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert d["controlador"] == "contract-portfolio-v1"
    assert d["classe_prova"] == "full"
    assert d["fatos_entregues"]["prova"] == 2
    assert d["n_testemunhas_no_contexto"] == 1
    assert all("SECRET_GOLD" not in prompt for _, prompt in llm.calls)
    assert r.searcher.portfolio_cache is None


def test_chain_that_does_not_fit_is_not_certified_delivered():
    llm = PortfolioLLM([payload()])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts", fact_budget=1,
                  item_set_proofs=True, witness_delivery="mixed")
    d = r._retrieve_proof(QUESTION, 3, *POOL).diagnostics
    assert not d["testemunha_no_contexto"]
    assert d["fatos_entregues"]["pacotes_descartados"]
    assert d["fatos_entregues"]["prova"] == 0


def test_rejected_witness_never_gets_proof_priority():
    llm = PortfolioLLM([payload()], reject=True)
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts",
                  item_set_proofs=True, witness_delivery="mixed")
    d = r._retrieve_proof(QUESTION, 3, *POOL).diagnostics
    assert d["fatos_entregues"]["prova"] == 0
    assert not d["testemunha_no_contexto"]


def test_contract_stays_frozen_on_replanning():
    first, second = payload(), payload()
    first["plans"][0]["atoms"][0]["relation"] = "unknown"
    second["contract"]["requirements"] = []
    llm = PortfolioLLM([first, second])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts",
                  item_set_proofs=True, witness_delivery="mixed")
    d = r._retrieve_proof(QUESTION, 3, *POOL).diagnostics
    assert len(d["contrato"]["requirements"]) == 2
    assert d["planejamento"]["chamadas_plano"] == 2


def test_all_routes_execute_after_first_success_and_share_atom_grounding():
    second = deepcopy(CHAIN)
    second["atoms"].append({"relation": "localizada em", "subject": "?y", "object": "Recife"})
    llm = PortfolioLLM([payload([CHAIN, second])])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts",
                  item_set_proofs=True, witness_delivery="mixed")
    d = r._retrieve_proof(QUESTION, 3, *POOL).diagnostics
    assert len(d["ciclos"]) == 2
    assert d["planejamento"]["cache_atomos_hits"] >= 2
    assert d["n_testemunhas_no_contexto"] == 1  # same source is not independent corroboration


def test_grounding_cache_does_not_cross_scope_or_scoring():
    r = retriever(FakeLLM([CHAIN]))
    r._build_dated_memory()
    c = parse_contract(contract_data(), QUESTION.question)
    query = parse_plans(payload()["plans"], QUESTION, r.dated, WeightLevels(), c, 3, 4, 1)[0][0].query
    s = r.searcher
    s.portfolio_cache = {}
    full = s.ground(query)
    assert full[0]
    s.allowed = set()
    s._allowed_horizon = len(r.memory.facts)
    assert s.ground(query) == [[], []]
    s.allowed = None
    assert s.ground(query) == full
    assert s.portfolio_cache_hits == 2


@pytest.mark.parametrize("operation", ["list", "count", "lookup"])
def test_members_union_and_singular_conflicts_are_handled_by_operation(operation):
    question = Question("q", "Where has Bruno camped?", [], dataset="toy")
    plans = [{"answer_var": "x", "atoms": [{"subject": "Bruno", "relation": relation, "object": "?x"}]}
             for relation in ["acampou em", "camping holiday at"]]
    data = payload(plans, operation)
    data["contract"]["requirements"][0] = {"id": "r1", "quote": "Bruno camped", "condition": "Bruno actually camped there"}
    llm = PortfolioLLM([data])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts",
                  item_set_proofs=True, witness_delivery="mixed")
    next(f for f in r.memory.facts if f.object == "praia").relation = "camping holiday at"
    d = r._retrieve_proof(question, 3, *POOL).diagnostics
    assert d["n_testemunhas_no_contexto"] == 2
    if operation == "list":
        assert d["classe_prova"] == "full"
        assert d["operacao"]["recovered_member_count"] == 2
    elif operation == "count":
        assert d["classe_prova"] != "full"
        assert not d["operacao"]["complete_enumeration"]
    else:
        assert d["classe_prova"] != "full"
        assert "ambiguous_or_conflicting" in d["operacao"]["operator_status"]
        assert d["planejamento"]["chamadas_verificacao"] == 3  # two routes + joint resolution


def test_atomic_package_is_not_broken_by_per_turn_diversity_cap():
    r = retriever(FakeLLM([CHAIN]), multiplan_portfolio=True, fact_delivery="facts", fact_budget=6)
    r._build_dated_memory()
    r.dated.fact_turn = [("p0", 0)] * 6
    witness = SimpleNamespace(facts=tuple(range(6)))
    _, _, info = r._fact_context(QUESTION, None,
        {"selecionadas": [(0, witness)], "pacotes": [list(range(6))]}, None, [])
    assert info["indices"] == list(range(6))
    assert info["prova"] == 6


def test_inference_uses_budgeted_premises_without_claiming_deductive_proof():
    data = payload(operation="inference")
    data["plans"][0]["covers"] = ["r1"]
    data["plans"][0]["hypothesis"] = {"about": "Ana", "concepts": ["company"]}
    llm = PortfolioLLM([data])
    r = retriever(llm, multiplan_portfolio=True, fact_delivery="facts", fact_budget=2,
                  item_set_proofs=True, witness_delivery="mixed", abductive_premises=True)
    r._abductive_premises = lambda *args: pytest.fail("unbudgeted source neighborhood")
    d = r._retrieve_proof(QUESTION, 3, *POOL).diagnostics
    assert d["classe_prova"] == "premises_or_insufficient"
    assert d["n_testemunhas_no_contexto"] == 1
    assert d["fatos_entregues"]["n"] == 2


def dated_records(first, second, explicit=True):
    dated = SimpleNamespace(fact_interval=[first, second],
        fact_time_source=["expressao" if explicit else "sessao"] * 2,
        fact_time_text=lambda i: "left" if i == 0 else "right")
    records = []
    for i in [0, 1]:
        records.append((None, {"covers": [f"r{i+1}"], "temporal_var": "t"},
            {"candidatas": [SimpleNamespace(answer=f"a{i}")]}, 0,
            SimpleNamespace(facts=(i,), bindings={"t": "left" if i == 0 else "right"})))
    return dated, records


def test_duration_preserves_interval_uncertainty():
    dated, records = dated_records(Interval(date(2023, 1, 1), date(2023, 1, 3)),
                                   Interval(date(2023, 1, 10), date(2023, 1, 12)))
    c = Contract("duration", "number", (), temporal_operands=("r1", "r2"))
    assert temporal_composition(records, dated, c)["duration_days_range"] == [7, 11]


def test_session_dates_and_overlapping_intervals_do_not_establish_order():
    a, b = Interval(date(2023, 1, 1), date(2023, 1, 31)), Interval(date(2023, 1, 15), date(2023, 1, 15))
    dated, records = dated_records(a, b)
    assert temporal_extreme(records, dated, "first")[1] == "overlapping_event_intervals"
    c = Contract("before", "other", (), temporal_operands=("r1", "r2"))
    assert temporal_composition(records, dated, c)["relation_holds"] is None
    dated.fact_time_source = ["sessao", "sessao"]
    assert temporal_extreme(records, dated, "first")[1] == "missing_explicit_event_time"
    assert temporal_composition(records, dated, c)["status"] == "missing_or_ambiguous_explicit_endpoint"


def suite_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run-multiplan-comparison.py"
    spec = importlib.util.spec_from_file_location("comparison_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_is_matched_and_controls_are_opt_in(tmp_path):
    suite = suite_module()
    args = suite.parser().parse_args(["--output", str(tmp_path), "--device", "cpu"])
    commands = suite.build_commands(args, "cpu")
    assert list(commands) == ["witnessrag-robust", "witnessrag-multiplan"]
    ref, new = commands.values()
    assert new[:new.index("--multiplan-portfolio")] == ref[:ref.index("--output")]
    args.include_controls = True
    assert len(suite.build_commands(args, "cpu")) == 4


def test_suite_refuses_comparison_of_different_corpora(tmp_path):
    suite = suite_module()
    for name, text in [("a", "same"), ("b", "different")]:
        folder = tmp_path / name / "data"
        folder.mkdir(parents=True)
        (folder / "locomo_corpus.json").write_text(text)
    with pytest.raises(ValueError, match="Corpora diferentes"):
        suite.compare_corpora(tmp_path, ["a", "b"])


def test_all_new_prompt_templates_format_without_missing_fields():
    text = PLAN_PROMPT.format(question="q", contract="null", vocabulary="", evidence="",
                              feedback="[]", max_plans=3, max_atoms=4)
    assert '"temporal_operands":["r1","r2"]' in text
    assert "QUESTION: q" in text
    assert CHECK_PROMPT.format(question="q", contract="{}", plan="{}", candidates="")
