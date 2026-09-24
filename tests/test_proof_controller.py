"""Controlador de prova (desenho v3): Registrar, pontuação, Planejar, Provar,
Verificar e Responder.

Nenhum teste usa servidor de modelo. Os pontos centrais: toda pergunta recebe
um plano, o rótulo do benchmark nunca é lido, o contexto só muda por uma prova
confirmada (até k_W trechos, nunca o prefixo protegido) ou por evidência
parcial de um plano composto (uma vaga), e sem peso de tempo ou importância a
ordem é exatamente a do híbrido.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from test_witness import ExactEmbedder
from wrag import config as C
from wrag import prompts
from wrag.data import Corpus, Passage, Question
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import LLMResult
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.confirm import parse_verdict, render_candidates
from wrag.witness.dated_memory import DatedMemory, arousal
from wrag.witness.memory import MemoryView
from wrag.witness.plan import CARDINALITY_ALL, default_plan, plan_from_data
from wrag.witness.query import Atom, ConjunctiveQuery
from wrag.witness.scoring import (MemoryScorer, Period, WeightLevels, Weights,
                                  proximity, resolve_period)
from wrag.witness.search import WitnessSearcher
from wrag.witness.timeline import Interval, format_interval, resolve_expression

# -- memória datada de brinquedo ------------------------------------------------

DIALOGUE = [
    ("p0", "8 May, 2023", "D1:1", "Ana", "I work at Atlas now, I am so happy!",
     ("Ana", "trabalha em", "Atlas", "")),
    ("p1", "20 June, 2023", "D2:1", "Ana", "Atlas is located in Recife.",
     ("Atlas", "localizada em", "Recife", "")),
    ("p2", "27 June, 2023", "D3:1", "Bruno", "I camped in the mountains last week.",
     ("Bruno", "acampou em", "montanhas", "last week")),
    ("p3", "6 August, 2023", "D4:1", "Bruno", "We camped at the beach.",
     ("Bruno", "acampou em", "praia", "")),
    ("p4", "1 September, 2023", "D5:1", "Carla", "I paint landscapes.",
     ("Carla", "pinta", "paisagens", "")),
    ("p5", "2 October, 2023", "D6:1", "Carla", "I read a novel.",
     ("Carla", "le", "romance", "")),
]


def dated_toy():
    passages = [Passage(pid=pid, title=f"chunk {pid}",
                        text=f"Session date: {day}\n[{turn}] {speaker}: {text}")
                for pid, day, turn, speaker, text, _f in DIALOGUE]
    corpus = Corpus(name="toy", passages=passages, questions=[])
    facts = [Fact(fid=f"f{i}", subject=s, relation=r, object=o, pid=row[0], time=t,
                  confidence=0.9)
             for i, row in enumerate(DIALOGUE) for (s, r, o, t) in [row[5]]]
    embedder = ExactEmbedder()
    kg = build_graph(corpus, ExtractionResult(facts=facts), embedder, C.GraphConfig(),
                     with_passage_nodes=True)
    return corpus, kg, embedder


class FakeLLM:
    """Planejador e verificador falsos, com registro dos prompts."""

    def __init__(self, plans, verdict=None):
        self.plans = list(plans)
        self.verdict = verdict if verdict is not None else "all"
        self.calls = []

    def chat(self, prompt, **kwargs):
        stage = kwargs.get("stage")
        self.calls.append((stage, prompt))
        if stage in {"witness.plan", "witness.replan_v3"}:
            plan = self.plans[min(len(self.plans) - 1,
                                  sum(s in {"witness.plan", "witness.replan_v3"}
                                      for s, _p in self.calls) - 1)]
            if plan == "filtered":
                return LLMResult(filtered=True, finish_reason="content_filter")
            return LLMResult(text=plan if isinstance(plan, str) else json.dumps(plan))
        if stage == "witness.confirm":
            if self.verdict == "all":
                ids = [f"A{i}" for i in range(1, 10) if f"A{i}:" in prompt]
                return LLMResult(text=json.dumps({"supported": ids, "rejected": []}))
            return LLMResult(text=json.dumps(self.verdict))
        raise AssertionError(f"unexpected stage {stage}")

    def stages(self):
        return [s for s, _p in self.calls]


def retriever(llm, **overrides):
    corpus, kg, embedder = dated_toy()
    run = C.RunConfig()
    run.witness = C.WitnessConfig(grounding_mode="exact", entity_match_threshold=0.5,
                                  candidates_per_atom=50, answer_set=True,
                                  proof_controller=True, hybrid_fallback=True,
                                  vocabulary_aware_compile=True)
    for key, value in overrides.items():
        setattr(run.witness, key, value)
    ctx = IndexContext(corpus, llm, embedder, run, kg=kg)
    r = WitnessRAGRetriever(ctx)
    r.index()
    return r


CHAIN = {"answer_var": "x", "atoms": [
    {"relation": "trabalha em", "subject": "Ana", "object": "?y"},
    {"relation": "localizada em", "subject": "?y", "object": "?x"}],
    "aggregation": "none", "expected_type": "place",
    "period": {"reference": "now", "text": ""}, "time_weight": "normal",
    "importance_weight": "normal", "fallback": "Ana employer city"}
POOL = (["p4", "p5", "p2", "p3", "p0", "p1"], [0.06, 0.05, 0.04, 0.03, 0.02, 0.01])
QUESTION = Question("q", "Where is the company Ana works at located?", ["SECRET_GOLD"],
                    gold_pids=["p1"], qtype="multi-hop")


# -- Registrar ------------------------------------------------------------------

def test_relative_expressions_resolve_against_the_session_date():
    assert format_interval(resolve_expression("last week", date(2023, 6, 27))) == \
        "19 June 2023 - 25 June 2023"
    assert format_interval(resolve_expression("yesterday", date(2023, 5, 8))) == "7 May 2023"
    assert format_interval(resolve_expression("two weekends ago", date(2023, 7, 17))) == \
        "8 July 2023 - 9 July 2023"
    assert format_interval(resolve_expression("last summer", date(2023, 6, 27))) == \
        "1 June 2022 - 31 August 2022"
    assert format_interval(resolve_expression("12 may", date(2023, 6, 27))) == "12 May 2023"
    assert resolve_expression("on a rock", date(2023, 6, 27)) is None
    assert resolve_expression("last week", None) is None


def test_dated_memory_dates_every_fact_and_keeps_its_source_turn():
    corpus, kg, _e = dated_toy()
    memory = DatedMemory(corpus, kg.facts)
    by_object = {f.object: i for i, f in enumerate(kg.facts)}
    assert memory.fact_time_text(by_object["montanhas"]) == "19 June 2023 - 25 June 2023"
    assert memory.fact_time_text(by_object["praia"]) == "6 August 2023"
    assert memory.fact_time_source[by_object["praia"]] == "sessao"
    assert memory.first == date(2023, 5, 8) and memory.last == date(2023, 10, 2)
    excerpt = memory.excerpt(by_object["montanhas"])
    assert "Bruno: I camped in the mountains" in excerpt and "27 June 2023" in excerpt
    assert memory.fact_importance[by_object["Atlas"]] > memory.fact_importance[by_object["Recife"]]
    assert 0.0 <= arousal("so happy!!") < 1.0


# -- pontuação ----------------------------------------------------------------------

def test_weights_keep_similarity_majority_and_levels_renormalise():
    with pytest.raises(ValueError):
        Weights(0.4, 0.3, 0.3)
    with pytest.raises(ValueError):
        Weights(0.7, 0.2, 0.2)
    levels = WeightLevels(time_normal=0.0, time_strong=0.4, importance_normal=0.0,
                          importance_strong=0.4)
    both = levels.weights("strong", "strong")
    assert abs(both.time + both.importance - 0.5) < 1e-9 and both.similarity == 0.5
    assert levels.weights("normal", "normal").pure_similarity
    assert levels.weights("bogus", "normal").pure_similarity


def test_pure_similarity_is_exactly_the_hybrid_order():
    corpus, kg, _e = dated_toy()
    scorer = MemoryScorer(DatedMemory(corpus, kg.facts))
    fused = dict(zip(*POOL))
    order = scorer.rank(fused, POOL[0], [p.pid for p in corpus.passages], Weights(),
                        Period("now"))
    assert order == POOL[0]


def test_window_period_prefers_passages_of_that_period():
    corpus, kg, _e = dated_toy()
    memory = DatedMemory(corpus, kg.facts)
    scorer = MemoryScorer(memory)
    period, repairs = resolve_period("window", "June 2023", memory)
    assert period.kind == "window" and not repairs
    fused = {"p2": 0.03, "p3": 0.03, "p4": 0.03}
    order = scorer.rank(fused, ["p3", "p4", "p2"], ["p2", "p3", "p4"],
                        Weights(0.7, 0.0, 0.3), period)
    assert order[0] == "p2"
    assert proximity(memory.passage_interval["p2"], period, scorer.scale(period)) == 1.0


def test_similarity_dominance_holds_for_any_valid_weights():
    corpus, kg, _e = dated_toy()
    scorer = MemoryScorer(DatedMemory(corpus, kg.facts))
    period, _ = resolve_period("window", "October 2023", scorer.memory)
    fused = {"p0": 1.0, "p5": 0.0}
    for weights in (Weights(0.5, 0.0, 0.5), Weights(0.5, 0.25, 0.25), Weights(0.5, 0.5, 0.0)):
        scores = scorer.passage_scores(fused, ["p0", "p5"], weights, period)
        assert scores["p5"] <= scores["p0"] + 1e-9


def test_unparsable_window_falls_back_to_now_with_a_repair():
    corpus, kg, _e = dated_toy()
    period, repairs = resolve_period("window", "some day", DatedMemory(corpus, kg.facts))
    assert period.kind == "now" and "janela_nao_interpretada" in repairs


# -- Planejar -----------------------------------------------------------------------

def test_plan_parsing_repairs_time_answers_and_derives_cardinality():
    corpus, kg, _e = dated_toy()
    memory = DatedMemory(corpus, kg.facts)
    question = Question("q", "When did Bruno camp in the mountains?", [])
    plan = plan_from_data({"answer_var": "t", "atoms": [
        {"relation": "acampou em", "subject": "Bruno", "object": "montanhas"}],
        "aggregation": "none", "period": {"reference": "now"}}, question, memory,
        WeightLevels())
    assert plan.valid and plan.executable and plan.query.atoms[0].time == "?t"
    assert "espaco_de_tempo_para_resposta" in plan.repairs
    plan = plan_from_data({"answer_var": "x", "atoms": [
        {"relation": "acampou em", "subject": "Bruno", "object": "?x"}],
        "aggregation": "set", "period": {"reference": "window", "text": "June 2023"},
        "time_weight": "strong", "importance_weight": "loud"}, question, memory, WeightLevels())
    assert plan.cardinality == CARDINALITY_ALL and plan.composite
    assert plan.period.kind == "window" and plan.weights.time == pytest.approx(0.3)
    assert any(r.startswith("peso_importancia_invalido") for r in plan.repairs)
    assert not plan_from_data("junk", question, memory, WeightLevels()).valid


def test_plan_prompt_has_no_category_vocabulary():
    for word in ("multi-hop", "single-hop", "open-domain", "locomo", "category", "benchmark"):
        assert word not in prompts.PLAN_TEMPLATE.lower()
        assert word not in prompts.CONFIRM_TEMPLATE.lower()


# -- Verificar ----------------------------------------------------------------------

def test_verdict_parsing_is_per_candidate_and_explicit_rejection_wins():
    verdict = parse_verdict({"supported": ["A1", "A2", "A9"],
                             "rejected": [{"id": "A2", "reason": "wrong_period"},
                                          {"id": "A3", "reason": "made_up"}]}, 3)
    assert verdict.supported == [0] and verdict.rejected == {1: "wrong_period",
                                                             2: "not_supported"}
    assert not parse_verdict("nope", 2).valid
    text = render_candidates([{"answer": "Recife", "facts": [
        {"triple": ("Atlas", "localizada em", "Recife"), "date": "20 June 2023",
         "excerpt": "[D2:1] Ana: Atlas is located in Recife."}]}])
    assert text.startswith("A1: Recife") and "event date: 20 June 2023" in text


# -- controlador ----------------------------------------------------------------------

def test_confirmed_chain_proof_enters_the_tail_and_keeps_the_prefix():
    llm = FakeLLM([CHAIN])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert d["controlador"] == "prova-v3" and d["rota"] == "prova"
    assert d["motivo_parada"] == "prova_confirmada"
    assert result.pids[0] == "p4" and {"p0", "p1"} <= set(result.pids)
    assert d["contexto_alterado_pelo_witness"] and d["testemunha_no_contexto"]
    assert llm.stages() == ["witness.plan", "witness.confirm"]
    assert d["planejamento"]["chamadas"] == 2


def test_rejected_proof_replans_and_never_changes_the_context_by_proof():
    verdict = {"supported": [], "rejected": [{"id": "A1", "reason": "wrong_entity"}]}
    llm = FakeLLM([CHAIN, dict(CHAIN, importance_weight="strong")], verdict=verdict)
    result = retriever(llm, proof_partial_evidence=False)._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert result.pids == POOL[0][:3]
    assert d["motivo_parada"] == "prova_recusada" and not d["contexto_alterado_pelo_witness"]
    assert llm.stages() == ["witness.plan", "witness.confirm",
                            "witness.replan_v3", "witness.confirm"]
    feedback_prompt = llm.calls[2][1]
    assert "proof rejected by the check (wrong_entity)" in feedback_prompt


def test_repeating_the_same_logical_plan_stops_the_cycle():
    verdict = {"supported": [], "rejected": [{"id": "A1", "reason": "wrong_entity"}]}
    llm = FakeLLM([CHAIN, dict(CHAIN, fallback="other words only")], verdict=verdict)
    result = retriever(llm, proof_partial_evidence=False)._retrieve_proof(QUESTION, 3, *POOL)
    assert llm.stages() == ["witness.plan", "witness.confirm", "witness.replan_v3"]
    assert result.diagnostics["ciclos"][-1]["resultado"] == "plano_repetido"


def test_one_fact_plan_changes_at_most_the_last_slot():
    plan = {"answer_var": "x", "atoms": [{"relation": "localizada em", "subject": "Atlas",
                                          "object": "?x"}], "aggregation": "none"}
    llm = FakeLLM([plan])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert result.pids[:2] == ["p4", "p5"] and result.pids[2] == "p1"


def test_proof_already_in_context_needs_no_verification():
    plan = {"answer_var": "x", "atoms": [{"relation": "pinta", "subject": "Carla",
                                          "object": "?x"}], "aggregation": "none"}
    llm = FakeLLM([plan])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert result.pids == POOL[0][:3]
    assert result.diagnostics["motivo_parada"] == "prova_ja_no_contexto"
    assert llm.stages() == ["witness.plan"]


def test_invalid_and_filtered_plans_fall_back_to_the_hybrid_context():
    llm = FakeLLM(["not json"])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert result.pids == POOL[0][:3] and result.diagnostics["motivo_parada"] == "plano_invalido"
    assert llm.stages() == ["witness.plan", "witness.replan_v3"]
    llm = FakeLLM(["filtered"])
    result = retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    assert result.pids == POOL[0][:3] and result.diagnostics["motivo_parada"] == "plano_filtrado"


def test_benchmark_label_and_gold_are_never_read():
    outputs = []
    for qtype in ("multi-hop", "single-hop", "temporal", ""):
        llm = FakeLLM([CHAIN])
        question = Question("q", QUESTION.question, ["SECRET_GOLD"], gold_pids=["p1"],
                            qtype=qtype)
        result = retriever(llm)._retrieve_proof(question, 3, *POOL)
        outputs.append((result.pids, [p for _s, p in llm.calls]))
        assert all("SECRET_GOLD" not in p and "multi-hop" not in p for _s, p in llm.calls)
    assert all(item == outputs[0] for item in outputs)


def test_time_variable_answers_with_the_resolved_event_date():
    plan = {"answer_var": "t", "atoms": [{"relation": "acampou em", "subject": "Bruno",
                                          "object": "montanhas", "time": "?t"}],
            "aggregation": "none"}
    result = retriever(FakeLLM([plan]))._retrieve_proof(
        Question("q", "When did Bruno camp in the mountains?", []), 3, *POOL)
    assert result.diagnostics["resposta_estrutural"] == "19 June 2023 - 25 June 2023"


def test_window_plan_prefers_the_fact_of_the_period():
    plan = {"answer_var": "x", "atoms": [{"relation": "acampou em", "subject": "Bruno",
                                          "object": "?x"}], "aggregation": "none",
            "period": {"reference": "window", "text": "June 2023"}, "time_weight": "strong"}
    result = retriever(FakeLLM([plan]))._retrieve_proof(
        Question("q", "Where did Bruno camp in June 2023?", []), 3, *POOL)
    assert result.diagnostics["resposta_estrutural"] == "montanhas"
    assert result.diagnostics["periodo"]["tipo"] == "window"


def test_evidence_facts_in_the_plan_prompt_are_dated():
    llm = FakeLLM([CHAIN])
    retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    prompt = llm.calls[0][1]
    assert "Facts found by the first search" in prompt
    assert "19 June 2023 - 25 June 2023" in prompt or "20 June 2023" in prompt


def test_retrieve_entry_point_uses_the_proof_controller():
    llm = FakeLLM([CHAIN])
    r = retriever(llm)
    result = r.retrieve(QUESTION, 3)
    assert result.diagnostics["controlador"] == "prova-v3"
    assert r.index_report()["memoria_datada"]["fatos"] == len(DIALOGUE)


# -- linha de comando -------------------------------------------------------------------

def test_cli_flags_reach_config_and_guard_the_reader(tmp_path):
    from test_pilot import parser
    from wrag.pilot import _run_config, make_plan
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--proof-controller",
                                "--evidence-reader", "--answer-set", "--proof-cycles", "1",
                                "--no-proof-verify", "--partial-evidence"])
    cfg = _run_config(make_plan(args, tmp_path)["settings"], 1)
    assert cfg.witness.proof_controller and cfg.witness.proof_cycles == 1
    assert not cfg.witness.proof_verify and cfg.witness.proof_partial_evidence
    default = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--proof-controller",
                                   "--evidence-reader", "--answer-set"])
    cfg = _run_config(make_plan(default, tmp_path / "d")["settings"], 1)
    assert cfg.witness.proof_cycles == 2 and cfg.witness.proof_verify
    assert not cfg.witness.proof_partial_evidence
    for bad in (["--proof-controller"],
                ["--proof-controller", "--evidence-reader"],
                ["--proof-controller", "--evidence-reader", "--answer-set", "--agnostic-router"],
                ["--proof-controller", "--evidence-reader", "--answer-set", "--selective-witness"],
                ["--proof-cycles", "2", "--evidence-reader"],
                ["--proof-controller", "--evidence-reader", "--answer-set", "--proof-cycles", "9"]):
        with pytest.raises(ValueError):
            make_plan(parser().parse_args(["--gpu", "3", "--dataset", "locomo"] + bad),
                      tmp_path / "bad")


def test_partial_evidence_is_opt_in_and_only_for_connected_plans(monkeypatch):
    # Without a proof, a connected plan may use the last slot for an agreed
    # probe only when the ablation is switched on; one-fact plans never do.
    no_proof = {"supported": [], "rejected": [{"id": "A1", "reason": "not_supported"}]}
    calls = []

    def fake_probe(self, question, query, baseline, k):
        calls.append(query.n_atoms)
        return list(baseline[:k - 1]) + ["p1"], {"adicionada": "p1"}

    monkeypatch.setattr(WitnessRAGRetriever, "_selective_probe_tail", fake_probe)
    default = retriever(FakeLLM([CHAIN], verdict=no_proof), proof_cycles=1)
    assert default._retrieve_proof(QUESTION, 3, *POOL).pids == POOL[0][:3] and not calls
    enabled = retriever(FakeLLM([CHAIN], verdict=no_proof), proof_cycles=1,
                        proof_partial_evidence=True)
    result = enabled._retrieve_proof(QUESTION, 3, *POOL)
    assert result.pids == ["p4", "p5", "p1"] and calls == [2]
    assert result.diagnostics["rota"] == "evidencia_parcial_sondas"
    single = {"answer_var": "x", "atoms": [{"relation": "acampou em", "subject": "Bruno",
                                            "object": "?x"}], "aggregation": "set"}
    calls.clear()
    retriever(FakeLLM([single], verdict=no_proof), proof_cycles=1,
              proof_partial_evidence=True)._retrieve_proof(QUESTION, 3, *POOL)
    assert calls == []


def test_proof_report_pairs_runs(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "proof_report", Path(__file__).resolve().parent.parent / "scripts" / "proof-report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write(root, rows):
        folder = root / "conversations" / "conv00" / "benchmark" / "x" / "locomo"
        folder.mkdir(parents=True)
        (folder / "witnessrag.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    base = [{"qid": f"locomo:conv-{c}:qa{i}", "tipo": t, "f1_locomo": 0.5, "bleu1_locomo": 0.4,
             "recuperadas": ["a", "b"], "resposta": "x", "recall@5": 1.0, "all_recall@5": 1.0,
             "diagnosticos": {"controlador": "selective-v1"},
             "uso_llm": {"por_estagio": {"qa": {"chamadas": 1}}}}
            for i, (c, t) in enumerate([(1, "single-hop"), (1, "multi-hop"), (2, "temporal"),
                                        (2, "open-domain")])]
    proof = [dict(r, f1_locomo=0.6 if r["tipo"] == "multi-hop" else 0.5,
                  diagnosticos={"controlador": "prova-v3", "rota": "base",
                                "motivo_parada": "sem_prova",
                                "contexto_alterado_pelo_witness": False,
                                "ciclos": [{"verificacao": {"chamada": True,
                                                            "confirmadas": [0],
                                                            "recusadas": {}}}]},
                  uso_llm={"por_estagio": {"qa": {"chamadas": 1},
                                           "witness.plan": {"chamadas": 1}}})
             for r in base]
    write(tmp_path / "proof", proof)
    write(tmp_path / "ref", base)
    report = module.main(["--proof", str(tmp_path / "proof"), "--reference",
                          str(tmp_path / "ref")])
    paired = report["paired_reference"]
    assert paired["n"] == 4 and paired["multi-hop"]["delta_f1"] == pytest.approx(0.1)
    assert paired["all"]["same_context"] == 1.0
    assert paired["all"]["same_context_same_answer"] == 1.0
    assert report["calls_by_stage"]["witness.plan"] == 1.0
    assert (tmp_path / "proof" / "proof_report.md").exists()


from test_pipeline import offline  # noqa: E402,F401  (fixture)


def test_end_to_end_runner_with_the_proof_controller(offline):
    from wrag.eval import runner
    from wrag.util import read_jsonl
    tmp, _llm, _questions = offline
    cfg = C.RunConfig(n_questions=3, top_k=2)
    cfg.witness.proof_controller = True
    cfg.witness.hybrid_fallback = True
    cfg.witness.answer_set = True
    root = runner.run(["sample"], ["witnessrag"], cfg, tag="proof")
    rows = read_jsonl(root / "sample" / "witnessrag.jsonl")
    assert len(rows) == 3
    for row in rows:
        diagnostics = row["diagnosticos"]
        assert diagnostics["controlador"] == "prova-v3"
        assert len(row["recuperadas"]) <= 2
        assert "witness.plan" in row["uso_llm"]["por_estagio"]


def test_review_regressions():
    # A proof accepted by Provar always fits the tail that Responder may change.
    fits = WitnessRAGRetriever._fits
    assert fits(["p2", "p3", "p6"], ["p0", "p1", "p2"], 2) is False  # p2 holds a free slot
    assert fits(["p2", "p3"], ["p0", "p1", "p2"], 2) is True
    assert fits(["p3"], ["p0", "p1", "p2"], 2) is True
    assert fits(["p3", "p4"], ["p0", "p1", "p2", "p5", "p6"], 2) is True
    assert fits(["p3", "p4"], ["p0", "p1", "p2", "p5", "p6"], 1) is False
    # Odd verifier output never raises.
    assert parse_verdict({"supported": True, "rejected": 2}, 2).supported == []
    assert parse_verdict({"supported": [{"id": "A2"}]}, 2).supported == [1]
    # Calendar overflow and month-only expressions.
    assert resolve_expression("3000 years ago", date(2023, 1, 1)) is None
    assert format_interval(resolve_expression("in October", date(2023, 11, 5))) == \
        "1 October 2023 - 31 October 2023"
    assert format_interval(resolve_expression("last weekend", date(2023, 7, 16))) == \
        "8 July 2023 - 9 July 2023"
    assert format_interval(resolve_expression("the week before 27 June 2023",
                                              date(2023, 7, 1))) == "20 June 2023 - 26 June 2023"
    # A "when" chain without a time slot dates its last fact; comparisons are sets.
    question = Question("q", "When did the company of Ana move?", [])
    plan = plan_from_data({"answer_var": "t", "atoms": [
        {"relation": "trabalha em", "subject": "Ana", "object": "?y"},
        {"relation": "move to", "subject": "?y", "object": "?z"}], "aggregation": "none"},
        question, None, WeightLevels())
    assert plan.query.answer_var == "t" and plan.query.atoms[1].time == "?t"
    clash = plan_from_data({"answer_var": "x", "atoms": [
        {"relation": "r", "subject": "Ana", "object": "?x", "time": "?x"}]}, question, None,
        WeightLevels())
    assert clash.query.atoms[0].time == "" and "variavel_de_tempo_colide_com_entidade" in clash.repairs
    compare = plan_from_data({"answer_var": "x", "atoms": [
        {"relation": "r", "subject": "Ana", "object": "?x"}], "aggregation": "max"},
        question, None, WeightLevels())
    assert compare.cardinality == CARDINALITY_ALL
