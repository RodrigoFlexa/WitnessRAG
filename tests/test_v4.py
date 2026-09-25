"""Design v4: typed variables, item-level set proofs, proof delivered as source
turns, abductive premises, the yes/no reason option of the reader and k_W
proportional to k.

Every option is off by default, and a run without them must reproduce design
v3 exactly (same prompts, same contexts). No test calls a model server.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from test_proof_controller import CHAIN, POOL, QUESTION, FakeLLM, retriever
from test_witness import ExactEmbedder
from wrag import config as C
from wrag import prompts
from wrag.data import Corpus, Passage, Question
from wrag.eval import reader as reader_mod
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import LLMResult
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness.plan import CARDINALITY_ALL, plan_from_data
from wrag.witness.scoring import WeightLevels

LEVELS = WeightLevels(0.0, 0.3, 0.0, 0.1)

# -- a memory where one person practises many things --------------------------

SPORTS = [
    ("s0", "8 May, 2023", "D1:1", "Omar", "I started kickboxing classes.", "kickboxing"),
    ("s1", "20 June, 2023", "D2:1", "Omar", "Taekwondo keeps me focused.", "taekwondo"),
    ("s2", "27 June, 2023", "D3:1", "Omar", "Basketball on Sundays with friends.", "basketball"),
    ("s3", "6 August, 2023", "D4:1", "Omar", "Chess helps me relax.", "chess"),
    ("s4", "1 September, 2023", "D5:1", "Omar", "Yoga in the morning.", "yoga"),
    ("s5", "2 October, 2023", "D6:1", "Lia", "I paint landscapes and love museums.", ""),
]


def sports_retriever(llm, **overrides):
    passages = [Passage(pid=pid, title=f"chunk {pid}",
                        text=f"Session date: {day}\n[{turn}] {speaker}: {text}")
                for pid, day, turn, speaker, text, _o in SPORTS]
    corpus = Corpus(name="toy", passages=passages, questions=[])
    facts = [Fact(fid=f"f{i}", subject="Omar", relation="pratica", object=obj, pid=pid,
                  confidence=0.9)
             for i, (pid, _d, _t, _s, _x, obj) in enumerate(SPORTS) if obj]
    facts.append(Fact(fid="f9", subject="Lia", relation="pinta", object="paisagens",
                      pid="s5", confidence=0.9))
    facts.append(Fact(fid="f10", subject="Lia", relation="ama", object="museus",
                      pid="s5", confidence=0.9))
    embedder = ExactEmbedder()
    kg = build_graph(corpus, ExtractionResult(facts=facts), embedder, C.GraphConfig(),
                     with_passage_nodes=True)
    run = C.RunConfig()
    run.witness = C.WitnessConfig(grounding_mode="exact", entity_match_threshold=0.5,
                                  candidates_per_atom=50, answer_set=True,
                                  proof_controller=True, hybrid_fallback=True,
                                  vocabulary_aware_compile=True)
    for key, value in overrides.items():
        setattr(run.witness, key, value)
    r = WitnessRAGRetriever(IndexContext(corpus, llm, embedder, run, kg=kg))
    r.index()
    return r


SET_PLAN = {"answer_var": "x", "atoms": [{"relation": "pratica", "subject": "Omar",
                                          "object": "?x"}],
            "aggregation": "set", "types": {"x": "martial art"},
            "period": {"reference": "now", "text": ""}, "time_weight": "normal",
            "importance_weight": "normal", "fallback": "Omar martial arts"}
SET_Q = Question("qs", "What martial arts has Omar done?", ["SECRET"], qtype="multi-hop")
SET_POOL = (["s5", "s4", "s3", "s2", "s1", "s0"], [0.06, 0.05, 0.04, 0.03, 0.02, 0.01])


class TypeAwareLLM(FakeLLM):
    """The verifier rejects members that are not martial arts."""

    def chat(self, prompt, **kwargs):
        if kwargs.get("stage") == "witness.confirm":
            self.calls.append(("witness.confirm", prompt))
            supported, rejected = [], []
            for number in range(1, 20):
                marker = f"A{number}: "
                if marker not in prompt:
                    continue
                answer = prompt.split(marker, 1)[1].splitlines()[0].strip()
                if answer in {"kickboxing", "taekwondo"}:
                    supported.append(f"A{number}")
                else:
                    rejected.append({"id": f"A{number}", "reason": "wrong_type"})
            return LLMResult(text=json.dumps({"supported": supported, "rejected": rejected}))
        return super().chat(prompt, **kwargs)


# -- plan ------------------------------------------------------------------------

def test_plan_reads_types_and_hypothesis_and_keeps_them_in_signature():
    plan = plan_from_data(dict(SET_PLAN), SET_Q, None, LEVELS)
    assert plan.query.types == {"x": "martial art"}
    assert "?x must be a martial art" in plan.describe()
    untyped = plan_from_data({k: v for k, v in SET_PLAN.items() if k != "types"},
                             SET_Q, None, LEVELS)
    assert plan.signature() != untyped.signature()
    # A type on a variable the plan does not use, a generic type, or an
    # over-long one is dropped; the plan stays valid.
    odd = plan_from_data(dict(SET_PLAN, types={"z": "gift", "x": "thing"}), SET_Q, None, LEVELS)
    assert odd.valid and odd.query.types == {}
    legacy = plan_from_data(dict(SET_PLAN, types=None, answer_type="martial art"),
                            SET_Q, None, LEVELS)
    assert legacy.query.types == {"x": "martial art"}
    hyp = plan_from_data(dict(SET_PLAN, hypothesis={"about": "Lia", "concepts": [
        "Lia pinta paisagens", "", 3]}), SET_Q, None, LEVELS)
    assert hyp.hypothesis == {"about": "Lia", "concepts": ["Lia pinta paisagens"]}
    assert plan_from_data(dict(SET_PLAN, hypothesis={"about": "Lia"}), SET_Q, None,
                          LEVELS).hypothesis == {}


def test_v3_prompt_is_unchanged_without_the_options_and_extended_with_them():
    llm = FakeLLM([CHAIN])
    retriever(llm)._retrieve_proof(QUESTION, 3, *POOL)
    v3_prompt = llm.calls[0][1]
    assert '"types"' not in v3_prompt and '"hypothesis"' not in v3_prompt
    assert prompts.plan_template(False, False) is prompts.PLAN_TEMPLATE
    llm4 = FakeLLM([CHAIN])
    retriever(llm4, typed_variables=True, abductive_premises=True)._retrieve_proof(
        QUESTION, 3, *POOL)
    v4_prompt = llm4.calls[0][1]
    # The fields are in the task, in the JSON shape and in the worked examples.
    assert '5. "types"' in v4_prompt and '6. "hypothesis"' in v4_prompt
    assert '"types": {"x": "kind named by the question, or omit"}' in v4_prompt
    assert '"subject":"Nira","object":"?x"}],"aggregation":"none","types":{"x":"musical instrument"}' in v4_prompt
    assert 'Question: "Would Paulo enjoy a jazz festival?"' in v4_prompt
    only_types = prompts.plan_template(True, False)
    assert '"hypothesis"' not in only_types and 'martial art' in only_types
    only_hyp = prompts.plan_template(False, True)
    assert '"types"' not in only_hyp and "Paulo" in only_hyp
    for variant in (only_types, only_hyp, prompts.plan_template(True, True)):
        variant.format(question="q", max_atoms=4, vocabulary="", evidence="", feedback="")
        for word in ("single-hop", "multi-hop", "open-domain", "category"):
            assert word not in variant.lower()


def test_a_type_is_ignored_when_the_option_is_off():
    llm = FakeLLM([SET_PLAN])
    r = sports_retriever(llm)
    result = r._retrieve_proof(SET_Q, 3, *SET_POOL)
    # v3: five members is not selective.
    assert result.diagnostics["motivo_parada"] == "respostas_demais"
    assert "trechos_extras" not in result.diagnostics


# -- item-level set proofs -----------------------------------------------------------

def test_item_set_proof_is_verified_per_member_and_delivered_as_turns():
    llm = TypeAwareLLM([SET_PLAN])
    r = sports_retriever(llm, typed_variables=True, item_set_proofs=True,
                         witness_delivery="excerpts")
    result = r._retrieve_proof(SET_Q, 3, *SET_POOL)
    d = result.diagnostics
    assert d["motivo_parada"] == "prova_confirmada" and d["rota"] == "prova_falas"
    # The k passages of the search are untouched.
    assert result.pids == SET_POOL[0][:3] and not d["contexto_alterado_pelo_witness"]
    block = d["trechos_extras"][0]["text"]
    assert d["trechos_extras"][0]["title"] == prompts.EXCERPT_BLOCK_TITLE
    assert "kickboxing" in block and "Taekwondo" in block
    assert "Basketball" not in block and "Chess" not in block
    assert "[D1:1] Omar: I started kickboxing classes." in block
    assert "Session date: 8 May, 2023" in block
    # Dialogue only: no extracted answer or triple that the reader could copy.
    assert "answer:" not in block and "pratica" not in block
    assert d["contexto_alterado_por_trechos"]
    # One plan and one verification; the verifier saw the type.
    assert llm.stages() == ["witness.plan", "witness.confirm"]
    assert "must be a martial art" in llm.calls[1][1]


def test_item_mode_with_pages_respects_k_w():
    llm = TypeAwareLLM([SET_PLAN])
    r = sports_retriever(llm, typed_variables=True, item_set_proofs=True)
    result = r._retrieve_proof(SET_Q, 3, *SET_POOL)
    d = result.diagnostics
    assert d["motivo_parada"] == "prova_confirmada"
    assert result.pids[0] == "s5"
    assert len(set(result.pids) - set(SET_POOL[0][:3])) <= 2


def test_untyped_set_with_too_many_members_keeps_the_v3_refusal():
    plan = {k: v for k, v in SET_PLAN.items() if k != "types"}
    llm = TypeAwareLLM([plan])
    r = sports_retriever(llm, typed_variables=True, item_set_proofs=True,
                         witness_delivery="excerpts", proof_set_max_items=3,
                         proof_cycles=1)
    untyped_q = Question("qu", "What has Omar done?", ["SECRET"], qtype="multi-hop")
    result = r._retrieve_proof(untyped_q, 3, *SET_POOL)
    assert result.diagnostics["motivo_parada"] == "respostas_demais"
    assert "witness.confirm" not in llm.stages()


def test_the_kind_named_by_the_question_types_a_plan_without_types():
    from wrag.witness.query import question_type_phrase
    assert question_type_phrase("What martial arts has John done?") == "martial art"
    assert question_type_phrase("What kind of books does Caroline have?") == "book"
    assert question_type_phrase("Which hobbies did Evan pursue?") == "hobby"
    for none in ("What did Melanie buy?", "What items has Melanie bought?",
                 "Where has Melanie camped?", "What is Caroline's identity?"):
        assert question_type_phrase(none) == ""
    plan = {k: v for k, v in SET_PLAN.items() if k != "types"}
    llm = TypeAwareLLM([plan])
    r = sports_retriever(llm, typed_variables=True, item_set_proofs=True,
                         witness_delivery="excerpts")
    result = r._retrieve_proof(SET_Q, 3, *SET_POOL)
    d = result.diagnostics
    assert d["plano_final"]["consulta"]["types"] == {"x": "martial art"}
    assert "tipo_da_pergunta" in d["plano_final"]["reparos"]
    assert d["motivo_parada"] == "prova_confirmada"
    # The same holds for a one-value plan: the kind goes to the verifier.
    llm_t = FakeLLM([dict(CHAIN)])
    off = retriever(llm_t, typed_variables=True)._retrieve_proof(
        Question("q", "Which city is Ana's employer in?", ["x"]), 3, *POOL)
    assert off.diagnostics["plano_final"]["consulta"]["types"] == {"x": "city"}


def test_value_plans_keep_v3_rules_under_item_mode():
    llm = FakeLLM([CHAIN])
    base = retriever(FakeLLM([CHAIN]))._retrieve_proof(QUESTION, 3, *POOL)
    v4 = retriever(llm, item_set_proofs=True, typed_variables=True)._retrieve_proof(
        QUESTION, 3, *POOL)
    assert v4.pids == base.pids
    assert v4.diagnostics["motivo_parada"] == base.diagnostics["motivo_parada"]


def test_excerpt_delivery_of_a_chain_keeps_the_passages():
    llm = FakeLLM([CHAIN])
    result = retriever(llm, witness_delivery="excerpts")._retrieve_proof(QUESTION, 3, *POOL)
    d = result.diagnostics
    assert result.pids == POOL[0][:3]
    assert d["motivo_parada"] == "prova_confirmada" and d["rota"] == "prova_falas"
    block = d["trechos_extras"][0]["text"]
    assert "Atlas is located in Recife." in block and "I work at Atlas now" in block
    assert llm.stages() == ["witness.plan", "witness.confirm"]


def test_a_rejected_proof_adds_no_excerpt():
    verdict = {"supported": [], "rejected": [{"id": "A1", "reason": "wrong_entity"}]}
    llm = FakeLLM([CHAIN], verdict=verdict)
    result = retriever(llm, witness_delivery="excerpts", proof_cycles=1)._retrieve_proof(
        QUESTION, 3, *POOL)
    assert "trechos_extras" not in result.diagnostics
    assert result.pids == POOL[0][:3]


# -- abductive premises ---------------------------------------------------------------

def test_abductive_premises_come_from_the_neighbourhood_of_the_person():
    plan = {"answer_var": "x", "atoms": [{"relation": "ama", "subject": "Lia", "object": "?x"}],
            "aggregation": "none",
            "hypothesis": {"about": "Lia", "concepts": ["Lia pinta paisagens"]}}
    question = Question("qa", "Would Lia enjoy an art class?", ["SECRET"], qtype="open-domain")
    llm = FakeLLM([plan])
    r = sports_retriever(llm, abductive_premises=True)
    result = r._retrieve_proof(question, 3, *SET_POOL)
    d = result.diagnostics
    assert result.pids == SET_POOL[0][:3]
    assert d["premissas"]["sobre"] == "Lia" and d["premissas"]["n_fatos_vizinhanca"] == 2
    extra = [b for b in d["trechos_extras"] if b["title"].startswith("Statements about Lia")]
    assert extra and "I paint landscapes" in extra[0]["text"]
    # Premises are support, not proof: no extra model call.
    assert llm.stages() == ["witness.plan"]
    off = sports_retriever(FakeLLM([plan]))._retrieve_proof(question, 3, *SET_POOL)
    assert "premissas" not in off.diagnostics


# -- reader ------------------------------------------------------------------------

class ReaderLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def chat(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return LLMResult(text=json.dumps({"answer": self.answer}))


def test_reader_appends_blocks_and_keeps_the_reason_only_with_the_option():
    corpus = Corpus(name="locomo", passages=[Passage(pid="p", title="t", text="Session date: 1 May, 2023\n[D1:1] A: hi")],
                    questions=[])
    question = Question("q", "Would Caroline pursue writing as a career?", ["x"],
                        dataset="locomo", qtype="open-domain")
    llm = ReaderLLM("Likely no; she wants to be a counselor")
    cfg = C.QAConfig(evidence_reader=True)
    plain = reader_mod.read(llm, corpus, question, ["p"], cfg,
                            extra_passages=[{"title": "Extra", "text": "BLOCK"}])
    assert plain.answer == "likely no"
    assert "[2] Extra\nBLOCK" in llm.prompts[-1]
    explained = reader_mod.read(llm, corpus, question, ["p"],
                                C.QAConfig(evidence_reader=True, yesno_rationale=True))
    assert explained.answer == "Likely no; she wants to be a counselor"
    assert "a semicolon and the reason" in llm.prompts[-1]
    assert "a semicolon and the reason" not in llm.prompts[0]


# -- k_W and flags ------------------------------------------------------------------

def test_edit_fraction_scales_k_w_and_reproduces_v3_at_k5():
    r = retriever(FakeLLM([CHAIN]), proof_edit_fraction=0.4)
    composite = plan_from_data(CHAIN, QUESTION, None, LEVELS)
    simple = plan_from_data({"answer_var": "x", "atoms": [
        {"relation": "localizada em", "subject": "Atlas", "object": "?x"}]}, QUESTION, None, LEVELS)
    assert (r._edit_limit(composite, 5), r._edit_limit(simple, 5)) == (2, 1)
    assert (r._edit_limit(composite, 20), r._edit_limit(simple, 20)) == (8, 4)
    fixed = retriever(FakeLLM([CHAIN]))
    assert (fixed._edit_limit(composite, 20), fixed._edit_limit(simple, 20)) == (2, 1)


def test_pilot_flags_reach_the_config_and_are_guarded(tmp_path):
    from wrag.pilot import _run_config, make_plan, parser
    base = ["--gpu", "0", "--dataset", "locomo", "--evidence-reader", "--answer-set"]
    args = parser().parse_args(base + ["--proof-controller", "--typed-variables",
                                       "--item-set-proofs", "--witness-delivery", "excerpts",
                                       "--abductive-premises", "--proof-edit-fraction", "0.4",
                                       "--yesno-rationale"])
    cfg = _run_config(make_plan(args, tmp_path)["settings"], 1)
    w = cfg.witness
    assert w.typed_variables and w.item_set_proofs and w.abductive_premises
    assert w.witness_delivery == "excerpts" and w.proof_edit_fraction == 0.4
    assert cfg.qa.yesno_rationale
    default = _run_config(make_plan(parser().parse_args(base + ["--proof-controller"]),
                                    tmp_path / "d")["settings"], 1)
    assert not default.witness.typed_variables and default.witness.witness_delivery == "pages"
    assert default.witness.proof_edit_fraction == 0.0 and not default.qa.yesno_rationale
    for bad in (["--typed-variables"], ["--witness-delivery", "excerpts"],
                ["--proof-controller", "--proof-edit-fraction", "2"]):
        with pytest.raises(ValueError):
            make_plan(parser().parse_args(base + bad), tmp_path / "bad")
    with pytest.raises(ValueError):
        make_plan(parser().parse_args(["--gpu", "0", "--dataset", "locomo",
                                       "--yesno-rationale"]), tmp_path / "bad2")


def test_a_value_proof_already_in_the_passages_adds_no_unverified_text():
    plan = {"answer_var": "x", "atoms": [{"relation": "pinta", "subject": "Carla",
                                          "object": "?x"}], "aggregation": "none"}
    llm = FakeLLM([plan])
    result = retriever(llm, witness_delivery="excerpts")._retrieve_proof(QUESTION, 3, *POOL)
    assert result.diagnostics["motivo_parada"] == "prova_ja_no_contexto"
    assert "trechos_extras" not in result.diagnostics
    assert llm.stages() == ["witness.plan"]


def test_mixed_delivery_swaps_what_fits_and_sends_the_rest_as_turns():
    llm = TypeAwareLLM([SET_PLAN])
    r = sports_retriever(llm, typed_variables=True, item_set_proofs=True,
                         witness_delivery="mixed", proof_max_new_passages=1)
    result = r._retrieve_proof(SET_Q, 3, *SET_POOL)
    d = result.diagnostics
    assert d["motivo_parada"] == "prova_confirmada" and d["rota"] == "prova_trechos_e_falas"
    new = set(result.pids) - set(SET_POOL[0][:3])
    assert len(new) == 1 and result.pids[:2] == SET_POOL[0][:2]
    block = d["trechos_extras"][0]["text"]
    placed = "s0" in new
    assert ("Taekwondo" in block) == placed and ("kickboxing" in block) == (not placed)
    # With room for both, mixed is the v3 swap and adds no text.
    llm2 = TypeAwareLLM([SET_PLAN])
    both = sports_retriever(llm2, typed_variables=True, item_set_proofs=True,
                            witness_delivery="mixed")._retrieve_proof(SET_Q, 3, *SET_POOL)
    assert {"s0", "s1"} <= set(both.pids) and "trechos_extras" not in both.diagnostics


def test_mixed_delivery_of_a_value_proof_is_the_v3_swap():
    v3 = retriever(FakeLLM([CHAIN]))._retrieve_proof(QUESTION, 3, *POOL)
    mixed = retriever(FakeLLM([CHAIN]), witness_delivery="mixed")._retrieve_proof(
        QUESTION, 3, *POOL)
    assert mixed.pids == v3.pids and "trechos_extras" not in mixed.diagnostics
