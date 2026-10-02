"""Temporal reference (plan -> execution -> reflection) and the reflection loop.

A scripted model replaces the LLM. No benchmark gold, supporting passage or
category is available to any component under test.
"""
import json

from test_witness import ExactEmbedder
from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import LLMResult
from wrag.methods.base import IndexContext
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.witness import reflection_loop as RL
from wrag.witness.local_plans_v2 import MultiOriginPlanner
from wrag.witness.temporal_reference import (incomplete_date, parse_reference, plan_reference,
                                             TemporalReference)


class Script:
    """Replies by stage; a callable receives the prompt."""
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def chat(self, prompt, system=None, params=None, stage="misc"):
        self.calls.append((stage, prompt))
        reply = self.replies.get(stage, "")
        text = reply(prompt) if callable(reply) else reply
        if isinstance(text, (dict, list)):
            text = json.dumps(text)
        return LLMResult(text=text or "", prompt_tokens=10, completion_tokens=5)

    def stages(self):
        return [stage for stage, _ in self.calls]


ROWS = [
    # subject, relation, object, statement, time, session
    ("John", "return from", "Chicago", "John returned from Chicago.", "yesterday", "10 August, 2023"),
    ("John", "meet", "teammates", "John met his teammates.", "", "15 August, 2023"),
    ("John", "visit", "museum", "John visited a museum.", "", "1 July, 2023"),
    ("Joanna", "travel to", "Woodhaven", "Joanna travelled to Woodhaven.", "July 2022", "2 August, 2022"),
    ("Joanna", "travel to", "Lisbon", "Joanna travelled to Lisbon.", "March 2023", "4 April, 2023"),
]


def build(llm, **witness):
    passages, facts = [], []
    for i, (subject, relation, obj, statement, when, session) in enumerate(ROWS):
        pid = f"p{i}"
        passages.append(Passage(pid=pid, title=pid,
                                text=f"Session date: {session}\n[D{i}:1] {subject}: {statement}"))
        facts.append(Fact(fid=f"f{i}", subject=subject, relation=relation, object=obj, pid=pid,
                          turn_id=f"D{i}:1", time=when, statement=statement,
                          kind="past"))
    corpus = Corpus(name="toy", passages=passages, questions=[])
    embedder = ExactEmbedder()
    kg = build_graph(corpus, ExtractionResult(facts=facts), embedder, C.GraphConfig(), with_passage_nodes=True)
    options = dict(local_plans=True, local_plan_version="v2", proof_controller=True, grounding_mode="exact",
                   fact_delivery="facts", fact_fill="question", answer_set=True, local_plan_beam=16,
                   local_plan_candidates=48, fact_time="both", temporal_reference=True)
    options.update(witness)
    cfg = C.RunConfig(witness=C.WitnessConfig(**options))
    r = WitnessRAGRetriever(IndexContext(corpus, llm, embedder, cfg, kg=kg))
    r.index()
    return r


def q(text):
    return Question(qid="q", dataset="locomo", question=text, answers=["SECRET_GOLD"])


def ref(**fields):
    base = {"target": "other", "event": "", "scope": "none", "window": "", "relation": "",
            "anchor_event": "", "order": "none", "granularity": "any"}
    base.update(fields)
    return base


def test_parse_reference_vocabulary_and_repairs():
    good = parse_reference(ref(target="when", scope="relative", relation="since",
                               anchor_event="moving to Porto", order="latest"))
    assert good.valid and good.target == "date" and good.scope == "event"
    assert good.relation == "after" and good.order == "last" and good.repairs
    empty_window = parse_reference(ref(scope="window", window=""))
    assert empty_window.scope == "none" and "window_without_period->none" in empty_window.repairs
    assert not parse_reference("not json").valid
    assert not TemporalReference().active()


def test_planner_prompt_sees_only_the_question_and_present_date():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned")})
    r = build(llm)
    r._build_dated_memory()
    plan_reference(r, "When did John return from Chicago?")
    plan_reference(r, "When did John return from Chicago?")
    prompts = [p for s, p in llm.calls if s == "plan.temporal_reference"]
    assert len(prompts) == 1  # cached per question
    assert "15 August 2023" in prompts[0] and "SECRET_GOLD" not in prompts[0]
    assert "teammates" not in prompts[0] and "Woodhaven" not in prompts[0]


def contract_of(r, text):
    c = MultiOriginPlanner(r, q(text)).contract
    return (c.operation, c.period.kind, str(c.period.interval), c.temporal_side, c.time_weight, c.reference_fact)


def test_reference_never_rewrites_the_search_contract():
    """v3 on conv03: LLM-driven plan changes cost 9.6 F1 on the 17 questions they touched."""
    cases = [("What did John do after returning from Chicago?",
              ref(scope="event", relation="after", anchor_event="returning from Chicago")),
             ("Where did Joanna travel that July?", ref(scope="window", window="July 2022", relation="within")),
             ("What did Joanna write on 2 August 2022?", ref(target="date", scope="window", window="2 August 2022")),
             ("When did John first visit a museum?", ref(target="date", order="first"))]
    for text, reply in cases:
        with_ref = build(Script({"plan.temporal_reference": reply}))
        without = build(Script({}), temporal_reference=False)
        assert contract_of(with_ref, text) == contract_of(without, text), text


def test_a_date_inside_the_question_is_a_scope_not_the_target():
    r = build(Script({"plan.temporal_reference": ref(target="date", scope="window", window="2 August 2022")}))
    planner = MultiOriginPlanner(r, q("What did Joanna write on 2 August 2022?"))
    assert planner.reference_applied == []
    r2 = build(Script({"plan.temporal_reference": ref(target="date", granularity="month")}))
    assert MultiOriginPlanner(r2, q("In which month did Joanna travel to Woodhaven?")).reference_applied == ["date_question"]


def test_date_question_lists_candidate_times_at_stated_precision():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned from Chicago",
                                                 granularity="day")})
    r = build(llm)
    result = r.retrieve(q("When did John return from Chicago?"))
    text = result.diagnostics["trechos_extras"][0]["text"]
    times = result.diagnostics["temporal_reference"]["candidate_times"]
    assert times and times[0]["fact"] == 0
    assert "the day before 10 August 2023" in times[0]["time"]
    # The reader's context stays the v2 context; times serve the reflection.
    assert "Times of the requested event" not in text


def test_time_reference_ablation_makes_no_call_and_no_temporal_block():
    llm = Script({"plan.temporal_reference": ref(target="date")})
    r = build(llm, study_ablation="no-time-reference")
    result = r.retrieve(q("When did John return from Chicago?"))
    assert "plan.temporal_reference" not in llm.stages()
    assert result.diagnostics.get("temporal_reference") is None
    assert "Times of the requested event" not in result.diagnostics["trechos_extras"][0]["text"]


def test_no_witness_keeps_question_only_reference_for_the_reflection():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned")})
    r = build(llm, study_ablation="no-witness")
    result = r.retrieve(q("When did John return from Chicago?"))
    temporal = result.diagnostics["temporal_reference"]
    assert temporal["reference"]["target"] == "date" and temporal["applied"] == ["date_question"]
    assert result.diagnostics["local_plans"]["generated"] == 0
    assert "Times of the requested event" not in result.diagnostics["trechos_extras"][0]["text"]


def test_date_signals():
    assert incomplete_date("the 17th") and incomplete_date("15th after his trip")
    assert not incomplete_date("17 August 2023") and not incomplete_date("the week before 9 June 2023")
    temporal = {"operation": "date", "now": "2023-08-15", "reference": ref(target="date"), "applied": ["date_question"],
                "candidate_times": [{"time": "9 August 2023", "start": "2023-08-09", "end": "2023-08-09"}]}
    marks, triggered = RL.signals(q("When did John return from Chicago?"), ["the 17th", "3 May 2023", "9 August 2023"], temporal)
    assert triggered and any("not a complete date" in m for m in marks)
    assert any('"3 May 2023" matches none' in m for m in marks)
    assert not any('"9 August 2023" matches none' in m for m in marks)
    marks, triggered = RL.signals(q("When did John return from Chicago?"), ["9 August 2023"], temporal)
    assert not triggered


# -- reflection loop -------------------------------------------------------------

def readers(joint, plain):
    def reply(prompt):
        return {"answer": joint if "Perform reflection internally" in prompt else plain}
    return reply


def qa_cfg():
    return C.QAConfig(evidence_reader=True, reader_reflection=True, reflection_loop=True, max_tokens=128)


def run_loop(llm, question="When did John return from Chicago?"):
    r = build(llm)
    retrieval = r.retrieve(q(question))
    return RL.reflective_read(r, r.corpus, q(question), retrieval, qa_cfg(), "witnessrag")


def verifier_calls(llm):
    return [p for s, p in llm.calls if s == "qa.reflection_verify"]


def test_agreement_without_signal_skips_the_verifier():
    llm = Script({"plan.temporal_reference": ref(target="date"),
                  "qa": readers("the day before 10 August 2023", "the day before 10 August 2023"),
                  "qa.reflection_verify": {"decision": "revise", "answer": "WRONG"}})
    _, reading = run_loop(llm)
    assert not verifier_calls(llm) and reading.answer == "the day before 10 August 2023"
    assert reading.reflection["final_source"] == "agreement" and not reading.reflection["triggered"]


def test_verifier_sees_ids_signals_reference_and_can_revise():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned from Chicago"),
                  "qa": readers("the 9th", "15 August 2023"),
                  "qa.reflection_verify": {"analysis": "F0 dates the return.", "decision": "revise",
                                           "answer": "the day before 10 August 2023", "evidence": ["F0", "F999"]}})
    result, reading = run_loop(llm)
    prompt = verifier_calls(llm)[0]
    assert "[F0]" in prompt and "SECRET_GOLD" not in prompt
    assert "The proposed answers disagree." in prompt and "not a complete date" in prompt
    assert "TIME REFERENCE" in prompt
    assert prompt.index("1. the 9th") < prompt.index("2. 15 August 2023")  # v2 (joint) reading first
    assert reading.answer == "the day before 10 August 2023"
    trace = result.diagnostics["reflection_loop"]
    assert trace["passes"][0]["evidence"] == [0]
    assert reading.reflection["final_source"] == "revised" and not trace["searched"]


def test_accept_keeps_the_exact_proposal_and_invalid_falls_back_to_v2_reading():
    llm = Script({"plan.temporal_reference": ref(target="date"),
                  "qa": readers("9 August 2023", "15 August 2023"),
                  "qa.reflection_verify": {"analysis": "", "decision": "accept", "answer": "9 august 2023."}})
    _, reading = run_loop(llm)
    assert reading.answer == "9 August 2023" and reading.reflection["final_source"] == "accepted_proposal"
    llm2 = Script({"plan.temporal_reference": ref(target="date"),
                   "qa": readers("9 August 2023", "10 August 2023"), "qa.reflection_verify": "garbage"})
    _, reading2 = run_loop(llm2)
    assert reading2.answer == "9 August 2023" and reading2.reflection["final_source"] == "fallback_default_reading"


def test_guard_rejects_abstention_over_an_answer():
    llm = Script({"plan.temporal_reference": ref(),
                  "qa": readers("insufficient information", "a museum"),
                  "qa.reflection_verify": {"analysis": "", "decision": "accept", "answer": "insufficient information"}})
    result, reading = run_loop(llm, "What did John visit?")
    assert reading.answer == "a museum" and reading.reflection["final_source"] == "abstention_rejected"
    assert not result.diagnostics["reflection_loop"]["searched"]


def test_guard_drops_added_qualifiers_but_lists_may_grow():
    llm = Script({"plan.temporal_reference": ref(),
                  "qa": readers("the museum hall", "a museum"),
                  "qa.reflection_verify": {"analysis": "", "decision": "revise", "answer": "a museum, a big building in town"}})
    _, reading = run_loop(llm, "What did John visit?")
    assert reading.answer == "a museum" and reading.reflection["final_source"] == "revision_qualifier_dropped"
    llm2 = Script({"plan.temporal_reference": ref(),
                   "qa": readers("Woodhaven", "Lisbon"),
                   "qa.reflection_verify": {"analysis": "", "decision": "revise", "answer": "Woodhaven, Lisbon"}})
    _, reading2 = run_loop(llm2, "Which places has Joanna travelled to?")
    assert reading2.answer == "Woodhaven, Lisbon"


def test_search_is_bounded_to_one_retry_and_second_pass_cannot_search():
    verdicts = iter([
        {"analysis": "No fact about the meeting.", "decision": "search", "answer": "",
         "missing": "John met his teammates", "evidence": ["F0"]},
        {"analysis": "F1 states it.", "decision": "search", "answer": "", "missing": "again"},
    ])
    llm = Script({"plan.temporal_reference": ref(),
                  "qa": readers("insufficient information", "insufficient information"),
                  "qa.reflection_verify": lambda prompt: next(verdicts)})
    result, reading = run_loop(llm, "What did John do after returning from Chicago?")
    prompts = verifier_calls(llm)
    assert len(prompts) == 2 and "A further search is not available" in prompts[1]
    trace = result.diagnostics["reflection_loop"]
    assert trace["searched"] and trace["hint"] == "John met his teammates"
    assert trace["passes"][1]["decision"] == "invalid"
    assert len(set(result.diagnostics["fatos_entregues"]["indices"])) <= 40
    assert reading.answer == "insufficient information"


def test_every_reading_abstains_and_verifier_agrees_then_search():
    verdicts = iter([
        {"analysis": "", "decision": "accept", "answer": "insufficient information"},
        {"analysis": "F1.", "decision": "revise", "answer": "met his teammates", "evidence": ["F1"]},
    ])
    llm = Script({"plan.temporal_reference": ref(),
                  "qa": readers("insufficient information", "insufficient information"),
                  "qa.reflection_verify": lambda prompt: next(verdicts)})
    result, reading = run_loop(llm, "What did John do after returning from Chicago?")
    assert result.diagnostics["reflection_loop"]["searched"]
    assert reading.answer == "met his teammates"


def test_order_needs_a_literal_cue_and_session_dates_are_not_event_times():
    from wrag.witness.temporal_reference import ground_order
    finish = ground_order(parse_reference(ref(target="date", order="last")), "When did Joanna finish her book?")
    assert finish.order == "none" and any("no_cue" in r for r in finish.repairs)
    first = ground_order(parse_reference(ref(target="date", order="first")), "When did Joanna first watch it?")
    assert first.order == "first"
    llm = Script({"plan.temporal_reference": ref(target="date", event="John met his teammates")})
    r = build(llm)
    result = r.retrieve(q("When did John meet his teammates?"))
    assert all(row["explicit"] for row in result.diagnostics["temporal_reference"]["candidate_times"])


def test_verifier_chooses_the_time_but_the_reader_keeps_its_form():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned from Chicago"),
                  "qa": readers("last Saturday", "the day before 10 August 2023"),
                  "qa.reflection_verify": {"analysis": "F0.", "decision": "revise", "answer": "9 August 2023"}})
    _, reading = run_loop(llm)
    assert reading.answer == "the day before 10 August 2023"
    assert reading.reflection["final_source"] == "revised_time_kept_reader_form"


def test_yes_no_revision_keeps_only_the_polarity():
    llm = Script({"plan.temporal_reference": ref(),
                  "qa": readers("no", "yes"),
                  "qa.reflection_verify": {"analysis": "F3.", "decision": "revise",
                                           "answer": "yes, both visited it, no doubt"}})
    _, reading = run_loop(llm, "Has Joanna been to Lisbon?")
    assert reading.answer == "yes" and reading.reflection["final_source"] == "revised_polarity"


def test_age_or_event_times_are_not_fragments_and_same_period_is_agreement():
    from datetime import date
    from wrag.witness.temporal_reference import same_period
    assert not incomplete_date("when she was 10") and not incomplete_date("after graduating")
    assert incomplete_date("last Friday") and incomplete_date("two days ago")
    now = date(2023, 9, 1)
    assert same_period(["the week before 24 August 2023", "14 August 2023 - 20 August 2023"], now)
    assert not same_period(["the week before 24 August 2023", "3 May 2023"], now)
    temporal = {"operation": "date", "now": "2023-09-01", "applied": ["date_question"], "reference": ref(target="date")}
    marks, triggered = RL.signals(q("When did Deborah go to a yoga retreat?"),
                                  ["the week before 24 August 2023", "14 August 2023 - 20 August 2023"], temporal)
    assert not triggered and "same period" in marks[0]
    marks, triggered = RL.signals(q("When did Jolene get her first console?"), ["when she was 10"], temporal)
    assert not triggered


def test_verifier_cannot_switch_to_another_form_of_the_same_time():
    llm = Script({"plan.temporal_reference": ref(target="date", event="John returned from Chicago"),
                  "qa": readers("the day before 10 August 2023", "last Thursday"),
                  "qa.reflection_verify": {"analysis": "", "decision": "accept", "answer": "9 August 2023"}})
    _, reading = run_loop(llm)
    assert reading.answer == "the day before 10 August 2023"
    assert reading.reflection["final_source"] == "same_time_kept_default_form"
