"""Document adaptation, provenance, role/date isolation and unchanged metrics."""
import json
from datetime import date
from types import SimpleNamespace

import pytest

from test_local_plans import NoLLM
from test_witness import ExactEmbedder
from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import LLM, LLMResult
from wrag.methods.base import IndexContext
from wrag.witness.query import Atom
from benchmarks.memoryagentbench.engine import WitnessEngine, standard_config
from benchmarks.memoryagentbench import source_memory as S
from benchmarks.memoryagentbench.__main__ import parser


def item(s="Emma", r="marry", o="Ethelred II", quote=None, turn="S0.0", **extra):
    return {"subject": s, "relation": r, "object": o,
            "quote": quote or "Emma, sister of Richard II, married Ethelred II.",
            "turn": turn, **extra}


def test_document_extraction_keeps_literal_qualifiers_and_directed_relation():
    text = "Emma, sister of Richard II, married Ethelred II."
    p = Passage("p", "", text)
    spans = S.source_spans(text, 0)
    facts, rejected = S.parse_source_facts({"memories": [item()]}, p, spans, conversation=False, limit=40)
    assert not rejected
    assert facts[0].triple == ("Emma", "marry", "Ethelred II")
    assert facts[0].statement == text and facts[0].turn_id == spans[0].sid
    assert not spans[0].when


def test_metadata_and_invented_source_are_rejected_without_deleting_real_facts():
    p = Passage("p", "", "Emma, sister of Richard II, married Ethelred II.")
    data = {"memories": [item(), item(s="User", r="mention", o="Emma"),
                          item(quote="Emma married Richard II."), item(turn="S999.0")]}
    facts, rejected = S.parse_source_facts(data, p, S.source_spans(p.text, 0), conversation=False, limit=40)
    assert len(facts) == 1
    assert rejected == {"ingestion_metadata": 1, "unsupported_source": 2}


def test_real_conversational_reading_is_retained_and_session_dates_are_real():
    text = "Chat Time: 2022/11/17 (Thu) 12:04\n{'role': 'user', 'content': 'I read The Sea Garden.'}"
    p = Passage("p", "", text)
    spans = S.source_spans(text, 0, conversation=True)
    actual = next(s for s in spans if "I read" in s.text)
    facts, rejected = S.parse_source_facts({"memories": [item(s="User", r="read", o="The Sea Garden",
               quote="I read The Sea Garden.", turn=actual.sid)]}, p, spans, conversation=True, limit=40)
    assert not rejected and facts[0].subject == "User"
    assert actual.when == date(2022, 11, 17) and actual.speaker == "User"
    assert "User:" in facts[0].statement
    assert S.source_spans("continued conversation.", 1, conversation=True,
                          initial_date=actual.when, initial_speaker=actual.speaker)[0].when == actual.when


def test_document_dates_are_not_inferred_from_years_or_ingestion_order():
    spans = S.source_spans("Document 1191:\nA fleet arrived in 1191.", 42)
    assert all(s.when is None for s in spans)
    assert all(s.speaker == "Source" for s in spans)
    assert all(s.text == "Document 1191:" or s.text == "A fleet arrived in 1191." for s in spans)


def test_span_offsets_are_literal_and_roles_do_not_cross_into_next_speaker():
    text = "  {'role': 'user', 'content': 'I read A.'}, {'role': 'assistant', 'content': 'Try B.'}  "
    spans = S.source_spans(text, 3, conversation=True)
    for span in spans:
        assert text[span.start:span.end] == span.text
    assert next(s for s in spans if "I read" in s.text).speaker == "User"
    assert next(s for s in spans if "Try B" in s.text).speaker == "Assistant"


def test_whole_sentence_batches_cover_source_exactly(monkeypatch):
    from wrag import ie
    monkeypatch.setattr(ie, "_load_window_tokenizer", lambda *args: SimpleNamespace(encode=lambda t, **kw: t.split()))
    spans = S.source_spans("One complete sentence. Another complete sentence. Final sentence.", 0)
    cfg = C.IEConfig(window_tokens=17)
    batches = S.source_batches(spans, cfg)
    assert len(batches) > 1
    assert [s for batch in batches for s in batch] == spans
    assert all(s.text.endswith(".") for batch in batches for s in batch)


class SourceLLM(LLM):
    name = "source-test"
    deployment = "unit-test"

    def __init__(self, invalid=False):
        super().__init__()
        self.invalid, self.calls = invalid, []

    def _complete(self, messages, params, stage):
        self.calls.append((messages, stage))
        return LLMResult(text=json.dumps({"bad": []} if self.invalid else {"memories": [item()]}))


def test_extraction_cache_includes_source_prompt_and_never_stores_invalid_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    p = Passage("p", "", "Emma, sister of Richard II, married Ethelred II.")
    cfg = C.IEConfig(window_tokens=0)
    spans = S.source_spans(p.text, 0)
    bad = SourceLLM(invalid=True)
    with pytest.raises(ValueError, match="memories list"):
        S.extract_source(p, bad, cfg, spans, conversation=False)
    assert not list(tmp_path.rglob("*.json"))
    llm = SourceLLM()
    first, _ = S.extract_source(p, llm, cfg, spans, conversation=False)
    second, _ = S.extract_source(p, llm, cfg, spans, conversation=False)
    assert first.facts == second.facts and len(llm.calls) == 1
    monkeypatch.setattr(S, "SYSTEM", S.SYSTEM + " changed")
    S.extract_source(p, llm, cfg, spans, conversation=False)
    assert len(llm.calls) == 2
    assert "<User>" not in str(llm.calls) and "SECRET_GOLD" not in str(llm.calls)


def test_truncated_json_retries_all_source_records_instead_of_registering_partial_facts(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    text = "Emma married Ethelred II. Richard ruled Normandy."
    p = Passage("p", "", text)
    spans = S.source_spans(text, 0)

    class TruncatedLLM(SourceLLM):
        def _complete(self, messages, params, stage):
            prompt = messages[-1]["content"]
            self.calls.append((messages, stage))
            if all(f"[{s.sid}]" in prompt for s in spans):
                return LLMResult(text='{"memories":[{"turn":"S0.0"', finish_reason="length")
            span = next(s for s in spans if f"[{s.sid}]" in prompt)
            entry = item(quote=span.text, turn=span.sid) if span == spans[0] else item(
                s="Richard", r="rule", o="Normandy", quote=span.text, turn=span.sid)
            return LLMResult(text=json.dumps({"memories": [entry]}), finish_reason="stop")

    llm = TruncatedLLM()
    result, rejected = S.extract_source(p, llm, C.IEConfig(window_tokens=0), spans, conversation=False)
    assert {f.turn_id for f in result.facts} == {s.sid for s in spans}
    assert len(llm.calls) == result.extraction_windows == 3
    assert rejected == {"schema_split_retries": 1}
    assert not result.blocked_pids
    # Only valid complete child responses enter the source cache.
    assert len(list((tmp_path / "mab-source").glob("*.json"))) == 2


def test_single_record_retry_reduces_item_limit_and_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    p = Passage("p", "", "Emma married Ethelred II.")
    spans = S.source_spans(p.text, 0)

    class LimitLLM(SourceLLM):
        def _complete(self, messages, params, stage):
            self.calls.append((messages, stage))
            if "At most 40 items" in messages[-1]["content"]:
                return LLMResult(text='{"memories":', finish_reason="length")
            return LLMResult(text=json.dumps({"memories": [item(quote=p.text)]}))

    llm = LimitLLM()
    result, rejected = S.extract_source(p, llm, C.IEConfig(window_tokens=0), spans, conversation=False)
    assert len(result.facts) == 1 and len(llm.calls) == 2
    assert "At most 20 items" in str(llm.calls[-1])
    assert rejected["schema_split_retries"] == 1
    bad = SourceLLM(invalid=True)
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "invalid")
    with pytest.raises(ValueError, match="memories list"):
        S.extract_source(p, bad, C.IEConfig(window_tokens=0), spans, conversation=False)
    assert len(bad.calls) == 8  # Four levels, each with one bounded format repair.


def test_quote_format_stall_recovers_literal_source_by_id_and_reuses_separate_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    text = 'The UNFCCC aims to "stabilize greenhouse gas concentrations".'
    p = Passage("p", "", text)
    spans = S.source_spans(text, 180)

    class QuoteLLM(SourceLLM):
        def _complete(self, messages, params, stage):
            self.calls.append((params.json_mode, stage, messages))
            if params.json_mode:
                return LLMResult(text='{"memories":[{"quote":"broken\\\\"',
                                 finish_reason="length", completion_tokens=4000)
            # The model generates a reference, never the supporting quote.
            row = {"turn": spans[0].sid, "subject": "UNFCCC", "relation": "aim",
                   "object": "stabilize greenhouse gas concentrations", "time": "", "kind": "past"}
            return LLMResult(text=json.dumps({"memories": [row]}), finish_reason="stop")

    llm = QuoteLLM()
    cfg = C.IEConfig(window_tokens=0, max_tokens=4000)
    result, rejected = S.extract_source(p, llm, cfg, spans, conversation=False)
    assert result.facts[0].statement == text and result.facts[0].turn_id == spans[0].sid
    assert result.facts[0].object == "stabilize greenhouse gas concentrations"
    assert [(mode, stage) for mode, stage, _ in llm.calls] == [
        (True, "index.openie"), (False, "index.openie.repair")]
    assert result.extraction_windows == 2 and rejected == {"source_reference_repairs": 1}
    assert not list((tmp_path / "mab-source").glob("*.json"))
    assert len(list((tmp_path / "mab-source-reference-repair").glob("*.json"))) == 1
    llm.calls.clear()
    cached, _ = S._reference_repair(p, llm, cfg, spans, conversation=False)
    assert not llm.calls and cached.facts == result.facts


def test_reference_repair_rejects_fabricated_record_and_never_caches_it(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    p = Passage("p", "", 'Emma said "hello".')

    class FabricatedLLM(SourceLLM):
        def _complete(self, messages, params, stage):
            return LLMResult(text=json.dumps({"memories": [
                {"turn": "S999.0", "subject": "Emma", "relation": "say", "object": "hello"}]}))

    with pytest.raises(ValueError, match="no valid source-grounded facts"):
        S._reference_repair(p, FabricatedLLM(), C.IEConfig(), S.source_spans(p.text, 0), conversation=False)
    assert not list(tmp_path.rglob("*.json"))


def test_reference_repair_preserves_real_conversational_speaker_and_literal_time(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    text = "Chat Time: 2022/11/17\n{'role': 'user', 'content': 'I read \"Cloud Garden\" last week.'}"
    p = Passage("p", "", text)
    spans = S.source_spans(text, 0, conversation=True)
    actual = next(s for s in spans if "I read" in s.text)

    class ConversationLLM(SourceLLM):
        def _complete(self, messages, params, stage):
            return LLMResult(text=json.dumps({"memories": [{"turn": actual.sid, "subject": "User",
                    "relation": "read", "object": "Cloud Garden", "time": "last week", "kind": "past"}]}))

    registered, _ = S._reference_repair(p, ConversationLLM(), C.IEConfig(), spans, conversation=True)
    assert registered.facts[0].statement == "User: " + actual.text
    assert registered.facts[0].time == "last week" and actual.when == date(2022, 11, 17)


def retriever(*, conversation=False, budget=2400):
    text = "Emma, sister of Richard II, married Ethelred II."
    p = Passage("p", "", text)
    spans = S.source_spans(text, 0, conversation=conversation)
    # Deliberately omit the marriage relation, simulating lossy extraction.
    facts = [Fact("f0", "Emma", "sister of", "Richard II", "p", statement="Emma is Richard II's sister.", turn_id="S0.0"),
             Fact("f1", "User", "mention", "Emma", "p", statement="User mentioned Emma.", turn_id="S0.0")]
    corpus = Corpus("source-test", [p], [])
    embedder = ExactEmbedder()
    cfg = standard_config(rerank="")
    cfg.witness.grounding_mode = "exact"
    cfg.witness.excerpt_max_chars = budget
    cfg.graph.merge_identity_variants = False
    kg = build_graph(corpus, ExtractionResult(facts=facts), embedder, cfg.graph, with_passage_nodes=True)
    ctx = IndexContext(corpus, NoLLM(), embedder, cfg, kg=kg)
    r = S.SourceWitnessRetriever(ctx, {"p": spans}, conversation=conversation)
    r.index()
    return r


def test_literal_sources_rescue_missing_relation_without_new_llm_or_gold():
    r = retriever()
    q = Question("q", "Who did Emma marry?", ["SECRET_GOLD"], dataset="toy")
    result = r.retrieve(q)
    diag = result.diagnostics
    delivered = "\n".join(b["text"] for b in diag["trechos_extras"])
    assert "Emma, sister of Richard II, married Ethelred II." in delivered
    assert "SECRET_GOLD" not in str(diag) and diag["planejamento"]["chamadas"] == 0
    src = diag["source_adaptation"]
    assert src["excerpt_chars"] <= src["excerpt_budget"] == 2400
    for row in src["excerpts"]:
        assert r.corpus.get(row["pid"]).text[row["start"]:row["end"]] == row["text"]
    assert r.dated.first is None and r.dated.last is None


def test_document_planner_rejects_ingestion_join_but_keeps_uncertain_relations():
    r = retriever()
    q = Question("q", "Who did Emma marry?", [])
    p = S.SourcePlanner(r, q)
    bad = p.candidate([Atom("mention", "?x", "Emma")])
    assert bad and not p.checks(bad, bad.witnesses[0])["projection_compatible"]
    other = p.candidate([Atom("sister of", "Emma", "?x")])
    assert other and p.checks(other, other.witnesses[0])["projection_compatible"]
    assert "predicate_alignment" in p.checks(other, other.witnesses[0])
    assert p.checks(other, other.witnesses[0])["predicate_coverage"] == 0


def test_conversational_planner_preserves_real_user_relations():
    r = retriever(conversation=True)
    p = S.SourcePlanner(r, Question("q", "Who mentioned Emma?", []))
    c = p.candidate([Atom("mention", "?x", "Emma")])
    assert c and not p.checks(c, c.witnesses[0])["ingestion_metadata"]


def test_adaptation_identity_and_default_isolation():
    cfg = standard_config()
    standard = WitnessEngine(NoLLM(), ExactEmbedder(), cfg)
    adapted = WitnessEngine(NoLLM(), ExactEmbedder(), cfg, adaptation="ar-source-v2")
    assert "adapter" not in standard.protocol_identity()
    assert adapted.protocol_identity()["adapter"] == "ar-source-v2"
    assert cfg.ie.style == "memory" and cfg.witness.local_plan_version == "v2"
    with pytest.raises(ValueError, match="four Accurate Retrieval"):
        adapted.prepare("key", "factconsolidation_sh_6k", ["Iris lives in Rome."], "")
    args = parser().parse_args(["run", "--output", "out", "--adaptation", "ar-source-v2",
                               "--splits", "Accurate_Retrieval"])
    assert args.adaptation == "ar-source-v2" and args.source_excerpt_chars == 2400


def test_ar_order_is_explicit_and_stable_within_each_source():
    samples = [SimpleNamespace(source=source, row_index=i) for source, i in
               (("eventqa_full", 4), ("longmemeval_s*", 19), ("ruler_qa2_421K", 1),
                ("eventqa_full", 2), ("longmemeval_s*", 17), ("ruler_qa1_197K", 0))]
    ordered = S.order_samples(samples)
    assert [(s.source, s.row_index) for s in ordered] == [
        ("ruler_qa1_197K", 0), ("ruler_qa2_421K", 1), ("longmemeval_s*", 17),
        ("longmemeval_s*", 19), ("eventqa_full", 2), ("eventqa_full", 4)]
    with pytest.raises(ValueError, match="AR sources only"):
        S.order_samples([SimpleNamespace(source="factconsolidation_sh_6k", row_index=0)])


def test_declared_source_budget_bounds_literal_rescue():
    r = retriever(budget=160)
    result = r.retrieve(Question("q", "Who did Emma marry?", []))
    info = result.diagnostics["source_adaptation"]
    assert info["excerpt_chars"] <= 160
    assert sum(len(row["text"]) for row in info["excerpts"]) <= 160
