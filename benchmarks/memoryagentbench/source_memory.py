"""AR source adapter: literal provenance, documentary facts and scoped plans.

The adapter consumes only incoming chunks and questions. Official splitting,
queries, output limits, scoring and conversational LoCoMo code remain unchanged.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from wrag import config as C
from wrag.ie import ExtractionResult, Fact
from wrag.llm import GenParams
from wrag.methods.witnessrag import WitnessRAGRetriever
from wrag.util import canonical_symbol, read_json, sha, write_json
from wrag.witness.dated_memory import DatedMemory, Turn, _terms
from wrag.witness.local_contract import SourceIndex, stems
from wrag.witness.local_plans_v2 import MultiOriginPlanner, retrieve_local_v2
from wrag.witness.scoring import MemoryScorer
from wrag.witness.timeline import Interval

VERSION = "ar-source-v2"
REFERENCE_REPAIR_VERSION = "source-record-reference-v1"
AR_ORDER = ("ruler_qa1_197K", "ruler_qa2_421K", "longmemeval_s*", "eventqa_full")
SYSTEM = "Extract source-grounded atomic facts. Return one JSON object only."
TEMPLATE = """Register the following {kind} source records for later retrieval.
Extract their CONTENT, not the act of storing, reading or presenting documents.
Cover informative clauses, including embedded relations, kinship, marriages,
causes, locations, dates, quantities, lists and qualifications. Split propositions.
For conversations, role fields identify real speakers: resolve I/my to that
speaker and retain who stated or recommended what; proposals are not events.
For documents, use the entities described, never an invented User or Assistant.
Keep exact names and quantities. Preserve the direction of subject/relation/object.
Do not infer a relationship merely because two names co-occur.
Each item must cite the record ID and an exact supporting quote from that record.
The quote must support the ENTIRE directed triple, including its qualifiers.
Use a short canonical relation; keep time expressions literally or leave time empty.
No external knowledge. Source IDs and ingestion order are not facts or event dates.
At most {limit} items. Format:
{{"memories":[{{"turn":"S0.0","subject":"...","relation":"...",
"object":"...","quote":"exact source words","time":"","kind":"past"}}]}}

Source records:
{text}
"""
REFERENCE_TEMPLATE = """Extract atomic facts from the following {kind} source records.
Extract document CONTENT, not ingestion actions. Cover embedded relations,
kinship, marriages, causes, dates, quantities and qualifications. Preserve exact
names and directed subject/relation/object relations. Do not infer a relation
merely because names co-occur. For conversations, resolve I/my to the real
speaker; preserve attribution and distinguish proposals from events.
Each entire triple must be explicitly supported by its cited record ID.
The original record is already stored: cite its ID in turn, DO NOT output a
quote field or copy supporting sentences. The program retrieves the original
record verbatim by that ID. This verifies provenance, not relational truth.
Use a short canonical relation, literal time expressions or empty time, and
kind past/plan/ongoing/said. No external knowledge or invented IDs/event dates.
At most {limit} items. Return one complete valid JSON object, without Markdown:
{{"memories":[{{"turn":"S0.0","subject":"...","relation":"...",
"object":"...","time":"","kind":"past"}}]}}

Input records (JSON strings are escaped once; read their actual content):
{records}
"""


def order_samples(samples):
    if any(s.source not in AR_ORDER for s in samples):
        raise ValueError("ar-source-v2 requires Accurate_Retrieval and AR sources only")
    return sorted(samples, key=lambda s: (AR_ORDER.index(s.source), s.row_index))


def conversational(source: str) -> bool:
    return source.startswith("longmemeval_")


def session_date(text: str, default: date | None = None) -> date | None:
    matches = list(re.finditer(r"Chat Time:\s*(\d{4})[/-](\d{2})[/-](\d{2})", text))
    if not matches:
        return default
    try:
        return date(*map(int, matches[-1].groups()))
    except ValueError:
        return default


def session_role(text: str, default: str = "Source") -> str:
    matches = list(re.finditer(r"['\"]role['\"]\s*:\s*['\"](user|assistant)['\"]", text, re.I))
    return matches[-1][1].capitalize() if matches else default


@dataclass(frozen=True)
class Span:
    sid: str
    text: str
    start: int
    end: int
    speaker: str = "Source"
    when: date | None = None


def source_spans(text: str, chunk_index: int, *, conversation: bool = False,
                 initial_date: date | None = None, initial_speaker: str = "Source") -> list[Span]:
    # Keep original offsets and bytes. Labels are outside the literal source.
    boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+|\n+", text)] + [len(text)]
    if conversation:
        boundaries += [m.start() for m in re.finditer(r"['\"]role['\"]\s*:\s*['\"](?:user|assistant)['\"]|Chat Time:", text, re.I)]
    boundaries = sorted(set(boundaries))
    spans = []
    current_date = initial_date
    for left, right in zip(boundaries, boundaries[1:]):
        raw = text[left:right]
        body = raw.strip()
        if not body:
            continue
        start = left + len(raw) - len(raw.lstrip())
        end = right - (len(raw) - len(raw.rstrip()))
        speaker = "Source"
        if conversation:
            current_date = session_date(text[:end], current_date)
            speaker = session_role(text[:end], initial_speaker)
        spans.append(Span(f"S{chunk_index}.{len(spans)}", body, start, end, speaker, current_date))
    return spans


def _literal(text: str) -> str:
    # Serialized conversations contain escaped whitespace/quotes; equivalence
    # here is only for locating evidence, never for scoring generated answers.
    text = re.sub(r"\\[nrt]", " ", text).replace('\\"', '"').replace("\\'", "'")
    return " ".join(text.split()).casefold()


def metadata_subject(subject: str) -> bool:
    return bool(re.fullmatch(r"(?:the\s+)?(?:user|assistant|source|document(?:\s+\d+)?)", subject.strip(), re.I))


def parse_source_facts(data, passage, spans, *, conversation: bool, limit: int):
    if not isinstance(data, dict) or not isinstance(data.get("memories"), list):
        raise ValueError("Source extraction requires a memories list")
    by_id = {s.sid: s for s in spans}
    facts, rejected, seen = [], Counter(), set()
    for item in data["memories"][:limit]:
        if not isinstance(item, dict):
            rejected["invalid_item"] += 1
            continue
        s, r, o = (str(item.get(k) or "").strip() for k in ("subject", "relation", "object"))
        span = by_id.get(str(item.get("turn") or ""))
        quote = str(item.get("quote") or "").strip()
        if not all((s, r, o, span, quote)) or _literal(quote) not in _literal(span.text):
            rejected["unsupported_source"] += 1
            continue
        if not conversation and metadata_subject(s):
            rejected["ingestion_metadata"] += 1
            continue
        key = (canonical_symbol(s), canonical_symbol(r), canonical_symbol(o), span.sid)
        if key in seen:
            continue
        seen.add(key)
        when = str(item.get("time") or "").strip()
        if when and _literal(when) not in _literal(span.text):
            rejected["nonliteral_time_removed"] += 1
            when = ""
        statement = f"{span.speaker}: {quote}" if conversation else quote
        kind = str(item.get("kind") or "past")
        facts.append(Fact(fid=sha(*key, passage.pid)[:16], subject=s, relation=r,
                          object=o, pid=passage.pid, statement=statement,
                          turn_id=span.sid, time=when,
                          kind=kind if kind in {"past", "plan", "ongoing", "said"} else "said"))
    return facts, dict(rejected)


def _reference_repair(passage, llm, cfg, spans, *, conversation: bool):
    """Recover malformed output using IDs, never synthesized quote text.

    Its separate cache cannot replace primary extraction entries or be read by
    the historical extractor. Both transports keep the same model and budget.
    """
    records = json.dumps([{"id": s.sid, "speaker": s.speaker,
                           "date": s.when.isoformat() if s.when else "",
                           "text": s.text} for s in spans], ensure_ascii=False)
    prompt = REFERENCE_TEMPLATE.format(kind="conversation" if conversation else "documentary",
                                        limit=cfg.max_triples_per_passage, records=records)
    identity = sha(REFERENCE_REPAIR_VERSION, SYSTEM, prompt, passage.pid, passage.text,
                   cfg.max_tokens, cfg.temperature, llm.name, getattr(llm, "deployment", ""),
                   llm.cache_identity() if hasattr(llm, "cache_identity") else "")
    path = C.CACHE_DIR / "mab-source-reference-repair" / f"{identity}.json"
    saved = read_json(path)
    cached = saved is not None
    if saved is None:
        response = llm.chat(prompt, system=SYSTEM,
                            params=GenParams(temperature=cfg.temperature,
                                             max_tokens=cfg.max_tokens, json_mode=False),
                            stage="index.openie.repair")
        if response.error:
            raise RuntimeError(f"Source reference repair failed: {response.error}")
        if response.finish_reason == "length":
            raise ValueError("Source reference repair reached its output limit")
        saved = {"filtered": response.filtered,
                 "data": None if response.filtered else response.json()}
    if saved["filtered"]:
        return ExtractionResult(blocked_pids=[passage.pid], blocked_windows=1,
                                extraction_windows=1), {"source_reference_repairs": 1}
    data = saved["data"]
    if not isinstance(data, dict) or not isinstance(data.get("memories"), list):
        raise ValueError("Source reference repair requires a memories list")
    by_id = {s.sid: s for s in spans}
    normalized = []
    for item in data["memories"]:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        span = by_id.get(str(item.get("turn") or ""))
        # Only a real ID supplied in this window can retrieve literal evidence.
        normalized.append({**item, "quote": span.text if span else ""})
    facts, rejected = parse_source_facts({"memories": normalized}, passage, spans,
                                         conversation=conversation, limit=cfg.max_triples_per_passage)
    if data["memories"] and not facts:
        raise ValueError("Source reference repair supplied no valid source-grounded facts")
    if not cached:
        write_json(path, saved)
    rejected["source_reference_repairs"] = 1
    if cached:
        rejected["source_reference_repair_cache_hits"] = 1
    return ExtractionResult(facts=facts, extraction_windows=1,
                            empty_pids=[] if facts else [passage.pid]), rejected


def _extract_source_window(passage, llm, cfg, spans, *, conversation: bool, depth: int = 0):
    kind = "conversation" if conversation else "documentary"
    rendered = "\n".join(f"[{s.sid}]" + (f" date={s.when.isoformat()}" if s.when else "") +
                         f" {s.speaker}: {s.text}" for s in spans)
    identity = sha(VERSION, SYSTEM, TEMPLATE, passage.pid, passage.text,
                   rendered, kind, cfg.max_tokens, cfg.max_triples_per_passage,
                   cfg.temperature, llm.name, getattr(llm, "deployment", ""),
                   llm.cache_identity() if hasattr(llm, "cache_identity") else "")
    path = C.CACHE_DIR / "mab-source" / f"{identity}.json"
    saved = read_json(path)
    if saved is None:
        result = llm.chat(TEMPLATE.format(kind=kind, limit=cfg.max_triples_per_passage,
                                         text=rendered), system=SYSTEM,
                          params=GenParams(temperature=cfg.temperature,
                                           max_tokens=cfg.max_tokens, json_mode=True),
                          stage="index.openie")
        if result.error:
            raise RuntimeError(f"Source extraction failed: {result.error}")
        saved = {"filtered": result.filtered, "data": None if result.filtered else result.json()}
        # A long list can repeat its supporting quote until the output cap cuts
        # the JSON. Retry smaller groups of intact records; never register a
        # truncated object or silently lose the remaining source records.
        if not saved["filtered"]:
            try:
                if result.finish_reason == "length":
                    raise ValueError("Source extraction JSON reached its output limit")
                parse_source_facts(saved["data"], passage, spans,
                                   conversation=conversation, limit=cfg.max_triples_per_passage)
            except ValueError as exc:
                # A tiny useful prefix despite thousands of generated tokens
                # indicates a formatting stall, not a source-window size issue.
                # Normal long-output truncations still use the existing splits.
                sparse = (result.completion_tokens > 0 and
                          len(result.text.strip()) < result.completion_tokens)
                if sparse or result.finish_reason != "length" or depth >= 3:
                    print(f"  source reference repair {passage.pid}/{spans[0].sid}: {exc}", flush=True)
                    try:
                        repaired, reasons = _reference_repair(passage, llm, cfg, spans,
                                                              conversation=conversation)
                    except ValueError as repair_error:
                        exc = repair_error
                    else:
                        repaired.extraction_windows += 1
                        return repaired, reasons
                if depth >= 3:
                    raise ValueError(f"Source extraction {passage.pid}/{spans[0].sid}: {exc}") from exc
                print(f"  source extraction retry {passage.pid}/{spans[0].sid} depth={depth + 1}: {exc}", flush=True)
                if len(spans) > 1:
                    middle = len(spans) // 2
                    groups, smaller = [spans[:middle], spans[middle:]], cfg
                else:
                    groups = [spans]
                    smaller = replace(cfg, max_triples_per_passage=max(1, cfg.max_triples_per_passage // 2))
                merged, rejected = ExtractionResult(extraction_windows=1), Counter(schema_split_retries=1)
                all_blocked = True
                for group in groups:
                    partial, reasons = _extract_source_window(passage, llm, smaller, group,
                                                              conversation=conversation, depth=depth + 1)
                    merged.facts.extend(partial.facts)
                    merged.extraction_windows += partial.extraction_windows
                    merged.blocked_windows += partial.blocked_windows
                    all_blocked = all_blocked and bool(partial.blocked_pids)
                    rejected.update(reasons)
                if all_blocked:
                    merged.blocked_pids = [passage.pid]
                return merged, dict(rejected)
        write_json(path, saved)
    facts, rejected = ([], {}) if saved["filtered"] else parse_source_facts(
        saved["data"], passage, spans, conversation=conversation,
        limit=cfg.max_triples_per_passage)
    out = ExtractionResult(facts=facts, blocked_pids=[passage.pid] if saved["filtered"] else [],
                           empty_pids=[passage.pid] if not facts and not saved["filtered"] else [],
                           extraction_windows=1, blocked_windows=int(saved["filtered"]))
    return out, rejected


def source_batches(spans, cfg):
    """Bound extraction by whole source sentences, preserving IDs and offsets.

    An oversized sentence remains intact; official chunking has the same rule.
    No text is discarded, and a sentence is never decoded from a cut token window.
    """
    if not cfg.window_tokens:
        return [spans] if spans else []
    from wrag.ie import _load_window_tokenizer
    tokenizer = _load_window_tokenizer(cfg.window_tokenizer, cfg.window_tokenizer_revision)
    batches, current, used = [], [], 0
    for span in spans:
        cost = len(tokenizer.encode(span.text, add_special_tokens=False)) + 12
        if current and used + cost > cfg.window_tokens:
            batches.append(current)
            current, used = [], 0
        current.append(span)
        used += cost
    if current:
        batches.append(current)
    return batches


def extract_source(passage, llm, cfg, spans, *, conversation: bool):
    windows = source_batches(spans, cfg)
    result, rejections = ExtractionResult(), Counter()
    seen = set()
    all_blocked = bool(windows)
    for records in windows:
        registered, rejected = _extract_source_window(passage, llm, cfg, records, conversation=conversation)
        rejections.update(rejected)
        for fact in registered.facts:
            if fact.fid not in seen:
                seen.add(fact.fid)
                result.facts.append(fact)
        result.blocked_windows += registered.blocked_windows
        result.extraction_windows += registered.extraction_windows
        all_blocked = all_blocked and bool(registered.blocked_pids)
    if all_blocked:
        result.blocked_pids = [passage.pid]
    elif not result.facts:
        result.empty_pids = [passage.pid]
    return result, dict(rejections)


class SourceDatedMemory(DatedMemory):
    def __init__(self, corpus, facts, records):
        super().__init__(corpus, [])
        self.records = records
        self.turns = {pid: [Turn(s.sid, s.speaker, s.text, s.when, -1) for s in spans]
                      for pid, spans in records.items()}
        dates = [s.when for spans in records.values() for s in spans if s.when]
        self.first, self.last = (min(dates), max(dates)) if dates else (None, None)
        for pid, spans in records.items():
            values = [s.when for s in spans if s.when]
            self.passage_interval[pid] = Interval(min(values), max(values)) if values else None
            self.passage_importance[pid] = 0.0
        self._turn_terms = {pid: [_terms(f"{s.speaker} {s.text}") for s in spans]
                            for pid, spans in records.items()}
        self.fact_interval = [None] * len(facts)
        self.fact_importance = np.zeros(len(facts), dtype=np.float32)
        self.fact_turn = [("", -1)] * len(facts)
        self.fact_time_source = [""] * len(facts)
        for i, f in enumerate(facts):
            self._register(i, f)


class SourcePlanner(MultiOriginPlanner):
    def __init__(self, r, question):
        super().__init__(r, question)
        # Remove full literal owner names as well as single-word names from
        # predicate hypotheses; their mention is not the requested relation.
        owner_terms = stems(" ".join(name for name, _ in self.mentions))
        self.reading.predicate_terms -= owner_terms
        if self.reading.predicate_terms:
            vec = r.ctx.embedder.encode([" ".join(sorted(self.reading.predicate_terms))])[0]
            self.relation_affinity = {rel: max(0., min(1., float(np.dot(vec, self.memory.relation_vectors[rid]))))
                                      for rel, rid in self.memory.relation_of.items()}
        else:
            self.relation_affinity = {}

    def checks(self, c, w):
        check = super().checks(c, w)
        ingestion = not self.r.source_conversation and any(
            metadata_subject(self.memory.facts[i].subject) for i in w.facts)
        check["ingestion_metadata"] = ingestion
        if ingestion:
            check["projection_compatible"] = False
        affinity = max((self.relation_affinity.get(canonical_symbol(a.relation), 0.)
                        for a in c.query.atoms), default=0.)
        check["predicate_alignment"] = max(check["predicate_coverage"], affinity)
        check["predicate_status"] = "lexical_or_embedding_hypothesis_not_verification"
        return check

    def rerank(self, candidates):
        ranked, scorer = super().rerank(candidates)
        adjusted = []
        for _, c, w, sem, temporal in ranked:
            check = self.checks(c, w)
            grade = int(check["projection_compatible"]) + int(check["required_conjunction"])
            # Semantic predicate alignment is a soft preference, not a veto
            # against paraphrases or a certificate of answer sufficiency.
            support = .45 * sem + .25 * check["owner_coverage"] + .30 * check["predicate_alignment"]
            weight = self.contract.time_weight
            adjusted.append((grade + (1 - weight) * support + weight * temporal, c, w, sem, temporal))
        adjusted.sort(key=lambda x: (-x[0], x[1].signature(), x[2].facts))
        return adjusted, scorer


class SourceWitnessRetriever(WitnessRAGRetriever):
    def __init__(self, ctx, records, *, conversation=False):
        super().__init__(ctx)
        self.source_records, self.source_conversation = records, conversation

    def _build_dated_memory(self):
        self.dated = SourceDatedMemory(self.corpus, self.memory.facts, self.source_records)
        cfg = self.ctx.run.witness
        self.scorer = MemoryScorer(self.dated, cfg.proof_min_scale_days, cfg.proof_point_scale_fraction)
        self.searcher.fact_times = [self.dated.fact_time_text(i) for i in range(len(self.memory.facts))]

    def _dialogue_block(self, fact_groups, window):
        # Exact source spans are rendered independently below under ONE budget.
        return "", []

    def _retrieve_proof(self, question, k, pool_pids, pool_scores):
        result = retrieve_local_v2(self, question, k, pool_pids, pool_scores,
                                   planner_class=SourcePlanner)
        diag = result.diagnostics
        source_index = self._local_source_index
        ranked = source_index.rank(stems(question.question))
        linked = {self.dated.fact_turn[i] for i in diag["fatos_entregues"].get("indices", [])}
        # Prefer query relevance, then link to delivered facts. Original source
        # retrieval can recover a premise omitted by the graph extraction.
        choices = sorted(ranked, key=lambda row: (-row[0],
                         (source_index.turns[row[1]][0], source_index.turns[row[1]][1]) not in linked,
                         row[1]))
        quotes, rows, chars, parents = [], [], 0, set()
        cap = self.ctx.run.witness.excerpt_max_chars
        for _, idx in choices:
            pid, pos, turn = source_index.turns[idx]
            if len(parents | {pid}) > k or len(rows) >= 8:
                continue
            span = self.source_records[pid][pos]
            header = f"[{pid}/{span.sid}; chars {span.start}:{span.end}] "
            if span.when:
                header += f"session date {span.when.isoformat()}; "
            remaining = cap - chars - len(header) - 1
            if remaining < 80:
                break
            body = span.text[:min(650, remaining)]
            line = header + body
            quotes.append(line)
            chars += len(line) + 1
            parents.add(pid)
            rows.append({"pid": pid, "source_id": span.sid, "start": span.start,
                         "end": span.start + len(body), "text": body,
                         "truncated": len(body) < len(span.text), "linked_fact": (pid, pos) in linked})
        # Replace the conversational rescue generated by local-v2 with the
        # bounded documentary/conversational source spans actually delivered.
        for block in diag["trechos_extras"]:
            text = re.sub(r"\n\nAdditional original turns .*?(?=\n\nPending checks:|\n\nSource mention groups|$)",
                          "", block["text"], flags=re.S)
            if quotes:
                text += "\n\nOriginal source excerpts (literal; graph joins remain hypotheses):\n" + "\n".join(quotes)
            block["text"] = text
        diag["source_adaptation"] = {"version": VERSION, "kind": "conversation" if self.source_conversation else "document",
                                     "excerpt_chars": chars, "excerpt_budget": cap, "excerpts": rows,
                                     "predicate_policy": "soft_relation_alignment; metadata_excluded_only_in_documents"}
        return result
