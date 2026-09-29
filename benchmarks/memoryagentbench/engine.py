"""Standard local-v2 WitnessRAG: sequential registration, then query reuse.

This adapter adds no gold-derived plans or rules. Serial numbers are observable
source provenance. Old and new facts remain available for the joint reflector;
no relation is assumed to be functional and no synthetic calendar date is used.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.ie import ExtractionResult, extract_corpus
from wrag.llm.base import usage_delta

from .data import memorize_message
from .protocol import digest


def standard_config(top_k: int = 10, fact_budget: int = 40,
                    rerank: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> C.RunConfig:
    cfg = C.RunConfig(dataset="memoryagentbench", n_questions=0, top_k=top_k,
                      methods=("witnessrag",), corpus_scope="isolated_context_per_sample")
    cfg.ie.style = "memory"
    cfg.ie.dialogue_mode = False
    cfg.ie.max_tokens = 4000
    cfg.ie.window_tokens = 512
    cfg.ie.window_overlap_tokens = 64
    cfg.ie.window_tokenizer = "Qwen/Qwen2.5-14B-Instruct"
    w = cfg.witness
    w.binding_aware_grounding = w.vocabulary_aware_compile = True
    w.hybrid_fallback = w.gap_context_rescue = w.answer_set = True
    w.proof_controller, w.proof_verify = True, False
    w.fact_delivery, w.fact_fill, w.fact_time = "facts", "question", "both"
    w.fact_budget, w.fact_rerank = fact_budget, rerank
    w.local_plans, w.local_plan_version = True, "v2"
    w.local_plan_beam, w.local_plan_candidates = 32, 96
    w.local_plan_executions, w.local_plan_starts = 4000, 12
    cfg.qa.reader_reflection = True
    return cfg


def annotate_serials(chunk: str, raw_context: str) -> str:
    """Label only actual numbered records present in this incoming chunk."""
    records = {int(m[1]): m[2] for m in re.finditer(r"(?m)^(\d+)\.\s+([^\n]+)", raw_context)}
    pattern = re.compile(r"(?<!\S)(\d+)\.\s+")
    def replacement(match):
        number = int(match[1])
        original = records.get(number)
        rest = chunk[match.end():]
        # Numeric expressions inside a fact must not become fake source IDs.
        if original and rest.startswith(original[:min(len(original), 24)]):
            return f"\n[MAB:{number}] Fact: {number}. "
        return match[0]
    return pattern.sub(replacement, chunk)


@dataclass
class Selection:
    context: str
    diagnostics: dict
    usage: dict
    retrieve_s: float
    filtered: bool = False


class WitnessEngine:
    def __init__(self, llm, embedder, cfg: C.RunConfig, conflict_recency_weight: float = .65):
        if not 0 <= conflict_recency_weight <= 1:
            raise ValueError("Conflict recency weight must be between 0 and 1")
        self.llm, self.embedder, self.cfg = llm, embedder, cfg
        self.conflict_recency_weight = conflict_recency_weight
        self.corpus = None
        self.retriever = None

    def protocol_identity(self):
        return {"conflict_recency_weight": self.conflict_recency_weight,
                "conflict_recency_policy": "ordinal-subject-relation-v2"}

    def prepare(self, context_key: str, source: str, chunks: list[str],
                raw_context: str) -> dict:
        from wrag.methods import build_context, build_methods
        # Reset first: a failed new context must never reuse a previous memory.
        self.corpus = self.retriever = None
        start, before = time.perf_counter(), self.llm.usage.snapshot()
        passages, extraction, cached_chunks = [], ExtractionResult(), 0
        short_key = digest(context_key)[:20]
        for index, chunk in enumerate(chunks):
            text = annotate_serials(chunk, raw_context) if source.startswith("factconsolidation_") else chunk
            passage = Passage(f"mab-{short_key}-c{index}", "",
                              memorize_message(source, text, index), sequence=index,
                              source_ids=(context_key,))
            passages.append(passage)
            # Only the current incoming message is visible to its registration
            # call. Graph consolidation after all chunks is allowed by §3.2.
            registration_before = self.llm.usage.snapshot()
            registered = extract_corpus(Corpus(f"mab-{short_key}-c{index}", [passage], []),
                                        self.llm, self.cfg.ie)
            if not usage_delta(self.llm.usage.snapshot(), registration_before)["total"].get("chamadas", 0):
                cached_chunks += 1
            extraction.facts.extend(registered.facts)
            extraction.entities_by_passage.update(registered.entities_by_passage)
            extraction.blocked_pids.extend(registered.blocked_pids)
            extraction.empty_pids.extend(registered.empty_pids)
            extraction.extraction_windows += registered.extraction_windows
            extraction.blocked_windows += registered.blocked_windows
            if (index + 1) % 25 == 0 or index + 1 == len(chunks):
                print(f"  memory chunks {index + 1}/{len(chunks)} facts={len(extraction.facts)}", flush=True)
        self.corpus = Corpus(f"mab-{short_key}", passages, [])
        ctx = build_context(self.corpus, self.cfg, llm=self.llm, embedder=self.embedder)
        ctx.extraction = extraction
        self._conflicts = source.startswith("factconsolidation_")
        if self._conflicts and self.conflict_recency_weight:
            from wrag.methods import ensure_graph
            from .recency import SerialWitnessRetriever
            graph_before, graph_started = self.llm.usage.snapshot(), time.perf_counter()
            ensure_graph(ctx, with_passage_nodes=True)
            ctx.shared_index_cost = {"seconds": time.perf_counter() - graph_started,
                                     "usage": usage_delta(self.llm.usage.snapshot(), graph_before)}
            self.retriever = SerialWitnessRetriever(ctx)
            self.retriever.conflict_recency_weight = self.conflict_recency_weight
            index_before, index_started = self.llm.usage.snapshot(), time.perf_counter()
            self.retriever.index()
            self.retriever.index_cost = {"seconds": time.perf_counter() - index_started,
                                        "usage": usage_delta(self.llm.usage.snapshot(), index_before)}
        else:
            self.retriever = build_methods(ctx, ["witnessrag"])["witnessrag"]
        return {"seconds": time.perf_counter() - start,
                "usage": usage_delta(self.llm.usage.snapshot(), before),
                "chunks": len(chunks), "extraction": extraction.stats(),
                "registration_cached_chunks": cached_chunks,
                "registration_new_chunks": len(chunks) - cached_chunks,
                "cost_scope": "observed_attempt; prior cached extraction cost is not charged again",
                "index": self.retriever.index_report()}

    def select(self, qid: str, text: str) -> Selection:
        if self.retriever is None or self.corpus is None:
            raise RuntimeError("No registered context")
        # Deliberately no gold answers, qtype, supporting IDs or decomposition.
        question = Question(qid, text, [], dataset="memoryagentbench")
        before, start = self.llm.usage.snapshot(), time.perf_counter()
        result = self.retriever.retrieve(question, self.cfg.top_k)
        diagnostics = dict(result.diagnostics or {})
        blocks = diagnostics.get("trechos_extras") or []
        if diagnostics.get("leitura_fatos"):
            context = "\n\n".join(b.get("text", "") for b in blocks)
        else:
            context = "\n\n".join(self.corpus.get(pid).full for pid in result.pids[:self.cfg.top_k])
        if self._conflicts:
            # Generated statements may omit their serial. Resolve ONLY source
            # turn IDs, or an unambiguous literal statement in the same passage.
            serials = []
            statements = {}
            unknown = 0
            for index in diagnostics.get("fatos_entregues", {}).get("indices", []):
                fact = self.retriever.memory.facts[index]
                raw = self.corpus.get(fact.pid).text
                matches = list(re.finditer(r"\[MAB:(\d+)\] Fact: (.*?)(?=\n\[MAB:|$)", raw, re.S))
                turn = re.fullmatch(r"MAB:(\d+)", fact.turn_id)
                numbered = [(int(m[1]), m[2].strip()) for m in matches]
                candidates = [n for n, body in numbered if turn and n == int(turn[1])]
                if not candidates and fact.statement:
                    candidates = [n for n, body in numbered if fact.statement.strip().casefold() in body.casefold()]
                if len(candidates) == 1:
                    statement = fact.statement or fact.verbalize()
                    serials.append(candidates[0])
                    statements.setdefault(statement, set()).add(candidates[0])
                else:
                    unknown += 1
            # Keep serial and claim together. A detached duplicate list gave
            # older unlabelled claims too much prominence in the initial pilot.
            for statement, numbers in statements.items():
                labels = ", ".join(str(n) for n in sorted(numbers))
                context = context.replace(f"- {statement}", f"- [source serial {labels}] {statement}")
            # Source-turn rescue may find an update missing from generated
            # triples. Its original record number is equally valid provenance.
            context = re.sub(r"\[MAB:(\d+)\]", r"[source serial \1]", context)
            diagnostics["serial_provenance"] = {"known": len(serials), "unknown": unknown}
            policy = getattr(self.retriever, "serial_policy", None)
            if policy is not None:
                # Put original, question-relevant records before hypotheses.
                # Replace the existing additional-turn rescue block, keeping
                # its character budget; no extra LLM call or gold is used.
                context = re.sub(r"\n\nAdditional original turns .*?(?=\n\nPending checks:|$)",
                                 "", context, flags=re.S)
                from wrag.witness.local_contract import SourceIndex, stems
                source_index = self.retriever._local_source_index
                terms = stems(text)
                lexical = SourceIndex.rank(source_index, terms)
                relevant = {i for score, i in lexical if score >= .5 * lexical[0][0]} if lexical else set()
                ranked = [(score, i) for score, i in source_index.rank(terms) if i in relevant][:4]
                focused = []
                for _, index in ranked:
                    pid, _, turn = source_index.turns[index]
                    match = re.fullmatch(r"MAB:(\d+)", turn.turn_id)
                    if match and int(match[1]) in policy.records.get(pid, {}):
                        focused.append((int(match[1]), turn.text[:650]))
                focused.sort(reverse=True)
                lines, used, admitted = [], 0, []
                for number, body in focused:
                    line = f"[source serial {number}] {body}"
                    if used + len(line) > self.cfg.witness.excerpt_max_chars:
                        continue
                    lines.append(line)
                    admitted.append(number)
                    used += len(line)
                if lines:
                    context = ("Question-relevant original records, newest first. Resolve conflicts "
                               "for the SAME subject and predicate before using retrieval hypotheses:\n" +
                               "\n".join(lines) + "\n\n" + context)
                context = re.sub(r"\[source serial (\d+)\]",
                                 lambda match: policy.version_label(int(match[1])), context)
                diagnostics["ordinal_recency"]["focused_source_serials"] = admitted
        return Selection(context, diagnostics, usage_delta(self.llm.usage.snapshot(), before),
                         time.perf_counter() - start, result.filtered)
