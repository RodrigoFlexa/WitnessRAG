"""Optional, query-scoped adaptive retrieval: one joint gate, at most one retry.

Generated hints are retrieval requests, never memory facts. Only actual original
fact indices may be retained, within the SAME total fact budget. No gold or
benchmark labels enter the controller. Standard reader prompts stay unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from wrag.data import Question


GATE_INSTRUCTION = """Adaptive retrieval control (overrides the earlier answer-only JSON format).
Reflect on whether the evidence minimally supports the requested answer. Return
your best short answer using the CURRENT evidence, and one of two decisions.
Use continue when relevant premises suffice, including a justified ordinary
inference. Correct answer type/units yourself; those do not require more retrieval.
Use replan ONLY for a specific missing personal fact, unresolved reference,
missing relation/qualifier, or contradictory claims that another memory could resolve.
Do not request more memory just to increase confidence or to answer a different question.
For replan, name the missing evidence, suggest at most TWO targeted searches
(use observed people/entities and relation words, never a guessed answer as a fact),
and keep at most EIGHT catalog IDs for useful partial premises. Keep IDs only
from the catalog. Do not preserve contradicted facts merely to support your draft.
There is at most ONE retry; the next reader must answer or abstain.
Return JSON: {"answer":"best current short answer or insufficient information",
"decision":"continue or replan","missing":"brief evidence gap or empty",
"searches":["targeted search"],"keep":[integer fact IDs]}.
No explanation or prose outside JSON. Search suggestions are hypotheses, not evidence.
The answer field must be the shortest COMPLETE requested value, usually a noun
phrase. Do not repeat the person, question, or supporting premises. Include only
items that answer the requested relation, not all nearby activities. For continue,
return empty missing, searches and keep fields. The catalog is for retention IDs;
it does not broaden the question or add another task to the answer.
"""


def parse_gate(data):
    if not isinstance(data, dict):
        return {"decision": "continue", "valid": False, "missing": "", "searches": [], "keep": []}
    searches = data.get("searches", [])
    keep = data.get("keep", [])
    missing = data.get("missing", "")
    valid = (data.get("decision") in {"continue", "replan"}
             and isinstance(data.get("answer"), str)
             and isinstance(missing, str) and len(missing) <= 400
             and isinstance(searches, list) and len(searches) <= 2
             and all(isinstance(s, str) and 0 < len(s.strip()) <= 200 for s in searches)
             and isinstance(keep, list) and len(keep) <= 8
             and all(type(i) is int and i >= 0 for i in keep))
    if data.get("decision") == "replan":
        valid = valid and bool(searches) and bool(missing.strip())
    return {"decision": data.get("decision") if valid else "continue", "valid": bool(valid),
            "missing": missing if valid else "", "searches": searches if valid else [],
            "keep": list(dict.fromkeys(keep)) if valid else []}


def catalog(retriever, retrieval):
    lines = ["Retainable fact catalog (IDs refer only to facts in the current evidence):"]
    for i in retrieval.diagnostics.get("fatos_entregues", {}).get("indices", []):
        f = retriever.memory.facts[i]
        lines.append(f"{i}: {f.subject} | {f.relation} | {f.object} [source={f.pid}/{f.turn_id}]")
    return "\n".join(lines)


def cached_fact_block(retriever, indices):
    lines = ["Retained partial premises (unverified; check new conflicting evidence):"]
    for i in indices:
        f = retriever.memory.facts[i]
        when = retriever.dated.fact_time_text(i)
        session = retriever.corpus.get(f.pid).session_time
        lines.append(f"- {f.statement or f.verbalize()} [fact={i}; source={f.pid}/{f.turn_id}; "
                     f"stated_time={f.time}; event={when}; session={session}; kind={f.kind}]")
    return {"title": "Query-local retained facts", "text": "\n".join(lines)}


def adaptive_read(retriever, corpus, question, retrieval, cfg, method=""):
    from wrag.eval.reader import read
    if not (retriever.ctx.run.witness.local_plans
            and retriever.ctx.run.witness.local_plan_version == "v2"
            and cfg.reader_reflection and cfg.evidence_reader
            and retrieval.diagnostics.get("leitura_fatos")):
        raise ValueError("Reflection replanning requires local-v2, fact delivery and the joint evidence reader")
    if retrieval.filtered:
        from wrag.eval.reader import ReadResult
        return retrieval, ReadResult(filtered=True)

    def reading(result, options, catalog_text=""):
        return read(retriever.ctx.llm, corpus, question, [], options, method=method,
                    extra_passages=result.diagnostics.get("trechos_extras"),
                    facts_mode=result.diagnostics.get("leitura_fatos") or False,
                    replan_catalog=catalog_text)

    first = reading(retrieval, cfg, catalog(retriever, retrieval))
    gate = first.reflection or {}
    initial_ids = retrieval.diagnostics.get("fatos_entregues", {}).get("indices", [])
    trace = {"version": "joint-replan-v1", "max_replans": 1, "requested": gate.get("decision") == "replan",
             "performed": False, "initial_answer": first.answer, "gate": gate,
             "initial_fact_indices": list(initial_ids), "kept_fact_indices": [], "new_fact_indices": [],
             "stop_reason": "sufficient" if gate.get("valid") else "invalid_control_no_retry"}
    if first.filtered or not trace["requested"]:
        result = copy.deepcopy(retrieval)
        result.diagnostics["reflection_replan"] = trace
        return result, first

    budget = retriever.ctx.run.witness.fact_budget
    # At most one quarter of the budget is reserved for old partial premises.
    allowed = set(initial_ids)
    kept = [i for i in gate["keep"] if i in allowed][:min(8, budget // 4)]
    trace["kept_fact_indices"] = kept
    trace["rejected_keep_ids"] = [i for i in gate["keep"] if i not in allowed]
    guidance = "\n".join(gate["searches"])
    query = Question(question.qid, question.question, [], dataset=question.dataset)
    private = copy.copy(retriever)
    private._reflection_search_hint = guidance
    private.ctx = replace(retriever.ctx, run=replace(retriever.ctx.run,
                          witness=replace(retriever.ctx.run.witness, fact_budget=budget-len(kept))))
    second = private.retrieve(query, retriever.ctx.run.top_k)
    trace["performed"] = True
    trace["guidance"] = guidance
    trace["retrieval_seconds"] = second.latency_s
    new_ids = second.diagnostics.get("fatos_entregues", {}).get("indices", [])
    trace["new_fact_indices"] = [i for i in new_ids if i not in allowed]
    if second.filtered:
        from wrag.eval.reader import ReadResult
        second.latency_s += retrieval.latency_s
        second.diagnostics["reflection_replan"] = {**trace, "stop_reason": "retry_filtered"}
        return second, ReadResult(filtered=True, prompt_tokens=first.prompt_tokens,
                                  completion_tokens=first.completion_tokens, latency_s=first.latency_s)

    # New literal source turns can matter even if no extracted fact is new.
    before_turns = retrieval.diagnostics.get("local_plans", {})
    after_turns = second.diagnostics.get("local_plans", {})
    old_turns = set(before_turns.get("source_turns", [])) | {r["turn_id"] for r in before_turns.get("additional_source_turns", [])}
    new_turns = set(after_turns.get("source_turns", [])) | {r["turn_id"] for r in after_turns.get("additional_source_turns", [])}
    if not trace["new_fact_indices"] and not (new_turns-old_turns):
        result = copy.deepcopy(retrieval)
        trace["stop_reason"] = "no_new_evidence"
        result.diagnostics["reflection_replan"] = trace
        result.latency_s += second.latency_s
        return result, first

    # Duplicate retained facts consume no second copy in the final packet.
    cached = [i for i in kept if i not in new_ids]
    second.diagnostics = copy.deepcopy(second.diagnostics)
    if cached:
        second.diagnostics["trechos_extras"].append(cached_fact_block(retriever, cached))
    info = second.diagnostics["fatos_entregues"]
    info["indices"] = list(new_ids) + cached
    info["fontes"] = list(info.get("fontes", [])) + [s for s in retrieval.diagnostics["fatos_entregues"].get("fontes", []) if s["indice"] in cached]
    info["orcamento"] = budget
    info["entregues"] = len(info["indices"])
    info["retained_from_first_pass"] = cached
    assert len(set(info["indices"])) <= budget
    # Give literal source turns their own session anchors. An ingestion/scan
    # timestamp is never used to date a quoted event or interpret "last year".
    turn_dates = {turn.turn_id: str(turn.when) for turns in retriever.dated.turns.values()
                  for turn in turns if turn.turn_id and turn.when} if hasattr(retriever.dated, "turns") else {}
    for block in second.diagnostics["trechos_extras"]:
        for turn_id in new_turns:
            if turn_id in turn_dates:
                block["text"] = block["text"].replace(f"[{turn_id}]", f"[{turn_id}; session={turn_dates[turn_id]}]")
    second.diagnostics["trechos_extras"].append({"title": "Final retry reading contract", "text":
        "Answer the ORIGINAL question with the shortest complete requested value. "
        "For calendar-time questions, resolve relative expressions using the supplied source session "
        "date when unambiguous, at the precision actually supported (e.g., a year). "
        "A session date anchors a relative expression; it does not itself establish an event date. "
        "Review retained partial premises against new evidence. No further search is available."})
    final = reading(second, replace(cfg, reflection_replan=False))
    final.prompt_tokens += first.prompt_tokens
    final.completion_tokens += first.completion_tokens
    final.latency_s += first.latency_s
    final.reflection = {"mode": "joint-replan-v1", "first": gate, "final": final.reflection}
    trace["stop_reason"] = "one_retry_exhausted"
    second.latency_s += retrieval.latency_s
    second.diagnostics["reflection_replan"] = trace
    second.diagnostics["planejamento"]["replanejamentos"] = 1
    return second, final
