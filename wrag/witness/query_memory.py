"""Question-oriented memory: advisory CQ, compact premises, cited interpretations.

The target CQ never changes the v2 executor. Validation checks structure and
provenance, not semantic truth. Derived conclusions never enter the fact graph.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from wrag.llm import GenParams
from wrag.util import sha
from wrag.witness.query import Atom, ConjunctiveQuery, is_var


TARGET_INSTRUCTION = (
    "Represent the question as a conjunctive query with an answer variable, "
    "preserving participants, roles, qualifiers and temporal constraints. "
    "The query is a hypothesis; do not supply an answer."
)
TARGET_FORMAT = (
    'Return JSON: {"answer_var":"x","atoms":[{"relation":"predicate",'
    '"subject":"entity or ?variable","object":"entity or ?variable",'
    '"time":"optional ?time variable"}],"expected_type":"answer type",'
    '"operation":"value|set|count|date|duration|first|last|comparison|yesno|inference",'
    '"constraints":["requirements not expressed by the positive atoms"]}. '
    "Bind the answer variable in an atom; use at most six atoms."
)
REFLECT_INSTRUCTION = (
    "Given the question, target query and facts, identify supported conclusions, "
    "conflicting claims and missing premises. Distinguish event dates from session "
    "dates and plans from occurrences. Use ordinary knowledge only as an explicit "
    "inference bridge. Preserve uncertainty. Cite fact IDs for each conclusion; "
    "keep observations and inferences separate. Session headings give when facts "
    "were said; only explicit event dates establish when events occurred."
)
REFLECT_FORMAT = (
    'Return JSON: {"conclusions":[{"text":"supported conclusion",'
    '"kind":"observation|inference","premises":["F0"],'
    '"bridge":"knowledge connecting premises to an inference; empty for observation"}],'
    '"conflicts":[{"text":"unresolved conflict or justified resolution",'
    '"premises":["F0","F1"]}],"missing":["missing evidence or unresolved constraint"]}. '
    "Use only IDs in FACTS. Empty arrays are valid. Do not repeat all facts."
)
OPERATIONS = {"value", "set", "count", "date", "duration", "first", "last",
              "comparison", "yesno", "inference"}


class UnboundTargetVariable(ValueError):
    def __init__(self, answer, candidates, data):
        super().__init__(f"Target answer variable {answer!r} is not bound; candidates={candidates}")
        self.candidates, self.data = candidates, data


def _strings(values, name):
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must be a list of nonempty strings")
    return list(dict.fromkeys(v.strip() for v in values))


def validate_target(data):
    if not isinstance(data, dict):
        raise ValueError("Target query must be a JSON object")
    operation = data.get("operation")
    if operation not in OPERATIONS:
        raise ValueError("Invalid target operation")
    answer = data.get("answer_var")
    if not (answer is None and operation == "yesno") and (
            not isinstance(answer, str) or not re.fullmatch(r"\??[A-Za-z_]\w*", answer)):
        raise ValueError("Invalid answer variable")
    answer_name = (answer or "").lstrip("?")
    rows = data.get("atoms")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 6:
        raise ValueError("Target query requires one to six atoms")
    atoms, repairs = [], []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid target atom")
        values = [row.get(k) for k in ("relation", "subject", "object")]
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Target atom requires relation, subject and object")
        when = row.get("time", "")
        if not isinstance(when, str):
            raise ValueError("Invalid target time")
        # The format example is sometimes copied literally. This denotes an
        # omitted optional field, not an event time or an existential variable.
        if re.fullmatch(r"optional \?[A-Za-z_]\w*(?: variable)?", when.strip()):
            repairs.append({"kind": "omitted_optional_time_placeholder", "from": when})
            when = ""
        for index in (1, 2):
            if answer_name and values[index].strip() == answer_name:
                repairs.append({"kind": "answer_variable_prefix", "term": values[index].strip()})
                values[index] = "?" + answer_name
        for term in [values[1], values[2], when]:
            if term.startswith("?") and not is_var(term):
                raise ValueError("Invalid atom variable")
        atoms.append(Atom(*[v.strip() for v in values], time=when.strip()))
    query = ConjunctiveQuery(answer_var=answer_name, atoms=atoms,
                             source="llm-advisory-target")
    if operation == "yesno":
        # Boolean CQs existentially quantify their variables and have no
        # projected entity/time head. This advisory representation is never
        # sent to the executor's single-variable CQ interface.
        if answer is not None:
            repairs.append({"kind": "boolean_projection", "from": answer, "to": None})
        query.answer_var = None
    elif query.answer_var not in query.variables():
        # LLMs sometimes retain the schema's default head `x` while using
        # a descriptive variable (?time, ?book) in the atom. With exactly one
        # bound variable, projection is unambiguous; no predicate is invented.
        variables = query.variables()
        if len(variables) != 1:
            raise UnboundTargetVariable(query.answer_var, variables, data)
        repairs.append({"kind": "unique_bound_variable_projection",
                        "from": query.answer_var, "to": variables[0]})
        query.answer_var = variables[0]
    expected = data.get("expected_type", "other")
    if not isinstance(expected, str) or not expected.strip():
        raise ValueError("Invalid expected answer type")
    normalized = {"answer_var": query.answer_var, "atoms": [a.to_dict() for a in atoms],
            "expected_type": expected.strip(), "operation": operation,
            "constraints": _strings(data.get("constraints", []), "constraints"),
            "status": "semantic_hypothesis_not_executed", "uses_annotations": False}
    # Keep a normalization audit when revalidating a shared, saved target.
    if repairs:
        normalized["normalizations"] = repairs
    elif data.get("normalizations"):
        normalized["normalizations"] = data["normalizations"]
    if data.get("generation_repairs"):
        normalized["generation_repairs"] = data["generation_repairs"]
    return normalized


def _json_result(result, stage):
    if result.filtered:
        raise RuntimeError(f"{stage} was filtered; no downstream answer generated")
    if result.exhausted or result.finish_reason == "length":
        raise ValueError(f"{stage} output was truncated; increase its output budget")
    return result.json()


def target_query(llm, question_text, max_tokens=384):
    prompt = TARGET_INSTRUCTION + "\n" + TARGET_FORMAT + "\nQUESTION: " + question_text
    return _validated_generation(llm, prompt, "memory.target", max_tokens, validate_target)


def _validated_generation(llm, prompt, stage, max_tokens, validator, fallback=None):
    """One bounded output repair; all calls and failed outputs remain audited."""
    params = GenParams(max_tokens=max_tokens, json_mode=True, exact_max_tokens=True)
    result = llm.chat(prompt, params=params, stage=stage)
    try:
        return validator(_json_result(result, stage))
    except ValueError as error:
        reason = str(error)
        projection_error = error if isinstance(error, UnboundTargetVariable) and error.candidates else None
    print(f"[repair] {stage}: {reason}", flush=True)
    repair_prompt = (prompt + "\nOUTPUT REPAIR: The previous output failed validation: " + reason +
        "\nCorrect the JSON using the original question and supplied evidence. "
        "For a target query, answer_var must name an existing atom variable that represents "
        "the requested answer, with a ? prefix in its atom term; correct a missing prefix "
        "or misplaced head without adding predicates. Omit optional time when unspecified. "
        "For a reflection, remove unsupported "
        "claims and cite only supplied fact IDs. Return complete JSON within the output budget. "
        "Do not add personal facts.\nPREVIOUS OUTPUT:\n" + result.text)
    if projection_error is not None:
        # Repair only the projection, not all atoms. Finite choices prevent the
        # schema's example head from being copied again and use fewer tokens.
        repair_prompt = ("QUESTION: " + prompt.partition("\nQUESTION: ")[2] +
            "\nPROJECTION REPAIR: Choose the existing variable that "
            "represents the answer requested by QUESTION. Keep the supplied query unchanged. "
            'Return ONLY JSON {"answer_var":"chosen variable without ?"}. '
            "Allowed choices: " + json.dumps(projection_error.candidates) +
            "\nSUPPLIED QUERY:\n" + json.dumps(projection_error.data, ensure_ascii=False))
    repaired = llm.chat(repair_prompt, params=params, stage=stage+".repair")
    repaired_data = _json_result(repaired, stage+".repair")
    if projection_error is not None:
        choice = repaired_data.get("answer_var") if isinstance(repaired_data, dict) else None
        if not isinstance(choice, str) or choice.lstrip("?") not in projection_error.candidates:
            raise ValueError("Projection repair did not select an allowed answer variable")
        repaired_data = {**projection_error.data, "answer_var":choice.lstrip("?")}
    repair_error = None
    try:
        value = validator(repaired_data)
    except ValueError as error:
        if fallback is None:
            raise
        value = fallback(repaired_data)
        repair_error = str(error)
        print(f"[discard] {stage}: {len(value['discarded_entries'])} unsupported entries removed", flush=True)
    value["generation_repairs"] = [{"stage":stage+".repair", "error":reason,
        "previous_output":result.text, "previous_output_sha256":sha(result.text)}]
    if repair_error is not None:
        value["generation_repairs"][0].update(repair_error=repair_error, repair_output=repaired.text)
    return value


@dataclass
class FactPacket:
    records: list[dict]
    text: str
    dropped_packages: list[list[int]]
    budget: int

    def to_dict(self):
        return {"budget": self.budget, "n_facts": len(self.records),
                "records": self.records, "text": self.text,
                "dropped_packages": self.dropped_packages,
                "source_ids": [r["turn_id"] for r in self.records if r["turn_id"]]}


def compact_packet(facts, dated, diagnostics, budget, body="statement"):
    """Hard cap; prioritize complete selected witness packages before fill facts."""
    if budget not in (10, 20, 40) or body not in ("statement", "triple"):
        raise ValueError("Expected a 10/20/40 fact budget and statement/triple body")
    indices = list(dict.fromkeys(diagnostics["fatos_entregues"]["indices"]))
    if any(type(i) is not int or not 0 <= i < len(facts) for i in indices):
        raise ValueError("Invalid delivered fact index")
    available = set(indices)
    selected, dropped, blocked = [], [], set()
    for row in diagnostics["local_plans"].get("selected", []):
        package = list(dict.fromkeys(row.get("package_facts", row.get("facts", []))))
        if not package or not set(package) <= available:
            raise ValueError("Selected witness package is absent from delivered facts")
        extra = [i for i in package if i not in selected]
        if len(selected) + len(extra) <= budget:
            selected.extend(extra)
        else:
            dropped.append(package)
            blocked.update(package)
    for i in indices:
        if len(selected) == budget:
            break
        if i not in selected and i not in blocked:
            selected.append(i)
    records = []
    for i in selected:
        fact = facts[i]
        pid, position = dated.fact_turn[i]
        turns = dated.turns.get(pid, [])
        turn = turns[position] if 0 <= position < len(turns) else None
        interval = dated.fact_interval[i]
        event = None
        if interval and dated.fact_time_source[i] == "expressao":
            event = {"start": str(interval.start), "end": str(interval.end),
                     "expression": fact.time}
        records.append({"id": f"F{i}", "index": i, "fid": fact.fid, "pid": fact.pid,
                        "turn_id": turn.turn_id if turn else "",
                        "speaker": turn.speaker if turn else "",
                        "session": str(turn.when) if turn and turn.when else "unknown",
                        "kind": fact.kind or "unknown", "event": event,
                        "time_expression": fact.time,
                        "triple": list(fact.triple),
                        "body": fact.statement if body == "statement" and fact.statement
                                else " | ".join(fact.triple)})
    lines = []
    for session in sorted({r["session"] for r in records}):
        lines.append(f"Session {session}:")
        for r in records:
            if r["session"] != session:
                continue
            suffix = ""
            if r["event"]:
                e = r["event"]
                span = e["start"] + (".." + e["end"] if e["start"] != e["end"] else "")
                suffix = f'; event={span}; said={e["expression"]}'
            elif r["time_expression"]:
                suffix = f'; stated time={r["time_expression"]} (unresolved)'
            lines.append(f'{r["id"]} ({r["kind"]}) {r["body"]}{suffix}')
    return FactPacket(records, "\n".join(lines), dropped, budget)


def validate_reflection(data, packet):
    if not isinstance(data, dict):
        raise ValueError("Reflection must be a JSON object")
    allowed = {r["id"] for r in packet.records}
    accepted = {"conclusions": [], "conflicts": [],
                "missing": _strings(data.get("missing", []), "missing")}
    for name in ("conclusions", "conflicts"):
        rows = data.get(name)
        if not isinstance(rows, list):
            raise ValueError(f"Missing reflection array: {name}")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Reflection entry requires a text")
            premises = _strings(row.get("premises"), "premises")
            if not premises or not set(premises) <= allowed:
                raise ValueError("Reflection cites absent or empty premises")
            entry = {"text": row["text"].strip(), "premises": premises}
            if name == "conclusions":
                kind, bridge = row.get("kind"), row.get("bridge", "")
                if kind not in ("observation", "inference") or not isinstance(bridge, str):
                    raise ValueError("Invalid reflection kind or bridge")
                if kind == "inference" and not bridge.strip():
                    raise ValueError("Inference requires an explicit knowledge bridge")
                entry.update(kind=kind, bridge=bridge.strip())
            elif len(premises) < 2:
                raise ValueError("Conflict requires at least two distinct premises")
            accepted[name].append(entry)
    accepted["status"] = "references_validated_semantics_not_verified"
    return accepted


def discard_invalid_reflection_entries(data, packet):
    """Conservative fallback after the single model repair, never invent citations.

    A malformed envelope still fails. Each surviving entry must independently
    pass the strict reference/kind/bridge contract; rejected claims are audited.
    """
    if not isinstance(data, dict) or any(not isinstance(data.get(name), list)
                                        for name in ("conclusions", "conflicts")):
        raise ValueError("Cannot recover a malformed reflection envelope")
    clean = {"conclusions":[], "conflicts":[], "missing":_strings(data.get("missing", []), "missing")}
    discarded = []
    for name in ("conclusions", "conflicts"):
        for index, row in enumerate(data[name]):
            probe = {"conclusions":[], "conflicts":[], "missing":[]}
            probe[name] = [row]
            try:
                accepted = validate_reflection(probe, packet)
            except ValueError as error:
                discarded.append({"array":name, "index":index, "entry":row, "error":str(error)})
            else:
                clean[name].extend(accepted[name])
    if not discarded:
        raise ValueError("Reflection failure cannot be recovered by discarding entries")
    clean["missing"].append("Some candidate interpretations lacked valid support and were omitted.")
    value = validate_reflection(clean, packet)
    value["discarded_entries"] = discarded
    return value


def reflect(llm, question_text, target, packet, max_tokens=1024):
    target_input = {k:v for k,v in target.items() if k not in {"generation_repairs", "normalizations"}}
    prompt = (REFLECT_INSTRUCTION + "\n" + REFLECT_FORMAT + "\nQUESTION: " + question_text +
              "\nTARGET QUERY: " + json.dumps(target_input, ensure_ascii=False) +
              "\nFACTS:\n" + (packet.text or "(no retrieved facts)"))
    return _validated_generation(llm, prompt, "memory.reflect", max_tokens,
                                  lambda data:validate_reflection(data, packet),
                                  fallback=lambda data:discard_invalid_reflection_entries(data, packet))


def reader_block(reflection):
    lines = ["Memory interpretations: references checked; semantic conclusions remain tentative."]
    for row in reflection["conclusions"]:
        line = f'{row["kind"]}: {row["text"]} [premises: {", ".join(row["premises"])}]'
        if row["bridge"]:
            line += " Bridge: " + row["bridge"]
        lines.append(line)
    for row in reflection["conflicts"]:
        lines.append(f'Conflict: {row["text"]} [premises: {", ".join(row["premises"])}]')
    lines += ["Missing: " + text for text in reflection["missing"]]
    return {"title": "Question-oriented memory conclusions", "text": "\n".join(lines)}
