"""Bounded evidence union for a single retry; never slice a candidate join."""
from __future__ import annotations

import copy
import re


def packages(packet):
    allowed = set(packet.diagnostics.get("fatos_entregues", {}).get("indices", []))
    return [list(dict.fromkeys(p.get("package_facts", p.get("facts", []))))
            for p in packet.diagnostics.get("local_plans", {}).get("selected", [])
            if p.get("package_facts", p.get("facts"))
            and set(p.get("package_facts", p.get("facts", []))) <= allowed]


def select_union(initial, retry, keep, budget):
    """Reserve 60% for initial premises, up to 40% for complementary evidence.

    Candidate packages are indivisible. A package that cannot fit is skipped,
    including its members in the subsequent independent-fact fill. Unused
    capacity goes back to complete initial packages and independent facts.
    """
    old = list(dict.fromkeys(initial.diagnostics["fatos_entregues"]["indices"]))
    fresh = list(dict.fromkeys(retry.diagnostics["fatos_entregues"]["indices"]))
    old_groups, new_groups = packages(initial), packages(retry)
    old_members = {i for group in old_groups for i in group}
    new_members = {i for group in new_groups for i in group}
    chosen = []
    quota = min(budget, max(1, (3 * budget + 4) // 5))

    def admit(group, limit):
        additions = [i for i in group if i not in chosen]
        if len(chosen) + len(additions) <= limit:
            chosen.extend(additions)

    # Preserve evidence chains before singleton preferences from the checker.
    for group in sorted(old_groups, key=lambda g: not bool(set(g) & set(keep))):
        admit(group, quota)
    for i in dict.fromkeys(list(keep) + old):
        if i in old and i not in old_members:
            admit([i], quota)
    for group in new_groups:
        if set(group) - set(old):
            admit(group, budget)
    for i in fresh:
        if i not in old and i not in new_members:
            admit([i], budget)
    for group in old_groups:
        admit(group, budget)
    for i in old:
        if i not in old_members:
            admit([i], budget)
    return chosen


def literal_sections(packet):
    """Read only the literal rescue section already delivered to the reader."""
    lines = []
    marker = "\n\nAdditional original turns (candidate support, not inferred facts):\n"
    for block in packet.diagnostics.get("trechos_extras", []):
        text = block.get("text", "")
        if marker in text:
            section = text.split(marker, 1)[1].split("\n\n", 1)[0]
            lines.extend(line for line in section.splitlines() if re.match(r"^\[[^\]]+\]", line))
    return lines


def compose_union(retriever, initial, retry, indices, approved_sources=None):
    """Render actual selected facts and bounded original quotations.

    No checker text, guesses or retrieved-but-undelivered rescue turns enter
    the reader. Each source section retains the standard character cap.
    """
    cap = retriever.ctx.run.witness.excerpt_max_chars
    original_text = initial.diagnostics["trechos_extras"][0]["text"]
    marker = "\n\nRetrieved facts (joins are unverified candidates):\n"
    prefix = original_text.split(marker, 1)[0] if marker in original_text else ""
    facts = retriever._render_fact_ids(indices)
    text = prefix + marker + facts
    primary, turns, used = [], [], 0
    dated = retriever.dated
    # Protect the literal premises actually printed in the first packet, not
    # just their extracted triples. Otherwise filler facts can exhaust the
    # quote cap before a retained bridge's decisive qualifier is printed.
    source_priority = initial.diagnostics.get("local_plans", {}).get("source_turns", [])
    rank = {tid: n for n, tid in enumerate(source_priority)}
    def source_rank(i):
        if not hasattr(dated, "fact_turn"):
            return len(rank)
        pid, pos = dated.fact_turn[i]
        candidates = dated.turns.get(pid, [])
        tid = candidates[pos].turn_id if 0 <= pos < len(candidates) else ""
        return rank.get(tid, len(rank))
    for i in sorted(indices, key=source_rank):
        if not hasattr(dated, "fact_turn"):
            break
        pid, pos = dated.fact_turn[i]
        candidates = dated.turns.get(pid, [])
        if not 0 <= pos < len(candidates):
            continue
        turn = candidates[pos]
        if approved_sources is not None and turn.turn_id not in source_priority:
            # Novel quotations are delivered only as the exact judged rescue
            # text below, not an expanded and previously unseen source body.
            continue
        if turn.turn_id in turns:
            continue
        # Use original passage lines, including released caption/temporal lines.
        lines = retriever.corpus.get(pid).text.splitlines()
        if not 0 <= turn.line < len(lines):
            continue
        body = [lines[turn.line]]
        for line in lines[turn.line + 1:]:
            if line.startswith((f"[{turn.turn_id}]", f"[{turn.turn_id} ")):
                body.append(line)
            else:
                break
        header = f"Session date: {turn.when or 'unknown'}"
        quote = header + "\n" + "\n".join(body)
        if used + len(quote) + bool(primary) > cap:
            continue
        primary.append(quote)
        used += len(quote) + (len(primary) > 1)
        turns.append(turn.turn_id)
    if primary:
        text += "\n\nOriginal source turns:\n" + "\n".join(primary)
    # Alternate initial/retry literal rescue, preserving useful initial cues.
    old_lines = literal_sections(initial)
    new_lines = (literal_sections(retry) if approved_sources is None else
                 [text for text in approved_sources.values()])
    rescue, rescue_ids, used = [], [], 0
    for n in range(max(len(old_lines), len(new_lines))):
        for candidates in (old_lines, new_lines):
            if n >= len(candidates):
                continue
            line = candidates[n]
            match = re.search(r"\[([^\]\s]+)", line)
            if not match:
                continue
            tid = match.group(1)
            if tid in turns or tid in rescue_ids or len(rescue) >= 4:
                continue
            if used + len(line) + bool(rescue) > cap:
                continue
            rescue.append(line)
            rescue_ids.append(tid)
            used += len(line) + (len(rescue) > 1)
    if rescue:
        text += "\n\nAdditional original turns (candidate support, not inferred facts):\n" + "\n".join(rescue)
    groups = [g for packet in (initial, retry)
              for g in packet.diagnostics.get("local_plans", {}).get("count_groups", [])
              if set(g.get("facts", [])) <= set(indices)]
    if groups:
        text += "\n\nSource mention groups for counting (not a certified count):\n" + "\n".join(
            str(g["identity"]) + " facts " + str(g["facts"]) for g in groups)
    pending = original_text.rsplit("\n\nPending checks: ", 1)
    if len(pending) == 2:
        text += "\n\nPending checks: " + pending[1]
    result = copy.deepcopy(retry)
    result.diagnostics["trechos_extras"] = [{"title": "Local logical retrieval", "text": text}]
    info = result.diagnostics["fatos_entregues"]
    sources = {s["indice"]: s for packet in (initial, retry)
               for s in packet.diagnostics["fatos_entregues"].get("fontes", [])}
    info.update(indices=indices, fontes=[sources[i] for i in indices], n=len(indices),
                entregues=len(indices), orcamento=retriever.ctx.run.witness.fact_budget)
    local = result.diagnostics.setdefault("local_plans", {})
    local["selected"] = [copy.deepcopy(p) for packet in (initial, retry)
                         for p in packet.diagnostics.get("local_plans", {}).get("selected", [])
                         if set(p.get("package_facts", p.get("facts", []))) <= set(indices)]
    local.update(source_turns=turns, additional_source_turns=[{"turn_id": tid} for tid in rescue_ids],
                 count_groups=groups)
    return result, {"primary_source_chars": len("\n".join(primary)),
                    "additional_source_chars": len("\n".join(rescue)),
                    "delivered_source_turns": turns + rescue_ids}
