"""Paired memory-budget experiment over a saved local-v2 execution.

Replaying saved executor output is valid because the new CQ is advisory and
never steers retrieval. All arms receive identical reader evidence. Historical
CPU time is reported separately from this experiment's new inference time.
"""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

from wrag import config as C
from wrag.data import load_dataset
from wrag.eval import locomo_official as LO
from wrag.eval import metrics as M
from wrag.eval.reader import read
from wrag.eval.reflection_study import atomic_json, file_lock
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import sum_usage, usage_delta
from wrag.util import sha
from wrag.witness.dated_memory import DatedMemory
from wrag.witness.query_memory import (REFLECT_FORMAT, REFLECT_INSTRUCTION, TARGET_FORMAT,
                                      TARGET_INSTRUCTION, compact_packet, reader_block,
                                      reflect, target_query, validate_target)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_questions(questions, limit):
    if not limit or limit >= len(questions):
        return list(questions)
    groups = {kind: [q for q in questions if q.qtype == kind]
              for kind in sorted({q.qtype for q in questions})}
    chosen = []
    while len(chosen) < limit:
        for group in groups.values():
            if group and len(chosen) < limit:
                chosen.append(group.pop(0))
    return chosen


def load_source(source_run, conversation=0, cache_dir=None, extraction_cache=None):
    """Identify the exact existing extraction by every delivered index/fid/pid."""
    source_run = Path(source_run).resolve()
    status_path = source_run / "run_status.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "invalid_aborted":
        raise ValueError("Source run was marked invalid")
    conv = source_run / "conversations" / f"conv{conversation:02d}"
    if not conv.is_dir():
        raise FileNotFoundError(f"Prepared conversation missing: {conv}")
    paths = sorted(conv.glob("benchmark/*/locomo/witnessrag.jsonl"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one saved WitnessRAG execution per conversation")
    source_path = paths[0]
    rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Saved execution is empty or has duplicate question IDs")
    corpus = load_dataset("locomo", data_dir=conv / "data")
    by_question = {q.qid: q for q in corpus.questions}
    if {r["qid"] for r in rows} != set(by_question):
        raise ValueError("Saved executor run must contain every prepared question")
    references = []
    for row in rows:
        q = by_question.get(row["qid"])
        if q is None or row["pergunta"] != q.question or row["respostas_ouro"] != q.answers:
            raise ValueError("Saved questions differ from the prepared corpus")
        d = row["diagnosticos"]
        if row.get("filtrada") or d.get("local_plans", {}).get("version") != "v2":
            raise ValueError("Requires unfiltered local-v2 executor results")
        if not d.get("leitura_fatos") or not d.get("trechos_extras"):
            raise ValueError("Saved execution lacks the actual fact reader context")
        sources = d["fatos_entregues"]["fontes"]
        if {s["indice"] for s in sources} != set(d["fatos_entregues"]["indices"]):
            raise ValueError("Delivered fact source map is incomplete")
        references.extend(sources)
    if extraction_cache:
        candidates = [Path(extraction_cache).resolve()]
    else:
        if cache_dir is None:
            pilot = source_run / "pilot.json"
            saved_cache = json.loads(pilot.read_text(encoding="utf-8")).get("settings", {}).get("cache_dir") if pilot.exists() else None
            cache_dir = Path(saved_cache) if saved_cache and Path(saved_cache).is_dir() else C.CACHE_DIR
        candidates = sorted((Path(cache_dir) / "openie").glob("locomo-*.json"))
    matches = {}
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if extraction_cache:
                raise ValueError("Explicit extraction cache is malformed")
            continue
        facts = data.get("facts", [])
        if all(type(s["indice"]) is int and 0 <= s["indice"] < len(facts)
               and facts[s["indice"]]["fid"] == s["fid"]
               and facts[s["indice"]]["pid"] == s["pid"] for s in references):
            matches[file_hash(path)] = (path, data)
    if len(matches) != 1:
        raise ValueError("Cannot identify a unique matching extraction cache; provide --extraction-cache")
    extraction_path, saved = next(iter(matches.values()))
    extraction = ExtractionResult(facts=[Fact(
        fid=f["fid"], subject=f["s"], relation=f["r"], object=f["o"], pid=f["pid"],
        confidence=f.get("conf", .9), time=f.get("t", ""), statement=f.get("st", ""),
        turn_id=f.get("turn", ""), kind=f.get("kind", "")) for f in saved["facts"]])
    if any(f.pid not in corpus.pids for f in extraction.facts):
        raise ValueError("Extraction includes facts outside the selected conversation")
    dated = DatedMemory(corpus, extraction.facts)
    for row in rows:
        d = row["diagnosticos"]
        context = "\n".join(b["text"] for b in d["trechos_extras"])
        for source in d["fatos_entregues"]["fontes"]:
            i = source["indice"]
            fact = extraction.facts[i]
            if (fact.statement or " | ".join(fact.triple)) not in context:
                raise ValueError("Saved reader context is truncated or does not contain a delivered fact")
            pid, pos = dated.fact_turn[i]
            turns = dated.turns.get(pid, [])
            turn_id = turns[pos].turn_id if 0 <= pos < len(turns) else ""
            if turn_id != source.get("turn_id", ""):
                raise ValueError("Saved fact provenance differs from the literal source")
    source_cfg = source_path.parents[1] / "run.json"
    run = json.loads(source_cfg.read_text(encoding="utf-8"))
    provenance = {"source_run": str(source_run), "conversation": conversation,
                  "source_predictions": str(source_path), "source_sha256": file_hash(source_path),
                  "source_configuration": run["config"],
                  "extraction_cache": str(extraction_path), "extraction_sha256": file_hash(extraction_path),
                  "corpus_hash": corpus.stats()["corpus_hash"],
                  "fact_sources_verified": True, "executor_mode": "saved_v2_execution_replay",
                  "memory_construction_cost": "historical_not_measured_here"}
    return corpus, extraction.facts, dated, {r["qid"]: r for r in rows}, provenance


def evaluate_variant(llm, corpus, question, source, target, packet, target_usage,
                     reflector_max_tokens=1024, reader_max_tokens=128, memory_only=False):
    before = llm.usage.snapshot()
    started = time.perf_counter()
    reflection = reflect(llm, question.question, target, packet, reflector_max_tokens)
    reflector_seconds = time.perf_counter() - started
    after_reflector = llm.usage.snapshot()
    reading = None
    if not memory_only:
        cfg = C.QAConfig(evidence_reader=True, reader_reflection=False,
                         temporal_annotations=True, answer_set=True,
                         answer_guard=False, max_tokens=reader_max_tokens)
        extras = copy.deepcopy(source["diagnosticos"]["trechos_extras"])
        extras.append(reader_block(reflection))
        reading = read(llm, corpus, question, [], cfg, method="query-memory",
                       extra_passages=extras,
                       facts_mode=source["diagnosticos"]["leitura_fatos"])
        if reading.filtered:
            raise RuntimeError("qa was filtered; this arm is incomplete")
    own_usage = usage_delta(llm.usage.snapshot(), before)
    memory_usage = sum_usage([target_usage, usage_delta(after_reflector, before)])
    reader_usage = usage_delta(llm.usage.snapshot(), after_reflector)
    record = {"qid": question.qid, "dataset": corpus.name, "pergunta": question.question,
              "tipo": question.qtype, "respostas_ouro": list(question.answers),
              "metodo": f"query-memory-{packet.budget}", "fact_budget": packet.budget,
              "memory_only": memory_only, "resposta": reading.answer if reading else None,
              "memory": {"target": target, "packet": packet.to_dict(), "reflection": reflection},
              "usage_memory": memory_usage, "usage_reader": reader_usage,
              "uso_llm": sum_usage([target_usage, own_usage]),
              "target_is_shared": True, "reflector_seconds": reflector_seconds,
              "reader_seconds": reading.latency_s if reading else 0,
              "historical_retrieval_seconds": source["latencia_recuperacao_s"],
              "reader_base_evidence_sha256": sha(source["diagnosticos"]["trechos_extras"]),
              "reference_validation_only": True}
    if reading:
        record.update(f1=M.token_f1(reading.answer, question.answers),
                      em=M.exact_match(reading.answer, question.answers))
        record.update(LO.score_record(record))
    return record


def token_count(usage):
    t = usage.get("total", {})
    return t.get("tokens_prompt", 0) + t.get("tokens_resposta", 0)


def read_results(output, budget):
    paths = sorted((Path(output) / f"facts-{budget}" / "results").glob("*.json"))
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate result IDs")
    return rows


def write_report(output, budgets, expected_ids, baseline, memory_only):
    output = Path(output)
    arms = {n: read_results(output, n) for n in budgets}
    expected = set(expected_ids)
    if any({r["qid"] for r in rows} - expected for rows in arms.values()):
        raise ValueError("Unexpected question in experiment output")
    common = set.intersection(*[{r["qid"] for r in rows} for rows in arms.values()])
    by_arm = {n: {r["qid"]: r for r in rows} for n, rows in arms.items()}
    for qid in common:
        examples = [by_arm[n][qid] for n in budgets]
        if any(r["pergunta"] != baseline[qid]["pergunta"] or r["tipo"] != baseline[qid]["tipo"] for r in examples):
            raise ValueError("Comparison question text or categories changed")
        if len({sha(r["memory"]["target"]) for r in examples}) != 1 or len({r["reader_base_evidence_sha256"] for r in examples}) != 1:
            raise ValueError("Comparison arms differ in target query or base reader evidence")
    def average(rows, function):
        return statistics.mean(function(r) for r in rows) if rows else None
    def summary(rows):
        return {"n": len(rows), "f1_locomo": None if memory_only else average(rows, lambda r:r["f1_locomo"]),
                "bleu1_locomo": None if memory_only else average(rows, lambda r:r["bleu1_locomo"]),
                "memory_tokens": average(rows, lambda r:token_count(r["usage_memory"])),
                "memory_input_tokens": average(rows, lambda r:r["usage_memory"]["total"].get("tokens_prompt", 0)),
                "memory_output_tokens": average(rows, lambda r:r["usage_memory"]["total"].get("tokens_resposta", 0)),
                "reader_tokens": average(rows, lambda r:token_count(r["usage_reader"])),
                "facts_delivered": average(rows, lambda r:r["memory"]["packet"]["n_facts"]),
                "reflector_seconds": average(rows, lambda r:r["reflector_seconds"]),
                "dropped_packages": sum(len(r["memory"]["packet"]["dropped_packages"]) for r in rows)}
    result = {"protocol": "advisory-cq-reflector-v1", "expected_n": len(expected),
              "paired_n": len(common), "complete": all({r["qid"] for r in rows} == expected for rows in arms.values()),
              "memory_only": memory_only, "arms": {str(n): summary([by_arm[n][q] for q in sorted(common)]) for n in budgets},
              "categories": {kind: {str(n): summary([by_arm[n][q] for q in sorted(common) if baseline[q]["tipo"] == kind]) for n in budgets}
                             for kind in sorted({baseline[q]["tipo"] for q in common})},
              "baseline_f1_locomo": average([baseline[q] for q in common], lambda r:r["f1_locomo"]),
              "historical_retrieval_seconds": average([baseline[q] for q in common], lambda r:r["latencia_recuperacao_s"]),
              "logical_usage_note": "Each arm includes one target query; physically it is generated once per question. Logical tokens include local cache hits.",
              "reader_control": "Same original 40-fact/source context in all arms; joint reader reflection disabled. Baseline had joint reader reflection."}
    result["paired_differences"] = {}
    if not memory_only and common:
        import itertools
        for a, b in itertools.combinations(budgets, 2):
            deltas = [by_arm[b][q]["f1_locomo"]-by_arm[a][q]["f1_locomo"] for q in sorted(common)]
            result["paired_differences"][f"{a}_to_{b}"] = {
                "n": len(deltas), "f1_delta": statistics.mean(deltas),
                "improved": sum(d > 1e-8 for d in deltas),
                "worsened": sum(d < -1e-8 for d in deltas),
                "tied": sum(abs(d) <= 1e-8 for d in deltas)}
    attempts = [json.loads(p.read_text(encoding="utf-8"))["usage"] for p in (output / "attempts").glob("*.json")]
    result["execution_attempt_usage"] = sum_usage(attempts)
    history = [json.loads(p.read_text(encoding="utf-8")) for p in (output/"manifest-history").glob("*.json")]
    result["implementation_history"] = [{"code_hash":m.get("code_hash"), "prompts_hash":m.get("prompts_hash")} for m in history]
    result["target_repairs"] = sum(len(by_arm[budgets[0]][q]["memory"]["target"].get("generation_repairs", [])) for q in common)
    result["reflector_repairs"] = {str(n):sum(len(r["memory"]["reflection"].get("generation_repairs", [])) for r in arms[n]) for n in budgets}
    result["discarded_reflector_entries"] = {str(n):sum(len(r["memory"]["reflection"].get("discarded_entries", [])) for r in arms[n]) for n in budgets}
    for n, rows in arms.items():
        folder = output / f"facts-{n}"
        folder.mkdir(exist_ok=True)
        ordered = sorted(rows, key=lambda r:expected_ids.index(r["qid"]))
        (folder / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in ordered), encoding="utf-8")
        atomic_json(folder / "summary.json", summary(rows))
    atomic_json(output / "comparison.json", result)
    lines = ["# Comparação de memória: 10, 20 e 40 fatos", "",
             f"Perguntas pareadas: {len(common)}/{len(expected)}. Completa: {result['complete']}. Apenas memória: {memory_only}.", "",
             "A consulta-alvo e a execução v2 são compartilhadas. O respondedor recebe os mesmos fatos e fontes originais em todas as variantes, mais a saída do respectivo reflector. A reflexão conjunta do reader está desligada.", "",
             "| Fatos (teto) | n | Fatos entregues | F1 LoCoMo | BLEU-1 | Tokens memória/pergunta | Tokens respondedor/pergunta |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for n in budgets:
        s = result["arms"][str(n)]
        fmt = lambda v: "—" if v is None else f"{v:.1f}"
        lines.append(f"| {n} | {s['n']} | {fmt(s['facts_delivered'])} | {fmt(100*s['f1_locomo'] if s['f1_locomo'] is not None else None)} | {fmt(100*s['bleu1_locomo'] if s['bleu1_locomo'] is not None else None)} | {fmt(s['memory_tokens'])} | {fmt(s['reader_tokens'])} |")
    if result["baseline_f1_locomo"] is not None:
        lines += ["", f"F1 da run original nas mesmas perguntas: {100*result['baseline_f1_locomo']:.2f}. Essa referência tinha reflexão dentro do reader; a comparação entre os três tetos é o controle pareado principal."]
    for pair, values in result["paired_differences"].items():
        lines += ["", f"{pair.replace('_to_', ' → ')}: Δ F1 {100*values['f1_delta']:+.2f} pp; {values['improved']} melhoraram, {values['worsened']} pioraram, {values['tied']} empataram."]
    lines += ["", "Tokens da memória incluem consulta-alvo e reflector, com entrada e saída. Tokens lógicos incluem cache. Não somar o custo da consulta-alvo três vezes para estimar o gasto físico. `execution_attempt_usage` registra as chamadas desta experiência, inclusive tentativas interrompidas, e distingue chamadas servidas pelo cache local.", "", "Busca e reranking usam resultados salvos; seu tempo original e o custo histórico da construção da memória permanecem separados. Nenhum embedding, reranking ou extração é executado novamente. Validação de IDs e schemas não certifica a verdade das inferências."]
    lines += ["", f"Reparos de saída nas perguntas pareadas: {result['target_repairs']} da consulta-alvo. Reparos do reflector por variante: {result['reflector_repairs']}. Seu custo já integra os tokens da memória."]
    lines += ["", f"Entradas do reflector descartadas por falharem na validação após o reparo: {result['discarded_reflector_entries']}. Somente conclusões/conflitos com referências válidas chegam ao reader; os descartes ficam auditados em cada resultado."]
    if history:
        lines += ["", "Correções de código durante a execução foram registradas em `manifest-history/`. Resultados já concluídos foram preservados; a comparação permanece pareada dentro de cada pergunta."]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result


def check_manifest(output, identity, resume, allow_code_update=False):
    path = Path(output) / "manifest.json"
    normalized = json.loads(json.dumps(identity, ensure_ascii=False))
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if resume and saved != normalized:
            # An interrupted, empty experiment may adopt a code-only bug fix.
            # Keep the original manifest and attempted-call cost for auditing.
            # Completed arms or a shared target forbid this migration.
            has_artifacts = any(Path(output).glob("facts-*/results/*.json")) or any((Path(output)/"shared").glob("*.json"))
            compatible = lambda value: {k:v for k,v in value.items() if k != "code_hash"}
            if (not has_artifacts or allow_code_update) and compatible(saved) == compatible(normalized):
                archive = Path(output)/"manifest-history"/(sha(saved)+".json")
                atomic_json(archive, saved)
                atomic_json(path, normalized)
                return
        if not resume or saved != normalized:
            raise ValueError("Existing experiment differs or --resume is missing; use a new output")
    elif any(Path(output).iterdir()):
        raise ValueError("Output contains files without a compatible manifest")
    else:
        atomic_json(path, normalized)


def run_experiment(llm, corpus, facts, dated, baseline, provenance, output,
                   budgets=(10,20,40), questions=0, body="statement", resume=False,
                   memory_only=False, target_max_tokens=384, reflector_max_tokens=1024,
                   reader_max_tokens=128, model="gpt-4o-mini", endpoint=None, code_hash="",
                   allow_code_update=False):
    output = Path(output)
    eligible = [q for q in corpus.questions if q.qid in baseline]
    chosen = choose_questions(eligible, questions)
    if not chosen or questions < 0 or not budgets or len(set(budgets)) != len(budgets) or any(n not in (10,20,40) for n in budgets):
        raise ValueError("Invalid questions or fact budgets")
    identity = {"protocol": "advisory-cq-reflector-v1", "source": provenance,
                "model": model, "endpoint": endpoint, "body": body,
                "budgets": list(budgets), "question_ids": [q.qid for q in chosen],
                "memory_only": memory_only, "code_hash": code_hash,
                "target_max_tokens": target_max_tokens, "reflector_max_tokens": reflector_max_tokens,
                "reader_max_tokens": reader_max_tokens,
                "prompts_hash": sha(TARGET_INSTRUCTION, TARGET_FORMAT, REFLECT_INSTRUCTION, REFLECT_FORMAT)}
    output.mkdir(parents=True, exist_ok=True)
    # Single writer: a second launcher must not race a pending API call.
    with file_lock(output.parent / (output.name + ".experiment.lock")):
        check_manifest(output, identity, resume, allow_code_update)
        llm.usage.reset()
        attempt_id = str(time.time_ns())
        try:
            for position, q in enumerate(chosen, 1):
                label = sha(q.qid)[:24]
                pending = [n for n in budgets if not (output / f"facts-{n}" / "results" / (label+".json")).exists()]
                if not pending:
                    continue
                target_path = output / "shared" / (label+".json")
                if target_path.exists():
                    shared = json.loads(target_path.read_text(encoding="utf-8"))
                    if shared["qid"] != q.qid or shared["question"] != q.question:
                        raise ValueError("Shared target question mismatch")
                    target = validate_target(shared["target"])
                else:
                    before = llm.usage.snapshot()
                    target = target_query(llm, q.question, target_max_tokens)
                    shared = {"qid": q.qid, "question": q.question, "target": target,
                              "usage": usage_delta(llm.usage.snapshot(), before)}
                    atomic_json(target_path, shared)
                for n in pending:
                    packet = compact_packet(facts, dated, baseline[q.qid]["diagnosticos"], n, body)
                    row = evaluate_variant(llm, corpus, q, baseline[q.qid], target, packet,
                                           shared["usage"], reflector_max_tokens, reader_max_tokens, memory_only)
                    row["implementation_code_hash"] = code_hash
                    atomic_json(output / f"facts-{n}" / "results" / (label+".json"), row)
                    print(f"[{position}/{len(chosen)}] facts={n} delivered={len(packet.records)} memory_tokens={token_count(row['usage_memory'])} reader_tokens={token_count(row['usage_reader'])}", flush=True)
                write_report(output, budgets, [item.qid for item in chosen], baseline, memory_only)
        finally:
            atomic_json(output / "attempts" / (attempt_id+".json"), {"usage": llm.usage.snapshot()})
            result = write_report(output, budgets, [q.qid for q in chosen], baseline, memory_only)
        return result
