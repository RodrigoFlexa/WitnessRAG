#!/usr/bin/env python3
"""Component study on one LoCoMo conversation or all ten, against an existing
vLLM (Qwen2.5-14B) or the Azure gateway (gpt-4.1-mini).

Variants (docs/requisitos-prova.md):
  full               witness search + proof requirements + reflection completion
  no-witness         facts ranked individually (no planned witness search)
  no-time-reference  requirement time windows ignored; rule time weights zeroed
  no-time-model      no resolved times at all (implies no-time-reference)
  no-reflection      requirements measured, nothing completed (same reader)

Sequential and resumable. Writes summary.json / summary.md after each variant,
paired by question ID, and, when --baseline is given, a paired comparison with
the earlier study on the same conversation.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {"full": "Completo", "no-witness": "Sem testemunhas",
            "no-time-reference": "Sem referencia temporal", "no-time-model": "Sem modelagem temporal",
            "no-reflection": "Sem reflexao"}
CATEGORIES = ("single-hop", "multi-hop", "temporal", "open-domain")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conversation", required=True, help="index 0-9 or 'all'")
    p.add_argument("--backend", choices=["vllm", "azure"], default="vllm")
    p.add_argument("--deployment", default="gpt-4-1-mini-petrobras", help="Azure deployment (backend azure)")
    p.add_argument("--embed-device", default="cuda", help="cuda or cpu (embeddings and reranker)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=None,
                   help="earlier study root (e.g. runs/locomo-ablation-qwen14b/facts40)")
    p.add_argument("--before", type=Path, default=None,
                   help="run root of the configuration BEFORE the changes, at the same fact budget "
                        "(conversations/convNN inside); default: <baseline>/full")
    p.add_argument("--fact-budget", type=int, default=40)
    p.add_argument("--requirement-threshold", type=float, default=0.5,
                   help="support threshold of the proof requirements (calibrated on the dev conversation only)")
    p.add_argument("--member-threshold", type=float, default=0.5,
                   help="set-member threshold of the enumeration (calibrated on the dev conversation only)")
    p.add_argument("--max-members", type=int, default=8)
    p.add_argument("--no-requirements", action="store_true",
                   help="v2-equivalent configuration (no planner requirements, no member scan)")
    p.add_argument("--member-scan", action="store_true",
                   help="set requirements: exhaustive scan of the named people's turns (member table)")
    p.add_argument("--no-gate", action="store_true",
                   help="run every variant even if 'full' does not beat the 'before' configuration")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--gpu", default="0")
    p.add_argument("--port", type=int, default=8096)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--dry-run", action="store_true")
    return p


def label(args):
    return "todas as conversas" if args.conversation == "all" else f"conv{int(args.conversation):02d}"


def model_name(args):
    return args.deployment if args.backend == "azure" else "Qwen2.5-14B"


def conv_dir(args, root):
    """The before-run rows of this study's conversation(s)."""
    if root is None:
        return None
    if args.conversation == "all":
        return root
    sub = root / "conversations" / f"conv{int(args.conversation):02d}"
    return sub if sub.exists() else root


def command(args, name):
    backend = (["--backend", "vllm", "--existing-server", "--port", str(args.port),
                "--model", "Qwen/Qwen2.5-14B-Instruct"] if args.backend == "vllm" else
               ["--backend", "azure", "--model", args.deployment])
    return [sys.executable, "-u", "-m", "wrag.pilot", *backend,
            # The tokenizer only cuts the 2048-token chunks: the same chunks for every backend.
            "--tokenizer-model", "Qwen/Qwen2.5-14B-Instruct", "--gpu", args.gpu,
            "--concurrency", str(args.concurrency), "--dataset", "locomo",
            "--locomo-conversation", str(args.conversation),
            "--methods", "witnessrag", "--embed-model", "BAAI/bge-m3", "--embed-device", args.embed_device,
            "--locomo-chunk-tokens", "2048", "--locomo-ie-window-tokens", "512", "--top-k", "5",
            "--qa-max-tokens", "128", "--witness-candidate-pool", "20", "--answer-set", "--temporal-annotations",
            "--evidence-reader", "--binding-aware-grounding", "--vocab-compile", "--hybrid-fallback",
            "--dialogue-ie", "--gap-context-rescue", "--proof-controller", "--ie-style", "memory",
            "--fact-delivery", "facts", "--fact-fill", "question", "--fact-time", "both", "--fact-budget", str(args.fact_budget),
            "--fact-rerank", "cross-encoder/ms-marco-MiniLM-L6-v2", "--local-plans", "--reader-reflection",
            "--no-proof-verify", "--local-plan-version", "v2", "--local-plan-beam", "32",
            "--local-plan-candidates", "96", "--local-plan-executions", "4000", "--local-plan-starts", "12",
            *([] if args.no_requirements else [
                "--requirements", "--requirement-threshold", str(args.requirement_threshold),
                "--requirement-member-threshold", str(args.member_threshold),
                "--requirement-max-members", str(args.max_members),
                *(["--member-scan"] if args.member_scan else [])]),
            "--cache-dir", str(args.cache), "--hours", str(args.hours),
            "--study-ablation", name, "--output", str(args.output / name)]


def records(root: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(root.glob("**/locomo/witnessrag.jsonl")):
        if "source-data" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rows[row["qid"]] = row
    return rows


def f1(rows):
    return 100 * statistics.mean(r["f1_locomo"] for r in rows) if rows else float("nan")


def calls(row):
    usage = (row.get("uso_llm") or {}).get("total") or {}
    return usage.get("chamadas", 0), usage.get("tokens_prompt", 0) + usage.get("tokens_resposta", 0)


def before_root(args):
    return args.before or (args.baseline / "full" if args.baseline else None)


def before_report(args):
    """F1 of the configuration BEFORE the v3 changes, read from an existing run."""
    root = before_root(args)
    if root is None:
        return None
    rows = list(records(conv_dir(args, root)).values())
    out = {"source": str(root), "fact_budget": args.fact_budget, "full": None, "all_conversations": None,
           "other_variants": {}}
    if rows:
        out["full"] = {"n": len(rows), "f1": f1(rows),
                       "by_category": {c: f1([r for r in rows if r["tipo"] == c]) for c in CATEGORIES}}
    every = list(records(root).values())
    if len(every) > len(rows):
        out["all_conversations"] = {"n": len(every), "f1": f1(every)}
    if args.baseline and args.fact_budget == 40:
        for name in VARIANTS:
            if name != "full":
                other = list(records(conv_dir(args, args.baseline / name)).values())
                if other:
                    out["other_variants"][name] = f1(other)
    return out


def render_before(before, where):
    if not before or not before.get("full"):
        return "ANTES das mudancas: rodada de referencia nao encontrada."
    full = before["full"]
    cats = ", ".join(f"{c} {full['by_category'][c]:.2f}" for c in CATEGORIES)
    lines = [f"ANTES das mudancas (v2, configuracao completa, {before['fact_budget']} fatos, {where}): "
             f"F1 {full['f1']:.2f} (n={full['n']}; {cats})"]
    if before.get("all_conversations"):
        a = before["all_conversations"]
        lines.append(f"ANTES das mudancas (v2, completo, {before['fact_budget']} fatos, todas as conversas): "
                     f"F1 {a['f1']:.2f} (n={a['n']})")
    if before.get("other_variants"):
        lines.append("Demais variantes v2 nesta conversa: " + ", ".join(
            f"{VARIANTS[n]} {v:.2f}" for n, v in before["other_variants"].items()))
    lines.append(f"Fonte: {before['source']}")
    return "\n".join(lines)


def gate(args):
    """Paired comparison of the new full configuration with the 'before' one."""
    root = before_root(args)
    if root is None:
        return None
    new = records(args.output / "full")
    old = records(conv_dir(args, root))
    common = [q for q in new if q in old]
    if not common:
        return None
    return {"n": len(common), "new": f1([new[q] for q in common]), "old": f1([old[q] for q in common])}


def summarize(args, names):
    data = {name: records(args.output / name) for name in names}
    full = data.get("full", {})
    out = {"conversation": label(args), "model": model_name(args), "fact_budget": args.fact_budget,
           "before": before_report(args), "variants": {}}
    for name, rows in data.items():
        values = list(rows.values())
        common = sorted(set(rows) & set(full))
        entry = {"label": VARIANTS[name], "n": len(values), "f1": f1(values),
                 "by_category": {c: {"n": sum(r["tipo"] == c for r in values),
                                     "f1": f1([r for r in values if r["tipo"] == c])} for c in CATEGORIES},
                 "llm_calls_per_question": statistics.mean(calls(r)[0] for r in values) if values else 0,
                 "llm_tokens_per_question": statistics.mean(calls(r)[1] for r in values) if values else 0}
        if name != "full" and common:
            deltas = [rows[q]["f1_locomo"] - full[q]["f1_locomo"] for q in common]
            entry["paired_vs_full"] = {"n": len(common), "delta": 100 * statistics.mean(deltas),
                                       "better": sum(d > 1e-9 for d in deltas),
                                       "worse": sum(d < -1e-9 for d in deltas)}
        loops = [r["diagnosticos"].get("reflection_loop") for r in values if r["diagnosticos"].get("reflection_loop")]
        if loops:
            entry["reflection"] = {
                "verifier_called": sum(1 for l in loops if l.get("passes")),
                "decisions": dict(collections.Counter(l["passes"][0]["decision"] if l.get("passes") else "not_called"
                                                      for l in loops)),
                "searched": sum(bool(l.get("searched")) for l in loops),
                "final_source": dict(collections.Counter(l.get("final_source") for l in loops)),
                "changed_from_first_reading": sum(bool((r.get("reflexao_leitor") or {}).get("changed_from_first_reading"))
                                                  for r in values)}
        refs = [r["diagnosticos"].get("temporal_reference") for r in values if r["diagnosticos"].get("temporal_reference")]
        if refs:
            entry["temporal_reference"] = {
                "active": sum(1 for t in refs if (t.get("reference") or {}).get("valid")
                              and ((t["reference"].get("target") in {"date", "duration"})
                                   or t["reference"].get("scope") != "none" or t["reference"].get("order") != "none")),
                "applied": dict(collections.Counter(a for t in refs for a in t.get("applied") or [])),
                "anchor": dict(collections.Counter((t.get("anchor") or {}).get("status") for t in refs if t.get("anchor"))),
                "with_candidate_times": sum(1 for t in refs if t.get("candidate_times"))}
        out["variants"][name] = entry
    if args.baseline and args.fact_budget == 40:
        old = {name: records(conv_dir(args, args.baseline / name)) for name in names}
        out["baseline"] = {}
        for name in names:
            common = sorted(set(old[name]) & set(data[name]))
            if common:
                deltas = [data[name][q]["f1_locomo"] - old[name][q]["f1_locomo"] for q in common]
                out["baseline"][name] = {"n": len(common), "old_f1": f1([old[name][q] for q in common]),
                                         "new_f1": f1([data[name][q] for q in common]),
                                         "better": sum(d > 1e-9 for d in deltas),
                                         "worse": sum(d < -1e-9 for d in deltas)}
    return out


def render(summary):
    lines = [f"# Estudo de componentes - LoCoMo {summary['conversation']} - {summary.get('model', 'Qwen2.5-14B')} - {summary.get('fact_budget', 40)} fatos",
             "", "```", render_before(summary.get("before"), summary["conversation"]), "```", "",
             "F1 oficial (%). Delta pareado contra o completo, mesmas perguntas.", "",
             "| Configuracao | n | F1 | Delta | +/- | Single | Multi | Temporal | Open | chamadas/q | tokens/q |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in summary["variants"].items():
        p = v.get("paired_vs_full")
        delta = f"{p['delta']:+.2f}" if p else ""
        bw = f"{p['better']}/{p['worse']}" if p else ""
        cats = " | ".join(f"{v['by_category'][c]['f1']:.2f} ({v['by_category'][c]['n']})" for c in CATEGORIES)
        lines.append(f"| {v['label']} | {v['n']} | {v['f1']:.2f} | {delta} | {bw} | {cats} | "
                     f"{v['llm_calls_per_question']:.1f} | {v['llm_tokens_per_question']:.0f} |")
    if summary.get("baseline"):
        lines += ["", "Mesma variante, estudo anterior (v2) vs atual (v3):", "",
                  "| Configuracao | n | F1 v2 | F1 v3 | melhor/pior |", "|---|---|---|---|---|"]
        for name, b in summary["baseline"].items():
            lines.append(f"| {VARIANTS[name]} | {b['n']} | {b['old_f1']:.2f} | {b['new_f1']:.2f} | {b['better']}/{b['worse']} |")
    for name, v in summary["variants"].items():
        if v.get("reflection") or v.get("temporal_reference"):
            lines += ["", f"{v['label']}: reflexao={json.dumps(v.get('reflection'), ensure_ascii=False)}",
                      f"referencia_temporal={json.dumps(v.get('temporal_reference'), ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


def write(args, names):
    summary = summarize(args, names)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = render(summary)
    (args.output / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    args = parser().parse_args()
    args.output, args.cache = args.output.resolve(), args.cache.resolve()
    names = [n for n in args.variants.split(",") if n]
    if not names or any(n not in VARIANTS for n in names) or ("full" not in names and len(names) > 1):
        raise ValueError("Unknown variant list; paired deltas need 'full'")
    jobs = {name: command(args, name) for name in names}
    print("=" * 100 + "\n" + render_before(before_report(args), label(args)) + "\n" + "=" * 100, flush=True)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    from benchmarks.memoryagentbench.protocol import code_hash
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"code_hash": code_hash(ROOT), "conversation": args.conversation, "variants": names,
                "fact_budget": args.fact_budget, "requirement_threshold": args.requirement_threshold,
                "member_threshold": args.member_threshold, "max_members": args.max_members,
                "member_scan": args.member_scan, "no_requirements": args.no_requirements,
                "backend": args.backend, "deployment": args.deployment if args.backend == "azure" else "",
                "port": args.port, "cache": str(args.cache)}
    state_path = args.output / "suite.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get("identity") != identity:
        raise ValueError("Resume would mix code versions or configurations")
    state = state or {"identity": identity, "variants": {}}
    if args.backend == "vllm":
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=15) as response:
            if "Qwen/Qwen2.5-14B-Instruct" not in {row["id"] for row in json.load(response)["data"]}:
                raise ValueError("Unexpected vLLM model")
    # Legacy WRAG_* values must not silently change the condition; the Azure
    # credentials (AZURE_OPENAI_*) are kept.
    env = {k: v for k, v in os.environ.items() if not k.startswith("WRAG_")}
    env.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED="42",
               WRAG_EMBED_MAX_SEQ_LENGTH="0", WRAG_EMBED_BATCH_SIZE="8",
               WRAG_LLM_CACHE="1", WRAG_EMBED_CACHE="1", WRAG_CONTINUE_ON_CONTENT_FILTER="1")
    if args.embed_device.startswith("cuda"):
        env["WRAG_EMBED_STRICT_DEVICE"] = "1"
    if args.backend == "azure":
        env.update(WRAG_LLM_BACKEND="azure", WRAG_AZURE_DEPLOYMENT=args.deployment,
                   WRAG_AZURE_CONCURRENCY=str(args.concurrency))
    for name, cmd in jobs.items():
        if state["variants"].get(name, {}).get("status") == "complete":
            continue
        if (args.output / name / "pilot.json").exists():
            cmd = cmd + ["--resume"]
        state["variants"][name] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        state_path.write_text(json.dumps(state, indent=2))
        print(f"\n{render_before(before_report(args), label(args))}\nINICIANDO {VARIANTS[name]} ({label(args)})", flush=True)
        with (args.output / f"{name}.launcher.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    rows = records(args.output / name)
                    root = before_root(args)
                    old_rows = records(conv_dir(args, root)) if root else {}
                    same = [q for q in rows if q in old_rows]
                    before = (f" | antes (v2 completo, {args.fact_budget} fatos, mesmas {len(same)}): "
                              f"{f1([old_rows[q] for q in same]):.2f}") if same else ""
                    print(f"  {VARIANTS[name]}: {len(rows)} perguntas, F1 {f1(list(rows.values())):.2f}{before}", flush=True)
        if process.returncode != 0:
            state["variants"][name] = {"status": "failed", "exit_code": process.returncode}
            state_path.write_text(json.dumps(state, indent=2))
            raise RuntimeError(f"{name} failed; see {name}.launcher.log")
        state["variants"][name] = {"status": "complete", "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        state_path.write_text(json.dumps(state, indent=2))
        print(write(args, [n for n in names if (args.output / n).exists()]), flush=True)
        if name == "full" and not args.no_gate:
            check = gate(args)
            if check and check["new"] <= check["old"]:
                state["status"] = "gated"
                state["gate"] = check
                state_path.write_text(json.dumps(state, indent=2))
                print(f"TRAVA: o completo novo ({check['new']:.2f}) nao superou o antes ({check['old']:.2f}) "
                      f"nas mesmas {check['n']} perguntas. Variantes restantes nao foram rodadas.", flush=True)
                return
            if check:
                print(f"TRAVA LIBERADA: completo novo {check['new']:.2f} > antes {check['old']:.2f} "
                      f"(n={check['n']}). Seguindo com as variantes.", flush=True)
    state["status"] = "complete"
    state_path.write_text(json.dumps(state, indent=2))
    print(write(args, names), flush=True)


if __name__ == "__main__":
    main()
