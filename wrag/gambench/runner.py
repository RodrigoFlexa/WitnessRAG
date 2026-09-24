"""
Execução de um recorte (HotpotQA 56k, uma tarefa do RULER, o NarrativeQA).

Para cada amostra, na ordem do GAM: páginas do protocolo -> o motor escolhe
até 5 -> o leitor do protocolo responde -> métrica do GAM. Cada amostra vira
uma linha de `predictions.jsonl` assim que termina; `--resume` pula as já
respondidas (as que terminaram em erro são tentadas de novo). A manifest
guarda tudo o que define a rodada e é conferida na retomada.

Semântica de falha, como no GAM:
* HotpotQA e NarrativeQA: amostra sem resposta (erro ou bloqueio do filtro de
  conteúdo) sai da média do F1 (o GAM só soma quem tem a chave "f1");
* RULER: erro e bloqueio contam como incorretos.
O relatório também mostra a média estrita (toda falha vale 0).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wrag import config as C
from wrag.gambench import data as D
from wrag.gambench import metrics as M
from wrag.gambench import protocol as P
from wrag.gambench import reader as R
from wrag.gambench.engines import ENGINES, IE_WINDOW_TOKENIZER, Engine, engine_config
from wrag.gambench.pages import Pages, load_tokenizer, split_pages, tokenizer_fingerprint
from wrag.gambench.report import split_report, write_split_report
from wrag.util import append_jsonl, get_logger, read_json, read_jsonl, write_json

log = get_logger("wrag.gambench.runner")

READER_STAGE = "gam.answer"
DONE_STATUSES = {"ok", "filtered"}
# Campos da manifest que precisam coincidir para retomar uma rodada.
IDENTITY_FIELDS = ("benchmark", "split", "engine", "protocolo", "motor_config", "llm",
                   "embeddings", "paginas", "dados")


@dataclass
class RunArgs:
    data_dir: Path
    benchmark: str
    split: str
    engine: str
    output: Path
    start_idx: int = 0
    end_idx: int | None = None
    batch: int = 8
    reader_seed: int | None = P.READER_SEED
    resume: bool = False
    retry_errors: bool = True
    max_consecutive_errors: int = 5
    page_tokenizer: str = P.PAGE_TOKENIZER
    page_tokenizer_revision: str = ""
    ie_tokenizer: str = ""
    ie_tokenizer_revision: str = ""


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _llm_identity(llm) -> dict[str, Any]:
    identity = {"backend": llm.name, "deployment": getattr(llm, "deployment", "")}
    if hasattr(llm, "cache_identity"):
        identity["cache_identity"] = llm.cache_identity()[:16]
    return identity


def _embedder_identity(embedder) -> dict[str, Any]:
    """O que define os vetores (o TF-IDF de teste muda de chave a cada corpus)."""
    return {"backend": embedder.name,
            "modelo": getattr(embedder, "model_name", "") or getattr(embedder, "deployment", ""),
            "max_seq_length": getattr(embedder, "max_seq_length", 0) or 0}


def _prompt_digest() -> str:
    text = "\x00".join((R.HOTPOT_TEMPLATE, R.NARRATIVEQA_TEMPLATE, R.RULER_TEMPLATE))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _data_fingerprint(data_dir: Path, benchmark: str, split: str) -> dict[str, Any]:
    manifest = read_json(data_dir / "gam_manifest.json", {}) or {}
    if benchmark == "hotpotqa":
        name = P.HOTPOT_SPLITS[split]
        return {"arquivo": name, "sha256": P.HOTPOT_FILES[name].sha256}
    if benchmark == "ruler":
        return {"arquivo": f"{split}.jsonl", "fonte_sha256": P.RULER_FILES[split].sha256}
    info = manifest.get("narrativeqa", {})
    return {"revisao": P.NARRATIVEQA_REVISION, "semente": P.NARRATIVEQA_SEED,
            "primeiros_indices": info.get("primeiros_indices")}


def build_manifest(args: RunArgs, llm, embedder, cfg, tokenizer, n_selected: int) -> dict[str, Any]:
    return {
        "benchmark": args.benchmark, "split": args.split, "engine": args.engine,
        "protocolo": {"artigo": P.GAM_PAPER, "gam_commit": P.GAM_COMMIT,
                      "paginas_tokens": P.PAGE_TOKENS, "top_k": P.TOP_K,
                      "leitor": {"temperatura": P.READER_TEMPERATURE,
                                 "max_tokens": P.READER_MAX_TOKENS, "seed": args.reader_seed,
                                 "system_prompt": None, "prompts_sha256": _prompt_digest()}},
        "motor_config": cfg.to_dict(),
        "llm": _llm_identity(llm),
        "embeddings": _embedder_identity(embedder),
        "paginas": {"tokenizador": args.page_tokenizer,
                    "revisao": args.page_tokenizer_revision,
                    **tokenizer_fingerprint(tokenizer)},
        "dados": _data_fingerprint(args.data_dir, args.benchmark, args.split),
        "selecao": {"start_idx": args.start_idx, "end_idx": args.end_idx,
                    "amostras": n_selected},
        "git_head": _git_head(),
    }


def _check_resume(path: Path, manifest: dict[str, Any]) -> None:
    previous = read_json(path)
    if not previous:
        return
    # Compara na forma gravada (tuplas viram listas no JSON).
    current = json.loads(json.dumps(manifest, ensure_ascii=False, default=str))
    diffs = [k for k in IDENTITY_FIELDS if previous.get(k) != current.get(k)]
    if diffs:
        raise SystemExit(
            f"--resume recusado: {path} foi gravado com outra configuração ({', '.join(diffs)}). "
            "Use outro diretório de saída ou apague o anterior.")


def _score(benchmark: str, split: str, response: str, answers: list[str],
           status: str) -> dict[str, Any]:
    if benchmark == "ruler":
        correct = status == "ok" and M.ruler_correct(response, answers)
        return {"correct": bool(correct), "accuracy": 1.0 if correct else 0.0,
                "ruler_oficial": M.ruler_official(split, response, answers) if status == "ok" else 0.0}
    if status != "ok":
        return {}
    return {"f1": M.gam_f1(response, answers), "em": M.exact_match(response, answers)}


def _grouped(samples: list[D.Sample]) -> list[D.Sample]:
    """Mesma lista, com as perguntas de um mesmo contexto em sequência.

    Só muda a ordem de processamento (o que permite reaproveitar a memória de
    um livro); cada pergunta é independente e o relatório ignora a ordem.
    """
    first: dict[str, int] = {}
    for i, sample in enumerate(samples):
        first.setdefault(sample.context_key, i)
    order = sorted(range(len(samples)), key=lambda i: (first[samples[i].context_key], i))
    return [samples[i] for i in order]


def run_split(args: RunArgs, llm=None, embedder=None, tokenizer=None) -> dict[str, Any]:
    from wrag.embed import get_embedder
    from wrag.eval.runner import _trim
    from wrag.llm import get_llm

    if args.engine not in ENGINES:
        raise ValueError(f"motor desconhecido: {args.engine}")
    llm = llm or get_llm()
    embedder = embedder or get_embedder()
    tokenizer = tokenizer or load_tokenizer(args.page_tokenizer, args.page_tokenizer_revision)
    cfg = engine_config(args.engine, P.TOP_K, args.ie_tokenizer or default_ie_tokenizer(),
                        args.ie_tokenizer_revision)

    samples = D.load(args.data_dir, args.benchmark, args.split)
    selected = list(D.iter_selected(samples, args.start_idx, args.end_idx))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args, llm, embedder, cfg, tokenizer, len(selected))
    manifest_path = output / "manifest.json"
    predictions = output / "predictions.jsonl"
    if args.resume:
        _check_resume(manifest_path, manifest)
    elif predictions.exists() and predictions.stat().st_size:
        raise SystemExit(f"{predictions} já existe; use --resume ou outro diretório.")
    write_json(manifest_path, manifest)

    previous = read_jsonl(predictions) if args.resume else []
    latest = {row["sid"]: row for row in previous}
    done = {sid for sid, row in latest.items()
            if row.get("status") in DONE_STATUSES or not args.retry_errors}
    pending = [s for s in _grouped(selected) if s.sid not in done]
    expected = len(selected)
    print(f"[{args.engine}] {args.benchmark}/{args.split}: {len(selected)} amostras, "
          f"{len(selected) - len(pending)} já feitas, {len(pending)} a fazer", flush=True)

    engine = Engine(args.engine, llm, embedder, cfg)
    params = R.reader_params(args.reader_seed)
    pages_cache: tuple[str, Pages] | None = None
    consecutive_errors = 0
    started_run = time.perf_counter()

    for offset in range(0, len(pending), max(1, args.batch)):
        batch = pending[offset: offset + max(1, args.batch)]
        prepared: list[dict[str, Any]] = []
        for sample in batch:
            row: dict[str, Any] = {
                "sid": sample.sid, "benchmark": sample.benchmark, "split": sample.split,
                "index": sample.index, "engine": args.engine, "question": sample.question,
                "answers": sample.answers, "status": "error", **sample.meta}
            try:
                t0 = time.perf_counter()
                if pages_cache is None or pages_cache[0] != sample.context_key:
                    pages_cache = (sample.context_key, split_pages(sample.context, tokenizer))
                pages = pages_cache[1]
                pages_s = time.perf_counter() - t0
                if not pages.pages:
                    raise ValueError("contexto vazio")
                selection = engine.select(sample.benchmark, sample.sid, sample.question,
                                          sample.context_key, pages.pages)
                context = R.join_context([pages.pages[i] for i in selection.pages])
                prompt = R.build_prompt(sample.benchmark, sample.question, context, sample.example)
                row.update({"n_paginas": len(pages.pages), "tokens_contexto": pages.context_tokens,
                            "paginas_escolhidas": selection.pages,
                            "motor_filtrado": selection.filtered,
                            "memoria_reaproveitada": selection.reused_index,
                            "tempo": {"paginas_s": round(pages_s, 3),
                                      "indice_s": round(selection.index_s, 3),
                                      "busca_s": round(selection.retrieve_s, 3)},
                            "uso_motor": selection.usage.get("total", {}),
                            "diagnostico": _trim(selection.diagnostics)})
                prepared.append({"row": row, "prompt": prompt})
            except Exception as exc:  # noqa: BLE001 - registrado e contado
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=6)
                row.update(_score(sample.benchmark, sample.split, "", sample.answers, "error"))
                append_jsonl(predictions, row)
                consecutive_errors += 1
                log.error("%s: %s", sample.sid, row["error"])
                if consecutive_errors >= args.max_consecutive_errors:
                    raise SystemExit(f"{consecutive_errors} erros seguidos; último: {row['error']}")

        if prepared:
            prompts = [item["prompt"] for item in prepared]
            try:
                results = llm.chat_many(prompts, params=params, stage=READER_STAGE,
                                        desc=f"leitor {args.split}")
            except Exception:  # noqa: BLE001 - isola a amostra que falhou
                results = []
                for prompt in prompts:
                    try:
                        results.append(llm.chat(prompt, params=params, stage=READER_STAGE))
                    except Exception as exc:  # noqa: BLE001
                        results.append(exc)
            for item, result in zip(prepared, results):
                row = item["row"]
                sample_answers = row["answers"]
                if isinstance(result, Exception):
                    row["error"] = f"{type(result).__name__}: {result}"
                    row["status"] = "error"
                    consecutive_errors += 1
                elif result.filtered:
                    row["status"] = "filtered"
                    row["response"] = ""
                    consecutive_errors = 0
                else:
                    row["status"] = "ok"
                    row["response"] = R.clean_response(result.text)
                    consecutive_errors = 0
                if not isinstance(result, Exception):
                    row["leitor"] = R.reader_record(result)
                row.update(_score(row["benchmark"], row["split"], row.get("response", ""),
                                  sample_answers, row["status"]))
                append_jsonl(predictions, row)
            if consecutive_errors >= args.max_consecutive_errors:
                raise SystemExit(f"{consecutive_errors} erros seguidos no leitor; "
                                 "verifique o backend e retome com --resume.")
        rows = read_jsonl(predictions)
        write_split_report(output, split_report(rows, args.benchmark, args.split, expected,
                                                manifest))
        done_now = len({r["sid"] for r in rows if r.get("status") in DONE_STATUSES})
        elapsed = time.perf_counter() - started_run
        print(f"  {done_now}/{expected} respondidas ({elapsed:.0f}s)", flush=True)

    report = split_report(read_jsonl(predictions), args.benchmark, args.split, expected, manifest)
    write_split_report(output, report)
    return report


def output_dir(root: Path, engine: str, benchmark: str, split: str) -> Path:
    return root / engine / benchmark / split


def default_ie_tokenizer() -> str:
    return os.environ.get("WRAG_TOKENIZER_MODEL", "") or IE_WINDOW_TOKENIZER
