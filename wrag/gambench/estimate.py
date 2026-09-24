"""
Tamanho do trabalho antes de gastar: amostras, páginas e chamadas por motor.

Não chama LLM nem embeddings. Os tokens de contexto vêm do tokenizador do
BGE-M3 (o das páginas). Para as tarefas do RULER, que têm 500 contextos de
128K cada, mede-se uma amostra espaçada e extrapola-se.

Os tokens do REGISTRAR são aproximados: janelas de 512 tokens com 64 de
sobreposição sobre cada página, uma chamada de NER e uma de OpenIE por janela,
com a sobrecarga medida dos próprios templates; o tamanho das respostas é uma
suposição declarada (NER_OUT, OPENIE_OUT). Serve para ordem de grandeza.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from wrag import prompts
from wrag.gambench import data as D
from wrag.gambench import protocol as P
from wrag.gambench.engines import IE_WINDOW_OVERLAP, IE_WINDOW_TOKENS
from wrag.gambench.pages import load_tokenizer

NER_OUT = 80          # tokens de resposta por janela (suposição)
OPENIE_OUT = 400      # tokens de resposta por janela (suposição)
PLAN_IN, PLAN_OUT = 3000, 350      # por chamada de planejamento (suposição)
CONFIRM_IN, CONFIRM_OUT = 2500, 80  # por verificação (suposição)
READER_OVERHEAD = 150


def _count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
    except Exception:  # noqa: BLE001 - só uma estimativa
        return max(1, len(text) // 4)


def _windows(tokens: int) -> int:
    if tokens <= IE_WINDOW_TOKENS:
        return 1
    step = IE_WINDOW_TOKENS - IE_WINDOW_OVERLAP
    return math.ceil((tokens - IE_WINDOW_TOKENS) / step) + 1


def _page_sizes(total_tokens: int) -> list[int]:
    full, rest = divmod(total_tokens, P.PAGE_TOKENS)
    return [P.PAGE_TOKENS] * full + ([rest] if rest else [])


def estimate(data_dir: Path, benchmarks: Iterable[str] = P.BENCHMARKS,
             ruler_per_task: int = 20) -> dict[str, Any]:
    tokenizer = load_tokenizer()
    ner_overhead = _count(prompts.NER_SYSTEM + prompts.NER_TEMPLATE.format(text=""))
    ie_overhead = _count(prompts.OPENIE_SYSTEM + prompts.OPENIE_TEMPLATE.format(
        text="", entities="[]", max_triples=40)) + 60
    out: dict[str, Any] = {"suposicoes": {"ner_out": NER_OUT, "openie_out": OPENIE_OUT,
                                          "plano": [PLAN_IN, PLAN_OUT],
                                          "verificacao": [CONFIRM_IN, CONFIRM_OUT],
                                          "sobrecarga_ner": ner_overhead,
                                          "sobrecarga_openie": ie_overhead},
                           "recortes": {}}
    for benchmark in benchmarks:
        for split in D.splits_of(benchmark):
            samples = D.load(data_dir, benchmark, split)
            n = len(samples)
            contexts: dict[str, str] = {}
            for sample in samples:
                contexts.setdefault(sample.context_key, sample.context)
            keys = list(contexts)
            if benchmark == "ruler" and ruler_per_task and len(keys) > ruler_per_task:
                step = len(keys) / ruler_per_task
                measured = [keys[int(i * step)] for i in range(ruler_per_task)]
            else:
                measured = keys
            tokens = {k: len(tokenizer.encode(contexts[k], add_special_tokens=False))
                      for k in measured}
            scale = len(keys) / max(1, len(measured))
            pages = sum(len(_page_sizes(t)) for t in tokens.values()) * scale
            windows = sum(_windows(size) for t in tokens.values() for size in _page_sizes(t)) * scale
            window_tokens = sum(min(size, IE_WINDOW_TOKENS) * _windows(size) for t in tokens.values()
                                for size in _page_sizes(t)) * scale
            reader_in = n * (P.TOP_K * P.PAGE_TOKENS + READER_OVERHEAD)
            ie_in = window_tokens * 2 + windows * (ner_overhead + ie_overhead)
            ie_out = windows * (NER_OUT + OPENIE_OUT)
            out["recortes"][f"{benchmark}/{split}"] = {
                "amostras": n, "contextos": len(keys), "contextos_medidos": len(measured),
                "tokens_contexto_media": round(mean(tokens.values())) if tokens else 0,
                "paginas": round(pages),
                "leitor": {"chamadas": n, "tokens_entrada": round(reader_in),
                           "tokens_saida_max": n * P.READER_MAX_TOKENS},
                "witnessrag_extra": {
                    "janelas": round(windows), "chamadas_registrar": round(2 * windows),
                    "tokens_entrada_registrar": round(ie_in), "tokens_saida_registrar": round(ie_out),
                    "chamadas_consulta_max": 4 * n,
                    "tokens_entrada_consulta_max": n * 2 * (PLAN_IN + CONFIRM_IN),
                    "tokens_saida_consulta_max": n * 2 * (PLAN_OUT + CONFIRM_OUT)},
            }
            print(f"{benchmark}/{split}: {n} amostras, {round(pages)} páginas", flush=True)
    return out


def render(report: dict[str, Any], price_in: float = 0.0, price_out: float = 0.0) -> str:
    lines = ["| recorte | amostras | tokens/contexto | páginas | leitor: chamadas | leitor: M tokens in | "
             "witnessrag: chamadas extra | witnessrag: M tokens in | witnessrag: M tokens out |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    totals = {"reader_in": 0.0, "reader_out": 0.0, "w_in": 0.0, "w_out": 0.0, "pages": 0.0,
              "calls": 0.0, "w_calls": 0.0}
    for name, r in report["recortes"].items():
        w = r["witnessrag_extra"]
        w_in = w["tokens_entrada_registrar"] + w["tokens_entrada_consulta_max"]
        w_out = w["tokens_saida_registrar"] + w["tokens_saida_consulta_max"]
        w_calls = w["chamadas_registrar"] + w["chamadas_consulta_max"]
        lines.append(f"| {name} | {r['amostras']} | {r['tokens_contexto_media']} | {r['paginas']} | "
                     f"{r['leitor']['chamadas']} | {r['leitor']['tokens_entrada'] / 1e6:.1f} | "
                     f"{w_calls} | {w_in / 1e6:.1f} | {w_out / 1e6:.1f} |")
        totals["reader_in"] += r["leitor"]["tokens_entrada"]
        totals["reader_out"] += r["leitor"]["tokens_saida_max"]
        totals["w_in"] += w_in
        totals["w_out"] += w_out
        totals["pages"] += r["paginas"]
        totals["calls"] += r["leitor"]["chamadas"]
        totals["w_calls"] += w_calls
    lines.append(f"| **total** | | | {round(totals['pages'])} | {round(totals['calls'])} | "
                 f"{totals['reader_in'] / 1e6:.1f} | {round(totals['w_calls'])} | "
                 f"{totals['w_in'] / 1e6:.1f} | {totals['w_out'] / 1e6:.1f} |")
    if price_in or price_out:
        reader_cost = (totals["reader_in"] * price_in + totals["reader_out"] * price_out) / 1e6
        witness_cost = (totals["w_in"] * price_in + totals["w_out"] * price_out) / 1e6
        lines += ["", f"Custo com US$ {price_in}/M entrada e US$ {price_out}/M saída: "
                      f"leitor (qualquer motor) ≈ US$ {reader_cost:,.0f}; "
                      f"extra do witnessrag ≈ US$ {witness_cost:,.0f} (sem cache)."]
    lines += ["", "O leitor é o mesmo para todos os motores (limite superior: 5 páginas cheias). "
              "O extra do witnessrag é o REGISTRAR (NER + OpenIE por janela) mais, no máximo, "
              "dois planos e duas verificações por pergunta; respostas do REGISTRAR são "
              "suposições (ver `suposicoes`)."]
    return "\n".join(lines) + "\n"
