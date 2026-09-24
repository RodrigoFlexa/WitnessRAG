"""
python -m wrag.gambench {prepare,estimate,run,report}

  prepare   baixa e prepara HotpotQA (56K/224K/448K), RULER 128K e NarrativeQA
            exatamente como o código de avaliação do GAM
  estimate  amostras, páginas, chamadas e tokens por motor, sem chamar nada
  run       executa recortes com um motor; retomável
  report    tabela no formato da Tabela 1(b) do GAM

O backend de LLM e o de embeddings vêm do ambiente (WRAG_LLM_BACKEND,
WRAG_EMBED_BACKEND, WRAG_EMBED_MODEL...), como nos outros pontos de entrada.
Ver scripts/run-gam-bench.sh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wrag.gambench import protocol as P
from wrag.gambench.engines import ENGINES


def _benchmarks(value: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items or items == ["all"]:
        return list(P.BENCHMARKS)
    bad = [v for v in items if v not in P.BENCHMARKS]
    if bad:
        raise argparse.ArgumentTypeError(f"benchmark desconhecido: {bad}")
    return items


def _seed(value: str) -> int | None:
    return None if value.lower() in {"none", "null", ""} else int(value)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m wrag.gambench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="baixa e prepara os dados")
    prep.add_argument("--data-dir", type=Path, default=Path("data/gam"))
    prep.add_argument("--benchmarks", type=_benchmarks, default=list(P.BENCHMARKS))
    prep.add_argument("--source-dir", type=Path, default=None,
                      help="copiar os arquivos daqui em vez de baixar (mesmos nomes)")
    prep.add_argument("--drop-raw", action="store_true",
                      help="apagar os parquet brutos depois de convertidos")

    est = sub.add_parser("estimate", help="tamanho do trabalho, sem chamar LLM")
    est.add_argument("--data-dir", type=Path, default=Path("data/gam"))
    est.add_argument("--benchmarks", type=_benchmarks, default=list(P.BENCHMARKS))
    est.add_argument("--ruler-per-task", type=int, default=20)
    est.add_argument("--price-in", type=float, default=0.0, help="US$ por 1M tokens de entrada")
    est.add_argument("--price-out", type=float, default=0.0, help="US$ por 1M tokens de saída")
    est.add_argument("--output", type=Path, default=None)

    run = sub.add_parser("run", help="executa recortes com um motor")
    run.add_argument("--data-dir", type=Path, default=Path("data/gam"))
    run.add_argument("--benchmark", choices=P.BENCHMARKS, required=True)
    run.add_argument("--splits", default="all",
                     help="all, ou lista: 56k,224k,448k | tarefas do RULER | test")
    run.add_argument("--engine", choices=ENGINES, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--start-idx", type=int, default=0)
    run.add_argument("--end-idx", type=int, default=None,
                     help="exclusivo; como no GAM. Omitido = todas as amostras (protocolo)")
    run.add_argument("--batch", type=int, default=8, help="respostas pedidas em paralelo")
    run.add_argument("--reader-seed", type=_seed, default=P.READER_SEED,
                     help="42 (padrão) ou none, como o GAM, que não envia seed")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-retry-errors", action="store_true")
    run.add_argument("--max-consecutive-errors", type=int, default=5)
    run.add_argument("--page-tokenizer-revision", default="")
    run.add_argument("--ie-tokenizer", default="",
                     help="tokenizador das janelas do REGISTRAR (padrão: WRAG_TOKENIZER_MODEL "
                          "ou Qwen/Qwen2.5-14B-Instruct)")
    run.add_argument("--ie-tokenizer-revision", default="")
    run.add_argument("--model-label", default="")

    rep = sub.add_parser("report", help="tabela no formato da Tabela 1(b)")
    rep.add_argument("--output-root", type=Path, required=True)
    rep.add_argument("--engines", default=",".join(ENGINES))
    rep.add_argument("--model-label", default="")
    return p


def _splits(benchmark: str, value: str) -> list[str]:
    from wrag.gambench.data import splits_of

    available = list(splits_of(benchmark))
    if value in {"", "all"}:
        return available
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in available]
    if bad:
        raise SystemExit(f"recortes desconhecidos para {benchmark}: {bad}; opções: {available}")
    return items


def main(argv: list[str] | None = None) -> int:
    from wrag.util import setup_logging

    setup_logging()
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from wrag.gambench.data import prepare

        manifest = prepare(args.data_dir, args.benchmarks, args.source_dir,
                           keep_raw=not args.drop_raw)
        print(json.dumps({k: v for k, v in manifest.items() if k in P.BENCHMARKS},
                         ensure_ascii=False, indent=1))
        return 0
    if args.command == "estimate":
        from wrag.gambench.estimate import estimate, render
        from wrag.util import write_json

        report = estimate(args.data_dir, args.benchmarks, args.ruler_per_task)
        text = render(report, args.price_in, args.price_out)
        print(text)
        if args.output:
            write_json(args.output.with_suffix(".json"), report)
            args.output.with_suffix(".md").write_text(text, encoding="utf-8")
        return 0
    if args.command == "run":
        from wrag.gambench.report import write_suite
        from wrag.gambench.runner import RunArgs, output_dir, run_split

        if args.batch < 1:
            raise SystemExit("--batch deve ser positivo")
        for split in _splits(args.benchmark, args.splits):
            run_split(RunArgs(
                data_dir=args.data_dir, benchmark=args.benchmark, split=split,
                engine=args.engine,
                output=output_dir(args.output_root, args.engine, args.benchmark, split),
                start_idx=args.start_idx, end_idx=args.end_idx, batch=args.batch,
                reader_seed=args.reader_seed, resume=args.resume,
                retry_errors=not args.no_retry_errors,
                max_consecutive_errors=args.max_consecutive_errors,
                page_tokenizer_revision=args.page_tokenizer_revision,
                ie_tokenizer=args.ie_tokenizer, ie_tokenizer_revision=args.ie_tokenizer_revision))
            write_suite(args.output_root, ENGINES, args.model_label)
        return 0
    if args.command == "report":
        from wrag.gambench.report import render_suite, write_suite

        engines = [e.strip() for e in args.engines.split(",") if e.strip()]
        report = write_suite(args.output_root, engines, args.model_label)
        print(render_suite(report))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
