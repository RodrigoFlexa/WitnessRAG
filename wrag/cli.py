"""
Linha de comando.

    python -m wrag.cli selftest                  # verificação offline, sem rede
    python -m wrag.cli prepare-data              # baixa os subconjuntos oficiais
    python -m wrag.cli diag-azure                # testa o gateway antes de gastar
    python -m wrag.cli run --datasets musique --methods dense,witnessrag -n 20
    python -m wrag.cli report runs/<id>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from wrag import config as C
from wrag.util import get_logger, setup_logging

log = get_logger("wrag.cli")

HIPPORAG_REPO = "https://github.com/OSU-NLP-Group/HippoRAG.git"
DATASET_FILES = [
    "musique.json", "musique_corpus.json",
    "2wikimultihopqa.json", "2wikimultihopqa_corpus.json",
    "hotpotqa.json", "hotpotqa_corpus.json",
    "sample.json", "sample_corpus.json",
]


# ---------------------------------------------------------------------------

def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Copia os subconjuntos de 1000 perguntas do repositório oficial do HippoRAG.

    São os MESMOS subconjuntos usados no HippoRAG e no HippoRAG 2, o que é a
    razão de não gerar amostra própria: qualquer amostragem nova tornaria os
    números incomparáveis com os publicados sem acrescentar nada.
    """
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in DATASET_FILES if not (C.DATA_DIR / f).exists()]
    if not missing:
        print(f"todos os datasets já estão em {C.DATA_DIR}")
        return 0

    target = C.CACHE_DIR / "hipporag_repo"
    if not target.exists():
        print(f"clonando {HIPPORAG_REPO} ...")
        subprocess.run(["git", "clone", "--depth", "1", HIPPORAG_REPO, str(target)], check=True)

    source = target / "reproduce" / "dataset"
    copied = 0
    for name in DATASET_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, C.DATA_DIR / name)
            copied += 1
        else:
            print(f"  ausente no repo: {name} (baixe de huggingface.co/datasets/osunlp/HippoRAG_v2)")
    print(f"{copied} arquivo(s) copiado(s) para {C.DATA_DIR}")
    return 0


def cmd_diag_azure(args: argparse.Namespace) -> int:
    """Uma chamada de ida e volta antes de gastar um lote inteiro.

    O 404 do Azure diz 'Resource Not Found' para três causas diferentes
    (deployment inexistente, endpoint errado, api-version antiga), e descobrir
    qual delas é no meio de uma indexação de 2000 passagens custa caro.
    """
    from wrag.llm import GenParams, get_llm

    try:
        llm = get_llm("azure")
    except Exception as exc:  # noqa: BLE001
        print(f"FALHA ao construir o cliente: {exc}")
        return 1

    print(f"deployment: {llm.deployment}")
    print(f"modo: {'reasoning' if llm.reasoning else 'chat'}")
    print(f"api-version: {C.AZURE_API_VERSION}")
    try:
        result = llm.chat('Responda exatamente com {"ok": true}.',
                          params=GenParams(max_tokens=64, json_mode=True), stage="diag")
    except Exception as exc:  # noqa: BLE001
        print(f"FALHA na chamada: {type(exc).__name__}: {exc}")
        return 1

    print(f"finish_reason: {result.finish_reason!r} | filtrado: {result.filtered}")
    print(f"texto: {result.text[:200]!r}")
    print(f"tokens: prompt={result.prompt_tokens} resposta={result.completion_tokens}")
    if not result.text and not result.filtered:
        print("\nResposta vazia sem bloqueio. Se o modo for reasoning, suba "
              "WRAG_AZURE_REASONING_MIN_TOKENS (tente 12000).")
        return 1

    if C.AZURE_EMBED_DEPLOYMENT:
        from wrag.embed import AzureEmbedder

        try:
            vectors = AzureEmbedder().encode(["teste de embedding"])
            print(f"embeddings: OK, dimensão {vectors.shape[1]}")
        except Exception as exc:  # noqa: BLE001
            print(f"embeddings: FALHA ({exc}). O código cairá para sentence-transformers ou TF-IDF.")
    else:
        print("embeddings: WRAG_AZURE_EMBED_DEPLOYMENT não definido; será usado local/TF-IDF.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from wrag.eval.runner import run

    if args.max_query_plans < 1:
        raise ValueError("--max-query-plans deve ser >= 1")
    if args.soft_obligations and not args.active_obligations:
        raise ValueError("--soft-obligations exige --active-obligations")
    cfg = C.RunConfig()
    cfg.n_questions = args.n
    cfg.top_k = args.top_k
    cfg.seed = args.seed
    cfg.subset_corpus = args.subset_corpus
    cfg.train_questions_path = args.train_questions
    cfg.witness.grounding_mode = args.grounding
    cfg.witness.binding_aware_grounding = args.binding_aware_grounding
    cfg.witness.verify_witnesses = args.verify_witnesses
    cfg.witness.dense_fallback = not args.no_dense_fallback
    cfg.witness.answer_set = args.answer_set
    cfg.qa.answer_set = args.answer_set
    cfg.witness.vocabulary_aware_compile = args.vocab_compile
    cfg.witness.query_plans = args.query_plans
    cfg.witness.max_query_plans = args.max_query_plans
    cfg.witness.active_frontier = args.active_frontier
    cfg.witness.active_obligations = args.active_obligations
    cfg.witness.active_context = args.active_context
    cfg.witness.active_operators = args.active_operators
    cfg.witness.soft_obligations = args.soft_obligations
    cfg.witness.proof_reader = args.proof_reader
    cfg.qa.operator_reader = args.active_operators
    cfg.qa.proof_reader = args.proof_reader
    cfg.witness.hybrid_fallback = args.hybrid_fallback
    cfg.ie.dialogue_mode = args.dialogue_ie
    cfg.graph.merge_relation_inflections = not args.no_relation_family_merge
    if args.budget is not None:
        cfg.witness.budget_fraction = args.budget
    if args.no_acquisition:
        cfg.witness.enable_acquisition = False
    if args.beam is not None:
        cfg.witness.beam_width = args.beam
    if args.exhaustive:
        if args.grounding != "exact":
            raise ValueError("--exhaustive requer --grounding exact")
        cfg.witness.beam_width = cfg.witness.candidates_per_atom = cfg.witness.max_witnesses = 0

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    root = run(datasets, methods, cfg, tag=args.tag, resume=not args.fresh, resume_dir=args.resume_dir)
    print(f"\nrodada concluída: {root}")
    print(f"relatório: {root / 'report.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from wrag.eval.report import build_report

    path = build_report(Path(args.run_dir))
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Roda o pipeline inteiro offline (backend stub, embeddings TF-IDF).

    Exercita indexação, recuperação e relatório sem APIs. Não valida qualidade
    científica nem garante funcionamento de provedores externos.
    """
    import os

    os.environ["WRAG_LLM_BACKEND"] = "stub"
    os.environ["WRAG_EMBED_BACKEND"] = "tfidf"
    C.LLM_BACKEND = "stub"
    C.EMBED_BACKEND = "tfidf"

    from wrag.eval.runner import run
    from wrag.witness.budget import (select_greedy, select_ilp, submodularity_counterexample,
                                     toy_instance)

    print("== contraexemplo de submodularidade ==")
    print(json.dumps(submodularity_counterexample(), ensure_ascii=False, indent=2))

    print("\n== instância de brinquedo da proposta (5 fatos, orçamento 3) ==")
    demands, costs = toy_instance()
    result = select_ilp(demands, costs, budget=3.0)
    print(f"ILP:    fatos={sorted(result.kept)} valor={result.value} status={result.status}")
    greedy = select_greedy(demands, costs, budget=3.0)
    print(f"guloso: fatos={sorted(greedy.kept)} valor={greedy.value}")
    if sorted(result.kept) != [1, 2, 3] or result.value != 8.0:
        print("  AVISO: a solução esperada era {e1,e2,e3} com valor 8")

    print(f"\n== pipeline em {args.n} perguntas de {args.dataset} ==")
    cfg = C.RunConfig()
    cfg.n_questions = args.n
    cfg.top_k = 5
    root = run([args.dataset], args.methods.split(","), cfg, tag="selftest", resume=False)
    print((root / "report.md").read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="wrag", description="Benchmark WITNESS-RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="copia os subconjuntos oficiais do HippoRAG"
                   ).set_defaults(func=cmd_prepare_data)
    sub.add_parser("diag-azure", help="testa o gateway Azure").set_defaults(func=cmd_diag_azure)

    p_run = sub.add_parser("run", help="roda o benchmark")
    p_run.add_argument("--datasets", default=",".join(C.DATASETS))
    p_run.add_argument("--methods", default=",".join(C.METHODS))
    p_run.add_argument("-n", type=int, default=C.N_QUESTIONS, help="perguntas por dataset")
    p_run.add_argument("--top-k", type=int, default=C.TOP_K)
    p_run.add_argument("--seed", type=int, default=C.SEED)
    p_run.add_argument("--budget", type=float, default=None,
                       help="fração da memória a preservar (ativa a seleção por ILP)")
    p_run.add_argument("--beam", type=int, default=None, help="largura do feixe da junção")
    p_run.add_argument("--binding-aware-grounding", action="store_true",
                       help="expande o próximo salto a partir das variáveis ligadas")
    p_run.add_argument("--answer-set", action="store_true",
                       help="responde o conjunto de atribuições certas, com prova por item, "
                            "habilita aggregation=count e usa o leitor ciente de conjunto")
    p_run.add_argument("--vocab-compile", action="store_true",
                       help="compila a pergunta com as relações e entidades do grafo no prompt")
    p_run.add_argument("--query-plans", action="store_true",
                       help="gera e revisa planos usando o retorno da busca no grafo")
    p_run.add_argument("--max-query-plans", type=int, default=5,
                       help="orçamento global de planos distintos por pergunta (padrão: 5)")
    p_run.add_argument("--active-frontier", action="store_true")
    p_run.add_argument("--active-obligations", action="store_true")
    p_run.add_argument("--active-context", action="store_true")
    p_run.add_argument("--active-operators", action="store_true")
    p_run.add_argument("--soft-obligations", action="store_true")
    p_run.add_argument("--proof-reader", action="store_true")
    p_run.add_argument("--hybrid-fallback", action="store_true",
                       help="fallback do WITNESS-RAG por fusão recíproca de postos (denso + BM25)")
    p_run.add_argument("--dialogue-ie", action="store_true",
                       help="extração adaptada a diálogo: falante como sujeito e tempo do fato")
    p_run.add_argument("--no-relation-family-merge", action="store_true",
                       help="ablação: preserva flexões como paint/painted separadamente")
    p_run.add_argument("--verify-witnesses", action="store_true",
                       help="verifica testemunhas nos textos antes de promover passagens")
    p_run.add_argument("--no-acquisition", action="store_true",
                       help="desliga a aquisição adaptativa (ablação)")
    p_run.add_argument("--grounding", choices=["exact", "semantic"],
                       default=C.WitnessConfig().grounding_mode,
                       help="aterramento dos fatos (padrão: semantic)")
    p_run.add_argument("--exhaustive", action="store_true", help="sem cortes de candidatos, feixe ou testemunhas")
    p_run.add_argument("--no-dense-fallback", action="store_true", help="avalia apenas evidência estrutural")
    p_run.add_argument("--subset-corpus", action="store_true", help="piloto com universo de busca reduzido")
    p_run.add_argument("--train-questions", default="", help="JSON de treino externo; aceita {dataset} no caminho")
    p_run.add_argument("--resume-dir", default=None, help="retoma uma rodada com a mesma configuração e código")
    p_run.add_argument("--tag", default="")
    p_run.add_argument("--fresh", action="store_true", help="ignora resultados anteriores")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="regera o relatório de uma rodada")
    p_report.add_argument("run_dir")
    p_report.set_defaults(func=cmd_report)

    p_self = sub.add_parser("selftest", help="pipeline completo offline")
    p_self.add_argument("-n", type=int, default=10)
    p_self.add_argument("--dataset", default="2wikimultihopqa")
    p_self.add_argument("--methods", default="dense,graphrag,hipporag,hipporag2,witnessrag")
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
