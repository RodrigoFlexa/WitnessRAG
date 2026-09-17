"""
Orquestração do benchmark com registros por pergunta e retomada verificável.

Extração compartilhada, indexação própria e seleção de memória têm custos
separados. Cada retomada registra a reindexação sem substituir o custo inicial.
A retomada exige o mesmo código, configuração, corpus e perguntas; resultados
completos são reutilizados. Custos parciais de uma etapa interrompida antes do
checkpoint podem não estar registrados.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from wrag import config as C
from wrag.data import Corpus, Question, load_dataset, _PassageIndex, _parse_question
from wrag.embed import get_embedder
from wrag.eval import metrics as M
from wrag.eval.reader import read
from wrag.llm import get_llm
from wrag.llm.base import usage_delta, sum_usage
from wrag.llm.filters import LEDGER, configure_ledger
from wrag.methods import build_context, build_methods
from wrag.util import append_jsonl, get_logger, progress, read_json, read_jsonl, write_json, canonical_symbol, sha

log = get_logger("wrag.eval.runner")


@dataclass
class RunPaths:
    root: Path

    def dataset_dir(self, dataset: str) -> Path:
        path = self.root / dataset
        path.mkdir(parents=True, exist_ok=True)
        return path

    def records(self, dataset: str, method: str) -> Path:
        return self.dataset_dir(dataset) / f"{method}.jsonl"

    @property
    def filtered(self) -> Path:
        return self.root / "filtered.jsonl"


def new_run(tag: str = "") -> RunPaths:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    root = C.RUNS_DIR / (f"{stamp}-{tag}" if tag else stamp)
    root.mkdir(parents=True, exist_ok=True)
    configure_ledger(root / "filtered.jsonl")
    return RunPaths(root=root)


def run_dataset(
    dataset: str,
    run_cfg: C.RunConfig,
    paths: RunPaths,
    methods: Sequence[str],
    resume: bool = True,
) -> dict[str, Any]:
    corpus = load_dataset(dataset, n_questions=run_cfg.n_questions, seed=run_cfg.seed,
                          subset_corpus=run_cfg.subset_corpus)
    metadata_path = paths.dataset_dir(dataset) / "corpus.json"
    metadata = corpus.stats()
    previous = read_json(metadata_path)
    if previous and previous != metadata:
        raise ValueError("corpus ou perguntas mudaram; use uma nova rodada")
    write_json(metadata_path, metadata)
    ctx = build_context(corpus, run_cfg)

    started = time.perf_counter()
    ctx.llm.usage.reset()
    retrievers = build_methods(ctx, list(methods))

    # Demandas de treino para a seleção de memória sob orçamento. Só carregam se
    # alguém pediu orçamento < 1, porque construí-las custa uma busca por
    # pergunta de treino.
    if run_cfg.witness.budget_fraction < 1.0:
        for name, retriever in retrievers.items():
            if hasattr(retriever, "set_train_demands"):
                before = ctx.llm.usage.snapshot()
                t = time.perf_counter()
                _attach_train_demands(dataset, run_cfg, ctx, retriever)
                retriever.selection_cost = {"seconds": time.perf_counter() - t,
                                           "usage": usage_delta(ctx.llm.usage.snapshot(), before)}
    index_seconds = time.perf_counter() - started
    index_usage = ctx.llm.usage.snapshot()
    index_path = paths.dataset_dir(dataset) / "index_attempts.json"
    attempts = read_json(index_path, []) if resume else []
    attempts.append({"seconds": index_seconds, "usage": index_usage,
                     "shared": getattr(ctx, "shared_index_cost", {}),
                     "methods": {name: {"indexacao": r.index_cost,
                                        "selecao": getattr(r, "selection_cost", {})}
                                 for name, r in retrievers.items()}})
    write_json(index_path, attempts)
    original = attempts[0]

    summary: dict[str, Any] = {
        "dataset": dataset,
        "corpus": corpus.stats(),
        "indexacao_s": round(original["seconds"], 1),
        "uso_indexacao": original["usage"],
        "indexacao_compartilhada": original["shared"],
        "tentativas_indexacao": attempts,
        "embedder": {"backend": ctx.embedder.name, "chave": ctx.embedder.cache_key()},
        "subset_corpus": run_cfg.subset_corpus,
        "corpus_scope": run_cfg.corpus_scope,
        "extracao": ctx.extraction.stats() if ctx.extraction else {},
        "grafo": ctx.kg.stats() if ctx.kg else {},
        "metodos": {},
    }

    write_json(paths.dataset_dir(dataset) / "summary.json", summary)
    if run_cfg.interleave_methods:
        try:
            _run_interleaved(retrievers, corpus, run_cfg, paths, resume, summary, original)
        finally:
            if "relational" in retrievers:
                retrievers["relational"].searcher.close()
        return summary
    for name, retriever in retrievers.items():
        summary["metodos"][name] = _run_method(name, retriever, corpus, run_cfg, paths, resume)
        summary["metodos"][name]["indice"] = retriever.index_report()
        summary["metodos"][name].update(original["methods"][name])
        write_json(paths.dataset_dir(dataset) / "summary.json", summary)
        if name == "relational":
            retriever.searcher.close()

    write_json(paths.dataset_dir(dataset) / "summary.json", summary)
    return summary


def _run_interleaved(retrievers, corpus, cfg, paths, resume, summary, original):
    """Checkpoint após cada método; comparação usa só IDs concluídos por todos.

    A ordem de perguntas é embaralhada antes da avaliação, sem usar respostas.
    Métodos são rotacionados para distribuir efeitos de posição/cache.
    """
    rows = {}
    done = {}
    names = list(retrievers)
    for name in names:
        path = paths.records(corpus.name, name)
        if not resume:
            path.write_text("", encoding="utf-8")
        rows[name] = read_jsonl(path)
        done[name] = {r["qid"] for r in rows[name]}
    questions = list(corpus.questions)
    random.Random(cfg.seed).shuffle(questions)
    started = time.perf_counter()
    newly_completed = 0
    for index, question in enumerate(progress(questions, desc=f"{corpus.name}/todos")):
        had_work = any(question.qid not in done[name] for name in names)
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            if question.qid in done[name]:
                continue
            record = _answer_one(name, retrievers[name], corpus, question, cfg)
            append_jsonl(paths.records(corpus.name, name), record)
            rows[name].append(record)
            done[name].add(question.qid)
            summary["metodos"][name] = {
                "n_perguntas": len(corpus.questions),
                "n_concluidas": len(rows[name]),
                "tempo_consulta_s": round(sum(r["latencia_recuperacao_s"] + r["latencia_leitura_s"]
                                               for r in rows[name]), 3),
                "uso_consulta": sum_usage(r.get("uso_llm", {}) for r in rows[name]),
                "custo_completo": all("uso_llm" in r for r in rows[name]),
                "indice": retrievers[name].index_report(),
                **original["methods"][name],
            }
            write_json(paths.dataset_dir(corpus.name) / "summary.json", summary)
        log.info("perguntas concluídas em todos os métodos: %d/%d", index + 1, len(questions))
        newly_completed += int(had_work)
        if newly_completed and newly_completed % 5 == 0:
            seconds_per_question = (time.perf_counter() - started) / newly_completed
            estimate = seconds_per_question * (len(questions) - index - 1)
            log.info("ritmo observado: %.1f s/pergunta para todos os métodos; restante estimado: %.2f h "
                     "(varia com a dificuldade das perguntas)", seconds_per_question, estimate / 3600)


def _attach_train_demands(dataset: str, run_cfg: C.RunConfig, ctx, retriever) -> None:
    """Monta o conjunto de demandas que orienta a seleção de memória.

    Preferência por perguntas de TREINO reais, disjuntas da avaliação. Num piloto
    com corpus subamostrado quase nenhuma é elegível — as passagens de apoio delas
    não estão no índice — e aí a fonte passa a ser a síntese sobre o grafo. As
    duas se somam quando ambas existem; o relatório registra quantas vieram de
    cada fonte, porque a conclusão que se pode tirar da curva de orçamento depende
    disso.
    """
    from wrag.witness.demands import build_demands, synthesize_demands

    evaluated = {q.qid for q in ctx.corpus.questions}
    evaluation_text = {canonical_symbol(q.question) for q in ctx.corpus.questions}
    known = {p.pid for p in ctx.corpus.passages}
    # Uma demanda cuja testemunha não existe na memória indexada não informa nada
    # sobre o que vale a pena preservar.
    pool = []
    if run_cfg.train_questions_path:
        raw = read_json(Path(run_cfg.train_questions_path.format(dataset=dataset)))
        if not isinstance(raw, list):
            raise ValueError("arquivo de treino deve conter uma lista de perguntas")
        index = _PassageIndex()
        for p in ctx.corpus.passages:
            index.add(p.title, p.text)
        parsed = [q for item in raw if (q := _parse_question(dataset, item, index)) is not None]
        if any(q.qid in evaluated or canonical_symbol(q.question) in evaluation_text for q in parsed):
            raise ValueError("treino e avaliação possuem perguntas sobrepostas")
        pool = [q for q in parsed if q.gold_pids and not q.missing_supports and set(q.gold_pids) <= known]
        pool = pool[:max(50, run_cfg.n_questions)]

    demands = []
    if pool:
        demands = build_demands(pool, retriever.searcher, ctx.llm, run_cfg.witness, dataset=dataset)
    n_training_demands = len(demands)
    synthetic = []
    if len(demands) < 50:
        synthetic = synthesize_demands(retriever.memory, n_demands=1000, seed=run_cfg.seed)
        log.info("demandas: %d de perguntas de treino + %d sintetizadas do grafo",
                 len(demands), len(synthetic))
        demands = demands + synthetic
    retriever.set_train_demands(demands)
    retriever.demand_sources = {"perguntas_treino_elegiveis": len(pool), "treino": n_training_demands,
                               "sintetizadas": len(synthetic), "anotacoes_avaliacao_usadas": False}
    retriever._select_memory(run_cfg.witness.budget_fraction)


def _run_method(
    name: str,
    retriever,
    corpus: Corpus,
    run_cfg: C.RunConfig,
    paths: RunPaths,
    resume: bool,
) -> dict[str, Any]:
    path = paths.records(corpus.name, name)
    if not resume:
        path.write_text("", encoding="utf-8")
    done = {row["qid"] for row in read_jsonl(path)} if resume else set()
    if done:
        log.info("%s/%s: retomando, %d perguntas já feitas", corpus.name, name, len(done))

    llm = retriever.ctx.llm
    llm.usage.reset()
    started = time.perf_counter()

    for question in progress([q for q in corpus.questions if q.qid not in done],
                             desc=f"{corpus.name}/{name}"):
        record = _answer_one(name, retriever, corpus, question, run_cfg)
        append_jsonl(path, record)

    elapsed = time.perf_counter() - started
    rows = read_jsonl(path)
    return {
        "n_perguntas": len(corpus.questions),
        "tempo_consulta_s": round(sum(r.get("latencia_recuperacao_s", 0) + r.get("latencia_leitura_s", 0)
                                       for r in rows), 3),
        "tempo_sessao_s": round(elapsed, 3),
        "uso_consulta": sum_usage(r.get("uso_llm", {}) for r in rows),
        "custo_completo": all("uso_llm" in r for r in rows),
    }


def _answer_one(name: str, retriever, corpus: Corpus, question: Question,
                run_cfg: C.RunConfig) -> dict[str, Any]:
    before = retriever.ctx.llm.usage.snapshot()
    retrieval = retriever.retrieve(question, run_cfg.top_k)
    reading = read(retriever.ctx.llm, corpus, question, retrieval.pids,
                   replace(run_cfg.qa, top_k=run_cfg.top_k), method=name,
                   proof_context=retrieval.diagnostics.get("leitura_provas"))

    diagnostics = retrieval.diagnostics
    witness_facts = []
    respostas = diagnostics.get("respostas") or []
    if respostas and respostas[0].get("testemunhas"):
        witness_facts = respostas[0]["testemunhas"][0].get("fatos", [])

    record = {
        "qid": question.qid,
        "dataset": corpus.name,
        "metodo": name,
        "pergunta": question.question,
        "respostas_ouro": question.answers,
        "passagens_ouro": question.gold_pids,
        "n_hops": question.n_hops,
        "n_hops_fonte": question.hop_source,
        "tipo": question.qtype,
        "recuperadas": retrieval.pids,
        "scores": [round(float(s), 6) for s in retrieval.scores],
        "resposta": reading.answer,
        "em": M.exact_match(reading.answer, question.answers),
        "f1": M.token_f1(reading.answer, question.answers),
        "recall@2": M.recall_at_k(retrieval.pids, question.gold_pids, 2) if run_cfg.top_k >= 2 else float("nan"),
        "recall@5": M.recall_at_k(retrieval.pids, question.gold_pids, 5) if run_cfg.top_k >= 5 else float("nan"),
        "all_recall@5": M.all_recall_at_k(retrieval.pids, question.gold_pids, 5) if run_cfg.top_k >= 5 else float("nan"),
        "cobertura_testemunha": (M.witness_coverage(witness_facts, question.evidences)
                                  if "resposta_estrutural" in diagnostics else float("nan")),
        "cobertura_extremos_relaxada": (M.endpoint_coverage(witness_facts, question.evidences)
                                        if "resposta_estrutural" in diagnostics else float("nan")),
        "em_estrutural": (M.exact_match(diagnostics.get("resposta_estrutural", ""), question.answers)
                           if "resposta_estrutural" in diagnostics else None),
        "testemunha_no_contexto": diagnostics.get("testemunha_no_contexto", False),
        "abstencao": M.is_abstention(reading.answer),
        "risco": diagnostics.get("risco"),
        "risco_bruto": diagnostics.get("risco_bruto"),
        "forma_consulta": diagnostics.get("forma"),
        "resposta_estrutural": diagnostics.get("resposta_estrutural"),
        "filtrada": bool(retrieval.filtered or reading.filtered),
        "latencia_recuperacao_s": round(retrieval.latency_s, 3),
        "latencia_leitura_s": round(reading.latency_s, 3),
        "diagnosticos": _trim(diagnostics),
        "uso_llm": usage_delta(retriever.ctx.llm.usage.snapshot(), before),
    }
    # O LoCoMo tem avaliador próprio: stemming, remoção de "and" e F1 por
    # sub-resposta na categoria 1. Fica ao lado do EM/F1 do harness, que continua
    # sendo o número comparável entre datasets. Sem NLTK, a coluna não existe.
    if corpus.name == "locomo":
        from wrag.eval import locomo_official as LO
        if LO.available():
            record.update(LO.score_record(record))
    return record


def _trim(diagnostics: dict[str, Any], max_chars: int = 6000) -> dict[str, Any]:
    """Diagnósticos completos são grandes; o JSONL não precisa deles inteiros.

    O que fica é o suficiente para análise de erro: a consulta compilada, a forma,
    a lacuna, e as testemunhas do topo.
    """
    text = json.dumps(diagnostics, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return diagnostics
    keep = {k: diagnostics[k] for k in
            ("consulta", "forma", "planos_compilados", "plano_escolhido", "planejamento",
             "lacuna", "fallback", "rodadas_aquisicao",
             "profundidade_alcancada", "risco", "risco_bruto", "score_estrutural", "n_testemunhas",
             "n_testemunhas_propostas", "agregacao_executada",
             "resposta_estrutural", "testemunha_no_contexto", "ordem_fallback_preservada",
             "contexto_alterado_pelo_witness", "grounding_mode", "exhaustive",
             "truncations", "aterramento", "exaustiva", "truncamentos",
             "modo_aterramento", "busca_exaustiva", "feixe_exaustivo", "cortes", "risco_calibrado",
             "limite_inferior_contagem", "empacotamento", "classe_prova",
             "prova_provisoria", "leitura_provas")
            if k in diagnostics}
    research = diagnostics.get("pesquisa_provas")
    if isinstance(research, dict):
        keep["pesquisa_provas"] = {
            **{k: v for k, v in research.items() if k not in {"fronteira", "sondas", "acoes"}},
            "fronteira": research.get("fronteira", [])[:10],
            "sondas": research.get("sondas", [])[:5],
            "acoes": research.get("acoes", [])[:3],
        }
    respostas = diagnostics.get("respostas")
    if respostas:
        keep["respostas"] = respostas[:1]
    # As decisões da verificação e as provas por item do conjunto são o que
    # estoura o limite. As CONTAGENS não podem sumir junto: são elas que dizem
    # onde o método parou, e o relatório agrega exatamente esses campos.
    verification = diagnostics.get("verificacao")
    if isinstance(verification, dict):
        keep["verificacao"] = {k: v for k, v in verification.items() if k != "decisoes"}
    answers = diagnostics.get("conjunto_resposta")
    if isinstance(answers, dict):
        keep["conjunto_resposta"] = {
            **{k: v for k, v in answers.items() if k != "itens"},
            "itens": [{"resposta": i.get("resposta"), "passagens": i.get("passagens")}
                      for i in answers.get("itens", [])],
        }
    keep["_truncado"] = True
    return keep


def run(
    datasets: Sequence[str],
    methods: Sequence[str],
    run_cfg: C.RunConfig | None = None,
    tag: str = "",
    resume: bool = True,
    resume_dir: str | Path | None = None,
) -> Path:
    if not datasets or not methods:
        raise ValueError("informe ao menos um dataset e um método")
    run_cfg = run_cfg or C.RunConfig()
    if run_cfg.top_k < 1 or run_cfg.n_questions < 1 or not 0 <= run_cfg.witness.budget_fraction <= 1:
        raise ValueError("top-k/n devem ser positivos e budget deve estar em [0,1]")
    run_cfg.methods = tuple(methods)
    run_cfg.qa.top_k = run_cfg.top_k
    run_cfg.dataset = datasets[0]
    llm = get_llm()
    embedder = get_embedder()

    manifest = {
        "iniciado_em": datetime.now(timezone.utc).isoformat(),
        "datasets": list(datasets),
        "metodos": list(methods),
        "config": run_cfg.to_dict(),
        "llm": {"backend": llm.name, "deployment": getattr(llm, "deployment", "")},
        "embedder": {"backend": embedder.name, "chave": embedder.cache_key()},
        "ambiente": {"python": sys.version.split()[0], "plataforma": platform.platform()},
        "protocolo_versao": 2,
        "codigo_hash": sha([(str(p.relative_to(C.ROOT)), p.read_text(encoding="utf-8"))
                             for p in sorted((C.ROOT / "wrag").rglob("*.py"))]),
    }
    if hasattr(llm, "cache_identity"):
        manifest["llm"]["provider_identity"] = llm.cache_identity()
    if resume_dir is not None:
        paths = RunPaths(Path(resume_dir).resolve())
        previous = read_json(paths.root / "run.json")
        if not previous:
            raise ValueError("rodada a retomar não contém run.json")
        for key in ("datasets", "metodos", "config", "llm", "protocolo_versao", "codigo_hash"):
            if sha(previous.get(key)) != sha(manifest.get(key)):
                raise ValueError(f"configuração incompatível ao retomar: {key}")
        if previous.get("embedder", {}).get("backend") != embedder.name:
            raise ValueError("provedor de embedding mudou")
        configure_ledger(paths.filtered)
        manifest = previous
    else:
        paths = new_run(tag)
    write_json(paths.root / "run.json", manifest)
    log.info("rodada em %s", paths.root)

    summaries = {}
    for dataset in datasets:
        summaries[dataset] = run_dataset(dataset, run_cfg, paths, methods, resume=resume)

    manifest["terminado_em"] = datetime.now(timezone.utc).isoformat()
    manifest["resumos"] = summaries
    manifest["filtro_conteudo"] = LEDGER.summary()
    write_json(paths.root / "run.json", manifest)

    from wrag.eval.report import build_report

    build_report(paths.root)
    return paths.root
