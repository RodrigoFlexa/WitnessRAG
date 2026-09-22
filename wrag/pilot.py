"""Piloto Linux/vLLM com GPU explícita, prazo e finalização de resultados parciais.

Executar: python -m wrag.pilot --gpu 5
Este módulo não importa torch/config antes de fixar o ambiente dos subprocessos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METHODS = "dense,bm25,graphrag,hipporag,hipporag2,relational,witnessrag"
LOCOMO_CONVERSATIONS = 10   # locomo10.json; `--locomo-conversation all` roda as dez


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, help="índice físico NVIDIA ou UUID; ex.: 5")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="repo HF ou diretório de pesos HF")
    p.add_argument("--model-revision", default="", help="commit HF para fixar pesos/tokenizer")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "half"])
    p.add_argument("--quantization", default="", help="opcional, conforme checkpoint e suporte vLLM")
    p.add_argument("--port", type=int, default=8085)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--vllm-python", default=sys.executable, help="Python do ambiente que contém vLLM")
    p.add_argument("--existing-server", action="store_true", help="usa servidor já ativo em localhost:port")
    p.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--embed-device", default="cuda:0", help="cuda:0 é a GPU selecionada após remapeamento; ou cpu")
    p.add_argument("--dataset", choices=["2wikimultihopqa", "musique", "hotpotqa", "narrativeqa", "ruler", "sample", "locomo"], default="2wikimultihopqa")
    p.add_argument("-n", "--questions", type=int, default=None,
                   help="padrão: 100; LoCoMo: todas as perguntas das categorias 1 a 4")
    p.add_argument("--locomo-conversation", default="0",
                   help="índice da conversa, começando em zero, ou 'all' para as dez em sequência")
    p.add_argument("--locomo-turns-per-passage", type=int, default=8)
    p.add_argument("--locomo-chunk-tokens", type=int, default=0,
                   help="0 usa blocos de falas; 2048 aproxima o RAG descrito no ZeroMem")
    p.add_argument("--locomo-ie-window-tokens", type=int, default=0,
                   help="janelas OpenIE internas; 512 preserva fatos em chunks longos sem ampliar o leitor")
    p.add_argument("--locomo-file", type=Path, help="opcional: locomo10.json local, sem download")
    p.add_argument("--distractors", type=int, default=300, help="passagens aleatórias adicionais ao corpus candidato")
    p.add_argument("--full-corpus", action="store_true",
                   help="usa todo o corpus oficial; recomendado para resultados finais de QA")
    p.add_argument("--corpus-token-budget", type=int, default=0,
                   help="HotpotQA: expande o corpus candidato até 56k/224k/448k tokens")
    p.add_argument("--corpus-passages", type=int, default=0,
                   help="tamanho exato do índice em passagens, incluindo apoios")
    p.add_argument("--max-passages", type=int, default=1500, help="falha se o corpus candidato exceder este teto")
    p.add_argument("--methods", default=None, help="LoCoMo: somente witnessrag; demais: todos os métodos")
    p.add_argument("--no-acquisition", action="store_true", help="ablação sem aquisição dirigida")
    p.add_argument("--binding-aware-grounding", action="store_true")
    p.add_argument("--verify-witnesses", action="store_true")
    p.add_argument("--top-k", type=int, default=5, help="passagens entregues ao leitor")
    p.add_argument("--answer-set", action="store_true",
                   help="resposta como conjunto de atribuições certas, com prova por item")
    p.add_argument("--vocab-compile", action="store_true",
                   help="compilação ancorada nas relações e entidades do grafo")
    p.add_argument("--query-plans", action="store_true",
                   help="gera e revisa planos usando o retorno da busca no grafo")
    p.add_argument("--max-query-plans", type=int, default=5,
                   help="orçamento global de planos distintos por pergunta (padrão: 5)")
    p.add_argument("--active-frontier", action="store_true",
                   help="busca complementar e aquisição sobre lacunas da prova")
    p.add_argument("--active-obligations", action="store_true",
                   help="confere se cada plano cobre as condições da pergunta")
    p.add_argument("--active-context", action="store_true",
                   help="seleciona pacotes de evidência com provas completas")
    p.add_argument("--active-operators", action="store_true",
                   help="leitura especializada em tempo e contagem; contagem estrutural parcial")
    p.add_argument("--soft-obligations", action="store_true",
                   help="planos parciais podem oferecer contexto, sem serem certificados")
    p.add_argument("--proof-reader", action="store_true",
                   help="leitor confere hipóteses do grafo nas passagens selecionadas")
    p.add_argument("--temporal-memory", action="store_true",
                   help="usa ordem/data estruturadas para completar contexto temporal, sem LLM")
    p.add_argument("--complementary-context", action="store_true",
                   help="troca no máximo a quinta passagem por uma faceta ausente, sem LLM")
    p.add_argument("--temporal-annotations", action="store_true",
                   help="normaliza tempo relativo usando a data da própria sessão LoCoMo")
    p.add_argument("--admit-provisional-witnesses", action="store_true",
                   help="usa testemunhas provisórias com proveniência como rota qualificada")
    p.add_argument("--plan-repair", action="store_true",
                   help="diagnostica junções e busca/replaneja obrigações não cobertas")
    p.add_argument("--hybrid-fallback", action="store_true",
                   help="fallback por fusão recíproca de postos (denso + BM25)")
    p.add_argument("--witness-candidate-pool", type=int, default=20,
                   help="candidatos explorados pelo Witness antes de entregar top-k ao leitor")
    p.add_argument("--dialogue-ie", action="store_true",
                   help="extração adaptada a diálogo: falante como sujeito e tempo do fato")
    p.add_argument("--no-relation-family-merge", action="store_true",
                   help="ablação: preserva flexões como paint/painted separadamente")
    p.add_argument("--hours", type=float, default=6.5, help="janela total; reserva 2 min para finalização")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="reaproveita caches (OpenIE, embeddings) de outra rodada; a extração é a parte cara")
    p.add_argument("--dry-run", action="store_true", help="mostra comandos sem baixar dados ou iniciar processos")
    p.add_argument("--resume", action="store_true",
                   help="retoma um --output parcial compatível e pula conversas já concluídas")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return p


def make_plan(args, output):
    if args.max_query_plans < 1:
        raise ValueError("--max-query-plans deve ser >= 1")
    if args.soft_obligations and not args.active_obligations:
        raise ValueError("--soft-obligations exige --active-obligations")
    if args.plan_repair and not (args.query_plans and args.active_obligations and args.active_frontier):
        raise ValueError("--plan-repair exige --query-plans, --active-obligations e --active-frontier")
    if args.questions is None and args.dataset != "locomo":
        args.questions = 100
    if args.methods is None:
        args.methods = "witnessrag" if args.dataset == "locomo" else DEFAULT_METHODS
    if args.locomo_file:
        args.locomo_file = args.locomo_file.resolve()
    if str(args.locomo_conversation).strip().lower() == "all":
        # Cada conversa é uma memória própria: as perguntas de uma nunca devem ser
        # respondidas com o diálogo de outra. São dez corpora e dez rodadas, e não
        # um corpus único — juntar tudo criaria distratores que o protocolo
        # publicado não tem, e "John" aparece em três conversas diferentes.
        args.locomo_conversation = "all"
    else:
        args.locomo_conversation = int(args.locomo_conversation)
        if args.locomo_conversation < 0:
            raise ValueError("índice LoCoMo deve ser >= 0 ou 'all'")
    if args.locomo_turns_per_passage < 1:
        raise ValueError("tamanho da passagem deve ser >= 1")
    if args.locomo_chunk_tokens < 0:
        raise ValueError("locomo-chunk-tokens deve ser >= 0")
    if args.locomo_ie_window_tokens < 0 or 0 < args.locomo_ie_window_tokens <= 64:
        raise ValueError("locomo-ie-window-tokens deve ser 0 ou maior que o overlap de 64")
    if not args.gpu.strip() or "," in args.gpu:
        raise ValueError("selecione uma GPU; este piloto usa tensor-parallel-size=1")
    if args.hours <= 2 / 60 or (args.questions is not None and args.questions < 1) or args.max_passages < 1 or args.distractors < 0:
        raise ValueError("prazo deve exceder 2 minutos; n/teto positivos; distratores >= 0")
    if not 0 < args.gpu_memory_utilization < 1 or args.concurrency < 1 or not 1 <= args.port <= 65535:
        raise ValueError("memória, concorrência ou porta inválidas")
    if args.max_model_len < 4096:
        raise ValueError("use contexto >= 4096; o leitor recebe passagens completas")
    if args.top_k < 1:
        raise ValueError("top-k deve ser >= 1")
    if args.witness_candidate_pool < 1:
        raise ValueError("witness-candidate-pool deve ser >= 1")
    if args.corpus_token_budget < 0:
        raise ValueError("corpus-token-budget deve ser >= 0")
    if args.corpus_passages < 0:
        raise ValueError("corpus-passages deve ser >= 0")
    if args.embed_device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("embed-device deve ser cpu ou cuda:0 na GPU remapeada")
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    available = {"dense", "bm25", "hybrid", "graphrag", "hipporag", "hipporag2", "relational",
                 "witnessrag", "witnessrag-lite", "witnessrag-annotated", "witnessrag-oracle"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - available:
        raise ValueError("lista de métodos inválida ou duplicada")
    env = {
        "CUDA_VISIBLE_DEVICES": args.gpu, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "WRAG_LLM_BACKEND": "vllm", "OPENAI_MODEL": args.model,
        "OPENAI_BASE_URL": f"http://127.0.0.1:{args.port}/v1", "OPENAI_API_KEY": "local-pilot",
        "WRAG_MODEL_REVISION": args.model_revision,
        "WRAG_AZURE_CONCURRENCY": str(args.concurrency), "WRAG_AZURE_MAX_RETRIES": "2",
        "WRAG_AZURE_TIMEOUT_S": "120", "WRAG_EMBED_BACKEND": "st",
        "WRAG_EMBED_MODEL": args.embed_model, "WRAG_EMBED_DEVICE": args.embed_device,
        "WRAG_EMBED_BATCH_SIZE": "32", "WRAG_SEED": str(args.seed),
        "WRAG_DATA_DIR": str(output / "data"), "WRAG_RUNS_DIR": str(output / "benchmark"),
        "WRAG_CACHE_DIR": str(args.cache_dir.resolve() if args.cache_dir else output / "cache"), "WRAG_NO_PROGRESS": "1", "PYTHONUNBUFFERED": "1",
        # O sampler FlashInfer compila kernels via JIT e resolve o nvcc por `which nvcc`,
        # que aqui aponta para CUDA 11.5 e rejeita --compress-mode=size (exige >= 12.8).
        # A decodificação é gulosa (temperature=0), logo o sampler nativo não altera saídas.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    command = [args.vllm_python, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
               "--served-model-name", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", args.dtype, "--max-model-len", str(args.max_model_len),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-num-seqs", str(args.concurrency), "--tensor-parallel-size", "1",
               "--generation-config", "vllm"]
    if args.model_revision:
        command += ["--revision", args.model_revision, "--tokenizer-revision", args.model_revision]
    if args.quantization:
        command += ["--quantization", args.quantization]
    frozen_source = os.environ.get("WRAG_FROZEN_MEMORY_SOURCE")
    frozen_identity = None
    if frozen_source:
        source = Path(frozen_source).resolve()
        manifests = sorted((source / "controlled").glob("*/memory.json"))
        if not manifests:
            raise FileNotFoundError(f"No frozen memory manifests under {source}")
        digest = hashlib.sha256()
        for manifest in manifests:
            digest.update(manifest.parent.name.encode("utf-8"))
            digest.update(manifest.read_bytes())
        frozen_identity = {"source": str(source), "manifests": len(manifests),
                           "sha256": digest.hexdigest()}
    return {"output": str(output), "env": env, "server_command": command, "methods": methods,
            "frozen_memory": frozen_identity,
            "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "scope": ("LoCoMo: conversa completa, QA single/multi/temporal/open-domain, adaptação textual"
                      if args.dataset == "locomo" else
                      "piloto com corpus candidato reduzido e distratores; adaptações locais dos artigos")}


def locomo_conversations(plan):
    """Índices a rodar: um só, ou as dez conversas do arquivo oficial."""
    settings = plan["settings"]
    if settings.get("locomo_conversation") != "all":
        return [int(settings.get("locomo_conversation", 0))]
    return list(range(LOCOMO_CONVERSATIONS))


def prepare_conversation(plan, index, output):
    """Prepara UMA conversa num diretório próprio, reusando o snapshot baixado."""
    from wrag.locomo import prepare
    settings = plan["settings"]
    shared = Path(plan["output"]) / "source-data" / "locomo10.json"
    source = settings.get("locomo_file") or (str(shared) if shared.exists() else None)
    return prepare(output, source, index, settings.get("locomo_turns_per_passage", 8),
                   settings.get("questions"), settings["seed"], settings["max_passages"],
                   settings.get("locomo_chunk_tokens") or None, settings["model"],
                   settings.get("model_revision") or None)


def prepare_data(plan):
    """Corpus fixo para todos; seleção não usa desempenho dos métodos."""
    from wrag.data import load_dataset
    from wrag.util import read_json, write_json
    settings = plan["settings"]
    if settings["dataset"] == "locomo":
        from wrag.locomo import prepare
        return prepare(plan["output"], settings.get("locomo_file"),
                       settings.get("locomo_conversation", 0), settings.get("locomo_turns_per_passage", 8),
                       settings.get("questions"), settings["seed"], settings["max_passages"],
                       settings.get("locomo_chunk_tokens") or None, settings["model"],
                       settings.get("model_revision") or None)
    data = Path(plan["output"]) / "data"
    source = Path(plan["output"]) / "source-data"
    source.mkdir(parents=True, exist_ok=True)
    name, seed = settings["dataset"], settings["seed"]
    sources = {}
    for filename in (f"{name}.json", f"{name}_corpus.json"):
        path = source / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/OSU-NLP-Group/HippoRAG/main/reproduce/dataset/{filename}"
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read()
            json.loads(payload)  # não guardar uma página HTML de erro
            path.write_bytes(payload)
        sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    n = settings["questions"]
    reduced = load_dataset(name, n_questions=n, seed=seed, data_dir=source, subset_corpus=True)
    full = load_dataset(name, n_questions=n, seed=seed, data_dir=source, subset_corpus=False)
    keep = {p.pid for p in reduced.passages}
    cap = settings["max_passages"]
    if len(keep) > cap:
        raise ValueError(f"corpus candidato tem {len(keep)} passagens > teto {cap}; reduza -n ou aumente --max-passages")
    extras = [p for p in full.passages if p.pid not in keep]
    random.Random(seed).shuffle(extras)
    if settings.get("full_corpus"):
        if len(full.passages) > cap:
            raise ValueError(f"corpus completo tem {len(full.passages)} passagens > teto {cap}")
        chosen = list(full.passages)
        tokens = None
    elif settings.get("corpus_passages"):
        target = settings["corpus_passages"]
        if target < len(reduced.passages):
            raise ValueError(f"corpus-passages={target} menor que {len(reduced.passages)} apoios/candidatos")
        if target > cap:
            raise ValueError(f"corpus-passages={target} excede max-passages={cap}")
        chosen = reduced.passages + extras[:target - len(reduced.passages)]
        if len(chosen) != target:
            raise ValueError(f"corpus possui somente {len(chosen)} passagens; solicitado {target}")
        tokens = None
    elif settings.get("corpus_token_budget"):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            settings["model"], revision=settings.get("model_revision") or None)
        chosen = list(reduced.passages)
        tokens = sum(len(tokenizer.encode(p.full, add_special_tokens=False)) for p in chosen)
        for passage in extras:
            if len(chosen) >= cap or tokens >= settings["corpus_token_budget"]:
                break
            chosen.append(passage)
            tokens += len(tokenizer.encode(passage.full, add_special_tokens=False))
        if tokens < settings["corpus_token_budget"]:
            raise ValueError(f"corpus possui {tokens} tokens, abaixo do orçamento solicitado")
    else:
        chosen = reduced.passages + extras[:min(settings["distractors"], cap - len(keep))]
        tokens = None
    ids = {q.qid for q in reduced.questions}
    raw = read_json(source / f"{name}.json")
    selected = [q for q in raw if str(q.get("id") or q.get("_id") or "") in ids]
    write_json(data / f"{name}.json", selected)
    write_json(data / f"{name}_corpus.json", [{"title": p.title, "text": p.text} for p in chosen])
    metadata = {"source_sha256": sources, "source_repo": "https://github.com/OSU-NLP-Group/HippoRAG",
                "seed": seed, "full_passages": len(full.passages), "candidate_passages": len(keep),
                "additional_distractors": len(chosen) - len(keep), "selected_passages": len(chosen),
                "corpus_tokens": tokens, "corpus_token_budget": settings.get("corpus_token_budget", 0),
                "corpus_passage_budget": settings.get("corpus_passages", 0),
                "questions": len(selected), "question_ids": sorted(ids),
                "full_corpus": bool(settings.get("full_corpus")),
                "corpus_reduced": len(chosen) < len(full.passages)}
    write_json(Path(plan["output"]) / "data_selection.json", metadata)
    return metadata


def worker(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    # Ambiente já foi aplicado pelo processo pai, antes de importar config.
    from wrag import config as C
    from wrag.eval.runner import run
    from wrag.llm import get_llm, GenParams
    from wrag.util import setup_logging, write_json
    setup_logging()
    settings = plan["settings"]
    print("Preparando corpus do piloto...", flush=True)
    every = settings["dataset"] == "locomo" and settings.get("locomo_conversation") == "all"
    # No modo "all" esta chamada serve para baixar e registrar o snapshot uma vez
    # (origem e SHA-256); cada conversa é preparada depois no seu diretório.
    metadata = prepare_conversation(plan, 0, Path(plan["output"])) if every else prepare_data(plan)
    print(json.dumps({k: v for k, v in metadata.items() if k not in {"question_ids", "question_mapping"}}, ensure_ascii=False), flush=True)
    import importlib.metadata
    versions = {}
    for package in ("openai", "torch", "transformers", "sentence-transformers", "numpy", "scipy", "networkx", "igraph"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    write_json(Path(plan["output"]) / "benchmark_versions.json", versions)
    llm = get_llm()
    probe = llm.chat('Return only JSON: {"ok": true}', params=GenParams(max_tokens=32, json_mode=True),
                     stage="pilot.preflight")
    if not isinstance(probe.json(), dict) or probe.json().get("ok") is not True:
        raise RuntimeError("preflight não retornou o JSON esperado; confira modelo/endpoint")
    # Mantém os demais hiperparâmetros consolidados: sem corte oculto de tokens/fatos.
    if settings["dataset"] == "locomo" and settings.get("locomo_conversation") == "all":
        roots = _run_every_conversation(plan, settings)
    else:
        resume_dir = None
        if settings.get("resume"):
            manifests = sorted((Path(plan["output"]) / "benchmark").glob("*/run.json"))
            if len(manifests) > 1:
                raise ValueError("retomada ambígua: mais de uma rodada em benchmark/")
            if manifests:
                resume_dir = manifests[0].parent
                print(f"Retomando rodada parcial: {resume_dir}", flush=True)
        roots = [str(run([settings["dataset"]], plan["methods"],
                         _run_config(settings, metadata["questions"]), tag="qwen-pilot",
                         resume=True, resume_dir=resume_dir))]
    write_json(Path(plan["output"]) / "completed.json",
               {"run_dir": roots[0], "run_dirs": roots})


def _run_config(settings, n_questions):
    from wrag import config as C
    cfg = C.RunConfig(n_questions=settings["questions"] or n_questions, seed=settings["seed"],
                      top_k=settings.get("top_k", 5),
                      interleave_methods=True, corpus_scope=("locomo_full_selected_conversation"
                      if settings["dataset"] == "locomo" else
                      "full_official_corpus" if settings.get("full_corpus") else
                      "pilot_candidates_plus_random_distractors"))
    cfg.witness.enable_acquisition = not settings["no_acquisition"]
    cfg.witness.binding_aware_grounding = settings.get("binding_aware_grounding", False)
    cfg.witness.verify_witnesses = settings.get("verify_witnesses", False)
    cfg.witness.answer_set = settings.get("answer_set", False)
    cfg.qa.answer_set = settings.get("answer_set", False)
    cfg.witness.vocabulary_aware_compile = settings.get("vocab_compile", False)
    cfg.witness.query_plans = settings.get("query_plans", False)
    cfg.witness.max_query_plans = settings.get("max_query_plans", 5)
    cfg.witness.active_frontier = settings.get("active_frontier", False)
    cfg.witness.active_obligations = settings.get("active_obligations", False)
    cfg.witness.active_context = settings.get("active_context", False)
    cfg.witness.active_operators = settings.get("active_operators", False)
    cfg.witness.soft_obligations = settings.get("soft_obligations", False)
    cfg.witness.proof_reader = settings.get("proof_reader", False)
    cfg.witness.temporal_memory = settings.get("temporal_memory", False)
    cfg.witness.complementary_context = settings.get("complementary_context", False)
    cfg.witness.admit_provisional_witnesses = settings.get("admit_provisional_witnesses", False)
    cfg.witness.plan_repair = settings.get("plan_repair", False)
    if cfg.witness.plan_repair:
        # Conv00: later replans did not yield a selected proof, while repeated
        # verification and targeted extraction dominated query latency.
        cfg.witness.max_replan_calls = 1
        cfg.witness.verification_max_witnesses = 2
        cfg.witness.acquisition_rounds = 1
        cfg.witness.acquisition_passages = 1
    cfg.qa.operator_reader = cfg.witness.active_operators
    cfg.qa.proof_reader = cfg.witness.proof_reader
    cfg.qa.answer_guard = settings.get("answer_guard", False)
    cfg.qa.temporal_annotations = settings.get("temporal_annotations", False)
    cfg.witness.hybrid_fallback = settings.get("hybrid_fallback", False)
    cfg.witness.candidate_pool_k = settings.get("witness_candidate_pool", 20)
    cfg.ie.dialogue_mode = settings.get("dialogue_ie", False)
    cfg.graph.merge_relation_inflections = not settings.get("no_relation_family_merge", False)
    cfg.ie.window_tokens = settings.get("locomo_ie_window_tokens", 0)
    cfg.ie.window_tokenizer = settings["model"] if cfg.ie.window_tokens else ""
    cfg.ie.window_tokenizer_revision = settings.get("model_revision", "") if cfg.ie.window_tokens else ""
    return cfg


def _run_every_conversation(plan, settings):
    """Uma rodada por conversa, no mesmo servidor e no mesmo cache de extração.

    Dez corpora separados, não um corpus único: cada conversa é a memória das
    suas próprias perguntas, e é assim que o LoCoMo é avaliado. O cache de
    OpenIE é chaveado pelo conteúdo do corpus, então as dez convivem sem colisão.

    O prazo é checado ENTRE conversas: interromper no meio de uma deixaria uma
    rodada parcial cujo denominador não é comparável com as outras.
    """
    from wrag import config as C
    from wrag.eval.runner import run
    from wrag.util import read_json, write_json

    output = Path(plan["output"])
    deadline = plan.get("deadline_epoch")
    progress_path = output / "conversations.json"
    previous = read_json(progress_path).get("conversas", []) if progress_path.exists() else []
    done_by_index = {int(item["conversa"]): item for item in previous
                     if _completed_conversation(item, plan["methods"])}
    roots = [item["run_dir"] for _, item in sorted(done_by_index.items())]
    done = [item for _, item in sorted(done_by_index.items())]
    if done:
        print(f"Retomada: {len(done)} conversa(s) completas serão preservadas.", flush=True)
    for index in locomo_conversations(plan):
        if index in done_by_index:
            continue
        if deadline is not None and time.time() >= deadline:
            print(f"Prazo atingido; conversas restantes não foram executadas.", flush=True)
            break
        conversation = output / "conversations" / f"conv{index:02d}"
        metadata = prepare_conversation(plan, index, conversation)
        print(f"\n=== conversa {index} ({metadata['sample_id']}): "
              f"{metadata['questions']} perguntas, {metadata['selected_passages']} blocos ===", flush=True)
        C.DATA_DIR = conversation / "data"
        C.RUNS_DIR = conversation / "benchmark"
        C.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        root = run([settings["dataset"]], plan["methods"],
                   _run_config(settings, metadata["questions"]), tag=f"conv{index:02d}")
        roots.append(str(root))
        done.append({"conversa": index, "sample_id": metadata["sample_id"],
                     "perguntas": metadata["questions"], "run_dir": str(root)})
        done.sort(key=lambda item: item["conversa"])
        write_json(progress_path, {"conversas": done})
    return roots


def _completed_conversation(item, methods):
    """Accept only a complete, readable run before skipping it on resume."""
    try:
        run_dir = Path(item["run_dir"])
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        dataset = report["datasets"]["locomo"]
        if not manifest.get("terminado_em"):
            return False
        expected = int(item["perguntas"])
        return all(int(dataset["metodos"][method]["n_avaliadas"]) == expected for method in methods)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def stop_owned(process):
    """Encerra somente o grupo de processos iniciado por este launcher."""
    if process is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_ready(server, port, model, deadline):
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError("vLLM encerrou durante a inicialização; consulte vllm.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
                models = json.load(response)
            if model in {x.get("id") for x in models.get("data", [])}:
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("servidor não ficou pronto dentro do prazo; consulte vllm.log")


def wait_port_free(port, timeout_s=45):
    """Aguarda a porta liberar para reduzir falhas transitórias entre rodadas.

    Em execuções sequenciais, o processo anterior pode levar alguns segundos para
    soltar o socket. Se outra aplicação estiver ocupando a porta de forma estável,
    o erro permanece explícito após o prazo.
    """
    end = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < end:
        try:
            with socket.socket() as check:
                check.bind(("127.0.0.1", port))
            return
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    if last_error is not None:
        raise OSError(last_error.errno,
                      f"porta {port} permaneceu ocupada por {timeout_s}s ({last_error})")
    raise OSError(f"porta {port} permaneceu ocupada por {timeout_s}s")


def launch(args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or ROOT / "runs" / f"local-pilot-{stamp}").resolve()
    plan = make_plan(args, output)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if os.name != "posix" and not args.existing_server:
        raise RuntimeError("execute o launcher no servidor Linux com NVIDIA/vLLM; use --dry-run para inspecionar")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("--output deve ser novo ou vazio; use --resume para uma execução parcial")
    if args.resume:
        old_plan_path = output / "pilot.json"
        if not old_plan_path.exists():
            raise ValueError("--resume exige um pilot.json no diretório de saída")
        old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
        _validate_resume(old_plan, plan)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "pilot.json"
    # O worker precisa do MESMO prazo para não começar uma conversa que não cabe:
    # ser morto no meio deixaria uma rodada parcial com denominador incomparável.
    plan["deadline_epoch"] = time.time() + args.hours * 3600 - 180
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {**os.environ, **plan["env"]}
    # 120s reservados para encerrar processos e gerar relatórios/gráficos.
    deadline = time.monotonic() + args.hours * 3600 - 120
    server = job = None
    status = "failed"
    print(f"Saída: {output}\nGPU: {args.gpu}; prazo: {args.hours} h; modelo: {args.model}", flush=True)
    try:
        # A resumed attempt must not report an exception left by an older
        # attempt as its current failure cause.
        for stale_name in ("worker-error.txt", "error.txt"):
            (output / stale_name).unlink(missing_ok=True)
        try:
            hardware = subprocess.run(["nvidia-smi", "-i", args.gpu,
                                       "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv"],
                                      capture_output=True, text=True, timeout=10)
            (output / "gpu.txt").write_text(hardware.stdout + hardware.stderr, encoding="utf-8")
        except (OSError, subprocess.SubprocessError):
            pass
        if not args.existing_server:
            wait_port_free(args.port)
            with (output / "vllm.log").open("w", encoding="utf-8") as log:
                server = subprocess.Popen(plan["server_command"], env=env, cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print("Aguardando vLLM (pesos ausentes podem exigir download)...", flush=True)
        wait_ready(server, args.port, args.model, min(deadline, time.monotonic() + 3600))
        if not args.existing_server:
            try:
                version = subprocess.run([args.vllm_python, "-c",
                                          "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
                                         capture_output=True, text=True, timeout=10)
                (output / "vllm-version.txt").write_text(version.stdout, encoding="utf-8")
            except (OSError, subprocess.SubprocessError):
                pass
        with (output / "benchmark.log").open("a" if args.resume else "w", encoding="utf-8") as log:
            job = subprocess.Popen([sys.executable, "-m", "wrag.pilot", "--gpu", args.gpu,
                                    "--worker", str(plan_path)], env=env, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        print(f"Benchmark em execução. Acompanhe: tail -f {output / 'benchmark.log'}", flush=True)
        while job.poll() is None:
            if time.monotonic() >= deadline:
                status = "time_limit"
                break
            if server is not None and server.poll() is not None:
                raise RuntimeError("vLLM encerrou durante o benchmark")
            time.sleep(2)
        if job.poll() is not None:
            status = "complete" if job.returncode == 0 else "failed"
            if status == "failed" and (output / "worker-error.txt").exists():
                detail = (output / "worker-error.txt").read_text(encoding="utf-8").strip().splitlines()
                if detail:
                    print(f"Falha no benchmark: {detail[-1]}", file=sys.stderr)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        print(f"Falha: {exc}", file=sys.stderr)
        (output / "error.txt").write_text(str(exc), encoding="utf-8")
    finally:
        stop_owned(job)
        stop_owned(server)
        (output / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        manifests = sorted((output / "benchmark").glob("*/run.json")) + \
            sorted((output / "conversations").glob("*/benchmark/*/run.json"))
        for manifest in manifests:
            try:
                subprocess.run([sys.executable, "-m", "wrag.eval.plots", str(manifest.parent)],
                               env=env, cwd=ROOT, check=True, timeout=80)
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"Falha na finalização: {exc}. Reexecute python -m wrag.eval.plots {manifest.parent}",
                      file=sys.stderr)
        if not manifests:
            print("Nenhuma rodada chegou à avaliação. Consulte logs; não há resultados de qualidade.")
    print(f"Estado: {status}. Resultados: {output}", flush=True)
    print_results(output, status)
    return 0 if status == "complete" else 2


def _validate_resume(old, new):
    """Reject changes that would mix incomparable results in one aggregate."""
    fields = ("model", "model_revision", "embed_model", "dataset", "questions",
              "locomo_conversation", "locomo_turns_per_passage", "locomo_chunk_tokens",
              "locomo_ie_window_tokens", "seed", "top_k", "witness_candidate_pool",
              "corpus_token_budget", "corpus_passages", "full_corpus",
              "answer_set", "vocab_compile", "query_plans", "max_query_plans",
              "active_frontier", "active_obligations", "active_context", "active_operators",
              "soft_obligations", "proof_reader", "plan_repair", "temporal_memory",
              "complementary_context", "temporal_annotations", "admit_provisional_witnesses",
              "hybrid_fallback", "dialogue_ie",
              "no_relation_family_merge",
              "binding_aware_grounding", "verify_witnesses", "no_acquisition")
    differences = [name for name in fields
                   if (old.get("settings", {}).get(name) or 0) !=
                      (new.get("settings", {}).get(name) or 0)]
    if old.get("methods") != new.get("methods"):
        differences.append("methods")
    if old.get("frozen_memory") != new.get("frozen_memory"):
        differences.append("frozen_memory")
    if differences:
        raise ValueError("--resume incompatível; mudou: " + ", ".join(differences))


def print_locomo_aggregate(output, status, run_dirs):
    """Tabela única das conversas rodadas, sem poluir a saída por conversa.

    A média é micro: toda pergunta pesa igual, conversas maiores pesam mais. As
    conversas continuam sendo corpora separados — isto é uma agregação de dez
    experimentos, não um experimento sobre um índice único.
    """
    from wrag.eval.locomo_official import aggregate_runs
    from wrag.util import write_json

    summary = aggregate_runs(run_dirs)
    official = summary.get("oficial_disponivel")
    keys = (("f1_locomo", "bleu1_locomo", "em_locomo") if official else ()) + \
        ("f1", "em", "recall@2", "recall@5", "all_recall@2", "all_recall@5")
    labels = (("F1ofic", "BLEU1", "EMofic") if official else ()) + \
        ("F1", "EM", "R@2", "R@5", "AR@2", "AR@5")

    def pct(value):
        return f"{100 * value:.2f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"

    total = next(iter(summary["metodos"].values()), {}).get("n", 0)
    # O arquivo final segue a mesma apresentação do terminal: somente a média
    # geral e as categorias, sem uma tabela por conversa.
    for values in summary["metodos"].values():
        values.pop("por_conversa", None)
    write_json(Path(output) / "locomo_agregado.json", summary)
    print(f"\nResultados ({'concluído' if status == 'complete' else 'PARCIAIS — ' + status}): "
          f"{summary['conversas']} conversa(s), {total} perguntas, média micro\n")
    print(f"{'método / categoria':<35} {'n':>5} " + " ".join(f"{x:>8}" for x in labels)
          + f" {'disparo':>8} {'mudouctx':>8}")
    for method, values in summary["metodos"].items():
        for label, block in [(method, values)] + [(f"  {k}", v) for k, v in values["por_categoria"].items()]:
            fire = pct(block["taxa_de_disparo"]) if "taxa_de_disparo" in block else "—"
            changed = pct(block["taxa_de_intervencao"]) if "taxa_de_intervencao" in block else "—"
            print(f"{label:<35} {block['n']:>5} "
                  + " ".join(f"{pct(block.get(k)):>8}" for k in keys)
                  + f" {fire:>8} {changed:>8}")
    if official:
        print("\nF1ofic/EMofic reproduzem task_eval/evaluation.py do LoCoMo; F1/EM são do harness.")
    print(f"Agregado: {Path(output) / 'locomo_agregado.json'}", flush=True)


def _weighted(categories, keys):
    """Média das categorias ponderada por n. Cada pergunta está em exatamente uma."""
    out = {}
    for key in keys:
        pairs = [(v.get(key), v.get("n", 0)) for v in categories.values()
                 if isinstance(v.get(key), (int, float)) and math.isfinite(v[key])]
        total = sum(n for _v, n in pairs)
        out[key] = sum(v * n for v, n in pairs) / total if total else float("nan")
    return out


def print_results(output, status):
    """Resumo do mesmo report.json usado no relatório; sem recalcular métricas."""
    def pct(value):
        return f"{100 * value:.2f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"

    conversations = sorted((Path(output) / "conversations").glob("*/benchmark/*/report.json"))
    if conversations:
        print_locomo_aggregate(output, status, [path.parent for path in conversations])
        return
    reports = sorted((Path(output) / "benchmark").glob("*/report.json"))
    if not reports:
        print("Resumo indisponível: nenhum report.json foi gerado.", flush=True)
    for path in reports:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            print(f"\nResultados ({'concluído' if status == 'complete' else 'PARCIAIS — ' + status}):")
            for dataset, data in report.get("datasets", {}).items():
                total = data.get("corpus", {}).get("n_questions", "?")
                categories = data.get("por_categoria", {})
                # No LoCoMo a coluna que conta é a do avaliador oficial: na
                # categoria 1 ele compara sub-respostas separadas por vírgula, e o
                # F1 por token do harness castiga cada item a mais de um conjunto
                # correto. Imprimir só o harness esconde exatamente o multi-hop.
                official = dataset == "locomo" and any(
                    "f1_locomo" in v for m in categories.values() for v in m.values())
                keys = (("f1_locomo", "bleu1_locomo", "em_locomo") if official else ()) + \
                    ("f1", "em", "recall@2", "recall@5", "all_recall@2", "all_recall@5")
                labels = (("F1ofic", "BLEU1", "EMofic") if official else ()) + \
                    ("F1", "EM", "R@2", "R@5", "AR@2", "AR@5")
                print(f"\n{dataset} — métricas em %, perguntas previstas: {total}")
                print(f"{'método / categoria':<35} {'n':>5} " + " ".join(f"{x:>8}" for x in labels))
                for method, result in data.get("metodos", {}).items():
                    values = dict(result.get("metricas", {}))
                    if official:
                        values.update(_weighted(categories.get(method, {}), ("f1_locomo", "em_locomo")))
                    print(f"{method:<35} {result.get('n_avaliadas', 0):>5} "
                          + " ".join(f"{pct(values.get(k)):>8}" for k in keys))
                    for category, values in categories.get(method, {}).items():
                        label = f"  {category}"
                        print(f"{label:<35} {values.get('n', 0):>5} "
                              + " ".join(f"{pct(values.get(k)):>8}" for k in keys))
                    stop = data.get("onde_parou", {}).get(method)
                    if stop:
                        print(f"{'  testemunha disparou em':<35} {pct(stop.get('taxa_de_disparo')):>8}% "
                              f"das perguntas; o resto veio do fallback")
                        print(f"{'  contexto realmente alterado em':<35} "
                              f"{pct(stop.get('taxa_de_intervencao')):>8}% das perguntas")
                if data.get("excluidas"):
                    print(f"Excluídas por filtro de conteúdo: {len(data['excluidas'])}")
                if dataset == "locomo":
                    print("F1ofic/EMofic reproduzem task_eval/evaluation.py do LoCoMo; F1/EM são do harness."
                          if official else
                          "F1/EM do harness; não são as métricas do avaliador oficial LoCoMo.")
            print(f"Relatório: {path.with_suffix('.md')}", flush=True)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            print(f"Não foi possível imprimir {path}: {exc}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        try:
            worker(args.worker)
        except Exception:
            plan = json.loads(args.worker.read_text(encoding="utf-8"))
            Path(plan["output"]).joinpath("worker-error.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
            raise
        return 0
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
