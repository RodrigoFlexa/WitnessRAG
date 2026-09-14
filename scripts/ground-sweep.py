#!/usr/bin/env python3
"""Reexecuta o aterramento das consultas já compiladas, variando parâmetros.

As consultas compiladas ficam gravadas em <run>/<dataset>/witnessrag.jsonl e os
fatos no cache de OpenIE, então dá para varrer o espaço de parâmetros da busca
sem LLM e sem GPU. Mede o que importa: quantas provas fecham.

    scripts/ground-sweep.py runs/qwen14b-pilot
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/qwen14b-pilot").resolve()

# A chave do cache da OpenIE inclui a identidade do provedor, que por sua vez
# hasheia OPENAI_BASE_URL. Reusar o env gravado pelo piloto é o que faz o cache
# bater e o harness rodar sem LLM nenhum.
_plan = json.loads((RUN / "pilot.json").read_text(encoding="utf-8"))
os.environ.update(_plan["env"])
os.environ["WRAG_EMBED_DEVICE"] = os.environ.get("SWEEP_EMBED_DEVICE", "cpu")
os.environ["WRAG_NO_PROGRESS"] = "1"

from wrag import config as C                                  # noqa: E402
from wrag.data import load_dataset                            # noqa: E402
from wrag.embed import get_embedder                           # noqa: E402
from wrag.graph import build_graph                            # noqa: E402
from wrag.ie import extract_corpus                            # noqa: E402
from wrag.llm import get_llm                                  # noqa: E402
from wrag.eval.metrics import exact_match                      # noqa: E402
from wrag.witness.memory import MemoryView                    # noqa: E402
from wrag.witness.provenance import rank_passages, score_answers  # noqa: E402
from wrag.witness.query import Atom, ConjunctiveQuery         # noqa: E402
from wrag.witness.search import WitnessSearcher               # noqa: E402


def load_queries(run: Path):
    bench = sorted((run / "benchmark").glob("*/"))[-1]
    out = []
    for ds in sorted(p for p in bench.iterdir() if p.is_dir() and p.name != "figures"):
        f = ds / "witnessrag.jsonl"
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            q = (r.get("diagnosticos") or {}).get("consulta") or {}
            if q.get("atoms"):
                out.append((ds.name, r, q))
    return out


def rebuild(query: dict) -> ConjunctiveQuery:
    return ConjunctiveQuery(
        answer_var=query.get("answer_var", "x"),
        atoms=[Atom(relation=a["relation"], subject=a["subject"], object=a["object"])
               for a in query["atoms"]],
        expected_type=query.get("expected_type", "other"),
        aggregation=query.get("aggregation", "none"),
        source="replay",
    )


def main() -> int:
    items = load_queries(RUN)
    if not items:
        print("nenhuma consulta compilada encontrada")
        return 1
    dataset = items[0][0]
    print(f"{len(items)} consultas compiladas de {dataset}")

    # O piloto já gravou o corpus reduzido (candidatas + distratores) em data/;
    # a chave do cache da OpenIE é o conteúdo exato dessas passagens.
    corpus = load_dataset(dataset, n_questions=100, seed=42,
                          data_dir=RUN / "data", subset_corpus=False)
    print(f"corpus: {len(corpus.passages)} passagens, {len(corpus.questions)} perguntas")
    extraction = extract_corpus(corpus, get_llm(), C.IEConfig())
    embedder = get_embedder()
    kg = build_graph(corpus, extraction, embedder, C.GraphConfig())
    print(f"grafo: {len(kg.facts)} fatos, {len(kg.entities)} entidades, {len(kg.relations)} relações\n")

    grids = [
        ("hoje (baseline)", dict()),
        ("entidade top=25", dict(entity_top_k=25)),
        ("relação 0.50", dict(relation_match_threshold=0.50)),
        ("entidade top=25 + relação 0.50", dict(entity_top_k=25, relation_match_threshold=0.50)),
        ("+ entidade 0.45", dict(entity_top_k=25, relation_match_threshold=0.50,
                                 entity_match_threshold=0.45)),
        ("+ candidatos/átomo 200", dict(entity_top_k=25, relation_match_threshold=0.50,
                                        entity_match_threshold=0.45, candidates_per_atom=200)),
    ]
    # Fechar mais provas não é o objetivo: uma prova pode fechar sobre um
    # aterramento errado. O que vale é resposta estrutural correta e testemunha
    # que cubra as passagens de apoio. Afrouxar limiar sem olhar isto é trapaça.
    n = len(items)
    print(f"{'configuração':<32} {'provas':>8} {'EM estrut.':>11} {'precisão':>9} {'R@5 test.':>10}")
    print("-" * 74)
    for label, over in grids:
        cfg = C.WitnessConfig()
        for key, value in over.items():
            if not hasattr(cfg, key):
                print(f"  (aviso: {key} não existe em WitnessConfig; ignorado)")
                continue
            setattr(cfg, key, value)
        memory = MemoryView(kg)
        searcher = WitnessSearcher(memory, embedder, cfg)
        done = correct = 0
        recall = 0.0
        for _, rec, raw in items:
            res = searcher.join(rebuild(raw))
            if not res.complete:
                continue
            done += 1
            cands = score_answers(res.witnesses, memory, cfg)
            if not cands:
                continue
            correct += int(exact_match(cands[0].answer, rec.get("respostas_ouro") or []) > 0)
            pids, _ = rank_passages(cands, 5)
            gold = set(rec.get("passagens_ouro") or [])
            if gold:
                recall += len(set(pids) & gold) / len(gold)
        prec = 100 * correct / done if done else 0.0
        print(f"{label:<32} {done:>4}/{n:<3} {100*correct/n:>10.1f}% {prec:>8.1f}% {100*recall/n:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
