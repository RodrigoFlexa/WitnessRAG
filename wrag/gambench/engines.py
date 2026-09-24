"""
Quem escolhe as páginas: a parte do pipeline que muda entre métodos.

Cada contexto (uma amostra do HotpotQA ou do RULER, um livro do NarrativeQA)
vira uma memória isolada com as páginas do protocolo. O método devolve até
`top_k` páginas, na ordem em que as entrega; o leitor do protocolo responde
com elas. Nenhum método vê as respostas: a pergunta passada adiante tem a
lista de respostas vazia e nenhum rótulo de tarefa.

Motores:

* ``rag``: busca densa BGE-M3, top-5. É a linha RAG do artigo do GAM
  ("uniformly partitioned into segments of 2,048 tokens, and the top-5
  retrieved segments are used"). Serve para calibrar o harness contra a
  Tabela 1(b). O código do GAM codifica páginas com max_length=512
  (FlagEmbedding); com WRAG_EMBED_MAX_SEQ_LENGTH=512 isso é reproduzido.
* ``hybrid``: o BUSCAR do WitnessRAG sozinho (BGE-M3 + BM25 por RRF).
* ``witnessrag-lite``: BUSCAR + política de contexto sem LLM.
* ``witnessrag``: o método completo (desenho v3): REGISTRAR sobre todas as
  páginas (NER + OpenIE em janelas de 512 tokens, grafo), depois
  BUSCAR -> PLANEJAR -> PROVAR -> VERIFICAR, dois ciclos. É a configuração
  registrada do LoCoMo (scripts/run-witness-proof-locomo.sh) sem as opções
  que só existem para conversa (extração de diálogo, anotações de data e o
  leitor de evidências: aqui o leitor é o do protocolo).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.embed import Embedder
from wrag.gambench import protocol as P
from wrag.llm import LLM
from wrag.llm.base import usage_delta

ENGINES = ("rag", "hybrid", "witnessrag-lite", "witnessrag")
_REGISTRY_NAME = {"rag": "dense", "hybrid": "hybrid", "witnessrag-lite": "witnessrag-lite",
                  "witnessrag": "witnessrag"}
# Janela de extração do REGISTRAR, a mesma da rodada registrada do LoCoMo.
IE_WINDOW_TOKENS = 512
IE_WINDOW_OVERLAP = 64
IE_WINDOW_TOKENIZER = "Qwen/Qwen2.5-14B-Instruct"


def engine_config(engine: str, top_k: int = P.TOP_K, ie_tokenizer: str = IE_WINDOW_TOKENIZER,
                  ie_tokenizer_revision: str = "") -> C.RunConfig:
    if engine not in ENGINES:
        raise ValueError(f"motor desconhecido: {engine}; opções: {', '.join(ENGINES)}")
    cfg = C.RunConfig(n_questions=1, top_k=top_k, methods=(_REGISTRY_NAME[engine],),
                      corpus_scope="isolated_context_per_sample")
    cfg.qa.top_k = top_k
    witness = cfg.witness
    witness.candidate_pool_k = 20
    if engine == "witnessrag-lite":
        witness.hybrid_fallback = True
        witness.complementary_context = True
    elif engine == "witnessrag":
        witness.binding_aware_grounding = True
        witness.vocabulary_aware_compile = True
        witness.hybrid_fallback = True
        witness.gap_context_rescue = True
        witness.answer_set = True
        witness.proof_controller = True
        witness.proof_cycles = 2
        witness.proof_verify = True
        witness.proof_partial_evidence = False
        cfg.ie.dialogue_mode = False
        cfg.ie.window_tokens = IE_WINDOW_TOKENS
        cfg.ie.window_overlap_tokens = IE_WINDOW_OVERLAP
        cfg.ie.window_tokenizer = ie_tokenizer
        cfg.ie.window_tokenizer_revision = ie_tokenizer_revision
    return cfg


@dataclass
class Selection:
    pages: list[int]                       # índices das páginas, na ordem entregue
    diagnostics: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    index_s: float = 0.0
    retrieve_s: float = 0.0
    reused_index: bool = False
    filtered: bool = False


class Engine:
    """Constrói a memória de um contexto e escolhe páginas para cada pergunta.

    A última memória construída fica em uso enquanto as perguntas seguintes
    tiverem a mesma `context_key` (NarrativeQA: várias perguntas por livro);
    é só economia, porque a recuperação não guarda estado entre perguntas.
    """

    def __init__(self, engine: str, llm: LLM, embedder: Embedder, cfg: C.RunConfig) -> None:
        self.name = engine
        self.llm = llm
        self.embedder = embedder
        self.cfg = cfg
        self._key: str | None = None
        self._retriever = None
        self._corpus: Corpus | None = None
        self._index_report: dict[str, Any] = {}

    def _build(self, benchmark: str, context_key: str, pages: list[str]) -> float:
        from wrag.methods import build_context, build_methods

        started = time.perf_counter()
        passages = [Passage(pid=f"{context_key}#p{i}", title="", text=text, sequence=i,
                            source_ids=(context_key,)) for i, text in enumerate(pages)]
        corpus = Corpus(f"gam-{benchmark}-{context_key}", passages, [])
        ctx = build_context(corpus, self.cfg, llm=self.llm, embedder=self.embedder)
        registry = _REGISTRY_NAME[self.name]
        self._retriever = build_methods(ctx, [registry])[registry]
        self._corpus = corpus
        self._key = context_key
        try:
            self._index_report = self._retriever.index_report()
        except Exception:  # noqa: BLE001 - relatório de índice é só diagnóstico
            self._index_report = {}
        if ctx.extraction is not None:
            self._index_report["extracao"] = ctx.extraction.stats()
        return time.perf_counter() - started

    def select(self, benchmark: str, sid: str, question_text: str, context_key: str,
               pages: list[str]) -> Selection:
        before = self.llm.usage.snapshot()
        reused = self._key == context_key and self._retriever is not None
        index_s = 0.0 if reused else self._build(benchmark, context_key, pages)
        assert self._retriever is not None and self._corpus is not None
        # Sem respostas, sem rótulo de tarefa: só o texto da pergunta.
        question = Question(qid=sid, question=question_text, answers=[],
                            dataset=f"gam-{benchmark}", qtype="")
        started = time.perf_counter()
        result = self._retriever.retrieve(question, self.cfg.top_k)
        retrieve_s = time.perf_counter() - started
        order = {p.pid: p.sequence for p in self._corpus.passages}
        chosen: list[int] = []
        for pid in result.pids:
            if pid in order and order[pid] not in chosen:
                chosen.append(order[pid])
            if len(chosen) >= self.cfg.top_k:
                break
        diagnostics = dict(result.diagnostics or {})
        if not reused and self._index_report:
            diagnostics["indice"] = self._index_report
        return Selection(pages=chosen, diagnostics=diagnostics,
                         usage=usage_delta(self.llm.usage.snapshot(), before),
                         index_s=index_s, retrieve_s=retrieve_s, reused_index=reused,
                         filtered=bool(result.filtered))
