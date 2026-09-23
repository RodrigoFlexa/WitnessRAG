"""
Configuração central do benchmark.

Toda variável tem um default que funciona sem servidor nenhum (backend `stub`,
embeddings `tfidf`), para que o pipeline inteiro possa ser testado offline antes
de gastar uma única chamada no gateway da Petrobras. O `.env` sobrescreve.

Regra de ouro deste arquivo: nada de leitura de env espalhada pelo código. Se um
parâmetro do experimento não aparece aqui, ele não é reproduzível.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """.env sem dependência externa. Não sobrescreve o que já veio do ambiente."""
    for candidate in (ROOT / ".env", Path.cwd() / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in ("1", "true", "yes", "on", "sim")


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
DATA_DIR = Path(_env("WRAG_DATA_DIR", str(ROOT / "data")))
CACHE_DIR = Path(_env("WRAG_CACHE_DIR", str(ROOT / ".cache")))
RUNS_DIR = Path(_env("WRAG_RUNS_DIR", str(ROOT / "runs")))

for _d in (DATA_DIR, CACHE_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Backend de LLM
# ---------------------------------------------------------------------------
# `azure` para o gateway corporativo, `openai` para a API pública,
# `stub` para rodar o pipeline inteiro sem rede (respostas determinísticas).
LLM_BACKEND = _env("WRAG_LLM_BACKEND", "stub")

AZURE_API_KEY_VAR = "AZURE_OPENAI_API_KEY"
AZURE_ENDPOINT_VAR = "AZURE_OPENAI_ENDPOINT"
AZURE_BASE_URL_VAR = "AZURE_OPENAI_BASE_URL"

AZURE_API_VERSION = _env("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_CA_BUNDLE = _env("AZURE_OPENAI_CA_BUNDLE", "")
AZURE_DEPLOYMENT = _env("WRAG_AZURE_DEPLOYMENT", "gpt-5-mini")
AZURE_EMBED_DEPLOYMENT = _env("WRAG_AZURE_EMBED_DEPLOYMENT", "")
AZURE_EMBED_API_VERSION = _env("WRAG_AZURE_EMBED_API_VERSION", AZURE_API_VERSION)

# Modelos de reasoning gastam orçamento pensando antes de escrever. Um teto
# pensado para uma resposta curta volta como content="" e finish_reason="length".
AZURE_REASONING_MIN_TOKENS = _env_int("WRAG_AZURE_REASONING_MIN_TOKENS", 8000)
AZURE_REASONING_EFFORT = _env("WRAG_AZURE_REASONING_EFFORT", "minimal")
AZURE_MAX_TOKENS = _env_int("WRAG_AZURE_MAX_TOKENS", 2048)
AZURE_CONCURRENCY = _env_int("WRAG_AZURE_CONCURRENCY", 8)
AZURE_MAX_RETRIES = _env_int("WRAG_AZURE_MAX_RETRIES", 6)
AZURE_BACKOFF_BASE = _env_float("WRAG_AZURE_BACKOFF_BASE", 2.0)
AZURE_BACKOFF_MAX = _env_float("WRAG_AZURE_BACKOFF_MAX", 60.0)
AZURE_TIMEOUT_S = _env_float("WRAG_AZURE_TIMEOUT_S", 600.0)

# O gateway da Petrobras costuma bloquear conteúdo por política. Bloqueio não é
# erro fatal: o item vira `filtered` e o lote segue. Ver wrag/llm/filters.py.
CONTINUE_ON_CONTENT_FILTER = _env_bool("WRAG_CONTINUE_ON_CONTENT_FILTER", True)
MAX_FILTER_RATE = _env_float("WRAG_MAX_FILTER_RATE", 0.25)
HEALTH_CHECK_CALLS = _env_int("WRAG_HEALTH_CHECK_CALLS", 50)

LLM_CACHE = _env_bool("WRAG_LLM_CACHE", True)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# `auto` decide em runtime: azure se houver deployment, senão sentence-transformers
# se o pacote e os pesos existirem, senão tfidf (sempre funciona, offline).
EMBED_BACKEND = _env("WRAG_EMBED_BACKEND", "auto")
EMBED_MODEL = _env("WRAG_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = _env_int("WRAG_EMBED_BATCH_SIZE", 64)
EMBED_DEVICE = _env("WRAG_EMBED_DEVICE", "cuda")
EMBED_CACHE = _env_bool("WRAG_EMBED_CACHE", True)


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
SEED = _env_int("WRAG_SEED", 42)
N_QUESTIONS = _env_int("WRAG_N_QUESTIONS", 100)
TOP_K = _env_int("WRAG_TOP_K", 5)
DATASETS = tuple(x for x in _env("WRAG_DATASETS", "musique,2wikimultihopqa,hotpotqa").split(",") if x)
METHODS = tuple(x for x in _env("WRAG_METHODS", "dense,bm25,graphrag,hipporag,hipporag2,relational,witnessrag").split(",") if x)


@dataclass
class IEConfig:
    """Extração. Compartilhada por GraphRAG, HippoRAG, HippoRAG2 e WITNESS-RAG.

    Isto é uma decisão experimental, não de engenharia: a proposta exige comparar
    o executor relacional com os *mesmos fatos extraídos*. Um extrator por método
    tornaria a diferença entre métodos inseparável da diferença entre extratores.
    """

    max_tokens: int = 1600
    temperature: float = 0.0
    two_step: bool = True          # NER e depois OpenIE, como no HippoRAG
    max_triples_per_passage: int = 40
    # O documento entregue ao leitor pode ser longo (2.048 tokens no protocolo
    # ZeroMem), sem obrigar o extrator a resumir tudo em apenas 40 fatos. Quando
    # positivo, OpenIE lê janelas sobrepostas e cada fato mantém como pid o
    # documento pai; isto muda o custo de indexação, não o orçamento do leitor.
    window_tokens: int = 0
    window_overlap_tokens: int = 64
    window_tokenizer: str = ""
    window_tokenizer_revision: str = ""
    # Extração adaptada a diálogo: o sujeito de uma fala em primeira pessoa é o
    # falante, e a data da sessão vira o escopo temporal do fato. Sem isso, num
    # corpus conversacional a maior parte dos fatos fica sem sujeito resolvível
    # ("I", "my kids") e nenhuma junção fecha. Muda a base F compartilhada por
    # todos os métodos com grafo, e por isso é uma condição do experimento.
    dialogue_mode: bool = False


@dataclass
class GraphConfig:
    merge_similar_entities: bool = False  # experimental: similaridade não prova identidade
    # Fusão por contenção lexical + confirmação semântica. Sem nenhuma
    # canonicalização a junção conjuntiva quebra em variações triviais de grafia
    # ("Juan Courten"/"Juan de Courten"), que é onde as cadeias morriam.
    merge_identity_variants: bool = True
    # Agrupa apenas flexões com a mesma sequência de stems; pode ser desligado
    # para medir a contribuição desta normalização mantendo a extração fixa.
    merge_relation_inflections: bool = True
    identity_merge_threshold: float = 0.80
    synonym_threshold: float = _env_float("WRAG_SYNONYM_THRESHOLD", 0.80)
    synonym_max_neighbors: int = 20
    ppr_damping: float = _env_float("WRAG_PPR_DAMPING", 0.50)
    ppr_tol: float = 1e-8
    ppr_max_iter: int = 100


@dataclass
class HippoRAG2Config:
    n_triples_retrieved: int = 20      # antes do filtro
    n_triples_kept: int = 5            # depois do filtro (top-5 do artigo)
    n_passages_seed: int = 10
    passage_node_weight: float = _env_float("WRAG_PASSAGE_NODE_WEIGHT", 0.05)
    use_recognition_memory: bool = True


@dataclass
class GraphRAGConfig:
    community_levels: int = 2
    min_community_size: int = 3
    build_community_reports: bool = True
    max_communities_reported: int = 400
    text_units_per_entity: int = 8


@dataclass
class WitnessConfig:
    """Hiperparâmetros do WITNESS-RAG.

    Os nomes seguem a notação da proposta: F é a base de fatos, W_t o conjunto de
    testemunhas de uma demanda t=(q,a), B o orçamento de memória.
    """

    # -- compilação da consulta
    max_atoms: int = 4
    compile_temperature: float = 0.0
    # Compilação ancorada no vocabulário existente: as relações e entidades mais
    # próximas da pergunta entram no prompt para que o compilador emita
    # predicados que EXISTEM na base. Sem isso o compilador inventa relações
    # ("identity", "destress method") que nenhum fato instancia, e o átomo morre
    # no aterramento. É sugestão, não restrição: o prompt continua livre.
    vocabulary_aware_compile: bool = False
    vocabulary_relations: int = 40
    vocabulary_entities: int = 20
    # Orçamento global de hipóteses distintas por pergunta. Com query_plans, a
    # primeira chamada gera no máximo duas; as restantes são criadas depois de
    # observar lacunas, rejeições e fatos adquiridos. Nada vem do gabarito.
    query_plans: bool = False
    max_query_plans: int = 5
    # Pesquisa de provas, cada componente independente para ablações pareadas.
    active_frontier: bool = False
    active_obligations: bool = False
    active_context: bool = False
    active_operators: bool = False
    soft_obligations: bool = False
    proof_reader: bool = False
    # Zero-LLM context policies.  Each has an independent switch so the paired
    # ablation identifies whether chronology or missing-facet coverage helped.
    temporal_memory: bool = False
    complementary_context: bool = False
    admit_provisional_witnesses: bool = False
    # Experimental recovery of failed plans. Kept separate from the established
    # controller so the historical baseline remains reproducible.
    plan_repair: bool = False
    # LLM calls for replanning are distinct from the budget of valid plans.
    max_replan_calls: int = 2
    conditional_verification_max_witnesses: int = 1
    active_frontier_queries: int = 5
    active_frontier_passages: int = 30

    # -- aterramento (grounding) dos átomos em fatos
    grounding_mode: str = "semantic"  # semantic: aproximação; exact: controle simbólico
    relation_match_threshold: float = 0.65
    candidates_per_atom: int = 60
    entity_match_threshold: float = 0.55
    # Quantos clusters de entidade uma constante da pergunta pode alcançar. Era 5
    # fixo no código; com milhares de entidades no corpus, o cluster certo cai
    # fora do top-5 e o átomo morre antes de qualquer teste de relação.
    entity_top_k: int = 5
    relation_weight: float = 0.45      # peso da similaridade de relação
    argument_weight: float = 0.35      # peso da similaridade de argumento constante
    verbalization_weight: float = 0.20  # peso da similaridade do átomo verbalizado

    # -- busca de testemunhas
    max_witnesses: int = 20
    beam_width: int = 400
    passage_penalty: float = 0.15      # λ: custo por passagem distinta na testemunha
    binding_aware_grounding: bool = False  # ablação: expandir vizinhos após ligar variáveis
    verify_witnesses: bool = False         # ablação: verificar no texto antes de promover
    verification_max_witnesses: int = 5    # teto de chamadas por pergunta

    # -- conjunto de respostas
    # Uma testemunha certifica UMA atribuição. A resposta de uma consulta
    # conjuntiva é o conjunto de atribuições certas, cada uma com a sua
    # proveniência. Com `answer_set`, a resposta estrutural é esse conjunto,
    # `aggregation="count"` passa a ser executável como |conjunto| e a
    # verificação cobre uma testemunha por resposta distinta, em vez das mais
    # baratas (que costumam ser todas da mesma resposta).
    answer_set: bool = False
    answer_set_max_items: int = 10

    # -- fallback de recuperação
    # Fusão recíproca de postos entre denso e BM25. Respostas conversacionais
    # dependem de nomes próprios raros que um vetor de bloco longo dilui.
    hybrid_fallback: bool = False
    hybrid_rrf_k: int = 60
    # Profundidade explorada antes do corte do contexto. O leitor continua
    # recebendo exatamente RunConfig.top_k documentos.
    candidate_pool_k: int = 20
    # One text-only search for a concrete missing join relation. No LLM call.
    gap_context_rescue: bool = False
    # Cost-bounded controller: hybrid retrieval for every question and a
    # single graph compilation only for questions labelled multi-hop. Graph
    # evidence may replace context only when a connected multi-atom join is
    # selective enough to be useful.
    selective_witness: bool = False
    selective_max_answers: int = 3
    selective_max_witnesses: int = 3
    selective_max_new_passages: int = 2
    selective_min_score: float = 0.45

    # -- proveniência e risco
    fact_confidence: float = 0.90      # p_e default de um fato extraído uma vez
    delta_interpretation: float = 0.10  # δ_interpretação da união
    delta_verbalization: float = 0.05   # δ_verbalização da união

    # -- aquisição adaptativa (VOI)
    acquisition_rounds: int = 2
    acquisition_passages: int = 3
    acquisition_lambda: float = 0.05
    enable_acquisition: bool = True
    dense_fallback: bool = True

    # -- seleção de memória sob orçamento (ILP)
    budget_fraction: float = 1.0       # 1.0 = memória cheia (sem seleção)
    ilp_time_limit_s: int = 120
    ilp_max_facts: int = 20000


@dataclass
class QAConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_k: int = TOP_K
    # Leitor ciente de conjunto: perguntas que pedem um conjunto ("o que", "quais",
    # "onde já") são respondidas com todos os itens que as passagens sustentam.
    # É o MESMO leitor para todos os métodos, então a comparação entre métodos
    # continua válida; o que muda é a comparação com rodadas anteriores.
    answer_set: bool = False
    operator_reader: bool = False
    proof_reader: bool = False
    answer_guard: bool = False
    # Render a temporal view only for temporal LoCoMo questions. The stored
    # corpus, embeddings, OpenIE and graph remain identical to the control.
    temporal_annotations: bool = False
    # One-call, operation-aware reading for conversational memory. Off by
    # default so earlier runs remain reproducible.
    evidence_reader: bool = False


@dataclass
class RunConfig:
    dataset: str = "musique"
    n_questions: int = N_QUESTIONS
    top_k: int = TOP_K
    seed: int = SEED
    subset_corpus: bool = False       # o piloto reduz perguntas, não o universo de busca
    train_questions_path: str = ""    # treino separado; nunca usar o restante da avaliação
    interleave_methods: bool = False  # todos os métodos por pergunta; útil sob prazo
    corpus_scope: str = "provided"   # origem/escopo dos arquivos; piloto pode fornecer corpus reduzido
    methods: tuple[str, ...] = DATASETS and METHODS
    ie: IEConfig = field(default_factory=IEConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    hippo2: HippoRAG2Config = field(default_factory=HippoRAG2Config)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    witness: WitnessConfig = field(default_factory=WitnessConfig)
    qa: QAConfig = field(default_factory=QAConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
