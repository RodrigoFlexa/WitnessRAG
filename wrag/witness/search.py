"""
Busca de testemunhas para consultas conjuntivas positivas sobre fatos dirigidos.

No modo exact, cada testemunha instancia todos os átomos com uma atribuição
consistente de variáveis. A equivalência entre respostas e testemunhas contidas
na memória vale sob essa semântica fixa. Completude exige ausência de cortes de
candidatos, feixe e saída (zero desativa cada limite). Os cortes são registrados.
O modo semantic é aproximado: similaridade não demonstra equivalência lógica,
não autoriza inverter relações e não é interpretada como probabilidade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from wrag import config as C
from wrag.embed import Embedder, cosine_topk
from wrag.graph import KnowledgeGraph
from wrag.ie import Fact
from wrag.util import get_logger, canonical_symbol as normalize
from wrag.witness.query import Atom, ConjunctiveQuery, is_var, var_name

log = get_logger("wrag.witness.search")

REVERSE_PENALTY = 0.85
EPS = 1e-6


@dataclass
class Grounding:
    """Um candidato de aterramento: este fato pode instanciar este átomo."""

    fact_index: int
    score: float
    reversed: bool = False


@dataclass
class Witness:
    """Uma testemunha: conjunto de fatos que, juntos, demonstram uma resposta."""

    facts: tuple[int, ...]
    bindings: dict[str, str]
    score: float                      # ∏ scores de casamento dos átomos
    cost: float                       # -Σ log score + λ·|passagens distintas|
    pids: tuple[str, ...]
    answer: str = ""

    def to_dict(self, kg: KnowledgeGraph) -> dict[str, Any]:
        return {
            "resposta": self.answer,
            "score": round(self.score, 4),
            "custo": round(self.cost, 4),
            "passagens": list(self.pids),
            "fatos": [list(kg.facts[i].triple) for i in self.facts],
        }


@dataclass
class Gap:
    """O átomo que a busca não conseguiu aterrar, com o que já estava ligado.

    É a entrada da aquisição adaptativa: saber QUAL relação falta e sobre QUAL
    entidade é o que transforma "não achei" em uma ação de leitura dirigida.
    """

    atom_index: int
    atom: Atom
    bound_subject: str = ""
    bound_object: str = ""
    depth_reached: int = 0

    def anchor(self) -> str:
        return self.bound_subject or self.bound_object or ""

    def probe(self) -> str:
        """Consulta textual para achar a passagem que provavelmente fecha o buraco."""
        return " ".join(x for x in (self.bound_subject, self.atom.relation, self.bound_object) if x)

    def to_dict(self) -> dict[str, Any]:
        return {"atomo": self.atom.to_dict(), "sujeito_ligado": self.bound_subject,
                "objeto_ligado": self.bound_object, "profundidade": self.depth_reached}


@dataclass
class SearchResult:
    witnesses: list[Witness] = field(default_factory=list)
    gap: Gap | None = None
    n_candidates: list[int] = field(default_factory=list)
    depth_reached: int = 0
    exhaustive: bool = False
    truncations: list[str] = field(default_factory=list)
    grounding_mode: str = "exact"

    @property
    def complete(self) -> bool:
        return bool(self.witnesses)


@dataclass
class _State:
    clusters: dict[str, int]
    surfaces: dict[str, str]
    facts: tuple[int, ...]
    log_score: float
    pids: frozenset[str]

    def cost(self, penalty: float) -> float:
        return -self.log_score + penalty * len(self.pids)


class WitnessSearcher:
    """Aterra e junta. Não conhece LLM: a aquisição adaptativa mora no retriever."""

    def __init__(self, kg: Any, embedder: Embedder, cfg: C.WitnessConfig | None = None) -> None:
        # `kg` é um KnowledgeGraph ou uma MemoryView com a mesma interface.
        self.kg = kg
        self.embedder = embedder
        self.cfg = cfg or C.WitnessConfig()
        if self.cfg.grounding_mode not in {"exact", "semantic"}:
            raise ValueError("grounding_mode deve ser exact ou semantic")
        if min(self.cfg.beam_width, self.cfg.candidates_per_atom, self.cfg.max_witnesses) < 0:
            raise ValueError("limites devem ser >= 0; zero significa sem corte")
        self._ground_truncated = False
        # Fatos indexados por CLUSTER, não por nó: a variável compartilhada de
        # uma consulta conjuntiva casa por identidade de entidade, e duas grafias
        # da mesma entidade precisam casar ou a testemunha nunca fecha.
        self._facts_by_cluster_subject: dict[int, list[int]] = {}
        self._facts_by_cluster_object: dict[int, list[int]] = {}
        self._journal: list[tuple[dict, int, int]] = []
        # Memória selecionada sob orçamento: quando definido, S ⊂ F e a busca só
        # enxerga o que foi preservado. É assim que a curva de qualidade por
        # unidade de armazenamento é medida — sem isso, o ILP seria decorativo.
        self.allowed: set[int] | None = None
        self._reactivated: set[int] = set()
        self._allowed_horizon: int = len(kg.facts)
        for i, fact in enumerate(kg.facts):
            if fact.subj_id >= 0:
                self._facts_by_cluster_subject.setdefault(self._identity(fact.subj_id), []).append(i)
            if fact.obj_id >= 0:
                self._facts_by_cluster_object.setdefault(self._identity(fact.obj_id), []).append(i)

    def _identity(self, eid: int) -> int:
        return eid if self.cfg.grounding_mode == "exact" else self.kg.cluster(eid)

    def _permitted(self, fid: int) -> bool:
        return (self.allowed is None or fid in self.allowed or fid in self._reactivated
                or fid >= self._allowed_horizon)

    # -- reindexação incremental reversível (usada pela aquisição) ----------

    def register_facts(self, indices: Iterable[int]) -> None:
        """Indexa fatos acrescentados à memória depois da construção."""
        kg = self.kg
        for i in indices:
            fact = kg.facts[i]
            for eid, table in ((fact.subj_id, self._facts_by_cluster_subject),
                               (fact.obj_id, self._facts_by_cluster_object)):
                if eid >= 0:
                    cluster = self._identity(eid)
                    bucket = table.setdefault(cluster, [])
                    self._journal.append((table, cluster, len(bucket)))
                    bucket.append(i)

    def rollback(self) -> None:
        """Desfaz a indexação incremental, em par com `MemoryView.reset()`."""
        self._reactivated.clear()
        for table, key, length in reversed(self._journal):
            bucket = table.get(key)
            if bucket is not None:
                del bucket[length:]
                if not bucket:
                    table.pop(key, None)
        self._journal.clear()

    # -- aterramento --------------------------------------------------------

    def match_entity(self, surface: str, top: int | None = None) -> list[tuple[int, float]]:
        """Clusters candidatos para uma constante. Casamento exato ganha 1.0."""
        kg = self.kg
        top = self.cfg.entity_top_k if top is None else top
        exact = kg.entity_id(surface)
        out: list[tuple[int, float]] = []
        if exact is not None:
            out.append((kg.cluster(exact), 1.0))
        if kg.entity_vectors.size:
            vector = self.embedder.encode([surface])[0]
            idx, sims = cosine_topk(vector, kg.entity_vectors, top)
            for eid, sim in zip(idx, sims):
                if float(sim) < self.cfg.entity_match_threshold:
                    continue
                cluster = kg.cluster(int(eid))
                if all(cluster != c for c, _ in out):
                    out.append((cluster, float(sim)))
        return out

    def ground(self, query: ConjunctiveQuery) -> list[list[Grounding]]:
        kg = self.kg
        cfg = self.cfg
        self._ground_truncated = False
        if not query.atoms or not kg.facts:
            return [[] for _ in query.atoms]

        if cfg.grounding_mode == "exact":
            out = []
            for atom in query.atoms:
                candidates = []
                for fid, fact in enumerate(kg.facts):
                    if not self._permitted(fid) or normalize(atom.relation) != normalize(fact.relation):
                        continue
                    if not atom.subject_is_var and normalize(atom.subject) != normalize(fact.subject):
                        continue
                    if not atom.object_is_var and normalize(atom.object) != normalize(fact.object):
                        continue
                    candidates.append(Grounding(fid, 1.0))
                if cfg.candidates_per_atom and len(candidates) > cfg.candidates_per_atom:
                    self._ground_truncated = True
                    candidates = candidates[:cfg.candidates_per_atom]
                out.append(candidates)
            return out

        # Um lote de embeddings por consulta: relações, verbalizações e
        # constantes de uma vez. Cada chamada extra aqui é latência online.
        relations = [a.relation for a in query.atoms]
        verbalizations = [a.verbalize() for a in query.atoms]
        rel_vectors = self.embedder.encode(relations)
        verb_vectors = self.embedder.encode(verbalizations)

        constant_clusters: dict[str, list[tuple[int, float]]] = {}
        for constant in query.constants():
            constant_clusters[constant] = self.match_entity(constant)

        out: list[list[Grounding]] = []
        for i, atom in enumerate(query.atoms):
            out.append(self._ground_atom(atom, rel_vectors[i], verb_vectors[i], constant_clusters))
        return out

    def _ground_atom(self, atom: Atom, rel_vector: np.ndarray, verb_vector: np.ndarray,
                     constant_clusters: dict[str, list[tuple[int, float]]]) -> list[Grounding]:
        kg, cfg = self.kg, self.cfg

        forward: set[int] = set()
        backward: set[int] = set()

        def pool_for(constant: str, as_subject: bool) -> None:
            for cluster, _sim in constant_clusters.get(constant, []):
                direct = (self._facts_by_cluster_subject if as_subject else self._facts_by_cluster_object)
                inverse = (self._facts_by_cluster_object if as_subject else self._facts_by_cluster_subject)
                forward.update(direct.get(cluster, []))
                # Uma relação inversa deve ser outro predicado explícito.

        if not atom.subject_is_var:
            pool_for(atom.subject, as_subject=True)
        if not atom.object_is_var:
            pool_for(atom.object, as_subject=False)

        if not forward and not backward:
            # Átomo sem constante (ou constante que não casou com nó nenhum):
            # o único acesso é o índice denso sobre a verbalização dos fatos.
            if kg.fact_vectors.size == 0:
                return []
            limit = cfg.candidates_per_atom * 2 if cfg.candidates_per_atom else len(kg.facts)
            idx, _ = cosine_topk(verb_vector, kg.fact_vectors, limit)
            self._ground_truncated |= len(idx) < len(kg.facts)
            forward = {int(i) for i in idx}

        if self.allowed is not None:
            # Fatos adquiridos em tempo de consulta (índice ≥ tamanho da memória
            # indexada) não passam pelo orçamento: eles são custo de consulta, e
            # o orçamento restringe armazenamento.
            forward = {i for i in forward if self._permitted(i)}
            backward = {i for i in backward if self._permitted(i)}

        scored: list[Grounding] = []
        for fid in forward | backward:
            fact = kg.facts[fid]
            base = self._score_fact(atom, fact, fid, rel_vector, verb_vector, constant_clusters,
                                    reversed_=False) if fid in forward else 0.0
            rev = self._score_fact(atom, fact, fid, rel_vector, verb_vector, constant_clusters,
                                   reversed_=True) * REVERSE_PENALTY if fid in backward else 0.0
            if base >= rev and base > 0:
                scored.append(Grounding(fid, base, reversed=False))
            elif rev > 0:
                scored.append(Grounding(fid, rev, reversed=True))

        scored.sort(key=lambda g: -g.score)
        if cfg.candidates_per_atom and len(scored) > cfg.candidates_per_atom:
            self._ground_truncated = True
            return scored[:cfg.candidates_per_atom]
        return scored

    def _score_fact(self, atom: Atom, fact: Fact, fid: int, rel_vector: np.ndarray,
                    verb_vector: np.ndarray,
                    constant_clusters: dict[str, list[tuple[int, float]]],
                    reversed_: bool) -> float:
        kg, cfg = self.kg, self.cfg

        rel_sim = 0.0
        if fact.rel_id >= 0 and kg.relation_vectors.size:
            rel_sim = float(np.dot(rel_vector, kg.relation_vectors[fact.rel_id]))
        if normalize(atom.relation) == normalize(fact.relation):
            rel_sim = 1.0
        if rel_sim < cfg.relation_match_threshold:
            return 0.0
        verb_sim = float(np.dot(verb_vector, kg.fact_vectors[fid])) if kg.fact_vectors.size else 0.0

        subject_eid = fact.obj_id if reversed_ else fact.subj_id
        object_eid = fact.subj_id if reversed_ else fact.obj_id

        arg_sims: list[float] = []
        for term, eid in ((atom.subject, subject_eid), (atom.object, object_eid)):
            if is_var(term):
                continue
            if eid < 0:
                return 0.0
            cluster = kg.cluster(eid)
            match = next((s for c, s in constant_clusters.get(term, []) if c == cluster), None)
            if match is None:
                return 0.0   # a constante não casa com este lado do fato
            arg_sims.append(match)
        arg_sim = float(np.mean(arg_sims)) if arg_sims else 0.0

        weights = (cfg.relation_weight, cfg.argument_weight if arg_sims else 0.0,
                   cfg.verbalization_weight)
        total = sum(weights) or 1.0
        score = (cfg.relation_weight * _clip(rel_sim)
                 + (cfg.argument_weight * arg_sim if arg_sims else 0.0)
                 + cfg.verbalization_weight * _clip(verb_sim)) / total
        return float(max(0.0, min(1.0, score)))

    # -- junção -------------------------------------------------------------

    def _bound_groundings(self, atom: Atom, state: _State,
                          vectors: tuple[np.ndarray, np.ndarray],
                          constants: dict[str, list[tuple[int, float]]]) -> list[Grounding] | None:
        """Aplica a ligação ANTES do top-k; não refaz identidade por cosseno.

        None significa que o estado ainda não liga variável deste átomo.
        Uma lista vazia significa que a vizinhança ligada não tem candidato.
        A pontuação mantém o átomo original para não contar novamente a
        confiança de uma entidade intermediária já escolhida.
        """
        pools = []
        for term, table in ((atom.subject, self._facts_by_cluster_subject),
                            (atom.object, self._facts_by_cluster_object)):
            if is_var(term) and var_name(term) in state.clusters:
                pools.append(set(table.get(state.clusters[var_name(term)], [])))
        if not pools:
            return None
        pool = set.intersection(*pools)
        scored = []
        for fid in sorted(pool):
            if not self._permitted(fid):
                continue
            score = self._score_fact(atom, self.kg.facts[fid], fid,
                                     vectors[0], vectors[1], constants, False)
            grounding = Grounding(fid, score)
            # Restrições repetidas (?x R ?x) também precedem o corte.
            if score > 0 and self._extend(state, atom, grounding) is not None:
                scored.append(grounding)
        scored.sort(key=lambda g: (-g.score, g.fact_index))
        limit = self.cfg.candidates_per_atom
        if limit and len(scored) > limit:
            self._ground_truncated = True
            scored = scored[:limit]
        return scored

    def join(self, query: ConjunctiveQuery,
             groundings: Sequence[Sequence[Grounding]] | None = None) -> SearchResult:
        cfg = self.cfg
        kg = self.kg
        if not query.atoms or query.answer_var not in query.variables() or query.aggregation != "none":
            return SearchResult()

        external_groundings = groundings is not None
        groundings = list(groundings) if external_groundings else self.ground(query)
        if len(groundings) != len(query.atoms):
            raise ValueError("é necessária uma lista de candidatos por átomo")
        n_candidates = [len(g) for g in groundings]

        # Ordem: mais constantes primeiro (menos ramificação), depois menos
        # candidatos. Para uma cadeia ancorada isto é a caminhada por camadas do
        # produto G × A_q; para interseção, é uma junção por variável.
        order = sorted(range(len(query.atoms)),
                       key=lambda i: (-query.atoms[i].n_constants, n_candidates[i]))
        adaptive = (cfg.binding_aware_grounding and cfg.grounding_mode == "semantic"
                    and not external_groundings)
        if adaptive:
            rel_vectors = self.embedder.encode([a.relation for a in query.atoms])
            verb_vectors = self.embedder.encode([a.verbalize() for a in query.atoms])
            constants = {c: self.match_entity(c) for c in query.constants()}
        bound_cache: dict[tuple, list[Grounding] | None] = {}

        states = [_State({}, {}, (), 0.0, frozenset())]
        depth_reached = 0
        gap: Gap | None = None
        truncations = ["candidatos_fornecidos"] if external_groundings else []
        if not external_groundings and self._ground_truncated:
            truncations.append("candidatos")
        exhaustive = cfg.grounding_mode == "exact" and not truncations

        remaining = list(order)
        for depth in range(len(order)):
            # Seguir a fronteira ligada evita avaliar uma relação genérica
            # desconectada antes de visitar o elo que a torna seletiva.
            if adaptive:
                bound = set(states[0].clusters) if states else set()
                ai = min(remaining, key=lambda i: (
                    -sum(is_var(t) and var_name(t) in bound
                         for t in (query.atoms[i].subject, query.atoms[i].object)),
                    -query.atoms[i].n_constants, n_candidates[i], i))
            else:
                ai = remaining[0]
            remaining.remove(ai)
            atom = query.atoms[ai]
            candidates = groundings[ai]
            nxt: list[_State] = []
            seen: dict[tuple, _State] = {}
            for state in states:
                local = candidates
                if adaptive:
                    key = (ai, tuple((var_name(t), state.clusters.get(var_name(t)))
                                     for t in (atom.subject, atom.object) if is_var(t)))
                    if key not in bound_cache:
                        bound_cache[key] = self._bound_groundings(
                            atom, state, (rel_vectors[ai], verb_vectors[ai]), constants)
                    conditioned = bound_cache[key]
                    if conditioned is not None:
                        local = conditioned
                for grounding in local:
                    extended = self._extend(state, atom, grounding)
                    if extended is None:
                        continue
                    key = (tuple(sorted(set(extended.facts))), tuple(sorted(extended.clusters.items())))
                    previous = seen.get(key)
                    if previous is None or extended.log_score > previous.log_score:
                        seen[key] = extended
            nxt = list(seen.values())
            if adaptive and self._ground_truncated and "candidatos" not in truncations:
                truncations.append("candidatos")

            if not nxt:
                gap = self._gap(ai, atom, states, depth)
                break

            nxt.sort(key=lambda s: s.cost(cfg.passage_penalty))
            if cfg.beam_width and len(nxt) > cfg.beam_width:
                exhaustive = False
                if "feixe" not in truncations:
                    truncations.append("feixe")
                nxt = nxt[: cfg.beam_width]
            states = nxt
            depth_reached = depth + 1

        if depth_reached < len(query.atoms):
            return SearchResult(witnesses=[], gap=gap, n_candidates=n_candidates,
                                depth_reached=depth_reached, exhaustive=exhaustive,
                                truncations=truncations, grounding_mode=cfg.grounding_mode)

        witnesses = self._to_witnesses(query, states)
        if cfg.max_witnesses and len(witnesses) > cfg.max_witnesses:
            witnesses = witnesses[:cfg.max_witnesses]
            exhaustive = False
            truncations.append("testemunhas")
        return SearchResult(witnesses=witnesses, gap=None, n_candidates=n_candidates,
                            depth_reached=depth_reached, exhaustive=exhaustive,
                            truncations=truncations, grounding_mode=cfg.grounding_mode)

    def _extend(self, state: _State, atom: Atom, grounding: Grounding) -> _State | None:
        kg = self.kg
        fact = kg.facts[grounding.fact_index]
        if not self._permitted(grounding.fact_index):
            return None
        if self.cfg.grounding_mode == "exact":
            if grounding.reversed or normalize(atom.relation) != normalize(fact.relation):
                return None
            if not atom.subject_is_var and normalize(atom.subject) != normalize(fact.subject):
                return None
            if not atom.object_is_var and normalize(atom.object) != normalize(fact.object):
                return None
        subject_eid = fact.obj_id if grounding.reversed else fact.subj_id
        object_eid = fact.subj_id if grounding.reversed else fact.obj_id

        clusters = dict(state.clusters)
        surfaces = dict(state.surfaces)
        for term, eid in ((atom.subject, subject_eid), (atom.object, object_eid)):
            if not is_var(term):
                continue
            if eid < 0:
                return None
            name = var_name(term)
            cluster = self._identity(eid)
            if name in clusters and clusters[name] != cluster:
                return None      # a variável compartilhada não bate: sem testemunha
            clusters[name] = cluster
            surfaces.setdefault(name, kg.entities[eid])

        return _State(
            clusters=clusters,
            surfaces=surfaces,
            facts=tuple(sorted(set(state.facts) | {grounding.fact_index})),
            log_score=state.log_score + math.log(max(grounding.score, EPS)),
            pids=state.pids | {fact.pid},
        )

    def _gap(self, atom_index: int, atom: Atom, states: Sequence[_State], depth: int) -> Gap:
        best = min(states, key=lambda s: s.cost(self.cfg.passage_penalty)) if states else None
        bound_subject = atom.subject if not atom.subject_is_var else ""
        bound_object = atom.object if not atom.object_is_var else ""
        if best is not None:
            if atom.subject_is_var:
                bound_subject = best.surfaces.get(var_name(atom.subject), "")
            if atom.object_is_var:
                bound_object = best.surfaces.get(var_name(atom.object), "")
        return Gap(atom_index=atom_index, atom=atom, bound_subject=bound_subject,
                   bound_object=bound_object, depth_reached=depth)

    def _to_witnesses(self, query: ConjunctiveQuery, states: Sequence[_State]) -> list[Witness]:
        cfg = self.cfg
        out: list[Witness] = []
        for state in states:
            answer = state.surfaces.get(query.answer_var, "")
            if not answer:
                continue
            out.append(Witness(
                facts=tuple(sorted(state.facts)),
                bindings=dict(state.surfaces),
                score=float(math.exp(state.log_score)),
                cost=state.cost(cfg.passage_penalty),
                pids=tuple(sorted(state.pids)),
                answer=answer,
            ))
        out.sort(key=lambda w: w.cost)
        return _minimal_only(out)


def _minimal_only(witnesses: list[Witness]) -> list[Witness]:
    """Mantém só testemunhas mínimas por inclusão, dentro de cada resposta.

    A definição da proposta é W ∈ W_{q,a} sse W ⊨ q(a) e W é mínimo por inclusão.
    Um conjunto de fatos que contém outro conjunto suficiente não é uma
    testemunha mínima nova; a proveniência retém apenas os conjuntos mínimos.
    """
    by_answer: dict[str, list[Witness]] = {}
    for witness in witnesses:                     # já vêm ordenadas por custo
        key = normalize(witness.answer)
        facts = set(witness.facts)
        group = by_answer.setdefault(key, [])
        if any(set(other.facts) <= facts for other in group):
            continue                              # existe demonstração menor
        group[:] = [o for o in group if not facts < set(o.facts)] + [witness]
    kept = [w for group in by_answer.values() for w in group]
    kept.sort(key=lambda w: w.cost)
    return kept


def _clip(x: float) -> float:
    return float(max(0.0, min(1.0, x)))
