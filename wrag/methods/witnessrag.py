"""
WITNESS-RAG: recuperação orientada a testemunhas de consultas conjuntivas.

Compila a pergunta, executa junções dirigidas e seleciona passagens contendo
testemunhas completas. Semantic é o padrão aproximado; exact é o controle simbólico.
Aquisição dirigida usa uma heurística de similaridade menos custo, ainda sem
um modelo probabilístico de valor da informação. A proveniência é auditável;
os scores de confiança não constituem certificados de risco.
A seleção offline otimiza demandas de treino separado ou sintetizadas sobre
o grafo. O orçamento limita fatos visíveis, sem reduzir a RAM física do índice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wrag import config as C
from wrag.data import Question
from wrag.embed import cosine_topk
from wrag.ie import Fact, extract_targeted
from wrag.llm.filters import LEDGER
from wrag.methods.base import IndexContext, RetrievalResult, Retriever, pad_with_dense
from wrag.methods.dense import DenseRetriever, HybridRetriever
from wrag.util import get_logger, canonical_symbol
from wrag.witness import budget as budget_mod
from wrag.witness.memory import MemoryView
from wrag.witness.provenance import (AnswerCandidate, answer_set, rank_passages,
                                     score_answers)
from wrag.witness.query import ConjunctiveQuery, compile_query, compile_query_plans
from wrag.witness.search import (Gap, SearchResult, WitnessSearcher, cover_answers,
                                 executable_aggregations)

log = get_logger("wrag.methods.witnessrag")


def preserve_fallback_order_if_same_set(
    pids: list[str], scores: list[float], fallback_pids: list[str],
    fallback_scores: list[float], k: int,
) -> tuple[list[str], list[float], bool]:
    """Avoid changing the reader prompt when proof found no new document.

    A structural ranking is useful when it promotes evidence missing from the
    fallback top-k. If both rankings contain exactly the same documents,
    reordering only perturbs an order-sensitive reader and cannot improve
    retrieval recall. Preserve the established fallback order in that case.
    """
    fallback_pids = fallback_pids[:k]
    fallback_scores = fallback_scores[:k]
    if len(pids) == len(fallback_pids) and set(pids) == set(fallback_pids):
        return list(fallback_pids), list(fallback_scores), pids != fallback_pids
    return pids, scores, False


@dataclass
class AcquisitionAction:
    """Uma ação candidata: reler uma passagem procurando uma relação específica."""

    pid: str
    title: str
    text: str
    relation: str
    anchor: str
    expected_gain: float
    cost: float

    @property
    def voi(self) -> float:
        return self.expected_gain - self.cost

    def to_dict(self) -> dict[str, Any]:
        return {"passagem": self.pid, "relacao": self.relation, "ancora": self.anchor,
                "ganho_esperado": round(self.expected_gain, 4), "voi": round(self.voi, 4)}


class WitnessRAGRetriever(Retriever):
    name = "witnessrag"
    uses_graph = True

    def __init__(self, ctx: IndexContext, compile_mode: str = "llm") -> None:
        super().__init__(ctx)
        self.compile_mode = compile_mode
        # O fallback é parte do método, não um comparador: quando a prova falha,
        # é ele que entrega o contexto. Trocá-lo por RRF muda o WITNESS-RAG, e
        # por isso a opção fica registrada no relatório.
        self._dense = (HybridRetriever(ctx, ctx.run.witness.hybrid_rrf_k)
                       if ctx.run.witness.hybrid_fallback else DenseRetriever(ctx))
        self.memory: MemoryView | None = None
        self.searcher: WitnessSearcher | None = None
        self.selection: budget_mod.SelectionResult | None = None
        self._acquisition_calls = 0
        self._acquired_facts = 0

    # -- indexação ----------------------------------------------------------

    def index(self) -> None:
        if self.ctx.kg is None:
            raise RuntimeError("o grafo compartilhado precisa ser construído antes dos métodos")
        self._dense.index()
        self.memory = MemoryView(self.kg)
        self.searcher = WitnessSearcher(self.memory, self.ctx.embedder, self.ctx.run.witness)

        # A seleção sob orçamento não roda aqui: ela precisa do conjunto de
        # demandas, que o runner monta depois da indexação (`set_train_demands`
        # seguido de `_select_memory`). Chamá-la agora só produziria um aviso.
        if self.ctx.run.witness.budget_fraction < 1.0:
            log.info("orçamento de memória %.2f pedido; seleção aguarda as demandas",
                     self.ctx.run.witness.budget_fraction)
        self.indexed = True

    def _select_memory(self, fraction: float) -> None:
        """Seleção de memória sob orçamento. Ver `wrag/witness/budget.py`.

        As demandas vêm de perguntas de TREINO. Otimizar a memória para as
        perguntas de teste demonstraria memorização do conjunto de avaliação e
        nada mais — a proposta é explícita quanto a isso.
        """
        assert self.memory is not None and self.searcher is not None
        demands = getattr(self, "_train_demands", None)
        demands = demands or []
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("budget_fraction deve estar entre zero e um")

        costs = {i: 1.0 for i in range(len(self.memory.facts))}
        budget = fraction * sum(costs.values())
        cfg = self.ctx.run.witness
        if len(self.memory.facts) > cfg.ilp_max_facts:
            log.warning("%d fatos acima de ilp_max_facts=%d; usando o guloso",
                        len(self.memory.facts), cfg.ilp_max_facts)
            self.selection = budget_mod.select_greedy(demands, costs, budget)
        else:
            self.selection = budget_mod.select_ilp(demands, costs, budget, cfg.ilp_time_limit_s)

        # Aplicar de fato: a busca passa a enxergar S ⊂ F. Calcular a seleção e
        # continuar consultando F inteira transformaria o ILP em enfeite.
        self.searcher.allowed = set(self.selection.kept)
        self.searcher._allowed_horizon = len(self.memory.facts)
        log.info("seleção de memória: %s (de %d fatos)",
                 self.selection.to_dict(), len(self.memory.facts))

    def set_train_demands(self, demands: list[budget_mod.Demand]) -> None:
        self._train_demands = demands

    # -- aquisição adaptativa (VOI) ----------------------------------------

    def _plan_acquisition(self, gap: Gap, question: Question) -> list[AcquisitionAction]:
        """VOI míope de um passo sobre ações de leitura dirigida.

            VOI(a) = E_y[max_S U(S|D,y,a)] − max_S U(S|D) − λ·custo(a)

        O primeiro termo é aproximado por: a demanda vale 1, e a probabilidade de
        a ação fechar a testemunha é aproximada pela similaridade densa entre a
        sonda (âncora + relação faltante) e a passagem. É uma aproximação, e é
        onde mais há espaço para melhorar o método: uma estimativa treinada de
        "esta passagem contém esta relação sobre esta entidade" substituiria a
        similaridade por algo com calibração.
        """
        cfg = self.ctx.run.witness
        probe = gap.probe().strip()
        if not probe:
            return []

        pids, scores = self._dense.search(probe, cfg.acquisition_passages * 3)
        # O ganho é normalizado pelo topo da própria consulta. Sem isso a escala
        # do recuperador decide se há aquisição: o cosseno do denso fica na casa
        # de 0,6 e passa por λ=0,05, mas a fusão recíproca de postos devolve
        # ~1/(60+posto) ≈ 0,016 e reprova TODA ação — foi o que desligou a
        # aquisição em silêncio (240 chamadas viraram 0) ao ligar o híbrido.
        # O preço é explícito: o ganho passa a ser relativo dentro da consulta, e
        # λ só corta a cauda. Continua sendo heurística, não valor da informação.
        raw = [max(0.0, float(s)) for s in scores]
        top = max(raw, default=0.0)
        actions: list[AcquisitionAction] = []
        for pid, score in zip(pids, raw):
            passage = self.corpus.get(pid)
            actions.append(AcquisitionAction(
                pid=pid, title=passage.title, text=passage.text,
                relation=gap.atom.relation, anchor=gap.anchor(),
                expected_gain=(score / top) if top > 0 else 0.0,
                cost=cfg.acquisition_lambda,
            ))
        actions = [a for a in actions if a.voi > 0]
        actions.sort(key=lambda a: -a.voi)
        return actions[: cfg.acquisition_passages]

    def _acquire(self, actions: list[AcquisitionAction], question: Question) -> int:
        assert self.memory is not None and self.searcher is not None
        if not actions:
            return 0
        facts = extract_targeted(
            self.ctx.llm,
            relation=actions[0].relation,
            passages=[(a.pid, a.title, a.text) for a in actions],
            anchor=actions[0].anchor or None,
            dataset=self.ctx.dataset,
            method=self.name,
            question_id=question.qid,
        )
        self._acquisition_calls += len(actions)
        if not facts:
            return 0
        def fact_key(f):
            return (canonical_symbol(f.subject), canonical_symbol(f.relation),
                    canonical_symbol(f.object), f.pid)
        reread = {fact_key(f) for f in facts}
        reactivated = {i for i, f in enumerate(self.memory.facts)
                       if fact_key(f) in reread and not self.searcher._permitted(i)}
        self.searcher._reactivated.update(reactivated)
        added = self.memory.add_facts(facts, self.ctx.embedder)
        self.searcher.register_facts(added)
        self._acquired_facts += len(added) + len(reactivated)
        return len(added) + len(reactivated)

    # -- vocabulário da compilação -----------------------------------------

    def _vocabulary(self, question: Question) -> str:
        """Relações e entidades do grafo mais próximas da pergunta.

        Sugestão, não esquema: o compilador continua livre para emitir outra
        relação. O que isto corrige é o caso medido em que ele inventa um
        predicado ("identity", "destress method") que nenhum fato instancia, e o
        átomo morre no aterramento sem nunca ter tido chance.
        """
        from wrag import prompts

        memory = self.memory
        cfg = self.ctx.run.witness
        if memory is None or not cfg.vocabulary_aware_compile:
            return ""
        vector = self.ctx.embedder.encode([question.question])[0]
        relations: list[str] = []
        if memory.relation_vectors.size and cfg.vocabulary_relations:
            idx, _ = cosine_topk(vector, memory.relation_vectors, cfg.vocabulary_relations)
            relations = [memory.relations[int(i)] for i in idx]
        entities: list[str] = []
        if memory.entity_vectors.size and cfg.vocabulary_entities:
            idx, _ = cosine_topk(vector, memory.entity_vectors, cfg.vocabulary_entities)
            entities = [memory.entities[int(i)] for i in idx]
        return prompts.format_vocabulary(relations, entities)

    # -- recuperação --------------------------------------------------------

    def _retrieve(self, question: Question, k: int) -> RetrievalResult:
        assert self.memory is not None and self.searcher is not None
        cfg = self.ctx.run.witness
        pool_k = max(k, cfg.candidate_pool_k)
        dense_pids, dense_scores = self._dense.search(question.question, pool_k)

        before_events = len(LEDGER.events)
        try:
            result = self._retrieve_inner(question, k, dense_pids, dense_scores)
            result.filtered |= any(e.item_id == question.qid and e.method == self.name
                                   for e in LEDGER.events[before_events:])
            result.diagnostics.setdefault("risco", 1.0)
            result.diagnostics.setdefault("risco_bruto", 99.0)
            result.diagnostics["risco_calibrado"] = False
            result.diagnostics.setdefault("testemunha_no_contexto", False)
            result.diagnostics.setdefault("resposta_estrutural", "")
            result.diagnostics["candidatos_explorados"] = len(dense_pids)
            result.diagnostics["documentos_entregues"] = len(result.pids)
            result.diagnostics["orcamento_leitor"] = k
            return result
        finally:
            if not getattr(cfg, "acquisition_persist", False):
                # A memória volta ao estado indexado. Sem isso, a pergunta 300
                # responderia com fatos adquiridos nas 299 anteriores, e o
                # resultado passaria a depender da ordem do dataset.
                self.searcher.rollback()
                self.memory.reset()

    def _retrieve_inner(self, question: Question, k: int, dense_pids: list[str],
                        dense_scores: list[float]) -> RetrievalResult:
        assert self.memory is not None and self.searcher is not None
        cfg = self.ctx.run.witness

        vocabulary = self._vocabulary(question) if self.compile_mode == "llm" else ""
        if cfg.query_plans:
            plans = compile_query_plans(
                self.ctx.llm, question, mode=self.compile_mode,
                max_atoms=cfg.max_atoms, max_plans=cfg.max_query_plans,
                temperature=cfg.compile_temperature, dataset=self.ctx.dataset,
                method=self.name, vocabulary=vocabulary)
        else:
            plans = [compile_query(
                self.ctx.llm, question, mode=self.compile_mode,
                max_atoms=cfg.max_atoms, temperature=cfg.compile_temperature,
                dataset=self.ctx.dataset, method=self.name, vocabulary=vocabulary)]

        if plans and all(query.filtered for query in plans):
            return RetrievalResult(pids=dense_pids[:k] if cfg.dense_fallback else [],
                                   scores=dense_scores[:k] if cfg.dense_fallback else [], filtered=True,
                                   diagnostics={"fallback": "denso (compilação bloqueada)"})

        attempts: list[tuple[int, ConjunctiveQuery, SearchResult]] = []
        plan_diagnostics: list[dict[str, Any]] = []
        for index, candidate in enumerate(plans):
            supported = (bool(candidate.atoms)
                         and candidate.aggregation in executable_aggregations(cfg.answer_set))
            candidate_result = self.searcher.join(candidate) if supported else SearchResult()
            if supported:
                attempts.append((index, candidate, candidate_result))
            plan_diagnostics.append({
                "indice": index, "consulta": candidate.to_dict(),
                "executavel": supported, "fechou": candidate_result.complete,
                "profundidade": candidate_result.depth_reached,
                "candidatos_por_atomo": candidate_result.n_candidates,
            })

        if not attempts:
            query = plans[0] if plans else ConjunctiveQuery(fallback=question.question)
            return RetrievalResult(pids=dense_pids[:k] if cfg.dense_fallback else [],
                                   scores=dense_scores[:k] if cfg.dense_fallback else [],
                                   diagnostics={"fallback": "consulta ausente, inválida ou fora do escopo",
                                                "suportada": False,
                                                "agregacao": query.aggregation,
                                                "consulta": query.to_dict(),
                                                "planos_compilados": plan_diagnostics})

        complete = [attempt for attempt in attempts if attempt[2].complete]
        if complete:
            chosen_index, query, result = complete[0]  # ordem declarada pelo compilador
        else:
            # Aquisição custa chamadas: aplique-a somente ao plano que avançou
            # proporcionalmente mais antes de encontrar uma lacuna. Empate
            # preserva a preferência declarada pelo compilador.
            chosen_index, query, result = max(
                attempts,
                key=lambda item: (item[2].depth_reached / max(1, item[1].n_atoms),
                                  -item[0]))

        rounds = 0
        acquisitions: list[dict[str, Any]] = []
        while (not result.complete and cfg.enable_acquisition
               and rounds < cfg.acquisition_rounds and result.gap is not None):
            actions = self._plan_acquisition(result.gap, question)
            if not actions:
                break
            acquisitions.append({"rodada": rounds + 1, "lacuna": result.gap.to_dict(),
                                 "acoes": [a.to_dict() for a in actions]})
            rounds += 1
            if self._acquire(actions, question) == 0:
                break
            result = self.searcher.join(query)

        diagnostics: dict[str, Any] = {
            "consulta": query.to_dict(),
            "forma": query.shape(),
            "n_candidatos_por_atomo": result.n_candidates,
            "profundidade_alcancada": result.depth_reached,
            "feixe_exaustivo": result.exhaustive,
            "busca_exaustiva": result.exhaustive,
            "cortes": result.truncations,
            "modo_aterramento": result.grounding_mode,
            "aterramento_condicionado": cfg.binding_aware_grounding,
            "aquisicao_tipo": "heuristica_de_lacuna_por_similaridade",
            "rodadas_aquisicao": rounds,
            "aquisicoes": acquisitions,
        }
        if cfg.query_plans:
            diagnostics["planos_compilados"] = plan_diagnostics
            diagnostics["plano_escolhido"] = chosen_index

        if not result.complete:
            # Candidatos parciais são diagnóstico, não autorização para promover.
            diagnostics["fallback"] = "denso (sem testemunha completa)"
            diagnostics["lacuna"] = result.gap.to_dict() if result.gap else None
            if not cfg.dense_fallback:
                return RetrievalResult(diagnostics=diagnostics)
            # Uma prova incompleta não emite certificado, então a evidência
            # parcial não pode deslocar o ranking denso — e medindo, ela não
            # merece: nas compositional que caem aqui, colocar as passagens
            # parciais na frente dá R@5 21.4 contra 65.7 do denso puro, e em 20
            # de 35 casos nenhuma das cinco passagens do denso sobrevivia.
            # Nenhuma mistura testada superou o denso, então a parcial só ocupa
            # as sobras e continua registrada no diagnóstico para análise.
            partial_pids, partial_scores = self._partial_passages(query, k)
            diagnostics["passagens_parciais"] = partial_pids[:k]
            pids, scores = pad_with_dense(dense_pids[:k], dense_scores[:k],
                                          partial_pids, partial_scores, k)
            return RetrievalResult(pids=pids, scores=scores, diagnostics=diagnostics)

        if cfg.verify_witnesses:
            from wrag.witness.verification import verify_witnesses
            proposed = result.witnesses
            fitting = [w for w in proposed if len(set(w.pids)) <= k]
            out_of_context = len(proposed) - len(fitting)
            if cfg.answer_set:
                # Verificar as cinco mais baratas costuma verificar cinco provas
                # da MESMA resposta e deixar o conjunto sem as outras. O rodízio
                # gasta o mesmo orçamento cobrindo respostas distintas.
                fitting = cover_answers(fitting, cfg.verification_max_witnesses)
            result.witnesses, verification = verify_witnesses(
                self.ctx.llm, self.corpus, self.memory, question, query, fitting,
                cfg.verification_max_witnesses, self.ctx.dataset)
            # Descartada por não caber no contexto é diferente de descartada pelo
            # teto de verificação; `nao_avaliadas` já conta a segunda.
            verification["fora_do_orcamento_contexto"] = out_of_context
            diagnostics["verificacao"] = verification
            diagnostics["n_testemunhas_propostas"] = len(proposed)
            if not result.complete:
                diagnostics["fallback"] = "denso (nenhuma testemunha aprovada)"
                return RetrievalResult(pids=dense_pids[:k] if cfg.dense_fallback else [],
                                       scores=dense_scores[:k] if cfg.dense_fallback else [],
                                       diagnostics=diagnostics)

        candidates = score_answers(result.witnesses, self.memory, cfg)
        pids, scores = rank_passages(candidates, k, cover_answers_first=cfg.answer_set)
        if cfg.dense_fallback:
            pids, scores = pad_with_dense(pids, scores, dense_pids, dense_scores, k)
            pids, scores, order_preserved = preserve_fallback_order_if_same_set(
                pids, scores, dense_pids, dense_scores, k)
            diagnostics["ordem_fallback_preservada"] = order_preserved
            diagnostics["contexto_alterado_pelo_witness"] = pids != dense_pids[:k]
        # O leitor só recebe estas passagens: não certificar uma prova truncada.
        delivered = [w for w in result.witnesses if set(w.pids) <= set(pids)]
        candidates = score_answers(delivered, self.memory, cfg)

        best = candidates[0] if candidates else None
        diagnostics.update({
            "n_testemunhas": len(result.witnesses),
            "n_testemunhas_no_contexto": len(delivered),
            "testemunha_no_contexto": bool(delivered),
            "respostas": [c.to_dict(self.memory) for c in candidates[:3]],
            "resposta_estrutural": best.answer if best else "",
            "risco": round(best.risk, 4) if best else 1.0,
            "risco_bruto": round(best.risk_raw, 4) if best else 99.0,
            "score_estrutural": round(best.score, 4) if best else 0.0,
        })
        if cfg.answer_set:
            # A resposta da consulta é o conjunto das atribuições certas, não a
            # testemunha mais barata. `count` é |conjunto|; note que contar exige
            # completude, e o conjunto não a certifica — o número é o que a
            # memória prova, e o diagnóstico registra isso.
            answers = answer_set(candidates, cfg.answer_set_max_items)
            diagnostics["conjunto_resposta"] = answers.to_dict(self.memory)
            diagnostics["resposta_estrutural"] = (answers.count if query.aggregation == "count"
                                                  else answers.text)
            diagnostics["risco"] = round(answers.risk, 4) if answers.items else 1.0
            diagnostics["agregacao_executada"] = query.aggregation
        return RetrievalResult(pids=pids, scores=scores, diagnostics=diagnostics)

    def _partial_passages(self, query: ConjunctiveQuery, k: int) -> tuple[list[str], list[float]]:
        """Passagens dos melhores candidatos de cada átomo, quando a junção falha.

        São candidatos sem validação semântica, mantidos para diagnóstico.
        """
        assert self.searcher is not None and self.memory is not None
        groundings = self.searcher.ground(query)
        scores: dict[str, float] = {}
        for atom_groundings in groundings:
            for grounding in atom_groundings[:5]:
                pid = self.memory.facts[grounding.fact_index].pid
                scores[pid] = max(scores.get(pid, 0.0), grounding.score)
        order = sorted(scores, key=lambda pid: -scores[pid])[:k]
        return order, [scores[pid] for pid in order]

    def index_report(self) -> dict:
        report: dict[str, Any] = {"grafo": self.kg.stats(),
                                  "modo_compilacao": self.compile_mode,
                                  "chamadas_aquisicao": self._acquisition_calls,
                                  "fatos_adquiridos": self._acquired_facts}
        if self.selection is not None:
            report["selecao_memoria"] = self.selection.to_dict()
            report["selecao_memoria"]["escopo"] = "mascara_de_fatos; corpus_e_vetores_permanecem_em_RAM"
        report["fontes_demandas"] = getattr(self, "demand_sources", {})
        return report


class WitnessRAGOracleRetriever(WitnessRAGRetriever):
    """Diagnóstico privilegiado por anotações. Nome legado; não é teto oracular.

    Usa gabarito/evidência para montar padrões e tradução heurística de perguntas
    decompostas. Nenhuma diferença numérica isola perfeitamente o erro do parser.
    """

    name = "witnessrag-oracle"

    def __init__(self, ctx: IndexContext) -> None:
        super().__init__(ctx, compile_mode="oracle")


class WitnessRAGAnnotatedRetriever(WitnessRAGOracleRetriever):
    name = "witnessrag-annotated"
