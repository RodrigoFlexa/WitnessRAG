"""
Conjunto de respostas certas, vocabulário da compilação, extração de diálogo e
fallback híbrido.

A instância de apoio é a mesma dos outros testes: Ana e Bruno trabalham na Atlas
e pesquisam Óptica. A consulta de interseção tem DUAS respostas certas, e é essa
a propriedade que os testes daqui exercitam — uma testemunha certifica uma
atribuição, e a resposta da consulta é o conjunto delas.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from wrag import config as C
from wrag import prompts
from wrag.data import Question
from wrag.eval.reader import read
from wrag.ie import Fact, _cache_path, _parse_triples
from wrag.llm.base import LLMResult
from wrag.util import normalize
from wrag.witness.provenance import answer_set, score_answers
from wrag.witness.query import Atom, ConjunctiveQuery, looks_like_answer_set
from wrag.witness.search import Witness, cover_answers, executable_aggregations

from test_witness import build_toy


def intersection_query():
    return ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "?x", "Atlas"),
        Atom("pesquisa", "?x", "Optica"),
    ])


# --- conjunto de respostas --------------------------------------------------

def test_answer_set_keeps_every_certain_answer():
    memory, searcher, _e, cfg = build_toy()
    result = searcher.join(intersection_query())
    candidates = score_answers(result.witnesses, memory, cfg)
    answers = answer_set(candidates, max_items=10)
    assert {normalize(c.answer) for c in answers.items} == {"ana", "bruno"}
    assert answers.truncated == 0
    # Cada item leva a própria demonstração: a lista é auditável item a item.
    payload = answers.to_dict(memory)
    assert all(item["passagens"] for item in payload["itens"])
    assert payload["completude_certificada"] is False


def test_answer_set_count_is_the_size_of_the_set():
    memory, searcher, _e, cfg = build_toy()
    result = searcher.join(intersection_query())
    answers = answer_set(score_answers(result.witnesses, memory, cfg), max_items=10)
    assert answers.count == "2"


def test_answer_set_records_what_the_limit_dropped():
    memory, searcher, _e, cfg = build_toy()
    result = searcher.join(intersection_query())
    answers = answer_set(score_answers(result.witnesses, memory, cfg), max_items=1)
    assert len(answers.items) == 1 and answers.truncated == 1


def test_count_is_executable_only_with_answer_set():
    assert "count" in executable_aggregations(True)
    assert "count" not in executable_aggregations(False)
    memory, searcher, _e, _c = build_toy()
    query = intersection_query()
    query.aggregation = "count"
    assert not searcher.join(query).complete       # padrão: fora do escopo
    searcher.cfg.answer_set = True
    assert searcher.join(query).complete


def test_set_is_executable_only_with_answer_set():
    assert "set" in executable_aggregations(True)
    assert "set" not in executable_aggregations(False)


@pytest.mark.parametrize("question", [
    "What activities does Melanie partake in?",
    "Which books did Caroline read?",
    "What has Melanie painted?",
])
def test_obvious_enumerations_are_detected_without_gold_labels(question):
    assert looks_like_answer_set(question)


@pytest.mark.parametrize("question", [
    "What sports team does Melanie support?",
    "What did Melanie paint yesterday?",
    "Which city did Caroline visit?",
])
def test_ambiguous_or_singular_questions_are_not_forced_to_sets(question):
    assert not looks_like_answer_set(question)


def test_comparison_stays_out_of_scope():
    # `max`/`min`/`compare` exigem ordenar valores extraídos como texto.
    assert executable_aggregations(True).isdisjoint({"max", "min", "compare"})


def test_cover_answers_spreads_the_budget_over_distinct_answers():
    def witness(answer: str, cost: float) -> Witness:
        return Witness(facts=(int(cost * 10),), bindings={}, score=0.5, cost=cost,
                       pids=("p0",), answer=answer)
    # Três provas de "Ana" mais baratas que a única de "Bruno".
    witnesses = [witness("Ana", 0.1), witness("Ana", 0.2), witness("Ana", 0.3),
                 witness("Bruno", 0.4)]
    assert {w.answer for w in cover_answers(witnesses, 2)} == {"Ana", "Bruno"}
    assert [w.answer for w in witnesses[:2]] == ["Ana", "Ana"]   # o corte por custo perderia Bruno


def test_context_covers_one_proof_per_answer_before_extra_proofs():
    """Com orçamento apertado, o contexto leva uma prova de cada resposta.

    Sem isso a melhor resposta gasta as k passagens com provas dela mesma e o
    leitor nunca vê as outras respostas certas — que são o conjunto pedido.
    """
    from wrag.witness.provenance import AnswerCandidate, rank_passages

    def witness(pid: str, cost: float) -> Witness:
        return Witness(facts=(0,), bindings={}, score=0.9, cost=cost, pids=(pid,), answer="a")

    ana = AnswerCandidate(answer="Ana", score=0.9,
                          witnesses=[witness("p0", 0.1), witness("p1", 0.2)])
    bruno = AnswerCandidate(answer="Bruno", score=0.5, witnesses=[witness("p2", 0.3)])
    assert rank_passages([ana, bruno], k=2)[0] == ["p0", "p1"]
    assert rank_passages([ana, bruno], k=2, cover_answers_first=True)[0] == ["p0", "p2"]


# --- extração de diálogo ----------------------------------------------------

def test_dialogue_triples_may_carry_time():
    parsed = _parse_triples({"triples": [["Ana", "moved from", "Sweden", "2023-05-01"],
                                         ["Ana", "works at", "Atlas"]]}, limit=10)
    assert parsed == [("Ana", "moved from", "Sweden", "2023-05-01"),
                      ("Ana", "works at", "Atlas", "")]


def test_time_enters_the_verbalization_but_not_the_triple():
    fact = Fact(fid="f", subject="Ana", relation="hiked", object="the trail",
                pid="p0", time="19 October 2023")
    assert fact.verbalize() == "Ana hiked the trail (19 October 2023)"
    assert fact.triple == ("Ana", "hiked", "the trail")      # métricas comparam triplas
    assert fact.to_dict()["t"] == "19 October 2023"


def test_stats_expose_relation_fragmentation():
    """Um predicado diferente por fato não sustenta junção, e precisa aparecer.

    Foi assim que a primeira versão do prompt de diálogo degenerou: relações de
    quase cinco palavras, 87% delas únicas, e o objeto virando o interlocutor.
    """
    from wrag.ie import ExtractionResult

    def facts(triples):
        return [Fact(fid=str(i), subject=s, relation=r, object=o, pid="p0")
                for i, (s, r, o) in enumerate(triples)]

    saudavel = ExtractionResult(facts=facts([
        ("Mel", "read", "Charlotte's Web"), ("Mel", "read", "Nothing is Impossible"),
        ("Caroline", "painted", "a sunset"), ("Mel", "painted", "a horse")])).stats()
    assert saudavel["relacoes_por_fato"] == 0.5 and saudavel["relacoes_unicas"] == 0
    assert saudavel["palavras_por_relacao"] == 1.0

    degenerado = ExtractionResult(facts=facts([
        ("Mel", "said running is a great way to destress", "Caroline"),
        ("Mel", "thank Caroline for the compliment", "true"),
        ("Mel", "ask Caroline about jobs", "true")])).stats()
    assert degenerado["relacoes_por_fato"] == 1.0        # um predicado por fato
    assert degenerado["relacoes_unicas"] == 3
    assert degenerado["palavras_por_relacao"] > 4
    assert degenerado["n_objetos_distintos"] == 2        # objetos colapsam


def test_dialogue_prompt_forbids_the_degenerate_shapes():
    template = prompts.OPENIE_DIALOGUE_TEMPLATE
    assert "one to four words" in template
    assert '"true", "false" or "yes"' in template
    assert "the fact is about the CONTENT, not about the listener" in template
    assert "reusable canonical form" in template
    assert "Scope unnamed relatives and possessions" in template


def test_cache_key_unchanged_while_dialogue_mode_is_off():
    """Um campo novo de configuração não pode invalidar extrações já pagas.

    A chave é recalculada aqui pela fórmula ANTERIOR ao modo diálogo. Se alguém
    acrescentar um campo a `IEConfig` sem excluí-lo da chave, ou mexer nos
    prompts de NER/OpenIE, este teste falha — e falhar é o comportamento certo,
    porque a consequência real seria reextrair todos os corpora já pagos.
    """
    from wrag.data import Corpus, Passage
    from wrag.llm.stub import StubLLM
    from wrag.util import sha

    corpus = Corpus(name="toy", passages=[Passage(pid="p0", title="t", text="x")], questions=[])
    llm = StubLLM(filter_rate=0)
    legacy = sha({
        "dataset": "toy",
        "passages": [(p.pid, p.title, p.text) for p in corpus.passages],
        "backend": llm.name,
        "deployment": getattr(llm, "deployment", ""),
        "provider_identity": llm.cache_identity() if hasattr(llm, "cache_identity") else "",
        "config": {"max_tokens": 1600, "temperature": 0.0, "two_step": True,
                   "max_triples_per_passage": 40},
        "prompts": [prompts.NER_SYSTEM, prompts.NER_TEMPLATE,
                    prompts.OPENIE_SYSTEM, prompts.OPENIE_TEMPLATE],
        "prompt_version": 3,
    })
    off = _cache_path(corpus, llm, C.IEConfig(dialogue_mode=False))
    on = _cache_path(corpus, llm, C.IEConfig(dialogue_mode=True))
    assert off.name == f"toy-{legacy[:16]}.json"
    assert on != off                       # outra extração, outro arquivo


# --- vocabulário da compilação ----------------------------------------------

def test_vocabulary_block_is_a_suggestion_not_a_schema():
    block = prompts.format_vocabulary(["plays", "read"], ["Melanie"])
    assert "plays; read" in block and "Melanie" in block
    assert "Invent a new relation only when none of them fits" in block
    assert prompts.format_vocabulary([], []) == ""


def test_compilation_sends_the_vocabulary(monkeypatch):
    from wrag.witness import query as Q

    seen = {}

    class Recorder:
        def chat(self, prompt, **kwargs):
            seen["prompt"] = prompt
            return LLMResult(text='{"answer_var": "x", "atoms": [{"relation": "plays",'
                                  ' "subject": "Melanie", "object": "?x"}]}')

    question = Question(qid="q", question="What instruments does Melanie play?", answers=["violin"])
    compiled = Q.compile_query(Recorder(), question, vocabulary=prompts.format_vocabulary(["plays"], []))
    assert "plays" in seen["prompt"] and compiled.atoms
    # O vocabulário vem do grafo extraído, nunca de anotação do dataset.
    assert compiled.to_dict()["uses_annotations"] is False


def test_plan_compilation_returns_distinct_ordered_hypotheses():
    from wrag.witness.query import compile_plans_with_llm

    class Planner:
        def chat(self, prompt, **kwargs):
            assert "up to 3 distinct candidate query plans" in prompt
            return LLMResult(text='{"plans": ['
                '{"answer_var":"x","atoms":[{"relation":"attended recently",'
                '"subject":"Caroline","object":"?x"}],"aggregation":"none"},'
                '{"answer_var":"x","atoms":[{"relation":"attended",'
                '"subject":"Caroline","object":"?x"}],"aggregation":"none"}]}' )

    plans = compile_plans_with_llm(
        Planner(), Question("q", "What workshop did Caroline attend recently?", []),
        max_plans=3)
    assert [p.atoms[0].relation for p in plans] == ["attended recently", "attended"]
    assert all(not p.to_dict()["uses_annotations"] for p in plans)


def test_plan_prompt_has_diverse_few_shot_structures_and_exact_output_contract():
    rendered = prompts.COMPILE_PLANS_TEMPLATE.format(
        max_plans=3, max_atoms=4, vocabulary="\nGRAPH VOCABULARY: orbit; discover",
        planning_feedback="", question="Who discovered the comet?")
    assert '"relation":"play"' in rendered                 # direct
    assert '"subject":"Omar","object":"?y"' in rendered  # chain
    assert rendered.count('"subject":"?x"') >= 3          # intersection/count
    assert '"aggregation":"set"' in rendered
    assert '"aggregation":"count"' in rendered
    assert "after(?x, conference)" in rendered              # explicit negative contrast
    assert "one JSON object with only the `plans` field" in rendered
    assert "Melanie" not in rendered and "Caroline" not in rendered
    # The corpus vocabulary is closest to the real input, after synthetic examples.
    assert rendered.index("GRAPH VOCABULARY") > rendered.index("Example 6")
    assert rendered.index("GRAPH VOCABULARY") < rendered.index("### INPUT")


def test_single_plan_prompt_uses_the_same_synthetic_structural_examples():
    rendered = prompts.COMPILE_TEMPLATE.format(
        max_atoms=4, vocabulary="\nGRAPH VOCABULARY: orbit; discover",
        question="What did someone find?")
    for marker in ("Nira", "Omar", "Northstar Institute", "Jun", "Mateo",
                   '"aggregation":"count"'):
        assert marker in rendered
    # Benchmark participants must not appear in compiler demonstrations.
    assert "Melanie" not in rendered
    assert "Caroline" not in rendered
    assert 'relation="play(Nira, ?x)"' in rendered
    assert "Return one JSON object only" in rendered
    assert rendered.index("GRAPH VOCABULARY") > rendered.index("How many apprentices")
    assert rendered.index("GRAPH VOCABULARY") < rendered.index("### INPUT")


def test_plan_compilation_rejects_artificial_boolean_atoms():
    from wrag.witness.query import compile_plans_with_llm

    class Planner:
        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"plans":[{"answer_var":"x","atoms":['
                '{"relation":"supports","subject":"?x","object":"Caroline"},'
                '{"relation":"negative experience","subject":"Caroline",'
                '"object":"true"}]}]}')

    plan = compile_plans_with_llm(Planner(), Question("q", "Who supports Caroline?", []))[0]
    assert not plan.atoms
    assert plan.validation_error == "constante_booleana_artificial"


def test_plan_compilation_rejects_arguments_embedded_in_relation():
    from wrag.witness.query import compile_plans_with_llm

    class Planner:
        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"plans":[{"answer_var":"x","atoms":['
                '{"relation":"create(Melanie, ?x)","subject":"Melanie",'
                '"object":"?x"}],"expected_type":"work","aggregation":"set",'
                '"fallback":"art"}]}')

    plan = compile_plans_with_llm(
        Planner(), Question("q", "What did Melanie create?", []))[0]
    assert not plan.atoms
    assert plan.validation_error == "relacao_contem_argumentos"


def test_plan_compilation_keeps_parenthetical_natural_qualifier():
    from wrag.witness.query import compile_plans_with_llm

    class Planner:
        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"plans":[{"answer_var":"x","atoms":['
                '{"relation":"go to beach (2023)","subject":"Melanie",'
                '"object":"?x"}],"expected_type":"number","aggregation":"count",'
                '"fallback":"beach"}]}')

    plan = compile_plans_with_llm(
        Planner(), Question("q", "How often in 2023?", []))[0]
    assert plan.validation_error == ""
    assert plan.atoms[0].relation == "go to beach (2023)"


# --- leitor ciente de conjunto ----------------------------------------------

def test_reader_template_follows_the_configuration():
    from wrag.data import Corpus, Passage

    corpus = Corpus(name="toy", questions=[],
                    passages=[Passage(pid="p0", title="t", text="Melanie plays clarinet and violin.")])
    question = Question(qid="q", question="What instruments does Melanie play?",
                        answers=["clarinet and violin"])
    seen = []

    class Recorder:
        def chat(self, prompt, **kwargs):
            seen.append(prompt)
            return LLMResult(text='{"answer": "clarinet, violin"}')

    read(Recorder(), corpus, question, ["p0"], C.QAConfig(answer_set=False))
    read(Recorder(), corpus, question, ["p0"], C.QAConfig(answer_set=True))
    assert "EVERY item the passages support" in seen[1]
    assert "EVERY item" not in seen[0]
    # O que muda é só a regra de completude: o resto do prompt continua igual
    # para todos os métodos, que é o que mantém a comparação entre eles válida.
    assert "Answer the question using only the passages below." in seen[0]
    assert "Answer the question using only the passages below." in seen[1]


# --- fallback híbrido -------------------------------------------------------

def test_hybrid_fuses_both_rankings_by_reciprocal_rank():
    """RRF sobre dois rankings discordantes: quem aparece nos dois sobe.

    Os dois recuperadores são substituídos por rankings fixos de propósito: o que
    está sob teste é a fusão, não a qualidade do TF-IDF ou do BM25 neste corpus.
    """
    from wrag.data import Corpus, Passage
    from wrag.embed import TfidfEmbedder
    from wrag.methods.base import IndexContext
    from wrag.methods.dense import HybridRetriever
    from wrag.llm.stub import StubLLM

    corpus = Corpus(name="toy", questions=[],
                    passages=[Passage(pid=f"p{i}", title=f"t{i}", text=f"texto {i}")
                              for i in range(4)])
    ctx = IndexContext(corpus=corpus, llm=StubLLM(filter_rate=0), embedder=TfidfEmbedder(),
                       run=C.RunConfig())
    hybrid = HybridRetriever(ctx, rrf_k=1)
    hybrid._dense.search = lambda text, k: (["p0", "p1", "p2"], [1.0, 0.9, 0.8])
    hybrid._bm25.search = lambda text, k: (["p3", "p1", "p0"], [9.0, 4.0, 1.0])
    pids, scores = hybrid.search("consulta", 4)
    # p0: 1/2 + 1/4; p1: 1/3 + 1/3; p3: 1/2; p2: 1/4.
    assert pids == ["p0", "p1", "p3", "p2"]
    assert scores[0] == pytest.approx(1 / 2 + 1 / 4)
    assert scores[1] == pytest.approx(1 / 3 + 1 / 3)
    # p3 não aparece no ranking denso e ainda assim entra: é o ponto da fusão.
    assert "p3" in pids[:3]


# --- ligação pela linha de comando ------------------------------------------

def test_cli_flags_reach_the_configuration(monkeypatch):
    """As quatro opções novas precisam chegar à configuração da rodada."""
    from wrag import cli
    from wrag.eval import runner

    captured = {}

    def fake_run(datasets, methods, cfg, **kwargs):
        captured["cfg"] = cfg
        return Path("/tmp/rodada-inexistente")

    monkeypatch.setattr(runner, "run", fake_run)
    cli.main(["run", "--datasets", "locomo", "--methods", "witnessrag", "--top-k", "10",
              "--answer-set", "--vocab-compile", "--query-plans",
              "--hybrid-fallback", "--dialogue-ie"])
    cfg = captured["cfg"]
    assert cfg.top_k == 10
    assert cfg.witness.answer_set and cfg.qa.answer_set
    assert cfg.witness.vocabulary_aware_compile and cfg.witness.hybrid_fallback
    assert cfg.witness.query_plans
    assert cfg.witness.max_query_plans == 5
    assert cfg.ie.dialogue_mode
    assert cfg.graph.merge_relation_inflections

    captured.clear()
    cli.main(["run", "--datasets", "locomo", "--methods", "witnessrag"])
    cfg = captured["cfg"]
    assert not cfg.witness.answer_set and not cfg.qa.answer_set
    assert not cfg.witness.vocabulary_aware_compile and not cfg.witness.hybrid_fallback
    assert not cfg.witness.query_plans
    assert not cfg.ie.dialogue_mode      # o padrão preserva as rodadas anteriores

    captured.clear()
    cli.main(["run", "--datasets", "locomo", "--methods", "witnessrag",
              "--no-relation-family-merge"])
    assert not captured["cfg"].graph.merge_relation_inflections


# --- integração no retriever ------------------------------------------------

def build_context(answer_set_on: bool):
    """Contexto mínimo sobre a instância de cinco fatos, com um compilador fixo."""
    from wrag.data import Corpus, Passage
    from wrag.graph import build_graph
    from wrag.ie import ExtractionResult
    from wrag.methods.base import IndexContext
    from test_witness import ExactEmbedder, TRIPLES

    class Compiler:
        """Devolve sempre a consulta de interseção; o compilador não está sob teste."""

        name = "fake"
        usage = type("U", (), {"snapshot": staticmethod(lambda: {})})()

        def chat(self, prompt, **kwargs):
            return LLMResult(text='{"answer_var": "x", "atoms": ['
                                  '{"relation": "trabalha em", "subject": "?x", "object": "Atlas"},'
                                  '{"relation": "pesquisa", "subject": "?x", "object": "Optica"}],'
                                  ' "aggregation": "none"}')

    passages = [Passage(pid=pid, title=f"t{i}", text=f"{s} {r} {o}.")
                for i, (s, r, o, pid) in enumerate(TRIPLES)]
    corpus = Corpus(name="toy", passages=passages, questions=[])
    facts = [Fact(fid=f"f{i}", subject=s, relation=r, object=o, pid=pid, confidence=0.9)
             for i, (s, r, o, pid) in enumerate(TRIPLES)]
    embedder = ExactEmbedder()
    extraction = ExtractionResult(facts=facts)
    kg = build_graph(corpus, extraction, embedder, C.GraphConfig(), with_passage_nodes=True)
    run = C.RunConfig(top_k=5)
    run.witness.grounding_mode = "exact"
    run.witness.entity_match_threshold = 0.5
    run.witness.enable_acquisition = False
    run.witness.answer_set = answer_set_on
    return IndexContext(corpus=corpus, llm=Compiler(), embedder=embedder, run=run,
                        extraction=extraction, kg=kg)


def run_retriever(answer_set_on: bool, aggregation: str = "none"):
    from wrag.methods.witnessrag import WitnessRAGRetriever

    ctx = build_context(answer_set_on)
    if aggregation != "none":
        original = ctx.llm.chat

        def chat(prompt, **kwargs):
            result = original(prompt, **kwargs)
            return LLMResult(text=result.text.replace('"aggregation": "none"',
                                                      f'"aggregation": "{aggregation}"'))
        ctx.llm.chat = chat
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    question = Question(qid="q", question="Quem trabalha na Atlas e pesquisa Óptica?",
                        answers=["Ana, Bruno"])
    return retriever.retrieve(question, ctx.run.top_k)


def test_retriever_answers_with_the_whole_set():
    plain = run_retriever(False).diagnostics
    grouped = run_retriever(True).diagnostics
    # Sem a opção, a resposta estrutural é UMA atribuição: metade do gabarito.
    assert normalize(plain["resposta_estrutural"]) in ("ana", "bruno")
    assert "conjunto_resposta" not in plain
    itens = {normalize(i["resposta"]) for i in grouped["conjunto_resposta"]["itens"]}
    assert itens == {"ana", "bruno"}
    assert normalize(grouped["resposta_estrutural"]) in ("ana bruno", "bruno ana")
    assert grouped["conjunto_resposta"]["completude_certificada"] is False


def test_retriever_executes_count_only_with_the_answer_set():
    grouped = run_retriever(True, aggregation="count").diagnostics
    assert grouped["resposta_estrutural"] == "2"
    assert grouped["agregacao_executada"] == "count"
    plain = run_retriever(False, aggregation="count").diagnostics
    assert plain["fallback"] == "consulta ausente, inválida ou fora do escopo"
    assert plain["agregacao"] == "count"


def test_retriever_chooses_the_first_plan_that_closes(monkeypatch):
    from wrag.methods.witnessrag import WitnessRAGRetriever

    ctx = build_context(answer_set_on=True)
    ctx.run.witness.query_plans = True
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    bad = ConjunctiveQuery(answer_var="x", atoms=[Atom("missing", "Ana", "?x")])
    good = intersection_query()
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans",
                        lambda *args, **kwargs: [bad, good])
    result = retriever.retrieve(
        Question("q", "Quem trabalha na Atlas e pesquisa Óptica?", ["Ana, Bruno"]), 5)
    assert result.diagnostics["plano_escolhido"] == 1
    assert [p["fechou"] for p in result.diagnostics["planos_compilados"]] == [False, True]
    assert "fallback" not in result.diagnostics


def test_retriever_replans_after_observed_gap_without_using_gold(monkeypatch):
    from wrag.methods.witnessrag import WitnessRAGRetriever

    ctx = build_context(answer_set_on=True)
    ctx.run.witness.query_plans = True
    ctx.run.witness.max_query_plans = 3
    ctx.run.witness.enable_acquisition = False
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    missing = ConjunctiveQuery(answer_var="x", atoms=[Atom("missing", "Ana", "?x")])
    good = intersection_query()
    feedback_seen = []

    def compile_plans(*args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return [good] if kwargs.get("feedback") else [missing]

    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans", compile_plans)
    result = retriever.retrieve(
        Question("q", "Quem trabalha na Atlas e pesquisa Óptica?", ["SECRET_GOLD"]), 5)
    assert "fallback" not in result.diagnostics
    assert result.diagnostics["plano_escolhido"] == 1
    assert result.diagnostics["planejamento"]["planos_distintos"] == 2
    assert result.diagnostics["planejamento"]["chamadas"] == 2
    assert feedback_seen[1] and "missing" in feedback_seen[1]
    assert "SECRET_GOLD" not in feedback_seen[1]


def test_acquired_candidate_facts_are_visible_to_replanning(monkeypatch):
    from wrag.methods.witnessrag import AcquisitionAction, WitnessRAGRetriever

    ctx = build_context(answer_set_on=True)
    ctx.run.witness.query_plans = True
    ctx.run.witness.max_query_plans = 3
    ctx.run.witness.enable_acquisition = True
    ctx.run.witness.acquisition_rounds = 1
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    missing = ConjunctiveQuery(answer_var="x", atoms=[Atom("missing", "Ana", "?x")])
    good = intersection_query()
    feedback_seen = []

    def compile_plans(*args, **kwargs):
        feedback_seen.append(kwargs.get("feedback", ""))
        return [good] if kwargs.get("feedback") else [missing]

    def acquire(_actions, _question):
        retriever._last_acquired_facts = [
            Fact("new", "Ana", "pesquisa", "Optica", "p2", acquired=True)]
        return 1

    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans", compile_plans)
    monkeypatch.setattr(retriever, "_plan_acquisition", lambda *_: [
        AcquisitionAction("p2", "t2", "Ana pesquisa Optica.", "missing", "Ana", 1.0, 0.0)])
    monkeypatch.setattr(retriever, "_acquire", acquire)
    result = retriever.retrieve(
        Question("q", "Quem trabalha na Atlas e pesquisa Óptica?", ["SECRET_GOLD"]), 5)
    assert "fallback" not in result.diagnostics
    assert '"candidate_facts_acquired"' in feedback_seen[1]
    assert '"relation":"pesquisa"' in feedback_seen[1]
    assert "SECRET_GOLD" not in feedback_seen[1]
    assert result.diagnostics["planejamento"]["fatos_candidatos_adquiridos"] == 1


def test_verifier_rejection_tries_next_complete_plan_with_global_budget(monkeypatch):
    from wrag.methods.witnessrag import WitnessRAGRetriever

    ctx = build_context(answer_set_on=True)
    ctx.run.witness.query_plans = True
    ctx.run.witness.max_query_plans = 2
    ctx.run.witness.verify_witnesses = True
    ctx.run.witness.verification_max_witnesses = 2
    ctx.run.witness.enable_acquisition = False
    first = ConjunctiveQuery(answer_var="x", atoms=[Atom("localizada em", "Atlas", "?x")])
    second = intersection_query()
    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans",
                        lambda *args, **kwargs: [first, second])

    class PlanJudge:
        name = "judge"
        usage = type("U", (), {"snapshot": staticmethod(lambda: {})})()

        def chat(self, prompt, **kwargs):
            if '"answer": "Recife"' in prompt:
                return LLMResult(text='{"supported":false,"answers_question":false,'
                                      '"evidence":[],"failure_type":"wrong_answer_type",'
                                      '"reason":"wrong plan"}')
            who = "Ana" if '"answer": "Ana"' in prompt else "Bruno"
            p_work, p_research = (("p0", "p2") if who == "Ana" else ("p3", "p4"))
            return LLMResult(text=(
                '{"supported":true,"answers_question":true,"failure_type":"supported",'
                '"reason":"supported","evidence":['
                f'{{"atom":0,"pid":"{p_work}","quote":"{who} trabalha em Atlas."}},'
                f'{{"atom":1,"pid":"{p_research}","quote":"{who} pesquisa Optica."}}]}}'))

    ctx.llm = PlanJudge()
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    result = retriever.retrieve(
        Question("q", "Quem trabalha na Atlas e pesquisa Óptica?", ["Ana"]), 5)
    assert "fallback" not in result.diagnostics
    assert result.diagnostics["plano_escolhido"] == 1
    assert result.diagnostics["verificacao"]["avaliadas"] == 2
    assert result.diagnostics["verificacao"]["aceitas"] == 1
    assert [d["plano"] for d in result.diagnostics["verificacao"]["decisoes"]] == [0, 1]


def test_verification_budget_is_reserved_for_a_future_replan(monkeypatch):
    from wrag.methods.witnessrag import WitnessRAGRetriever

    ctx = build_context(answer_set_on=True)
    ctx.run.witness.query_plans = True
    ctx.run.witness.max_query_plans = 2
    ctx.run.witness.verify_witnesses = True
    ctx.run.witness.verification_max_witnesses = 2
    ctx.run.witness.enable_acquisition = False
    first = ConjunctiveQuery(answer_var="x", atoms=[Atom("localizada em", "Atlas", "?x")])
    second = intersection_query()

    def compile_plans(*args, **kwargs):
        return [second] if kwargs.get("feedback") else [first]

    def verify(_llm, _corpus, _memory, _question, query, witnesses, limit, _dataset):
        evaluated = min(limit, len(witnesses))
        accepted = witnesses[:evaluated] if len(query.atoms) == 2 else []
        return accepted, {
            "avaliadas": evaluated, "aceitas": len(accepted), "nao_avaliadas": 0,
            "rejeicoes_por_tipo": {} if accepted else {"wrong_answer_type": evaluated},
            "decisoes": [{"supported": bool(accepted)} for _ in range(evaluated)],
        }

    monkeypatch.setattr("wrag.methods.witnessrag.compile_query_plans", compile_plans)
    monkeypatch.setattr("wrag.witness.verification.verify_witnesses", verify)
    retriever = WitnessRAGRetriever(ctx)
    retriever.index()
    result = retriever.retrieve(
        Question("q", "Quem trabalha na Atlas e pesquisa Óptica?", ["Ana"]), 5)

    assert "fallback" not in result.diagnostics
    assert result.diagnostics["plano_escolhido"] == 1
    assert result.diagnostics["planejamento"]["chamadas"] == 2
    assert result.diagnostics["verificacao"]["avaliadas"] == 2


def test_acquisition_survives_a_fallback_with_another_score_scale():
    """A aquisição não pode depender da escala do recuperador de reserva.

    O cosseno do denso fica perto de 0,6 e passa por λ=0,05; a fusão recíproca de
    postos devolve ~0,016 e reprovaria toda ação. Normalizar pelo topo da
    consulta mantém a ORDEM das ações e torna λ um corte de cauda.
    """
    from wrag.methods.witnessrag import WitnessRAGRetriever
    from wrag.witness.search import Gap

    for scale, nome in ((1.0, "cosseno"), (0.016, "rrf")):
        ctx = build_context(answer_set_on=True)
        retriever = WitnessRAGRetriever(ctx)
        retriever.index()
        retriever._dense.search = lambda text, k, s=scale: (
            ["p0", "p1", "p2"], [0.60 * s, 0.40 * s, 0.20 * s])
        gap = Gap(atom_index=0, atom=Atom("pesquisa", "Ana", "?x"), bound_subject="Ana")
        question = Question(qid="q", question="O que Ana pesquisa?", answers=["Optica"])
        actions = retriever._plan_acquisition(gap, question)
        assert actions, f"nenhuma ação planejada na escala {nome}"
        assert [a.pid for a in actions] == ["p0", "p1", "p2"]
        assert actions[0].expected_gain == pytest.approx(1.0)
