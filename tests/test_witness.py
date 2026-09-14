"""
Verificação do núcleo, em instâncias pequenas onde a resposta certa é conhecida.

Os testes são deliberadamente sobre a instância de cinco fatos da proposta:

    e1 Ana trabalha na Atlas   e2 Atlas fica em Recife   e3 Ana pesquisa Óptica
    e4 Bruno trabalha na Atlas e5 Bruno pesquisa Óptica

É pequena o bastante para enumerar à mão e grande o bastante para conter o caso
que separa este método dos que ranqueiam por relevância: a consulta de
interseção, em que recuperar um funcionário da Atlas e outra pessoa que pesquisa
Óptica não responde nada.

Rode com `pytest tests/` ou `python tests/test_witness.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from wrag import config as C
from wrag.data import Corpus, Passage, Question
from wrag.embed import Embedder
from wrag.graph import build_graph
from wrag.ie import ExtractionResult, Fact
from wrag.llm.base import parse_json_loose
from wrag.llm.filters import is_content_filter_error
from wrag.util import normalize
from wrag.witness.budget import (Demand, select_greedy, select_ilp,
                                 submodularity_counterexample, toy_instance, utility)
from wrag.witness.memory import MemoryView
from wrag.witness.provenance import rank_passages, score_answers
from wrag.witness.query import Atom, ConjunctiveQuery, from_evidences
from wrag.witness.search import WitnessSearcher


class ExactEmbedder(Embedder):
    """Embedder de teste: 1.0 para strings normalizadas iguais, 0 caso contrário.

    Torna o aterramento determinístico, o que é o ponto: um teste que dependesse
    da qualidade de um modelo de embedding estaria testando o modelo, não a
    junção.
    """

    name = "exact"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.dim = 64

    def _encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            key = normalize(text)
            index = self.vocab.setdefault(key, len(self.vocab) % self.dim)
            out[row, index] = 1.0
        return out

    def _encode_cached(self, texts, desc):
        # Vocabulário incremental é local à instância; não reutilizar outro teste.
        return self._encode(texts)


TRIPLES = [
    ("Ana", "trabalha em", "Atlas", "p0"),
    ("Atlas", "localizada em", "Recife", "p1"),
    ("Ana", "pesquisa", "Optica", "p2"),
    ("Bruno", "trabalha em", "Atlas", "p3"),
    ("Bruno", "pesquisa", "Optica", "p4"),
]


def build_toy():
    passages = [Passage(pid=f"p{i}", title=f"t{i}", text=f"{s} {r} {o}.")
                for i, (s, r, o, _pid) in enumerate(TRIPLES)]
    corpus = Corpus(name="toy", passages=passages, questions=[])
    facts = [Fact(fid=f"f{i}", subject=s, relation=r, object=o, pid=pid, confidence=0.9)
             for i, (s, r, o, pid) in enumerate(TRIPLES)]
    embedder = ExactEmbedder()
    kg = build_graph(corpus, ExtractionResult(facts=facts), embedder, C.GraphConfig(),
                     with_passage_nodes=True)
    memory = MemoryView(kg)
    cfg = C.WitnessConfig(grounding_mode="exact", entity_match_threshold=0.5, candidates_per_atom=50)
    return memory, WitnessSearcher(memory, embedder, cfg), embedder, cfg


def facts_of(memory, witness):
    return {memory.facts[i].fid for i in witness.facts}


# ---------------------------------------------------------------------------

def test_single_hop():
    memory, searcher, _e, _c = build_toy()
    query = ConjunctiveQuery(answer_var="x",
                             atoms=[Atom("trabalha em", "Ana", "?x")])
    result = searcher.join(query)
    assert result.complete, "single-hop deveria fechar"
    assert normalize(result.witnesses[0].answer) == "atlas"
    assert facts_of(memory, result.witnesses[0]) == {"f0"}


def test_chain_two_hops():
    """q2(c) = ∃i trabalhaEm(Ana, i) ∧ localizadaEm(i, c) → Recife, testemunha {e1,e2}."""
    memory, searcher, _e, _c = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?i"),
        Atom("localizada em", "?i", "?x"),
    ])
    result = searcher.join(query)
    assert result.complete
    best = result.witnesses[0]
    assert normalize(best.answer) == "recife"
    assert facts_of(memory, best) == {"f0", "f1"}


def test_intersection_requires_same_variable():
    """q3(p) = trabalhaEm(p, Atlas) ∧ pesquisa(p, Óptica).

    As respostas são Ana e Bruno, com testemunhas {e1,e3} e {e4,e5}. O que NÃO
    pode aparecer é uma testemunha cruzada {e1,e5}: a variável compartilhada
    proíbe misturar o emprego de Ana com a pesquisa de Bruno. É exatamente o erro
    que um ranking por relevância comete.
    """
    memory, searcher, _e, cfg = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "?x", "Atlas"),
        Atom("pesquisa", "?x", "Optica"),
    ])
    result = searcher.join(query)
    assert result.complete
    answers = {normalize(w.answer) for w in result.witnesses}
    assert answers == {"ana", "bruno"}, answers
    sets = [facts_of(memory, w) for w in result.witnesses]
    assert {"f0", "f2"} in sets and {"f3", "f4"} in sets
    for s in sets:
        assert s in ({"f0", "f2"}, {"f3", "f4"}), f"testemunha cruzada: {s}"


def test_no_witness_when_relation_absent():
    memory, searcher, _e, _c = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?i"),
        Atom("fundada por", "?i", "?x"),
    ])
    result = searcher.join(query)
    assert not result.complete
    assert result.gap is not None
    # A lacuna precisa dizer QUAL relação faltou e sobre QUAL entidade; é o que a
    # aquisição adaptativa consome.
    assert result.gap.atom.relation == "fundada por"
    assert normalize(result.gap.bound_subject) == "atlas"


def test_provenance_and_passages():
    memory, searcher, _e, cfg = build_toy()
    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "?x", "Atlas"),
        Atom("pesquisa", "?x", "Optica"),
    ])
    result = searcher.join(query)
    candidates = score_answers(result.witnesses, memory, cfg)
    assert len(candidates) == 2
    for candidate in candidates:
        assert 0.0 < candidate.score <= 1.0
        assert candidate.risk_raw >= cfg.delta_interpretation + cfg.delta_verbalization
        # Duas evidências independentes: o limite da união soma dois δ_e.
        assert len(candidate.witnesses[0].facts) == 2

    pids, scores = rank_passages(candidates, k=4)
    assert set(pids) <= {"p0", "p2", "p3", "p4"}
    assert len(pids) == len(scores)


def test_memory_rollback():
    """A aquisição escreve e desfaz: sem isso, a pergunta N herda fatos da N-1."""
    memory, searcher, embedder, _c = build_toy()
    before = (len(memory.facts), len(memory.entities), len(memory.relations))

    new = [Fact(fid="fx", subject="Atlas", relation="fundada por", object="Clara", pid="p1")]
    added = memory.add_facts(new, embedder)
    searcher.register_facts(added)
    assert len(memory.facts) == before[0] + 1
    assert memory.entity_id("Clara") is not None

    query = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("trabalha em", "Ana", "?i"),
        Atom("fundada por", "?i", "?x"),
    ])
    result = searcher.join(query)
    assert result.complete, "o fato adquirido deveria fechar a testemunha"
    assert normalize(result.witnesses[0].answer) == "clara"

    searcher.rollback()
    memory.reset()
    assert (len(memory.facts), len(memory.entities), len(memory.relations)) == before
    assert memory.entity_id("Clara") is None
    assert not searcher.join(query).complete


def test_minimality():
    """Uma testemunha que contém outra suficiente não é testemunha nova."""
    memory, searcher, embedder, cfg = build_toy()
    # Duas fontes para o mesmo fato: duas demonstrações da mesma resposta.
    extra = [Fact(fid="f0b", subject="Ana", relation="trabalha em", object="Atlas", pid="p9")]
    added = memory.add_facts(extra, embedder)
    searcher.register_facts(added)
    query = ConjunctiveQuery(answer_var="x", atoms=[Atom("trabalha em", "Ana", "?x")])
    result = searcher.join(query)
    sizes = {len(w.facts) for w in result.witnesses}
    assert sizes == {1}, f"testemunhas não mínimas: {sizes}"
    atlas = [w for w in result.witnesses if normalize(w.answer) == "atlas"]
    assert len(atlas) == 2, "duas fontes = duas demonstrações alternativas"
    assert {w.pids for w in atlas} == {("p0",), ("p9",)}
    # O aterramento é suave, então o átomo também casa fracamente com fatos de
    # outra relação sobre a mesma entidade ("Ana pesquisa Óptica"). Isso é
    # intencional — o extrator é estocástico — e o preço é pago na proveniência,
    # onde o score baixo derruba a demonstração.
    outros = [w for w in result.witnesses if normalize(w.answer) != "atlas"]
    assert all(w.score <= atlas[0].score for w in outros)


def test_budget_toy_instance():
    """A instância verificada na proposta: {e1,e2,e3}, valor 8, orçamento 3."""
    demands, costs = toy_instance()
    result = select_ilp(demands, costs, budget=3.0)
    assert sorted(result.kept) == [1, 2, 3], result.kept
    assert result.value == 8.0
    assert utility(demands, {1, 2, 3}) == 8.0
    assert utility(demands, {1, 4, 5}) == 5.0     # perde a cidade de Ana


def test_submodularity_counterexample():
    out = submodularity_counterexample()
    assert out["ganho_sozinho"] == 0.0
    assert out["ganho_com_e1"] == 1.0
    assert out["submodular"] == 0.0


def test_budget_binds():
    """Com orçamento apertado, a utilidade cai — e o ILP não estoura o teto."""
    demands, costs = toy_instance()
    result = select_ilp(demands, costs, budget=2.0)
    assert sum(costs[e] for e in result.kept) <= 2.0
    assert result.value < 8.0
    greedy = select_greedy(demands, costs, budget=2.0)
    assert greedy.value <= result.value, "o guloso não pode superar o ótimo"


def test_query_shape_classification():
    chain = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("mother", "Lothair II", "?y"), Atom("date of death", "?y", "?x")])
    intersection = ConjunctiveQuery(answer_var="x", atoms=[
        Atom("works at", "?x", "Atlas"), Atom("researches", "?x", "Optics")])
    single = ConjunctiveQuery(answer_var="x", atoms=[Atom("works at", "Ana", "?x")])
    assert chain.shape() == "chain"
    assert intersection.shape() == "intersection"
    assert single.shape() == "single-hop"


def test_from_evidences_builds_chain():
    """As triplas anotadas do 2Wiki viram a consulta cuja testemunha é a anotação."""
    question = Question(
        qid="q", question="When did Lothair II's mother die?", answers=["20 March 851"],
        evidences=[("Lothair II", "mother", "Ermengarde of Tours"),
                   ("Ermengarde of Tours", "date of death", "20 March 851")],
    )
    query = from_evidences(question)
    assert query.n_atoms == 2
    assert query.shape() == "chain"
    assert query.atoms[0].subject == "Lothair II"
    assert query.atoms[0].object.startswith("?")           # entidade intermediária
    assert query.atoms[1].object == "?x"                   # a resposta


def test_json_parser_tolerance():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Claro! {"a": [1,2]} espero ter ajudado') == {"a": [1, 2]}
    assert parse_json_loose('{"a": "chave } dentro de string"}') == {"a": "chave } dentro de string"}
    assert parse_json_loose("sem json aqui") is None


def test_content_filter_detection():
    class Fake(Exception):
        pass

    policy = Fake("Error code: 400 - {'error': {'code': 'content_filter'}}")
    param = Fake("Error code: 400 - unsupported parameter: 'temperature'")
    assert is_content_filter_error(policy)
    assert not is_content_filter_error(param), "erro de parâmetro não pode virar bloqueio"


def test_metrics():
    from wrag.eval import metrics as M

    assert M.all_recall_at_k(["a", "b", "c"], ["a", "b"], 5) == 1.0
    assert M.all_recall_at_k(["a", "c"], ["a", "b"], 5) == 0.0
    assert M.recall_at_k(["a", "c"], ["a", "b"], 5) == 0.5
    assert M.exact_match("The Recife", ["recife"]) == 1.0
    assert 0.6 < M.token_f1("20 March 851", ["March 851"]) < 1.0
    assert M.witness_coverage([["Lothair II", "mother", "Ermengarde"]],
                              [("Lothair II", "mother", "Ermengarde")]) == 1.0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FALHA {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  ERRO  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
