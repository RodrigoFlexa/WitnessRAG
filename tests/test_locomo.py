import copy
import json
from pathlib import Path

import pytest

from wrag.data import load_dataset
from wrag.locomo import convert
from wrag.pilot import make_plan, parser, prepare_data
from wrag.util import read_json


def sample():
    return [{"sample_id": "conv-test", "conversation": {
        "session_2_date_time": "2023-05-10", "session_2": [
            {"dia_id": "D2:1", "speaker": "Ana", "text": "I live in Recife."}],
        "session_1_date_time": "2023-05-01", "session_1": [
            {"dia_id": "D1:1", "speaker": "Ana", "text": "I work at Atlas."},
            {"dia_id": "D1:2", "speaker": "Bruno", "text": "Here is my drawing.",
             "blip_caption": "A blue bird"}],
        "speaker_a": "Ana", "speaker_b": "Bruno"},
        "qa": [{"question": f"Question category {c}?", "answer": "SECRET_GOLD",
                "category": c, "evidence": ["D1:1; D2:1"]} for c in range(1, 6)],
        "session_summary": "SECRET_SUMMARY", "observation": "SECRET_OBSERVATION"},
        {"sample_id": "second", "conversation": {}, "qa": []}]


def test_categories_complete_conversation_and_evidence_mapping(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(sample()), encoding="utf-8")
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--locomo-file", str(source)])
    plan = make_plan(args, tmp_path / "output")
    assert plan["methods"] == ["witnessrag"] and plan["settings"]["questions"] is None
    meta = prepare_data(plan)
    corpus = load_dataset("locomo", data_dir=tmp_path / "output/data")
    assert [q.qtype for q in corpus.questions] == ["multi-hop", "single-hop"]
    assert len(corpus.passages) == 2
    assert all(len(q.gold_pids) == 2 for q in corpus.questions)
    assert meta["questions_by_type"] == {"multi-hop": 1, "single-hop": 1}
    text = "\n".join(corpus.texts())
    assert "Ana: I work at Atlas." in text and "2023-05-01" in text and "A blue bird" in text
    assert "SECRET" not in text and "Question category" not in text
    assert "session_1" in corpus.passages[0].title


def test_sampling_never_removes_dialogue_and_ignores_qa_when_chunking():
    raw = sample()
    q, passages, _ = convert(raw, turns_per_passage=1)
    raw[0]["qa"][0]["answer"] = "different answer"
    one, other_passages, _ = convert(raw, turns_per_passage=1, n_questions=1)
    assert len(one) == 1 and len(q) == 2 and passages == other_passages
    assert len(passages) == 3


def test_token_chunking_is_dialogue_only_and_maps_supports():
    questions, passages, metadata = convert(
        sample(), chunk_tokens=12, token_counter=lambda text: len(text.split()))
    assert len(passages) > 1 and metadata["chunk_tokens"] == 12
    assert metadata["tokenizer_name"] == "Qwen/Qwen2.5-14B-Instruct"
    assert all(q["paragraphs"] for q in questions)
    joined = "\n".join(p["text"] for p in passages)
    assert "SECRET" not in joined and "Question category" not in joined


@pytest.mark.parametrize("error", ["missing", "duplicate", "empty_evidence", "index"])
def test_invalid_annotations_fail_explicitly(error):
    raw = copy.deepcopy(sample())
    if error == "missing": raw[0]["qa"][0]["evidence"] = ["D999:1"]
    if error == "empty_evidence": raw[0]["qa"][0]["evidence"] = []
    if error == "duplicate": raw[0]["conversation"]["session_2"][0]["dia_id"] = "D1:1"
    with pytest.raises(ValueError):
        convert(raw, conversation_index=99 if error == "index" else 0)


def test_known_upstream_evidence_typo_is_repaired_and_audited():
    raw = [{"sample_id": "conv-43", "conversation": {"session_11": [
        {"dia_id": "D11:26", "speaker": "A", "text": "The book was The Alchemist."}
    ]}, "qa": [
        {"question": "ignored", "answer": "x", "category": 2, "evidence": []}
        for _ in range(18)
    ] + [{"question": "Which book?", "answer": "The Alchemist", "category": 4,
          "evidence": ["D:11:26"]}]}]
    questions, _passages, metadata = convert(raw)
    assert len(questions) == 1
    assert metadata["evidence_repairs"] == [
        {"qa_index": 18, "original": "D:11:26", "replacement": "D11:26"}]
    assert metadata["question_mapping"][questions[0]["id"]]["evidence_dialog_ids"] == ["D11:26"]


def test_other_pilot_defaults_unchanged(tmp_path):
    plan = make_plan(parser().parse_args(["--gpu", "3"]), tmp_path)
    assert plan["settings"]["questions"] == 100
    assert "hipporag2" in plan["methods"] and len(plan["methods"]) == 7


def test_locomo_witness_pipeline_and_category_report(tmp_path, monkeypatch):
    from wrag import config as C, methods
    from wrag.eval import runner
    from wrag.embed import TfidfEmbedder
    from wrag.llm.stub import StubLLM
    from wrag.util import write_json
    questions, passages, _ = convert(sample())
    write_json(tmp_path / "locomo.json", questions)
    write_json(tmp_path / "locomo_corpus.json", passages)
    for name, path in (("DATA_DIR", tmp_path), ("CACHE_DIR", tmp_path / "cache"),
                       ("RUNS_DIR", tmp_path / "runs")):
        monkeypatch.setattr(C, name, path)
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    llm, emb = StubLLM(filter_rate=0), TfidfEmbedder()
    for module in (runner, methods):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
        monkeypatch.setattr(module, "get_embedder", lambda: emb)
    cfg = C.RunConfig(n_questions=2, corpus_scope="locomo_full_selected_conversation")
    cfg.witness.enable_acquisition = False
    root = runner.run(["locomo"], ["witnessrag"], cfg, tag="locomo-offline")
    report = read_json(root / "report.json")["datasets"]["locomo"]
    assert set(report["metodos"]) == {"witnessrag"}
    assert report["por_categoria"]["witnessrag"]["single-hop"]["n"] == 1
    assert report["por_categoria"]["witnessrag"]["multi-hop"]["n"] == 1
    # A coluna oficial acompanha a do harness, sem substituí-la.
    categoria = report["por_categoria"]["witnessrag"]["multi-hop"]
    assert {"f1", "f1_locomo", "em_locomo"} <= set(categoria)
    assert (root / "report.md").read_text(encoding="utf-8").count("F1 ofic.") == 1


# --- avaliador oficial ------------------------------------------------------

official = pytest.importorskip("wrag.eval.locomo_official")
pytest.importorskip("nltk", reason="o avaliador oficial usa nltk.stem.PorterStemmer")


def test_official_normalization_drops_and_and_commas():
    assert official.normalize_answer("clarinet, and the violin") == "clarinet violin"


def test_official_single_hop_f1_uses_stemming():
    # "creating" e "create" colapsam no mesmo radical; o F1 do harness não faria isso.
    assert official.question_score("creating art", "create art", 4) == 1.0


def test_official_multi_hop_splits_only_on_commas():
    # Categoria 1: média sobre o ouro do melhor F1 previsto. Metade do ouro, meio ponto.
    assert official.question_score("clarinet", "clarinet, violin", 1) == 0.5


def test_official_multi_hop_does_not_penalize_extra_items():
    # Assimetria do avaliador oficial: prever itens a mais é de graça.
    assert official.question_score("clarinet, violin, drums", "clarinet, violin", 1) == 1.0


def test_official_exact_match_ignores_order():
    assert official.exact_match_score("Matt Patterson, Summer Sounds",
                                      "Summer Sounds, Matt Patterson") == 1.0


def test_official_evidence_recall_is_per_evidence():
    assert official.evidence_recall(["D1:1", "D1:2"], ["D1:1", "D3:9"]) == 0.5


def test_score_record_dispatches_by_question_type():
    record = {"qid": "q", "tipo": "multi-hop", "resposta": "clarinet",
              "respostas_ouro": ["clarinet, violin"]}
    scores = official.score_record(record)
    assert scores["categoria_locomo"] == 1 and scores["f1_locomo"] == 0.5


def test_locomo_pipeline_with_the_new_options(tmp_path, monkeypatch):
    """Rodada offline com conjunto de respostas, vocabulário, híbrido e diálogo.

    Verifica que as quatro opções atravessam CLI, configuração, extração, busca e
    relatório sem quebrar o pipeline. Não afirma nada sobre qualidade: o LLM é o
    stub determinístico.
    """
    from wrag import config as C, methods
    from wrag.eval import runner
    from wrag.embed import TfidfEmbedder
    from wrag.llm.stub import StubLLM
    from wrag.util import write_json
    questions, passages, _ = convert(sample())
    write_json(tmp_path / "locomo.json", questions)
    write_json(tmp_path / "locomo_corpus.json", passages)
    for name, path in (("DATA_DIR", tmp_path), ("CACHE_DIR", tmp_path / "cache"),
                       ("RUNS_DIR", tmp_path / "runs")):
        monkeypatch.setattr(C, name, path)
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    llm, emb = StubLLM(filter_rate=0), TfidfEmbedder()
    for module in (runner, methods):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
        monkeypatch.setattr(module, "get_embedder", lambda: emb)
    cfg = C.RunConfig(n_questions=2, top_k=3, corpus_scope="locomo_full_selected_conversation")
    cfg.witness.enable_acquisition = False
    cfg.witness.answer_set = True
    cfg.qa.answer_set = True
    cfg.witness.vocabulary_aware_compile = True
    cfg.witness.hybrid_fallback = True
    cfg.ie.dialogue_mode = True
    root = runner.run(["locomo"], ["witnessrag"], cfg, tag="locomo-opcoes")
    report = read_json(root / "report.json")["datasets"]["locomo"]
    parou = report["onde_parou"]["witnessrag"]
    assert parou["n"] == 2 and 0.0 <= parou["taxa_de_disparo"] <= 1.0
    assert sum(parou["por_parada"].values()) == 2
    assert "Onde a pergunta parou" in (root / "report.md").read_text(encoding="utf-8")
    # A configuração da rodada fica registrada: uma opção ligada tem que aparecer.
    manifesto = read_json(root / "run.json")["config"]
    assert manifesto["witness"]["answer_set"] and manifesto["ie"]["dialogue_mode"]
    assert manifesto["witness"]["hybrid_fallback"] and manifesto["qa"]["answer_set"]


# --- todas as conversas -----------------------------------------------------

def two_conversations():
    """Duas conversas com identificadores próprios, no formato do locomo10.json."""
    primeira = sample()[0]
    segunda = copy.deepcopy(primeira)
    segunda["sample_id"] = "conv-test-2"
    return [primeira, segunda]


def test_all_selects_every_conversation(tmp_path):
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo",
                                "--locomo-conversation", "all"])
    plan = make_plan(args, tmp_path)
    assert plan["settings"]["locomo_conversation"] == "all"
    from wrag.pilot import LOCOMO_CONVERSATIONS, locomo_conversations
    assert locomo_conversations(plan) == list(range(LOCOMO_CONVERSATIONS))

    single = make_plan(parser().parse_args(["--gpu", "3", "--dataset", "locomo",
                                            "--locomo-conversation", "3"]), tmp_path)
    assert single["settings"]["locomo_conversation"] == 3
    assert locomo_conversations(single) == [3]


def test_every_conversation_runs_on_its_own_corpus(tmp_path, monkeypatch):
    """Dez memórias separadas, não um índice único: uma rodada por conversa.

    As perguntas de uma conversa nunca podem ser respondidas com o diálogo de
    outra — no arquivo oficial "John" fala em três conversas diferentes.
    """
    from wrag import config as C, methods
    from wrag.eval import runner
    from wrag.embed import TfidfEmbedder
    from wrag.llm.stub import StubLLM
    from wrag.pilot import _run_every_conversation
    from wrag.util import write_json

    source = tmp_path / "locomo10.json"
    write_json(source, two_conversations())
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--methods", "witnessrag",
                                "--locomo-conversation", "all", "--locomo-file", str(source)])
    plan = make_plan(args, tmp_path / "saida")
    monkeypatch.setattr("wrag.pilot.LOCOMO_CONVERSATIONS", 2)
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(C, "EMBED_CACHE", False)
    llm, emb = StubLLM(filter_rate=0), TfidfEmbedder()
    for module in (runner, methods):
        monkeypatch.setattr(module, "get_llm", lambda: llm)
        monkeypatch.setattr(module, "get_embedder", lambda: emb)

    roots = _run_every_conversation(plan, plan["settings"])
    assert len(roots) == 2
    corpora = set()
    for index, root in enumerate(roots):
        assert f"conv{index:02d}" in root
        summary = read_json(Path(root) / "locomo" / "summary.json")
        corpora.add(summary["corpus"]["corpus_hash"])
        # Cada rodada enxerga apenas as passagens da sua conversa.
        assert summary["corpus"]["n_passages"] == 2
    # Corpora distintos, um por conversa: é o que impede responder a pergunta de
    # uma conversa com o diálogo de outra.
    assert len(corpora) == 2

    conversas = read_json(tmp_path / "saida" / "conversations.json")["conversas"]
    assert [c["conversa"] for c in conversas] == [0, 1]
    assert [c["sample_id"] for c in conversas] == ["conv-test", "conv-test-2"]

    # Uma retomada íntegra pula as duas rodadas já terminadas.
    monkeypatch.setattr(runner, "run", lambda *_a, **_k: pytest.fail("não deveria rerodar"))
    assert _run_every_conversation(plan, plan["settings"]) == roots


def test_deadline_stops_between_conversations(tmp_path, monkeypatch):
    """Prazo estourado não começa conversa nova: rodada parcial não é comparável."""
    from wrag.pilot import _run_every_conversation
    from wrag.util import write_json

    write_json(tmp_path / "locomo10.json", two_conversations())
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--methods", "witnessrag",
                                "--locomo-conversation", "all",
                                "--locomo-file", str(tmp_path / "locomo10.json")])
    plan = make_plan(args, tmp_path / "saida")
    plan["deadline_epoch"] = 0.0          # já vencido
    assert _run_every_conversation(plan, plan["settings"]) == []


def test_aggregate_micro_averages_over_questions(tmp_path, capsys):
    """A média é micro: a conversa com mais perguntas pesa mais."""
    from wrag.eval.locomo_official import aggregate_runs
    from wrag.util import write_json

    def run_dir(name, rows):
        path = tmp_path / name / "locomo"
        path.mkdir(parents=True)
        (path / "witnessrag.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return path.parent

    def row(qid, tipo, pred, gold):
        return {"qid": qid, "tipo": tipo, "resposta": pred, "respostas_ouro": [gold],
                "f1": 1.0, "em": 1.0, "recall@5": 1.0, "all_recall@5": 1.0}

    grande = run_dir("a", [row(f"locomo:conv-a:qa{i}", "single-hop", "x", "x") for i in range(3)])
    pequena = run_dir("b", [row("locomo:conv-b:qa0", "single-hop", "y", "z")])
    summary = aggregate_runs([grande, pequena])
    values = summary["metodos"]["witnessrag"]
    assert values["n"] == 4
    assert set(values["por_conversa"]) == {"conv-a", "conv-b"}
    if summary["oficial_disponivel"]:
        # 3 acertos e 1 erro: micro dá 0,75, não a média 0,5 entre conversas.
        assert values["f1_locomo"] == pytest.approx(0.75)

    from wrag.pilot import print_locomo_aggregate
    print_locomo_aggregate(tmp_path / "final", "complete", [grande, pequena])
    terminal = capsys.readouterr().out
    assert "Por conversa" not in terminal and "2 conversa(s), 4 perguntas" in terminal
    saved = read_json(tmp_path / "final" / "locomo_agregado.json")
    assert "por_conversa" not in saved["metodos"]["witnessrag"]
