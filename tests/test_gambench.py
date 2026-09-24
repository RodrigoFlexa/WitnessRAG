"""Protocolo do GAM (HotpotQA, RULER 128K, NarrativeQA).

O que é conferido: métricas e prompts idênticos aos do código de avaliação
do GAM, páginas com as mesmas regras, a seleção do NarrativeQA, o split do
RULER, e uma rodada ponta a ponta com o backend determinístico, incluindo
retomada, falhas, a tabela no formato do artigo e a garantia de que o motor
nunca vê respostas.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wrag import config as C
from wrag.gambench import data as D
from wrag.gambench import metrics as M
from wrag.gambench import protocol as P
from wrag.gambench import reader as R
from wrag.gambench.pages import split_pages
from wrag.gambench.report import render_suite, suite_report, table_row


# -- métricas -----------------------------------------------------------------

def test_f1_is_longbench_max_over_references():
    assert M.gam_f1("The Chief of Protocol.", ["Chief of Protocol"]) == 1.0
    assert M.gam_f1("", ["anything"]) == 0.0
    assert M.gam_f1("Paris France", ["France", "Lyon"]) == pytest.approx(2 / 3)
    assert M.gam_f1("an apple", ["the apple"]) == 1.0      # artigos saem
    assert M.qa_f1_score("x b b", "b b c") == pytest.approx(2 / 3)
    assert M.qa_f1_score("a b b", "b b c") == pytest.approx(0.8)   # "a" é artigo


def test_ruler_accuracy_requires_every_output():
    outputs = ["5713268", "8913550", "4674549", "6996728"]
    assert M.ruler_correct("5713268, 8913550, 4674549 and 6996728", outputs)
    assert not M.ruler_correct("5713268, 8913550, 4674549", outputs)
    assert not M.ruler_correct("", outputs) and not M.ruler_correct("x", [])
    # regra 2: pontuação vira espaço
    assert M.ruler_correct("the value is c44540a1 cace 4a67", ["c44540a1-cace-4a67"])
    # regra 3: todas as palavras com mais de dois caracteres
    assert M.ruler_correct("Scott and Ed were both American", ["American nationality of"]) is False
    assert M.ruler_correct("it is located in france near normandy", ["Normandy, France"])
    # respostas repetidas contam uma vez (o GAM usa set)
    assert M.ruler_correct("France", ["France", "France", "France", "France"])
    # diagnóstico oficial dá crédito parcial
    assert M.ruler_official("niah_multivalue", "5713268 8913550", outputs) == 0.5
    assert M.ruler_official("qa_1", "France", ["Paris", "France"]) == 1.0


# -- prompts e parâmetros do leitor ------------------------------------------------

def test_reader_prompts_are_the_gam_literals():
    hotpot = R.build_prompt("hotpotqa", "Q?", "CTX")
    assert hotpot == ("You are a careful multi-hop reading assistant. \nUse the given Context. \n"
                      "Answer with ONLY the final answer string; no extra words.\n\nQuestion:\nQ?"
                      "\n\nContext:\nCTX\n\nAnswer:\n")
    nqa = R.build_prompt("narrativeqa", "Q?", "CTX")
    assert nqa.startswith("You are a careful reading assistant. \nUse the given Context. \n")
    ruler = R.build_prompt("ruler", "Question: Find all X.", "CTX", example="EX")
    assert ruler == ("Read the text below and answer a question. Context: CTX\n\n"
                     "Question:\nQuestion: Find all X.\n\nHere is the example:\nEX\n\nAnswer:")
    assert R.build_prompt("ruler", "What is it?", "C") == \
        "Read the text below and answer a question. Context: C\n\nQuestion:\nWhat is it?\n\nAnswer:"
    assert R.clean_response("<think>x</think>  Paris \n") == "Paris"
    params = R.reader_params()
    assert (params.temperature, params.max_tokens, params.json_mode, params.exact_max_tokens) == \
        (0.3, 256, False, True)
    assert R.reader_params(None).seed is None


def test_azure_honors_the_exact_answer_budget(tmp_path, monkeypatch):
    from wrag.llm.azure import AzureLLM
    from wrag.llm.base import GenParams
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(AzureLLM, "_make_client", lambda _: None)
    llm = AzureLLM(deployment="gpt-4-1-mini-petrobras", max_tokens=2048)
    messages = [{"role": "user", "content": "hi"}]
    assert llm.build_kwargs(messages, R.reader_params())["max_tokens"] == 256
    assert llm.build_kwargs(messages, R.reader_params())["temperature"] == 0.3
    # sem a bandeira, o teto do backend continua valendo (chaves de cache antigas)
    assert llm.build_kwargs(messages, GenParams(max_tokens=256))["max_tokens"] == 2048


# -- páginas ----------------------------------------------------------------------

class WordTokenizer:
    """Um token por palavra; decode junta com espaço."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split(" "):
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids)


def test_pages_follow_the_gam_rules():
    tok = WordTokenizer()
    single = split_pages(" a b c ", tok, max_tokens=10)
    assert single.pages == ["[Session 1]\n a b c "]          # sem strip, número 1
    many = split_pages("a b c d e f g", tok, max_tokens=3)
    assert many.pages == ["[Session 0]\na b c", "[Session 1]\nd e f", "[Session 2]\ng"]
    assert many.context_tokens == 7
    # pedaço só de espaços é pulado sem consumir número
    skip = split_pages("a b c    ", tok, max_tokens=3)
    assert [p.split("\n")[0] for p in skip.pages] == ["[Session 0]"]
    assert split_pages("", tok).pages == []


def test_pages_with_the_bge_m3_tokenizer():
    pytest.importorskip("transformers")
    from wrag.gambench.pages import load_tokenizer
    try:
        tok = load_tokenizer()
    except Exception as exc:  # noqa: BLE001 - sem rede nem cache local
        pytest.skip(f"tokenizador do BGE-M3 indisponível: {exc}")
    text = " ".join(f"Document {i}: The quick brown fox number {i} jumps." for i in range(900))
    pages = split_pages(text, tok)
    assert len(pages.pages) == -(-pages.context_tokens // P.PAGE_TOKENS)
    assert pages.pages[0].startswith("[Session 0]\nDocument 0:")
    for page in pages.pages[:-1]:
        body = page.split("\n", 1)[1]
        assert abs(len(tok.encode(body, add_special_tokens=False)) - P.PAGE_TOKENS) <= 3


# -- dados --------------------------------------------------------------------------

def test_narrativeqa_selection_reproduces_gam():
    chosen = D.narrativeqa_selection(P.NARRATIVEQA_TEST_SIZE)
    assert tuple(chosen[:5]) == P.NARRATIVEQA_FIRST_INDICES
    assert len(chosen) == 300 and len(set(chosen)) == 300


def test_ruler_split_input_matches_gam_on_each_family():
    niah = ("A special magic number is hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the number afterwards.\nHAY. One of the special magic numbers for "
            "k-1 is: 42.\nWhat is the special magic number for k-1 mentioned in the provided text? "
            "The special magic number for k-1 mentioned in the provided text is")
    ex, ins, ctx, q = D.split_input(niah, "niah_single_2")
    assert ex == "" and ins.startswith("A special magic number") and ctx.startswith("HAY.")
    assert q.startswith("What is the special magic number for k-1")
    vt_instr = "Memorize and track the chain(s) of variable assignment hidden in the following text."
    vt = (f"{vt_instr}\n\nEXAMPLE VAR A = 1\nQuestion: ex? Answer: A\n{vt_instr}\n\n"
          "noise VAR B = 7 noise VAR C = VAR B\nQuestion: Find all variables that are assigned "
          "the value 7 in the text above. Answer:")
    ex, ins, ctx, q = D.split_input(vt, "vt")
    assert ex.startswith(vt_instr) and "EXAMPLE" in ex and "Question: ex?" in ex
    assert ctx == "noise VAR B = 7 noise VAR C = VAR B"
    assert q.startswith("Question: Find all variables")
    qa = ("The following are given documents.\n\nDocument 1:\nNormandy is in France.\n\n"
          "Answer the question based on the given documents. Only give me the answer and do not "
          "output any other words.\n\nQuestion: Where is Normandy? Answer:")
    ex, ins, ctx, q = D.split_input(qa, "qa_1")
    assert q == "Question: Where is Normandy? Answer:" and ctx == ""
    assert D.split_input("   ", "cwe") == ("", "", "", "")
    assert D.split_input("free text", "unknown") == ("", "", "free text", "")


# -- ponta a ponta ------------------------------------------------------------------

def _write_data(root: Path) -> None:
    hay = " ".join(f"Filler sentence number {i} about nothing." for i in range(60))
    hotpot = [{"index": 7, "input": "Which city hosts the Zorvex Tower?", "answers": ["Quellburg"],
               "num_docs": 2, "context": f"Document 1:\n{hay}\nDocument 2:\nThe Zorvex Tower is in "
                                        f"Quellburg. Quellburg Quellburg.\n{hay}"},
              {"index": 9, "input": "Who built the Amber Gate?", "answers": ["Tovan Reel"],
               "num_docs": 2, "context": f"Document 1:\nThe Amber Gate was built by Tovan Reel. "
                                        f"Tovan Reel Tovan Reel.\n{hay}"}]
    (root / "hotpotqa").mkdir(parents=True)
    (root / "hotpotqa" / "eval_400.json").write_text(json.dumps(hotpot))
    (root / "ruler").mkdir()
    rows = []
    for i, (key, value) in enumerate([("red-fox", "1111111"), ("blue-owl", "2222222"),
                                       ("grey-cat", "3333333")]):
        rows.append({"example": "", "instruction": "x",
                     "context": f"{hay} One of the special magic numbers for {key} is: {value}. {hay}",
                     "question": f"What is the special magic number for {key} mentioned in the "
                                 "provided text? The special magic number for", "index": i,
                     "outputs": [value], "length": 1000})
    (root / "ruler" / "niah_single_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "narrativeqa").mkdir()
    (root / "narrativeqa" / "documents.jsonl").write_text(
        json.dumps({"document_id": "d1", "text": f"{hay} Mira Solen sails to Kethra. {hay}"}) + "\n")
    questions = [{"order": 0, "index": 5, "document_id": "d1", "question": "Where does Mira sail?",
                  "answers": ["to Kethra", "Kethra"]},
                 {"order": 1, "index": 2, "document_id": "d1", "question": "Who sails to Kethra?",
                  "answers": ["Mira Solen"]}]
    (root / "narrativeqa" / "questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in questions))


@pytest.fixture()
def stub_env(tmp_path, monkeypatch):
    from wrag.embed import TfidfEmbedder
    from wrag.llm.stub import StubLLM
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(C, "LLM_CACHE", False)
    monkeypatch.setenv("WRAG_NO_PROGRESS", "1")
    data = tmp_path / "data"
    _write_data(data)
    return data, StubLLM(), TfidfEmbedder()


def _args(data, out, benchmark, split, engine="rag", **kw):
    from wrag.gambench.runner import RunArgs
    return RunArgs(data_dir=data, benchmark=benchmark, split=split, engine=engine,
                   output=out / engine / benchmark / split, **kw)


def test_end_to_end_three_benchmarks_and_table(stub_env, tmp_path):
    from wrag.gambench.runner import run_split
    data, llm, embedder = stub_env
    out = tmp_path / "runs"
    tok = WordTokenizer()
    hot = run_split(_args(data, out, "hotpotqa", "56k"), llm, embedder, tok)
    assert hot["registradas"] == 2 and hot["respondidas"] == 2 and hot["metrica"] == "f1"
    ruler = run_split(_args(data, out, "ruler", "niah_single_1"), llm, embedder, tok)
    assert ruler["valor"] == 100.0 and ruler["ruler_oficial"] == 100.0
    nqa = run_split(_args(data, out, "narrativeqa", "test"), llm, embedder, tok)
    rows = [json.loads(l) for l in (out / "rag/narrativeqa/test/predictions.jsonl").open()]
    assert [r["sid"] for r in rows] == ["narrativeqa-d1-5", "narrativeqa-d1-2"]
    assert rows[1]["memoria_reaproveitada"] is True       # mesmo livro, memória reusada
    assert all(len(r["paginas_escolhidas"]) <= 5 for r in rows)
    assert nqa["respondidas"] == 2

    report = suite_report(out, ["rag"], "stub")
    table = report["motores"]["rag"]["tabela"]
    # recortes minúsculos: marcados como fora do protocolo (asterisco)
    assert table["RULER Retri."]["valor"] == 100.0 and not table["RULER Retri."]["completo"]
    assert table["RULER MT"]["valor"] is None
    text = render_suite(report)
    assert "100.00*" in text and "GAM, GPT-4o-mini (artigo)" in text


def test_resume_skips_answered_and_retries_errors(stub_env, tmp_path, monkeypatch):
    from wrag.gambench import runner
    data, llm, embedder = stub_env
    out = tmp_path / "runs"
    tok = WordTokenizer()
    calls = {"n": 0}
    original = runner.Engine.select

    def flaky(self, benchmark, sid, *a, **kw):
        calls["n"] += 1
        if sid == "niah_single_1-1" and calls["n"] < 3:
            raise RuntimeError("falha transitória")
        return original(self, benchmark, sid, *a, **kw)

    monkeypatch.setattr(runner.Engine, "select", flaky)
    first = runner.run_split(_args(data, out, "ruler", "niah_single_1"), llm, embedder, tok)
    assert first["erros"] == 1 and first["valor"] == pytest.approx(200 / 3, abs=0.01)
    assert not first["completo"]
    with pytest.raises(SystemExit):   # sem --resume não sobrescreve
        runner.run_split(_args(data, out, "ruler", "niah_single_1"), llm, embedder, tok)
    second = runner.run_split(_args(data, out, "ruler", "niah_single_1", resume=True),
                              llm, embedder, tok)
    assert second["erros"] == 0 and second["valor"] == 100.0 and second["completo"]
    # a retomada refez só a amostra com erro
    assert calls["n"] == 4


def test_resume_refuses_a_different_configuration(stub_env, tmp_path):
    from wrag.gambench.runner import run_split
    data, llm, embedder = stub_env
    out = tmp_path / "runs"
    tok = WordTokenizer()
    run_split(_args(data, out, "hotpotqa", "56k", end_idx=1), llm, embedder, tok)
    with pytest.raises(SystemExit, match="outra configuração"):
        run_split(_args(data, out, "hotpotqa", "56k", resume=True, reader_seed=None),
                  llm, embedder, tok)


def test_filtered_answers_follow_gam_semantics(stub_env, tmp_path):
    from wrag.gambench.runner import run_split
    from wrag.llm.stub import StubLLM
    data, _llm, embedder = stub_env
    out = tmp_path / "runs"
    tok = WordTokenizer()
    blocked = StubLLM(filter_rate=1.0)
    hot = run_split(_args(data, out, "hotpotqa", "56k"), blocked, embedder, tok)
    # F1 do GAM: bloqueadas saem da média; estrito: contam zero
    assert hot["bloqueadas"] == 2 and hot["valor"] is None and hot["valor_estrito"] == 0.0
    ruler = run_split(_args(data, out, "ruler", "niah_single_1"), blocked, embedder, tok)
    assert ruler["valor"] == 0.0 and ruler["bloqueadas"] == 3


def test_engine_never_sees_answers_or_task(stub_env, tmp_path, monkeypatch):
    from wrag.gambench.runner import run_split
    from wrag.methods.base import Retriever
    data, llm, embedder = stub_env
    seen = []
    original = Retriever.retrieve

    def spy(self, question, k=None):
        seen.append((question.answers, question.qtype, question.question))
        return original(self, question, k)

    monkeypatch.setattr(Retriever, "retrieve", spy)
    for engine in ("rag", "hybrid", "witnessrag-lite"):
        run_split(_args(data, tmp_path / "runs", "ruler", "niah_single_1", engine=engine),
                  llm, embedder, WordTokenizer())
    assert seen and all(a == [] and t == "" for a, t, _q in seen)


def test_witnessrag_engine_runs_the_proof_controller(stub_env, tmp_path, monkeypatch):
    from wrag.gambench import engines
    from wrag.gambench.runner import run_split
    data, llm, embedder = stub_env
    # janelas do REGISTRAR com o tokenizador de palavras (sem baixar o do Qwen)
    monkeypatch.setattr("wrag.ie._load_window_tokenizer", lambda *_a, **_k: WordTokenizer())
    cfg = engines.engine_config("witnessrag")
    assert cfg.witness.proof_controller and cfg.ie.window_tokens == 512
    assert not cfg.ie.dialogue_mode and not cfg.qa.evidence_reader
    report = run_split(_args(data, tmp_path / "runs", "ruler", "niah_single_1",
                             engine="witnessrag"), llm, embedder, WordTokenizer())
    rows = [json.loads(l) for l in
            (tmp_path / "runs/witnessrag/ruler/niah_single_1/predictions.jsonl").open()]
    assert report["respondidas"] == 3
    assert all(r["diagnostico"].get("controlador") == "prova-v3" for r in rows)
    assert all(r["uso_motor"].get("chamadas", 0) > 0 for r in rows)


def test_table_groups_ruler_tasks_like_the_paper():
    reports = {("ruler", t): {"valor": 90.0, "completo": True, "protocolo_completo": True,
                              "registradas": 500} for t in P.RULER_GROUPS["Retri."]}
    reports[("ruler", "cwe")] = {"valor": 40.0, "completo": True, "protocolo_completo": True,
                                 "registradas": 500}
    reports[("ruler", "fwe")] = {"valor": 50.0, "completo": True, "protocolo_completo": True,
                                 "registradas": 500}
    row = table_row(reports)
    assert row["RULER Retri."]["valor"] == 90.0 and row["RULER Retri."]["completo"]
    assert row["RULER AGG."]["valor"] == 45.0 and row["RULER AGG."]["completo"]
    assert row["RULER QA"]["valor"] is None
    assert P.column_of("ruler", "vt") == "RULER MT" and P.column_of("hotpotqa", "448k") == "HotpotQA 448K"
    assert sum(len(t) for t in P.RULER_GROUPS.values()) == len(P.RULER_TASKS) == 13
