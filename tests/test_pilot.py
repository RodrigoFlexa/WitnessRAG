"""Piloto: isolamento de GPU, integração HTTP, corpus e gráficos pareados."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wrag import config as C
from wrag.pilot import make_plan, parser, prepare_data, wait_ready
from wrag.util import write_json, append_jsonl, read_json


def test_terminal_summary_uses_report_metrics_and_marks_partial(tmp_path, capsys):
    from wrag.pilot import print_results
    write_json(tmp_path / "benchmark/run/report.json", {"datasets": {"locomo": {
        "corpus": {"n_questions": 102}, "excluidas": ["q1"],
        "metodos": {"witnessrag": {"n_avaliadas": 10,
            "metricas": {"f1": .425, "em": .3, "recall@5": .8, "all_recall@5": .6}}},
        "por_categoria": {"witnessrag": {"single-hop": {"n": 7, "f1": .5},
                                         "multi-hop": {"n": 3, "f1": .25}}}}}})
    print_results(tmp_path, "time_limit")
    output = capsys.readouterr().out
    assert "PARCIAIS" in output and "102" in output
    assert "42.50" in output and "single-hop" in output and "multi-hop" in output
    assert "Excluídas por filtro de conteúdo: 1" in output
    assert "report.md" in output


def test_gpu_plan_isolated_from_parent_and_uses_hf_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    args = parser().parse_args(["--gpu", "5", "--model-revision", "abc123"])
    plan = make_plan(args, tmp_path)
    assert plan["env"]["CUDA_VISIBLE_DEVICES"] == "5"
    assert plan["env"]["WRAG_EMBED_DEVICE"] == "cuda:0"
    assert plan["env"]["OPENAI_MODEL"] == "Qwen/Qwen2.5-14B-Instruct"
    assert "--generation-config" in plan["server_command"]
    assert plan["server_command"][-4:] == ["--revision", "abc123", "--tokenizer-revision", "abc123"]
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    with pytest.raises(ValueError):
        make_plan(parser().parse_args(["--gpu", "5,6"]), tmp_path)


def test_resume_rejects_metric_changing_configuration(tmp_path):
    from wrag.pilot import _validate_resume
    old = make_plan(parser().parse_args(["--gpu", "3", "--dataset", "locomo",
                                         "--top-k", "15"]), tmp_path)
    same = make_plan(parser().parse_args(["--gpu", "4", "--dataset", "locomo",
                                          "--top-k", "15", "--resume"]), tmp_path)
    _validate_resume(old, same)  # GPU e --resume não alteram a comparação.
    changed = make_plan(parser().parse_args(["--gpu", "3", "--dataset", "locomo",
                                             "--top-k", "5"]), tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        _validate_resume(old, changed)


def test_reader_budget_and_witness_candidate_pool_are_independent(tmp_path):
    from wrag.pilot import _run_config
    args = parser().parse_args(["--gpu", "3", "--dataset", "locomo", "--top-k", "5",
                                "--witness-candidate-pool", "20",
                                "--locomo-ie-window-tokens", "512", "--max-query-plans", "5"])
    plan = make_plan(args, tmp_path)
    cfg = _run_config(plan["settings"], 10)
    assert cfg.top_k == 5 and cfg.qa.top_k == 5
    assert cfg.witness.candidate_pool_k == 20
    assert cfg.witness.max_query_plans == 5
    assert cfg.ie.window_tokens == 512 and cfg.ie.window_tokenizer == args.model


def test_compatible_backend_uses_model_task_cap_and_endpoint_cache(tmp_path, monkeypatch):
    from wrag.llm.openai_compat import OpenAICompatLLM
    from wrag.llm.base import GenParams
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(OpenAICompatLLM, "_make_client", lambda _: None)
    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8085/v1")
    llm = OpenAICompatLLM()
    kwargs = llm.build_kwargs([{"role": "user", "content": "hello"}], GenParams(max_tokens=32, json_mode=True))
    assert kwargs["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert kwargs["max_tokens"] == 32 and "reasoning_effort" not in kwargs
    first = llm._cache_path(kwargs)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8086/v1")
    second = OpenAICompatLLM()._cache_path(kwargs)
    assert first != second


def test_local_http_chat_and_model_readiness(monkeypatch):
    pytest.importorskip("openai")
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from wrag.llm.openai_compat import OpenAICompatLLM
    from wrag.llm.base import GenParams
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def reply(self, obj):
            payload = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            self.reply({"data": [{"id": "test-qwen"}]})
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.reply({"id": "test", "object": "chat.completion", "created": 0, "model": "test-qwen",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"ok": true}'},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
        monkeypatch.setenv("OPENAI_MODEL", "test-qwen")
        wait_ready(None, port, "test-qwen", time.monotonic() + 5)
        result = OpenAICompatLLM(use_cache=False).chat("Return JSON", params=GenParams(max_tokens=32, json_mode=True))
        assert result.json() == {"ok": True}
        assert requests[0]["model"] == "test-qwen" and requests[0]["max_tokens"] == 32
        assert result.prompt_tokens == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_pilot_data_keeps_support_and_adds_distractors(tmp_path):
    from wrag.data import load_dataset
    args = parser().parse_args(["--gpu", "5", "--dataset", "sample", "-n", "1", "--distractors", "2"])
    plan = make_plan(args, tmp_path)
    passages = [{"title": f"T{i}", "text": f"Entity {i} has a fact."} for i in range(5)]
    questions = [{"id": "q1", "question": "What fact?", "answer": "fact",
                  "paragraphs": [{**passages[0], "is_supporting": True}]}]
    write_json(tmp_path / "source-data" / "sample.json", questions)
    write_json(tmp_path / "source-data" / "sample_corpus.json", passages)
    meta = prepare_data(plan)
    corpus = load_dataset("sample", data_dir=tmp_path / "data")
    assert meta["candidate_passages"] == 1 and meta["additional_distractors"] == 2
    assert len(corpus.passages) == 3
    assert set(corpus.questions[0].gold_pids) <= {p.pid for p in corpus.passages}
    assert meta["corpus_reduced"] is True


def test_plots_use_common_ids_and_skip_missing_methods(tmp_path):
    pytest.importorskip("matplotlib")
    from wrag.eval.plots import plot_run
    from test_regressions import metric_row
    write_json(tmp_path / "run.json", {"datasets": ["toy"], "metodos": ["dense", "witnessrag"],
                                       "llm": {"backend": "stub"}})
    a = {**metric_row("a"), "latencia_recuperacao_s": 1, "latencia_leitura_s": 2}
    append_jsonl(tmp_path / "toy" / "dense.jsonl", a)
    append_jsonl(tmp_path / "toy" / "dense.jsonl", {**a, "qid": "unpaired", "latencia_recuperacao_s": 1000})
    append_jsonl(tmp_path / "toy" / "witnessrag.jsonl", a)
    target = plot_run(tmp_path)
    data = read_json(target / "plot_data.json")["toy"]
    assert data["question_ids"] == ["a"]
    assert data["latency_seconds_per_question"] == [3, 3]
    assert (target / "toy-qualidade.png").exists()
    write_json(tmp_path / "run.json", {"datasets": ["toy"], "metodos": ["dense", "missing"]})
    empty = plot_run(tmp_path)
    assert read_json(empty / "plot_data.json") == {}
