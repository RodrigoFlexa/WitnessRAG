"""Backend OpenAI público: plano sem segredo, cliente e perfil de tokens."""

import json

import pytest

from wrag.llm.base import GenParams
from wrag.llm.openai_compat import OpenAICompatLLM, is_public_openai
from wrag.pilot import make_plan, parser


def test_public_openai_detection():
    assert is_public_openai("")
    assert is_public_openai("https://api.openai.com/v1")
    assert not is_public_openai("http://127.0.0.1:8095/v1")


def test_pilot_plan_openai_never_stores_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    args = parser().parse_args(["--gpu", "0", "--backend", "openai", "--model", "gpt-4o-mini",
                                "--dataset", "locomo", "--locomo-conversation", "all",
                                "--methods", "witnessrag", "--embed-device", "cpu"])
    plan = make_plan(args, tmp_path)
    assert plan["server_command"] == []
    assert plan["env"]["WRAG_LLM_BACKEND"] == "openai"
    assert plan["env"]["OPENAI_MODEL"] == "gpt-4o-mini"
    assert "OPENAI_BASE_URL" not in plan["env"]
    assert "sk-test-secret" not in json.dumps(plan)


def test_public_api_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatLLM(deployment="gpt-4o-mini", use_cache=False)


def test_token_profile_public_vs_local(monkeypatch):
    messages = [{"role": "user", "content": "oi"}]
    params = GenParams(max_tokens=64)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    public = OpenAICompatLLM(deployment="gpt-4o-mini", use_cache=False)
    kwargs = public.build_kwargs(messages, params)
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] >= 64 and "max_completion_tokens" not in kwargs
    assert kwargs["temperature"] == params.temperature

    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8095/v1")
    local = OpenAICompatLLM(deployment="Qwen/Qwen2.5-14B-Instruct", use_cache=False)
    assert local.build_kwargs(messages, params)["max_tokens"] == 64
