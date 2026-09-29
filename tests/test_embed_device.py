"""A CUDA benchmark must fail visibly instead of silently using CPU."""
import sys
from types import SimpleNamespace

import pytest

from wrag import config as C
from wrag import embed as E


def fake_model(device):
    return SimpleNamespace(device=device, get_sentence_embedding_dimension=lambda: 3)


def test_strict_device_refuses_cpu_retry_on_cuda_load_error(monkeypatch):
    calls = []
    def load(name, device):
        calls.append(device)
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load))
    monkeypatch.setattr(C, "EMBED_STRICT_DEVICE", True)
    with pytest.raises(RuntimeError, match="forbids CPU fallback"):
        E.SentenceTransformerEmbedder(device="cuda")
    assert calls == ["cuda"]


def test_legacy_device_fallback_is_preserved(monkeypatch):
    calls = []
    def load(name, device):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA unavailable")
        return fake_model("cpu")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load))
    monkeypatch.setattr(C, "EMBED_STRICT_DEVICE", False)
    result = E.SentenceTransformerEmbedder(device="cuda")
    assert calls == ["cuda", "cpu"] and result._model.device == "cpu"


@pytest.mark.parametrize("actual,success", [("cpu", False), ("cuda:0", True)])
def test_strict_device_checks_where_the_model_really_loaded(monkeypatch, actual, success):
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=lambda name, device: fake_model(actual)))
    monkeypatch.setattr(C, "EMBED_STRICT_DEVICE", True)
    if success:
        assert E.SentenceTransformerEmbedder(device="cuda")._model.device == actual
    else:
        with pytest.raises(RuntimeError, match="device mismatch"):
            E.SentenceTransformerEmbedder(device="cuda")


def test_factory_propagates_strict_cuda_failure(monkeypatch):
    def load(name, device):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load))
    monkeypatch.setattr(C, "EMBED_STRICT_DEVICE", True)
    monkeypatch.setattr(C, "EMBED_DEVICE", "cuda")
    monkeypatch.setattr(E, "_EMBEDDER", None)
    with pytest.raises(RuntimeError, match="forbids CPU fallback"):
        E.get_embedder("st", force_new=True)
