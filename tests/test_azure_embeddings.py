"""Mock the remote embedding API: limits, pooling, ordering and provider cache."""
from types import SimpleNamespace

import numpy as np
import pytest
import tiktoken

from wrag.embed import AzureEmbedder
from wrag import config as C


def embedder(mock_create):
    e = object.__new__(AzureEmbedder)
    e.deployment = "embedding-3-small-global"
    e.batch_size = 2
    e.dim = 0
    e._client = SimpleNamespace(embeddings=SimpleNamespace(create=mock_create))
    return e


def test_preserves_oversized_input_by_pooling_and_restores_response_order():
    received = []
    def create(**kwargs):
        batch = kwargs["input"]
        received.extend(batch)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[len(t), 1.0])
                                     for i, t in reversed(list(enumerate(batch)))])
    e = embedder(create)
    text = " memory" * 9000
    out = e._encode(["short", text, ""])
    lengths = [len(t) for t in received]
    assert lengths == [1, 8191, 809, 1]
    assert all(0 < n <= 8191 for n in lengths)
    assert sum(lengths[1:3]) == len(tiktoken.get_encoding("cl100k_base").encode(text))
    expected = (8191 ** 2 + 809 ** 2) / 9000
    assert out.shape == (3, 2) and out[1, 0] == pytest.approx(expected)
    assert out[0, 0] == out[2, 0] == 1 and np.isfinite(out).all()


def test_missing_response_indices_fail_instead_of_assigning_wrong_vectors():
    e = embedder(lambda **kw: SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[1, 2])]))
    with pytest.raises(RuntimeError, match="indices"):
        e._encode(["first", "second"])


def test_cache_separates_endpoints_without_persisting_secrets(monkeypatch):
    e = embedder(None)
    monkeypatch.setenv(C.AZURE_BASE_URL_VAR, "https://one.invalid")
    first = e.cache_key()
    monkeypatch.setenv(C.AZURE_BASE_URL_VAR, "https://two.invalid")
    second = e.cache_key()
    assert first != second and "https" not in second
