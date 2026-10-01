"""A resumed run must receive the same numeric ranks as the first run."""
import pytest

from wrag import config as C
from wrag.witness.rerank import Reranker


def test_cold_and_restarted_reranker_return_identical_scores(tmp_path,monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(C,"CACHE_DIR",tmp_path)
    class Model:
        def predict(self,pairs,**kwargs):
            assert pairs == [("question","first"),("question","second")]
            return [10.123456789, 10.123456790]
    cold=Reranker("test-model",device="cpu")
    cold._model=Model()
    original=cold.score("question",["first","second"])
    restarted=Reranker("test-model",device="cpu")
    def forbidden(): raise AssertionError("cached reranking must not load the model")
    monkeypatch.setattr(restarted,"_load",forbidden)
    assert restarted.score("question",["first","second"]) == original
    assert cold.score("question",["first","second"]) == original
    assert sorted(range(2),key=lambda i:(-original[i],i)) == [0,1]
