"""
Camada de embedding plugável, com cache em disco.

Três provedores, escolhidos por `WRAG_EMBED_BACKEND` (default `auto`):

* **azure** — usa um deployment de embedding do próprio gateway. Não exige GPU,
  mas cada indexação custa chamadas.
* **sentence-transformers** — modelo local (bge-large-en-v1.5 por padrão, o
  mesmo que já aparece no seu `.env`). Custo zero por vetor, exige GPU e pesos.
* **tfidf** — TF-IDF por caractere+palavra ajustado no próprio corpus, reduzido
  por SVD. Não é um embedding semântico e não deve aparecer em nenhuma tabela
  do artigo; existe para que o pipeline rode offline de ponta a ponta.

`auto` tenta nessa ordem e cai para o próximo em silêncio, registrando no log
qual escolheu. O relatório grava o provedor efetivo em `run.json`: uma tabela
que não diz qual retriever produziu os vetores não é comparável com nenhuma
outra.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from wrag import config as C
from wrag.util import get_logger, progress, sha

log = get_logger("wrag.embed")


class Embedder(ABC):
    name: str = "abstract"
    dim: int = 0

    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def encode(self, texts: Sequence[str], desc: str = "") -> np.ndarray:
        """Vetores L2-normalizados, uma linha por texto. Similaridade é produto
        interno, então todo o resto do código pode usar `A @ B.T` direto."""
        if not texts:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)
        vectors = self._encode_cached(list(texts), desc)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def _encode_cached(self, texts: list[str], desc: str) -> np.ndarray:
        if not C.EMBED_CACHE:
            return self._encode(texts)
        cache_dir = C.CACHE_DIR / "embed" / self.cache_key()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Um arquivo por lote de textos idênticos: o corpus inteiro reindexado
        # bate no mesmo arquivo, e um rerun de método não paga nada.
        path = cache_dir / f"{sha(texts)}.npy"
        if path.exists():
            try:
                return np.load(path)
            except (ValueError, OSError):
                pass
        vectors = self._encode(texts)
        try:
            np.save(path, vectors)
        except OSError as exc:
            log.warning("não consegui gravar cache de embedding: %s", exc)
        return vectors

    def cache_key(self) -> str:
        return f"{self.name}"


class AzureEmbedder(Embedder):
    name = "azure"

    def __init__(self, deployment: str | None = None, batch_size: int | None = None) -> None:
        from openai import AzureOpenAI

        self.deployment = deployment or C.AZURE_EMBED_DEPLOYMENT
        if not self.deployment:
            raise RuntimeError("WRAG_AZURE_EMBED_DEPLOYMENT não definido")
        self.batch_size = batch_size or C.EMBED_BATCH_SIZE
        api_key = os.environ.get(C.AZURE_API_KEY_VAR, "").strip()
        base_url = os.environ.get(C.AZURE_BASE_URL_VAR, "").strip()
        endpoint = os.environ.get(C.AZURE_ENDPOINT_VAR, "").strip()
        kwargs = {"api_key": api_key, "api_version": C.AZURE_EMBED_API_VERSION, "max_retries": 4}
        if base_url:
            kwargs["base_url"] = base_url
        else:
            kwargs["azure_endpoint"] = endpoint
        if C.AZURE_CA_BUNDLE:
            import httpx

            path = Path(C.AZURE_CA_BUNDLE)
            if not path.is_absolute():
                path = C.ROOT / path
            kwargs["http_client"] = httpx.Client(verify=str(path), timeout=300.0)
        self._client = AzureOpenAI(**kwargs)
        self.dim = 0

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in progress(range(0, len(texts), self.batch_size), desc="embed(azure)"):
            batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            response = self._client.embeddings.create(model=self.deployment, input=batch)
            out.extend(item.embedding for item in response.data)
        array = np.asarray(out, dtype=np.float32)
        self.dim = array.shape[1] if array.size else 0
        return array

    def cache_key(self) -> str:
        return f"azure-{self.deployment}"


class SentenceTransformerEmbedder(Embedder):
    name = "st"

    def __init__(self, model: str | None = None, device: str | None = None,
                 batch_size: int | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model or C.EMBED_MODEL
        device = device or C.EMBED_DEVICE
        try:
            self._model = SentenceTransformer(self.model_name, device=device)
        except Exception:  # noqa: BLE001 - cai para CPU se a GPU não estiver lá
            log.warning("não consegui usar device=%s; caindo para cpu", device)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        self.batch_size = batch_size or C.EMBED_BATCH_SIZE
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), batch_size=self.batch_size,
                               show_progress_bar=len(texts) > 512,
                               convert_to_numpy=True, normalize_embeddings=False),
            dtype=np.float32,
        )

    def cache_key(self) -> str:
        return "st-" + self.model_name.replace("/", "_")


class TfidfEmbedder(Embedder):
    """Fallback offline. Ajusta no primeiro `encode` e congela o vocabulário.

    Não é um retriever de artigo. É o que permite verificar recall@k, junção de
    testemunhas e relatório sem rede — e é o que aparece nos testes.
    """

    name = "tfidf"

    def __init__(self, dim: int = 256) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            lowercase=True, sublinear_tf=True, analyzer="word",
            ngram_range=(1, 2), min_df=1, max_features=200_000,
        )
        self._svd: TruncatedSVD | None = None
        self._fitted = False
        self._lock = threading.Lock()
        self._corpus: list[str] = []

    def fit(self, corpus: Sequence[str]) -> None:
        """Ajusta explicitamente no corpus da rodada. Chamado pelo índice."""
        with self._lock:
            from sklearn.decomposition import TruncatedSVD

            self._corpus = [t for t in corpus if t.strip()] or ["vazio"]
            matrix = self._vectorizer.fit_transform(self._corpus)
            n_components = int(min(self.dim, max(2, min(matrix.shape) - 1)))
            if matrix.shape[1] < 2:
                self._svd = None
                self.dim = matrix.shape[1]
            else:
                self._svd = TruncatedSVD(n_components=min(n_components, matrix.shape[1]), random_state=C.SEED)
                self._svd.fit(matrix)
                self.dim = self._svd.components_.shape[0]
            self._fitted = True
            log.info("TfidfEmbedder ajustado: %d docs, %d dims", len(self._corpus), self.dim)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(list(texts))
        matrix = self._vectorizer.transform([t if t.strip() else "vazio" for t in texts])
        return np.asarray(self._svd.transform(matrix) if self._svd is not None else matrix.toarray(),
                          dtype=np.float32)

    def _encode_cached(self, texts: list[str], desc: str) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return super()._encode_cached(texts, desc)

    def cache_key(self) -> str:
        # O vocabulário depende do corpus ajustado, então o cache tem que
        # depender dele também. Sem isso, dois datasets compartilhariam vetores.
        return "tfidf-v2-" + sha(self._corpus, self.dim, C.SEED)[:16]


_EMBEDDER: Embedder | None = None


def get_embedder(backend: str | None = None, force_new: bool = False) -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is not None and not force_new:
        return _EMBEDDER

    backend = (backend or C.EMBED_BACKEND).lower()
    order = [backend] if backend != "auto" else ["azure", "st", "tfidf"]

    last_error: Exception | None = None
    for candidate in order:
        try:
            if candidate == "azure":
                if not C.AZURE_EMBED_DEPLOYMENT:
                    raise RuntimeError("sem deployment de embedding configurado")
                _EMBEDDER = AzureEmbedder()
            elif candidate in ("st", "sentence-transformers"):
                _EMBEDDER = SentenceTransformerEmbedder()
            elif candidate == "tfidf":
                _EMBEDDER = TfidfEmbedder()
            else:
                raise ValueError(f"provedor de embedding desconhecido: {candidate!r}")
            log.info("embeddings: %s (%s)", _EMBEDDER.name, _EMBEDDER.cache_key())
            return _EMBEDDER
        except Exception as exc:  # noqa: BLE001 - a queda para o próximo é o ponto
            last_error = exc
            log.warning("provedor de embedding %r indisponível (%s)", candidate, exc)

    raise RuntimeError(f"nenhum provedor de embedding disponível: {last_error}")


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k por produto interno em vetores normalizados. Devolve (índices, scores)."""
    if matrix.size == 0 or k <= 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)
    scores = matrix @ query.reshape(-1)
    k = int(min(k, scores.shape[0]))
    idx = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]
