"""
Embedding utilities: OpenAI path (optional) + deterministic mock fallback.
Normalize all vectors for cosine similarity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import hashlib
import numpy as np

# sklearn is used only for the mock embedding (HashingVectorizer)
from sklearn.feature_extraction.text import HashingVectorizer


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def _mock_embedding(text: str, n_dim: int = 384) -> np.ndarray:
    hv = HashingVectorizer(n_features=n_dim, alternate_sign=False, norm=None)
    vec = hv.transform([text]).toarray().astype(np.float32)[0]

    # Add deterministic noise
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
    rng = np.random.default_rng(h % (2**32))
    vec = vec + rng.normal(0, 0.01, size=n_dim).astype(np.float32)

    return _l2_normalize(vec.reshape(1, -1))[0]


@dataclass
class Embedder:
    use_openai: bool = False
    model: str = "text-embedding-ada-002"
    fallback_model: str = "text-embedding-3-small"
    dim: int = 384  # used only by mock

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Mock embedding path
        if not self.use_openai:
            embs = np.vstack([_mock_embedding(t, n_dim=self.dim) for t in texts])
            return _l2_normalize(embs)

        # OpenAI embedding path
        try:
            from openai import OpenAI
            client = OpenAI()

            try:
                resp = client.embeddings.create(
                    model=self.model,
                    input=texts
                )
            except Exception:
                resp = client.embeddings.create(
                    model=self.fallback_model,
                    input=texts
                )

            embs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return _l2_normalize(embs)

        except Exception:
            # fallback to mock
            embs = np.vstack([_mock_embedding(t, n_dim=self.dim) for t in texts])
            return _l2_normalize(embs)
