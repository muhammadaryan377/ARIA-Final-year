"""Embedding abstraction for Copilot RAG.

Local-first by design: reuses ARIA's existing document_processor
EmbeddingGenerator (sentence-transformers, offline). When that model is not
installed/available, a deterministic hashing embedder keeps lexical + fuzzy
retrieval working with zero cloud dependency — so Local mode never needs the
internet and never silently calls out to a cloud embedding API.

Cloud embeddings are deliberately NOT required: the retriever always has a
working local path. A cloud embedder can be plugged in later behind the same
Embedder interface.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

log = logging.getLogger(__name__)

_HASH_DIM = 256


class Embedder(Protocol):
    """Common interface every embedder implements."""

    name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic, dependency-free embedder (bag-of-words hashing).

    Always available, fully offline. Used as the fallback when no neural
    embedding model is present, and as the stable basis for lexical scoring.
    """

    name = "hashing"
    dimension = _HASH_DIM

    @staticmethod
    def _tokens(text: str) -> list[str]:
        out: list[str] = []
        for w in re.findall(r"[A-Za-z0-9_]+", str(text or "").lower()):
            out.append(w)
            out.extend(p for p in w.split("_") if p)
        return out

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for tok in self._tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder:
    """Local neural embedder backed by ARIA's document_processor generator."""

    def __init__(self, model_name: str = "minilm", cache_dir: str | None = None):
        from document_processor.embeddings import EmbeddingGenerator

        self._gen = EmbeddingGenerator(model_name=model_name, cache_dir=cache_dir)
        self._gen.load_model()
        self.dimension = self._gen.get_dimension()
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._gen.generate(texts)

    def embed_one(self, text: str) -> list[float]:
        return self._gen.generate_single(text)


def create_embedder(model_name: str = "minilm",
                    cache_dir: str | None = None,
                    prefer_local: bool = True) -> Embedder:
    """Return the best available embedder, never failing to return one.

    Local neural model when importable (offline-capable), otherwise the
    deterministic hashing embedder. `prefer_local` documents intent: this
    function never reaches for a cloud embedding API.
    """
    if prefer_local:
        try:
            return SentenceTransformerEmbedder(model_name=model_name, cache_dir=cache_dir)
        except Exception as exc:
            log.info("Local neural embedder unavailable (%s); using hashing embedder.", exc)
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors (0.0 when degenerate)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)