"""Deterministic hash-derived embeddings for tests.

Same text always yields the same unit vector, so retrieval tests can assert
exact orderings without loading a model.
"""

from __future__ import annotations

import hashlib
import math

from porsuk.core.models import EmbedResult
from porsuk.core.registry import register


def _vector(text: str, dim: int, normalize: bool = True) -> tuple[float, ...]:
    """Expand a counter-salted digest so every component stays distinct.

    A single sha256 gives 32 bytes; indexing it modulo its length would
    repeat the same pattern once dim exceeds 32, making high-dimensional
    vectors degenerate. Salting with a counter avoids that.

    `normalize` mirrors the real embedders: sentence_transformers does not
    normalise by default, and the flag is part of the embedding fingerprint,
    so the fake has to be able to reproduce both behaviours.
    """
    raw: list[float] = []
    counter = 0
    while len(raw) < dim:
        digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        raw.extend(b / 255.0 - 0.5 for b in digest)
        counter += 1
    raw = raw[:dim]
    if not normalize:
        return tuple(raw)
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return tuple(x / norm for x in raw)


@register("embedder", "fake")
class FakeEmbedder:
    def __init__(self, dim: int = 8, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(dense=tuple(_vector(t, self.dim, self.normalize) for t in texts))

    def embed_query(self, text: str) -> EmbedResult:
        return EmbedResult(dense=(_vector(text, self.dim, self.normalize),))
