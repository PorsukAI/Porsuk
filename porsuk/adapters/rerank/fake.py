"""Deterministic reranker for tests - reverses the top_n and stamps a score."""

from __future__ import annotations

import dataclasses

from porsuk.core.models import ChunkHit
from porsuk.core.registry import register


@register("reranker", "fake")
class FakeReranker:
    def rerank(self, query: str, hits: list[ChunkHit], *, top_n: int) -> list[ChunkHit]:
        top = list(reversed(hits[:top_n]))
        return [dataclasses.replace(h, rerank_score=1.0 / (i + 1)) for i, h in enumerate(top)]
