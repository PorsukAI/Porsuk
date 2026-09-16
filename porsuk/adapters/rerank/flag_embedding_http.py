"""Cross-encoder reranker (çapraz kodlayıcı yeniden sıralayıcı) over HTTP.

bge-reranker-v2-m3, served by the same FlagEmbedding process as bge-m3's
sparse head - one Python process, two jobs. The strongest single
lever on ranking quality per the Turkish RAG literature, but the
GPU cost is real, so whether it is on by default is a measurement decision.
"""

from __future__ import annotations

import dataclasses

import httpx

from porsuk.core.models import ChunkHit
from porsuk.core.registry import register


@register("reranker", "flag_embedding_http")
class FlagEmbeddingReranker:
    def __init__(self, *, base_url: str, api_key: str | None = None, timeout: float = 60.0) -> None:
        self._url = base_url.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, hits: list[ChunkHit], *, top_n: int) -> list[ChunkHit]:
        candidates = hits[:top_n]
        if not candidates:
            return []
        resp = self._client.post(
            self._url,
            headers=self._headers,
            json={"query": query, "texts": [h.chunk.text for h in candidates]},
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]
        scored = sorted(zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True)
        return [dataclasses.replace(h, rerank_score=float(s)) for h, s in scored]
