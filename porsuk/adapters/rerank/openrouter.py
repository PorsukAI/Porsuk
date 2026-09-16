"""Cross-encoder reranker (çapraz kodlayıcı yeniden sıralayıcı) over OpenRouter's
dedicated /v1/rerank endpoint.

Distinct from flag_embedding_http.py: same job (score query/chunk pairs), a
different wire contract. OpenRouter's rerank endpoint is not the OpenAI chat/
embeddings shape - it takes {model, query, documents} and returns
{results: [{index, relevance_score}]}, already sorted by relevance and
carrying a "document" echo we don't need. Verified directly against
qwen/qwen3-reranker-8b (Fireworks-served) before writing this.
"""

from __future__ import annotations

import dataclasses

import httpx

from porsuk.core.models import ChunkHit
from porsuk.core.registry import register


@register("reranker", "openrouter")
class OpenRouterReranker:
    def __init__(
        self, *, model: str, base_url: str, api_key: str | None = None, timeout: float = 60.0
    ) -> None:
        self._model = model
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
            json={
                "model": self._model,
                "query": query,
                "documents": [h.chunk.text for h in candidates],
            },
        )
        resp.raise_for_status()
        results = sorted(resp.json()["results"], key=lambda r: r["relevance_score"], reverse=True)
        return [
            dataclasses.replace(candidates[r["index"]], rerank_score=float(r["relevance_score"]))
            for r in results
        ]
