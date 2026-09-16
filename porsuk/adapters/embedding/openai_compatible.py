"""Embedder over any OpenAI-compatible /v1/embeddings, with an optional
separate sparse-weights endpoint (`sparse_base_url`); with none given,
`EmbedResult.sparse` is None and only dense vectors are returned.
"""

from __future__ import annotations

import math
import time

import httpx

from porsuk.core.models import EmbedResult
from porsuk.core.registry import register

# Indexing 5000 documents means tens of thousands of requests; a hosted
# endpoint (or a tunnel in front of one) will drop some to rate limits or
# transient 5xx. Retry with backoff rather than fail the whole run on one.
# A transient blip recovers within a few tries; more just delays the
# `failed` mark the pipeline gives a document whose endpoint is truly down.
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 15.0


@register("embedder", "openai_compatible")
class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str,
        dim: int,
        base_url: str,
        api_key: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        timeout: float = 60.0,
        sparse_base_url: str | None = None,
    ) -> None:
        self.dim = dim
        self._model = model
        self._url = base_url.rstrip("/") + "/embeddings"
        self._sparse_url = sparse_base_url.rstrip("/") + "/embed" if sparse_base_url else None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._normalize = normalize
        self._batch_size = batch_size
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _request(self, url: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.post(url, headers=self._headers, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                # Only a connection-level failure needs a fresh client; a
                # 5xx leaves the pool usable, so reuse it.
                if isinstance(exc, httpx.TransportError):
                    self._client.close()
                    self._client = httpx.Client(timeout=self._timeout)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(min(_BACKOFF_BASE_S * 2**attempt, _BACKOFF_CAP_S))
        raise RuntimeError(f"request to {url} failed after {_MAX_RETRIES} tries: {last_exc}")

    def _dense(self, inputs: list[str]) -> list[tuple[float, ...]]:
        rows = self._request(self._url, {"model": self._model, "input": inputs})["data"]
        vectors = [tuple(r["embedding"]) for r in rows]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(
                f"embedding endpoint returned dim {len(vectors[0])}, config says {self.dim} "
                f"- wrong model, or a mismatched matryoshka/truncation setting"
            )
        return [self._maybe_normalize(v) for v in vectors]

    def _sparse(self, inputs: list[str]) -> list[dict[int, float]]:
        rows = self._request(self._sparse_url, {"texts": inputs})["sparse"]
        return [{int(k): float(v) for k, v in weights.items()} for weights in rows]

    def _maybe_normalize(self, vec: tuple[float, ...]) -> tuple[float, ...]:
        if not self._normalize:
            return vec
        norm = math.sqrt(sum(c * c for c in vec)) or 1.0
        return tuple(c / norm for c in vec)

    def _embed(self, texts: list[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(dense=())
        dense: list[tuple[float, ...]] = []
        sparse: list[dict[int, float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            dense.extend(self._dense(batch))
            if self._sparse_url is not None:
                sparse.extend(self._sparse(batch))
        return EmbedResult(
            dense=tuple(dense),
            sparse=tuple(sparse) if self._sparse_url is not None else None,
        )

    def embed_documents(self, texts: list[str]) -> EmbedResult:
        return self._embed(texts)

    def embed_query(self, text: str) -> EmbedResult:
        return self._embed([text])
