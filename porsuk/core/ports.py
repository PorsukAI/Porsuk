"""Ports: the only abstractions core code knows about.

Protocols, not ABCs: an adapter conforms structurally and needs no
inheritance, which lets third-party classes be adapters directly and makes
test doubles trivial.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from porsuk.core.models import (
    Chunk,
    ChunkHit,
    DocumentHit,
    DocumentProfile,
    DocumentSummary,
    EmbedResult,
    Filters,
    ParseOutcome,
)


@runtime_checkable
class LLMProvider(Protocol):
    model: str

    def complete(self, prompt: str, *, max_tokens: int = 512, enable_thinking: bool = True) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> EmbedResult: ...

    def embed_query(self, text: str) -> EmbedResult: ...


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collections(self, *, embedding_fingerprint: str) -> None: ...

    def stored_fingerprint(self) -> str | None: ...

    def upsert_profiles(self, profiles: list[DocumentProfile], vectors: EmbedResult) -> None: ...

    def chunks_for_document(self, document_id: str) -> list[Chunk]: ...

    def get_profile(self, document_id: str) -> DocumentProfile | None:
        """The one profile for `document_id`, or `None` if it isn't
        indexed. The file-serving endpoint ("kaynağı aç") uses
        this to resolve a doc_id to its on-disk `path` without scrolling
        every document in the corpus."""
        ...

    def upsert_chunks(self, chunks: list[Chunk], vectors: EmbedResult) -> None: ...

    def search_profiles(
        self, vector: EmbedResult, *, k: int, filters: Filters
    ) -> list[DocumentHit]: ...

    def search_chunks(
        self,
        vector: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]: ...

    def search_chunks_sparse(
        self,
        sparse: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]: ...

    def search_chunks_bm25(
        self,
        query_terms: dict[int, float],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]: ...

    def search_chunks_text(
        self,
        terms: list[str],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]: ...

    def list_documents(self, filters: Filters, *, limit: int) -> list[DocumentSummary]: ...

    def set_job_id(self, document_id: str, job_id: str | None) -> None:
        """Re-stamp every chunk and profile for `document_id` with `job_id`.
        `None` moves the document into the unscoped base
        corpus, visible from every chat: "promote this upload to permanent"."""
        ...


@runtime_checkable
class Parser(Protocol):
    name: str
    cost: int

    def can_handle(self, path: str) -> bool: ...

    def parse(self, path: str) -> ParseOutcome: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ChunkHit], *, top_n: int) -> list[ChunkHit]: ...
