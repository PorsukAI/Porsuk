"""Brute-force in-memory store.

Reference implementation for the VectorStore port. The Qdrant adapter must
match its observable behaviour; this one is the executable definition of it.
"""

from __future__ import annotations

import math
from dataclasses import replace

from porsuk.core.models import (
    Chunk,
    ChunkHit,
    DocumentHit,
    DocumentProfile,
    DocumentSummary,
    EmbedResult,
    Filters,
)
from porsuk.core.registry import register
from porsuk.retrieval.bm25 import tokenize as bm25_tokenize

_BM25_K1 = 1.5
_BM25_B = 0.75


class FingerprintMismatchError(RuntimeError):
    """Raised when an index built with one embedder is opened with another."""


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """True cosine, not a bare dot product.

    FakeEmbedder returns unit vectors, for which a dot product would suffice,
    but a real embedder need not normalise (sentence_transformers does not by
    default) and this adapter is the reference the Qdrant adapter must match.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _profile_passes(profile: DocumentProfile, filters: Filters) -> bool:
    if filters.doc_types and profile.doc_type not in filters.doc_types:
        return False
    if filters.languages and profile.language not in filters.languages:
        return False
    if filters.topics and not set(filters.topics) & set(profile.topics):
        return False
    if filters.path_prefix and not profile.path.startswith(filters.path_prefix):
        return False
    if filters.min_parse_quality is not None and profile.parse_quality < filters.min_parse_quality:
        return False
    if filters.date_from is not None or filters.date_to is not None:
        # A profile with no modified_at (no filesystem date captured) is
        # excluded from a date-range query, not treated as always-in-range:
        # matches QdrantStore's behaviour (a missing payload field never
        # matches a DatetimeRange condition).
        if profile.modified_at is None:
            return False
        if filters.date_from is not None and profile.modified_at < filters.date_from:
            return False
        if filters.date_to is not None and profile.modified_at > filters.date_to:
            return False
    if filters.job_ids and profile.job_id not in filters.job_ids:
        return False
    if filters.unscoped_excludes_other_jobs and profile.job_id is not None:
        return False
    return True


def _chunk_passes(chunk: Chunk, filters: Filters) -> bool:
    if filters.languages and chunk.language not in filters.languages:
        return False
    if filters.job_ids and chunk.job_id not in filters.job_ids:
        return False
    if filters.unscoped_excludes_other_jobs and chunk.job_id is not None:
        return False
    return True


@register("store", "inmemory")
class InMemoryStore:
    def __init__(self, *, dim: int | None = None) -> None:
        # `dim` is accepted and ignored: the container passes the embedder's
        # dimension to every store, and a brute-force store does not need it.
        self._dim = dim
        self._fingerprint: str | None = None
        self._profiles: dict[str, tuple[DocumentProfile, tuple[float, ...]]] = {}
        self._chunks: dict[str, tuple[Chunk, tuple[float, ...]]] = {}
        self._sparse: dict[str, dict[int, float]] = {}
        self._bm25_tf: dict[str, dict[int, int]] = {}

    def ensure_collections(self, *, embedding_fingerprint: str) -> None:
        if self._fingerprint is None:
            self._fingerprint = embedding_fingerprint
            return
        if self._fingerprint != embedding_fingerprint:
            raise FingerprintMismatchError(
                f"this index was built with {self._fingerprint!r} but is being "
                f"opened with {embedding_fingerprint!r}. Reindex, or correct the "
                f"model in the config."
            )

    def stored_fingerprint(self) -> str | None:
        return self._fingerprint

    def upsert_profiles(self, profiles: list[DocumentProfile], vectors: EmbedResult) -> None:
        for profile, vector in zip(profiles, vectors.dense, strict=True):
            self._profiles[profile.document_id] = (profile, vector)

    def upsert_chunks(self, chunks: list[Chunk], vectors: EmbedResult) -> None:
        sparse = vectors.sparse if vectors.sparse is not None else [None] * len(chunks)
        for chunk, vector, sp in zip(chunks, vectors.dense, sparse, strict=True):
            self._chunks[chunk.chunk_id] = (chunk, vector)
            if sp is not None:
                self._sparse[chunk.chunk_id] = dict(sp)
            # BM25 term frequencies are derived from the chunk text, not from
            # the embedder: they are written for every chunk.
            self._bm25_tf[chunk.chunk_id] = bm25_tokenize(chunk.text)

    def sparse_vector_for(self, chunk_id: str) -> dict[int, float] | None:
        return self._sparse.get(chunk_id)

    def get_profile(self, document_id: str) -> DocumentProfile | None:
        entry = self._profiles.get(document_id)
        return entry[0] if entry is not None else None

    def set_job_id(self, document_id: str, job_id: str | None) -> None:
        for chunk_id, (chunk, vector) in list(self._chunks.items()):
            if chunk.document_id == document_id:
                self._chunks[chunk_id] = (replace(chunk, job_id=job_id), vector)
        if document_id in self._profiles:
            profile, vector = self._profiles[document_id]
            self._profiles[document_id] = (replace(profile, job_id=job_id), vector)

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        return sorted(
            (c for c, _ in self._chunks.values() if c.document_id == document_id),
            key=lambda c: c.char_span[0],
        )

    def list_documents(self, filters: Filters, *, limit: int) -> list[DocumentSummary]:
        rows = [
            DocumentSummary(
                document_id=p.document_id,
                path=p.path,
                filename=p.filename,
                doc_type=p.doc_type,
                language=p.language,
                page_count=p.page_count,
                parse_quality=p.parse_quality,
                topics=p.topics,
            )
            for p, _ in self._profiles.values()
            if _profile_passes(p, filters)
        ]
        rows.sort(key=lambda r: r.filename)
        return rows[:limit]

    def search_profiles(
        self, vector: EmbedResult, *, k: int, filters: Filters
    ) -> list[DocumentHit]:
        (query,) = vector.dense
        scored = [
            DocumentHit(
                profile=profile,
                score=_cosine(query, stored),
                retrieval_method="semantic",
            )
            for profile, stored in self._profiles.values()
            if _profile_passes(profile, filters)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def all_chunk_ids(self) -> list[str]:
        return list(self._chunks)

    def all_profile_ids(self) -> list[str]:
        return list(self._profiles)

    def search_chunks(
        self,
        vector: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        (query,) = vector.dense
        allowed = set(doc_ids) if doc_ids is not None else None
        scored = [
            ChunkHit(
                chunk=chunk,
                score=_cosine(query, stored),
                retrieval_method="semantic",
            )
            for chunk, stored in self._chunks.values()
            if (allowed is None or chunk.document_id in allowed) and _chunk_passes(chunk, filters)
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def search_chunks_sparse(
        self,
        sparse: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Keyword search (anahtar kelime araması): dot product over the stored
        {token_id: weight} maps. The reference the Qdrant adapter must match.
        """
        query = sparse.sparse[0] if sparse.sparse else {}
        allowed = set(doc_ids) if doc_ids is not None else None
        scored = []
        for chunk_id, (chunk, _dense) in self._chunks.items():
            if allowed is not None and chunk.document_id not in allowed:
                continue
            if not _chunk_passes(chunk, filters):
                continue
            weights = self._sparse.get(chunk_id, {})
            dot = sum(query.get(t, 0.0) * w for t, w in weights.items())
            if dot > 0:
                scored.append(ChunkHit(chunk=chunk, score=dot, retrieval_method="keyword"))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def search_chunks_text(
        self,
        terms: list[str],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Plain word-membership search (`keyword_backend=text`):
        no scoring model, rank = count of distinct query words the chunk
        contains (ties: total occurrences). The reference the Qdrant adapter
        must match.
        """
        lowered = [t.lower() for t in terms if t]
        if not lowered:
            return []
        allowed = set(doc_ids) if doc_ids is not None else None
        scored: list[tuple[int, int, ChunkHit]] = []
        for chunk, _dense in self._chunks.values():
            if allowed is not None and chunk.document_id not in allowed:
                continue
            if not _chunk_passes(chunk, filters):
                continue
            body = chunk.text.lower()
            distinct = sum(1 for t in lowered if t in body)
            if distinct == 0:
                continue
            occurrences = sum(body.count(t) for t in lowered)
            scored.append(
                (distinct, occurrences, ChunkHit(chunk=chunk, score=float(distinct),
                                                 retrieval_method="keyword"))
            )
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [hit for _d, _o, hit in scored[:k]]

    def search_chunks_bm25(
        self,
        query_terms: dict[int, float],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Keyword search over classic BM25. Okapi BM25 with the
        textbook k1=1.5 / b=0.75; IDF over the filtered candidate set (Qdrant's
        `Modifier.IDF` is corpus-wide, but the reference computes it over the
        same rows the query can see so `doc_ids`-scoped results still rank
        sensibly). The Qdrant adapter must match this observable ordering.
        """
        if not query_terms:
            return []
        allowed = set(doc_ids) if doc_ids is not None else None
        candidates = [
            (cid, chunk)
            for cid, (chunk, _dense) in self._chunks.items()
            if (allowed is None or chunk.document_id in allowed) and _chunk_passes(chunk, filters)
        ]
        if not candidates:
            return []
        n = len(candidates)
        lengths = {cid: sum(self._bm25_tf.get(cid, {}).values()) or 1 for cid, _ in candidates}
        avgdl = sum(lengths.values()) / n
        df: dict[int, int] = {}
        for cid, _ in candidates:
            for tid in self._bm25_tf.get(cid, {}):
                if tid in query_terms:
                    df[tid] = df.get(tid, 0) + 1
        scored: list[ChunkHit] = []
        for cid, chunk in candidates:
            tf = self._bm25_tf.get(cid, {})
            dl = lengths[cid]
            score = 0.0
            for tid in query_terms:
                freq = tf.get(tid, 0)
                if freq == 0:
                    continue
                idf = math.log(1 + (n - df[tid] + 0.5) / (df[tid] + 0.5))
                score += idf * (freq * (_BM25_K1 + 1)) / (
                    freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                )
            if score > 0:
                scored.append(ChunkHit(chunk=chunk, score=score, retrieval_method="keyword"))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
