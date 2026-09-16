"""Qdrant vector store: two collections (doc_profiles, doc_chunks) plus a
one-point _meta collection holding the embedding fingerprint, so an open
against a mismatched fingerprint raises rather than silently returning
nonsense. Only `doc_chunks` carries a named sparse vector (`"sparse"`, for
keyword search); `doc_profiles` and `_meta` use the unnamed default dense
vector. `url=":memory:"` runs entirely in-process (used by the tests); any
other value is treated as a server URL.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client import models as qm

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

_PROFILES = "doc_profiles"
_CHUNKS = "doc_chunks"
_META = "_meta"
_META_POINT_ID = 0


@register("store", "qdrant")
class QdrantStore:
    def __init__(self, *, url: str, dim: int) -> None:
        if url == ":memory:":
            self._client = QdrantClient(location=":memory:")
            self._local = True
        else:
            self._client = QdrantClient(url=url)
            self._local = False
        self._dim = dim

    # ----- lifecycle ----------------------------------------------------

    def ensure_collections(self, *, embedding_fingerprint: str) -> None:
        existing = self.stored_fingerprint()
        if existing is not None and existing != embedding_fingerprint:
            raise ValueError(
                f"this index was built with {existing!r}; opening with "
                f"{embedding_fingerprint!r}. Re-index or fix the model."
            )
        if not self._client.collection_exists(_PROFILES):
            self._client.create_collection(
                _PROFILES,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
            )
        if not self._client.collection_exists(_CHUNKS):
            # doc_chunks carries three vectors on every point:
            #   ""       the default dense vector (bge-m3 dense, semantic route)
            #   "sparse" bge-m3's learned lexical weights (keyword_backend=sparse)
            #   "bm25"   per-chunk term frequencies, corpus IDF applied by
            #              Modifier.IDF at query time (keyword_backend=bm25)
            # Declaring sparse_vectors_config alongside a plain VectorParams
            # keeps the dense one as the default: upserts pass vector={"": ...}
            # and dense search needs no `using=`.
            self._client.create_collection(
                _CHUNKS,
                vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
                sparse_vectors_config={
                    "sparse": qm.SparseVectorParams(),
                    "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
                },
            )
            # keyword_backend=text: a plain word-membership search over the
            # chunk body (MatchText), no scoring model at all: rank is "how
            # many query words the chunk contains". A text payload index makes
            # MatchText fast on a server; the local in-process Qdrant ignores
            # payload indexes (and warns), so only create it on a real server.
            if not self._local:
                self._client.create_payload_index(
                    _CHUNKS,
                    field_name="text",
                    field_schema=qm.TextIndexParams(
                        type=qm.TextIndexType.TEXT,
                        tokenizer=qm.TokenizerType.MULTILINGUAL,
                        lowercase=True,
                    ),
                )
        if existing is None:
            self._client.create_collection(
                _META,
                vectors_config=qm.VectorParams(size=1, distance=qm.Distance.COSINE),
            )
            self._client.upsert(
                _META,
                points=[
                    qm.PointStruct(
                        id=_META_POINT_ID,
                        vector=[0.0],
                        payload={"fingerprint": embedding_fingerprint},
                    )
                ],
            )

    def stored_fingerprint(self) -> str | None:
        if not self._client.collection_exists(_META):
            return None
        points = self._client.retrieve(_META, ids=[_META_POINT_ID], with_payload=True)
        return points[0].payload["fingerprint"] if points else None

    def drop_all(self) -> None:
        for name in (_PROFILES, _CHUNKS, _META):
            if self._client.collection_exists(name):
                self._client.delete_collection(name)

    # ----- writes -----------------------------------------------------

    def upsert_profiles(self, profiles: list[DocumentProfile], vectors: EmbedResult) -> None:
        points = [
            qm.PointStruct(
                id=_point_id(p.document_id),
                vector=list(vec),
                payload={
                    "document_id": p.document_id,
                    "path": p.path,
                    "filename": p.filename,
                    "doc_type": p.doc_type,
                    "language": p.language,
                    "summary": p.summary,
                    "topics": list(p.topics),
                    "entities": list(p.entities),
                    "profile_level": p.profile_level,
                    "parse_quality": p.parse_quality,
                    "parser_used": p.parser_used,
                    "image_heavy": p.image_heavy,
                    "content_hash": p.content_hash,
                    "page_count": p.page_count,
                    "size": p.size,
                    "job_id": p.job_id,
                    "modified_at": p.modified_at.isoformat() if p.modified_at else None,
                },
            )
            for p, vec in zip(profiles, vectors.dense, strict=True)
        ]
        self._client.upsert(_PROFILES, points=points)

    def upsert_chunks(self, chunks: list[Chunk], vectors: EmbedResult) -> None:
        # bge-m3's sparse (lexical) weights become a native Qdrant sparse
        # vector "sparse" on the point. The dense vector is passed
        # as vector={"": [...]}, the named form doc_chunks now requires,
        # and "sparse" is added only when the embedder produced weights.
        sparse = vectors.sparse if vectors.sparse is not None else [None] * len(chunks)
        points = [
            qm.PointStruct(
                id=_point_id(c.chunk_id),
                vector=_chunk_vector(vec, sp, bm25_tokenize(c.text)),
                payload={
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "text": c.text,
                    "page_no": c.page_no,
                    "language": c.language,
                    "section_title": c.section_title,
                    "section_path": c.section_path,
                    "char_span": list(c.char_span),
                    "job_id": c.job_id,
                },
            )
            for c, vec, sp in zip(chunks, vectors.dense, sparse, strict=True)
        ]
        self._client.upsert(_CHUNKS, points=points)

    def sparse_vector_for(self, chunk_id: str) -> dict[int, float] | None:
        """The stored {token_id: weight} map for one chunk, or None.

        Reads the native sparse vector back via `retrieve(with_vectors=True)`:
        in 1.19.0 `point.vector` is `{"": [...], "sparse": SparseVector}`
        when a sparse vector is present and a bare dense list when it is not.
        Not part of the VectorStore port - a seam and a test hook.
        """
        points = self._client.retrieve(_CHUNKS, ids=[_point_id(chunk_id)], with_vectors=True)
        if not points:
            return None
        vec = points[0].vector
        if not isinstance(vec, dict):
            return None
        sv = vec.get("sparse")
        if sv is None:
            return None
        return {int(i): float(v) for i, v in zip(sv.indices, sv.values, strict=True)}

    # ----- reads -----------------------------------------------------

    def set_job_id(self, document_id: str, job_id: str | None) -> None:
        """Re-stamp every chunk and the profile for `document_id` with
        `job_id` ("kalıcı yap"). A payload-only update, no
        vectors are touched, so this is cheap even for a large document."""
        doc_filter = qm.Filter(
            must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
        )
        self._client.set_payload(_CHUNKS, payload={"job_id": job_id}, points=doc_filter)
        self._client.set_payload(
            _PROFILES,
            payload={"job_id": job_id},
            points=qm.Filter(
                must=[
                    qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))
                ]
            ),
        )

    def get_profile(self, document_id: str) -> DocumentProfile | None:
        points = self._client.retrieve(
            _PROFILES, ids=[_point_id(document_id)], with_payload=True
        )
        return _payload_to_profile(points[0].payload) if points else None

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        """Every chunk of one document, in char_span order: the adjacency
        neighbour expansion (komşu genişletme) walks. A plain scroll
        with a document_id filter; `with_payload=True` and no `with_vectors` is
        unaffected by doc_chunks being a named-vector collection.
        """
        points = self._client.scroll(
            _CHUNKS,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
            ),
            limit=10_000,
            with_payload=True,
        )[0]
        return sorted(
            (_payload_to_chunk(p.payload) for p in points),
            key=lambda c: c.char_span[0],
        )

    def list_documents(self, filters: Filters, *, limit: int) -> list[DocumentSummary]:
        """Metadata rows for every stored profile passing `filters`.

        Scroll all matching profiles, then sort by filename and cap at `limit`:
        the inmemory adapter (the reference) does sort-then-cap, so
        `scroll(limit=limit)` here would return a different set of rows on a large
        corpus. `chunks_for_document` already scrolls 10_000 the same way.
        `_to_filter` without `doc_ids` covers language/doc_type/topic/path/quality.
        """
        points, _ = self._client.scroll(
            _PROFILES,
            scroll_filter=_to_filter(filters),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
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
            for p in (_payload_to_profile(pt.payload) for pt in points)
        ]
        rows.sort(key=lambda r: r.filename)
        return rows[:limit]

    def search_profiles(
        self, vector: EmbedResult, *, k: int, filters: Filters
    ) -> list[DocumentHit]:
        points = self._client.query_points(
            _PROFILES,
            query=list(vector.dense[0]),
            limit=k,
            query_filter=_to_filter(filters),
            with_payload=True,
        ).points
        return [
            DocumentHit(
                profile=_payload_to_profile(p.payload),
                score=p.score,
                retrieval_method="semantic",
            )
            for p in points
        ]

    def search_chunks(
        self,
        vector: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        points = self._client.query_points(
            _CHUNKS,
            query=list(vector.dense[0]),
            limit=k,
            query_filter=_to_filter(filters, doc_ids=doc_ids),
            with_payload=True,
        ).points
        return [
            ChunkHit(
                chunk=_payload_to_chunk(p.payload),
                score=p.score,
                retrieval_method="semantic",
            )
            for p in points
        ]

    def search_chunks_sparse(
        self,
        sparse: EmbedResult,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Keyword search (anahtar kelime araması) over the native
        sparse vectors. The query's lexical weights arrive as `sparse.sparse[0]`;
        `dense` is not touched and may be empty for a sparse-only query.
        """
        weights = sparse.sparse[0] if sparse.sparse else {}
        points = self._client.query_points(
            _CHUNKS,
            query=qm.SparseVector(indices=list(weights), values=list(weights.values())),
            using="sparse",
            limit=k,
            query_filter=_to_filter(filters, doc_ids=doc_ids),
            with_payload=True,
        ).points
        return [
            ChunkHit(
                chunk=_payload_to_chunk(p.payload),
                score=p.score,
                retrieval_method="keyword",
            )
            for p in points
        ]

    def search_chunks_bm25(
        self,
        query_terms: dict[int, float],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Keyword search over classic BM25. The "bm25" sparse
        vector holds per-chunk term frequencies; `Modifier.IDF` on the
        collection makes Qdrant apply corpus inverse-document-frequency at
        query time, so the query vector is just the term set with weight 1.
        """
        if not query_terms:
            return []
        points = self._client.query_points(
            _CHUNKS,
            query=qm.SparseVector(
                indices=list(query_terms), values=list(query_terms.values())
            ),
            using="bm25",
            limit=k,
            query_filter=_to_filter(filters, doc_ids=doc_ids),
            with_payload=True,
        ).points
        return [
            ChunkHit(
                chunk=_payload_to_chunk(p.payload),
                score=p.score,
                retrieval_method="keyword",
            )
            for p in points
        ]

    def search_chunks_text(
        self,
        terms: list[str],
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Plain word-membership search (`keyword_backend=text`).

        No scoring model, not BM25, not a learned sparse vector. Qdrant's
        `MatchText` on the `text` payload index is a filter: a chunk either
        contains a query word or it does not. We OR the per-word conditions,
        scroll every match, and rank by *how many distinct query words the
        chunk contains* (ties broken by total occurrences). This is the
        closest thing to `grep` the store offers; the reranker fixes the
        order afterwards.
        """
        terms = [t for t in terms if t]
        if not terms:
            return []
        base = _to_filter(filters, doc_ids=doc_ids)
        must = list(base.must) if base and base.must else []
        # `should` with no `min_should` means "at least one": the OR we want.
        flt = qm.Filter(
            must=must or None,
            should=[qm.FieldCondition(key="text", match=qm.MatchText(text=t)) for t in terms],
        )
        matched: list = []
        offset = None
        while True:
            page, offset = self._client.scroll(
                _CHUNKS,
                scroll_filter=flt,
                limit=256,
                with_payload=True,
                offset=offset,
            )
            matched.extend(page)
            if offset is None or len(matched) >= 2000:
                break
        lowered = [t.lower() for t in terms]
        scored: list[tuple[int, int, ChunkHit]] = []
        for p in matched:
            body = (p.payload.get("text") or "").lower()
            distinct = sum(1 for t in lowered if t in body)
            if distinct == 0:
                continue
            occurrences = sum(body.count(t) for t in lowered)
            scored.append(
                (distinct, occurrences, ChunkHit(
                    chunk=_payload_to_chunk(p.payload),
                    score=float(distinct),
                    retrieval_method="keyword",
                ))
            )
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [hit for _d, _o, hit in scored[:k]]


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _chunk_vector(
    dense: tuple[float, ...],
    sparse: dict[int, float] | None,
    bm25_tf: dict[int, int],
) -> dict[str, object]:
    """The named-vector dict a doc_chunks point upsert takes: the default
    dense vector under "", the bge-m3 "sparse" vector when present, and the
    "bm25" term-frequency vector always (BM25 is text-derived, not
    embedder-derived).
    """
    vec: dict[str, object] = {"": list(dense)}
    if sparse:
        vec["sparse"] = qm.SparseVector(indices=list(sparse), values=list(sparse.values()))
    if bm25_tf:
        vec["bm25"] = qm.SparseVector(
            indices=list(bm25_tf), values=[float(v) for v in bm25_tf.values()]
        )
    return vec


def _to_filter(f: Filters, *, doc_ids: list[str] | None = None) -> qm.Filter | None:
    must: list[qm.FieldCondition] = []
    if doc_ids:
        must.append(qm.FieldCondition(key="document_id", match=qm.MatchAny(any=list(doc_ids))))
    if f.doc_types:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=list(f.doc_types))))
    if f.languages:
        must.append(qm.FieldCondition(key="language", match=qm.MatchAny(any=list(f.languages))))
    if f.job_ids:
        must.append(qm.FieldCondition(key="job_id", match=qm.MatchAny(any=list(f.job_ids))))
    if f.unscoped_excludes_other_jobs:
        # A chat with no scope sees the base corpus (job_id unset)
        # but not another chat's uploaded documents (job_id set).
        #
        # IsEmptyCondition, not IsNullCondition: a freshly upserted profile
        # stores job_id as an actual JSON null (job_id key present, value
        # null); IsNullCondition matches that. But `set_job_id`'s "kalıcı
        # yap" promotion goes through `set_payload({"job_id": None})`, and
        # Qdrant's set_payload with a None value DELETES the payload key
        # rather than storing null, so the key is then simply absent.
        # IsNullCondition does NOT match a missing key, only a present-and-
        # null one, so a promoted document silently stopped matching this
        # filter and vanished from every unscoped chat. IsEmptyCondition
        # matches both "missing" and "null", covering both origins.
        must.append(qm.IsEmptyCondition(is_empty=qm.PayloadField(key="job_id")))
    if f.path_prefix:
        must.append(qm.FieldCondition(key="path", match=qm.MatchText(text=f.path_prefix)))
    if f.topics:
        must.append(qm.FieldCondition(key="topics", match=qm.MatchAny(any=list(f.topics))))
    if f.min_parse_quality is not None:
        must.append(qm.FieldCondition(key="parse_quality", range=qm.Range(gte=f.min_parse_quality)))
    if f.date_from is not None or f.date_to is not None:
        # A profile with no modified_at (parser never produced a date) has
        # no payload value for the field at all: a DatetimeRange condition
        # on a missing field does not match in Qdrant, which is exactly the
        # "exclude, don't treat as always-in-range" behaviour wanted here.
        must.append(
            qm.FieldCondition(
                key="modified_at",
                range=qm.DatetimeRange(
                    gte=f.date_from.isoformat() if f.date_from else None,
                    lte=f.date_to.isoformat() if f.date_to else None,
                ),
            )
        )
    return qm.Filter(must=must) if must else None


def _payload_to_profile(pl: dict) -> DocumentProfile:
    modified_at_raw = pl.get("modified_at")
    return DocumentProfile(
        document_id=pl["document_id"],
        path=pl["path"],
        filename=pl["filename"],
        doc_type=pl["doc_type"],
        language=pl.get("language"),
        created_at=None,
        modified_at=datetime.fromisoformat(modified_at_raw) if modified_at_raw else None,
        size=pl.get("size", 0),
        page_count=pl.get("page_count", 0),
        summary=pl.get("summary", ""),
        topics=tuple(pl.get("topics", ())),
        entities=tuple(pl.get("entities", ())),
        profile_level=pl.get("profile_level", "cheap"),
        profile_source=frozenset(),
        parse_quality=pl.get("parse_quality", 0.0),
        parse_quality_components=None,
        parser_used=pl.get("parser_used", ""),
        image_heavy=pl.get("image_heavy", False),
        content_hash=pl.get("content_hash", ""),
        job_id=pl.get("job_id"),
    )


def _payload_to_chunk(pl: dict) -> Chunk:
    span = pl.get("char_span", [0, 0])
    return Chunk(
        chunk_id=pl["chunk_id"],
        document_id=pl["document_id"],
        text=pl["text"],
        page_no=pl.get("page_no"),
        language=pl.get("language"),
        section_title=pl.get("section_title"),
        section_path=pl.get("section_path"),
        char_span=(span[0], span[1]),
        job_id=pl.get("job_id"),
    )
