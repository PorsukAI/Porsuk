"""The seven retrieval tools the agent calls. `build_tools` binds a
`Retriever` and a `VectorStore` into seven `StructuredTool`s built as closures
(via `from_function`, since each closes over `retriever`, `store`, `cfg` and a
shared mutable `seen` dict rather than being a plain module-level function).
`seen: dict[str, Chunk]` threads chunk-returning tools to `expand_context`, so
it can widen only a chunk the agent has already retrieved without needing a
`document_id` argument.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from porsuk.core.models import Filters

if TYPE_CHECKING:
    from porsuk.core.config import AgentConfig
    from porsuk.core.models import Chunk, ChunkHit

_TRUNCATION_MARKER = "\n[doküman devam ediyor — get_document_outline ile bölüm seç]"


def _hit_payload(hit: ChunkHit) -> dict:
    return {
        "doc_id": hit.chunk.document_id,
        "chunk_id": hit.chunk.chunk_id,
        "page": hit.chunk.page_no,
        "section": hit.chunk.section_path,
        "text": hit.chunk.text,
        "score": round(hit.rerank_score if hit.rerank_score is not None else hit.score, 4),
    }


class _InvalidDate(ValueError):
    """Raised by `_parse_date`, caught at each tool boundary and turned
    into a `{"error": ...}` row instead of propagating, so a malformed date
    from the model reads like any other empty/error tool result rather than
    crashing the turn."""


def _parse_date(value: str | None, *, field: str) -> datetime | None:
    """`YYYY-MM-DD` (or a full ISO datetime) -> an aware UTC datetime. `None`
    passes through: the parameter is optional. A naive datetime (bare date,
    no offset) is assumed UTC rather than left comparable-but-wrong against
    `Filters.date_from`/`date_to`, which are always aware."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _InvalidDate(
            f"{field}: {value!r} is not a valid ISO date (use YYYY-MM-DD)"
        ) from None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def build_tools(
    retriever,
    store,
    *,
    cfg: AgentConfig,
    lang: str | None = None,
    scope: str | None = None,
) -> list:
    """Eight tools bound to `retriever` / `store`. `lang`, when set, filters
    every search tool to that language. `scope`, when set, is an
    index job ID: it restricts *every* tool to that job's documents,
    the four search tools via `search_filters`, and the three browse
    tools (`list_documents`, `get_document`, `get_document_outline`) by
    filtering `job_id` after the fetch, so a scoped chat cannot list or read
    another job's uploads.

    Unset (`scope=None`) means the base corpus, NOT every job's uploads:
    another chat's in-chat uploaded documents (`job_id` set)
    must not leak into a chat that never uploaded anything. Only documents
    indexed outside any chat (`porsuk index`, `job_id=None`) are visible.
    This is `unscoped_excludes_other_jobs=True`, not `job_ids=(...)`: an
    unscoped chat has no job id of its own to name.
    """

    search_filters = Filters(
        languages=(lang,) if lang else (),
        job_ids=(scope,) if scope else (),
        unscoped_excludes_other_jobs=scope is None,
    )
    seen: dict[str, Chunk] = {}

    def _record(hits: list[ChunkHit]) -> None:
        for h in hits:
            seen[h.chunk.chunk_id] = h.chunk

    def search_documents(
        query: str,
        method: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Doküman düzeyinde arama, hangi belgeler bu konuyla ilgili? Returns
        one row per document: doc_id, file, language, topics, score. Start here,
        then drill in with semantic_search / keyword_search restricted to the
        doc_ids this returns. `date_from`/`date_to` (YYYY-MM-DD) restrict to
        documents last modified in that range: a document with no captured
        date is excluded by either bound, not treated as always-in-range."""
        try:
            filters = dataclasses.replace(
                search_filters,
                date_from=_parse_date(date_from, field="date_from"),
                date_to=_parse_date(date_to, field="date_to"),
            )
        except _InvalidDate as exc:
            return [{"error": str(exc)}]
        hits = retriever.search_profiles(query, k=10, filters=filters)
        return [
            {
                "doc_id": h.profile.document_id,
                "file": h.profile.filename,
                "language": h.profile.language,
                "topics": list(h.profile.topics),
                "score": round(h.score, 4),
            }
            for h in hits
        ]

    def semantic_search(query: str, doc_ids: list[str] | None = None) -> list[dict]:
        """Kavramsal arama, bge-m3 dense. Anlamca yakın pasajları getirir,
        farklı kelimelerle yazılmış olsalar bile. Returns chunks with doc_id,
        chunk_id, section, page, text, score. Pass doc_ids to scope it."""
        hits = retriever.semantic_search(query, k=10, filters=search_filters, doc_ids=doc_ids)
        _record(hits)
        return [_hit_payload(h) for h in hits]

    def keyword_search(query: str, doc_ids: list[str] | None = None) -> list[dict]:
        """Kelime araması, bge-m3 sparse (lexical). Tam terim / kod / madde
        numarası ararken kullan. Returns chunks with doc_id, chunk_id, section,
        page, text, score. Pass doc_ids to scope it."""
        hits = retriever.keyword_search(query, k=10, filters=search_filters, doc_ids=doc_ids)
        _record(hits)
        return [_hit_payload(h) for h in hits]

    def get_document(doc_id: str, section: str | None = None) -> list[dict]:
        """Bir belgenin tam metnini (veya bir bölümünü) getirir. `section` verilirse
        section_path'i o önekle başlayan chunk'lar tutulur. Uzun metin
        max_context_chars'ta kesilir, o zaman get_document_outline ile bir bölüm seç."""
        chunks = store.chunks_for_document(doc_id)
        if scope is not None:
            chunks = [c for c in chunks if c.job_id == scope]
        else:
            chunks = [c for c in chunks if c.job_id is None]
        if not chunks:
            return [{"error": "doc_id not found"}]
        if section is not None:
            chunks = [
                c
                for c in chunks
                if c.section_path is not None and c.section_path.startswith(section)
            ]
            if not chunks:
                return [
                    {
                        "error": f"no chunks match section prefix {section!r} — "
                        "call get_document_outline"
                    }
                ]
        joined = "\n".join(c.text for c in chunks)
        truncated = len(joined) > cfg.max_context_chars
        if truncated:
            joined = joined[: cfg.max_context_chars] + _TRUNCATION_MARKER
        for c in chunks:
            seen[c.chunk_id] = c
        return [{"doc_id": doc_id, "section": section, "text": joined, "truncated": truncated}]

    def get_document_outline(doc_id: str) -> list[dict]:
        """Bir belgenin bölüm başlıklarını (section_path) belge sırasında,
        tekrarsız döndürür. get_document'e verilecek `section` önekini buradan seç."""
        chunks = store.chunks_for_document(doc_id)
        if scope is not None:
            chunks = [c for c in chunks if c.job_id == scope]
        else:
            chunks = [c for c in chunks if c.job_id is None]
        ordered: list[str] = []
        for c in chunks:
            if c.section_path is not None and c.section_path not in ordered:
                ordered.append(c.section_path)
        return [{"section_path": s} for s in ordered]

    def list_documents(
        lang: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict]:
        """Korpustaki belgeleri listeler (en fazla 50), dosya adına göre sıralı.
        `lang` / `doc_type` ile daralt. `date_from`/`date_to` (YYYY-MM-DD) son
        değişiklik tarihine göre daraltır, tarihi bilinmeyen bir belge her iki
        sınırda da hariç tutulur, aralıkta say­ılmaz. Returns file, doc_id,
        doc_type, language, topics, pages."""
        try:
            parsed_from = _parse_date(date_from, field="date_from")
            parsed_to = _parse_date(date_to, field="date_to")
        except _InvalidDate as exc:
            return [{"error": str(exc)}]
        rows = store.list_documents(
            Filters(
                languages=(lang,) if lang else (),
                doc_types=(doc_type,) if doc_type else (),
                job_ids=(scope,) if scope else (),
                unscoped_excludes_other_jobs=scope is None,
                date_from=parsed_from,
                date_to=parsed_to,
            ),
            limit=50,
        )
        return [
            {
                "file": r.filename,
                "doc_id": r.document_id,
                "doc_type": r.doc_type,
                "language": r.language,
                "topics": list(r.topics),
                "pages": r.page_count,
            }
            for r in rows
        ]

    def expand_context(
        chunk_id: str,
        before_chars: int | None = None,
        after_chars: int | None = None,
    ) -> list[dict]:
        """Daha önce bir aramada görülen bir chunk'ın çevresini genişletir,
        komşu chunk'ların metnini ekleyerek. Önce bir arama çalıştır; chunk_id
        görülmemişse hata döner."""
        before = before_chars if before_chars is not None else cfg.expand_before_chars
        after = after_chars if after_chars is not None else cfg.expand_after_chars
        if chunk_id not in seen:
            return [{"error": "chunk_id not seen yet — run a search first"}]
        target = seen[chunk_id]
        # target comes only from `seen`, populated exclusively by scoped
        # searches / scoped get_document, so it is in-scope by construction;
        # document_id is path-derived and job dirs are disjoint so no
        # cross-job collision.
        doc_chunks = store.chunks_for_document(target.document_id)
        idx = next((i for i, c in enumerate(doc_chunks) if c.chunk_id == chunk_id), None)
        if idx is None:
            return [{"error": "chunk not in its document — index may have changed"}]

        left = ""
        i = idx - 1
        while i >= 0 and len(left) < before:
            left = doc_chunks[i].text + left
            i -= 1

        right = ""
        j = idx + 1
        while j < len(doc_chunks) and len(right) < after:
            right = right + doc_chunks[j].text
            j += 1

        return [
            {
                "chunk_id": chunk_id,
                "doc_id": target.document_id,
                "page": target.page_no,
                "section": target.section_path,
                "text": left + target.text + right,
                "before_chars": before,
                "after_chars": after,
            }
        ]

    return [
        StructuredTool.from_function(
            func=search_documents,
            name="search_documents",
            description=search_documents.__doc__,
        ),
        StructuredTool.from_function(
            func=semantic_search,
            name="semantic_search",
            description=semantic_search.__doc__,
        ),
        StructuredTool.from_function(
            func=keyword_search,
            name="keyword_search",
            description=keyword_search.__doc__,
        ),
        StructuredTool.from_function(
            func=get_document,
            name="get_document",
            description=get_document.__doc__,
        ),
        StructuredTool.from_function(
            func=get_document_outline,
            name="get_document_outline",
            description=get_document_outline.__doc__,
        ),
        StructuredTool.from_function(
            func=list_documents,
            name="list_documents",
            description=list_documents.__doc__,
        ),
        StructuredTool.from_function(
            func=expand_context,
            name="expand_context",
            description=expand_context.__doc__,
        ),
    ]
