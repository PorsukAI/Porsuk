"""Shared JSON shapes for the CLI and the HTTP API."""

from __future__ import annotations

# Stores that keep their data on disk between an `index` run and a `search` /
# `ask` run. Anything else is warned about: the two are separate processes and
# an in-process store starts empty.
_PERSISTENT_STORES = frozenset({"qdrant"})


def _hit_dict(h) -> dict:
    return {
        "chunk_id": h.chunk.chunk_id,
        "document_id": h.chunk.document_id,
        "score": round(h.score, 4),
        "rerank_score": round(h.rerank_score, 4) if h.rerank_score is not None else None,
        "retrieval_method": h.retrieval_method,
        "section_path": h.chunk.section_path,
        "page_no": h.chunk.page_no,
        "language": h.chunk.language,
        "text": h.chunk.text,
    }
