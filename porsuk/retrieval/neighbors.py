"""Retrieval-time neighbour-chunk expansion (komşu genişletme).

A chunk boundary can split a clause from its condition - common in contracts
("Ödemeler 30 gün içinde yapılır." / next chunk: "Şu kadar ki, ...") - and
the second half reverses the first. Expansion happens at query time only:
the schema does not change and no re-index is needed, because adjacency is
already given by char_span order within a document_id. Width 0 disables it;
the effect is measured, not assumed.
"""

from __future__ import annotations

import dataclasses

from porsuk.core.models import ChunkHit
from porsuk.core.ports import VectorStore


def expand_neighbours(hits: list[ChunkHit], store: VectorStore, *, width: int) -> list[ChunkHit]:
    if width <= 0:
        return hits
    claimed: set[str] = set()
    out: list[ChunkHit] = []
    for hit in hits:
        siblings = store.chunks_for_document(hit.chunk.document_id)
        idx = next(
            (i for i, c in enumerate(siblings) if c.chunk_id == hit.chunk.chunk_id),
            None,
        )
        if idx is None:
            out.append(hit)
            continue
        lo = max(0, idx - width)
        hi = min(len(siblings), idx + width + 1)
        window = [
            c
            for c in siblings[lo:hi]
            if c.chunk_id not in claimed or c.chunk_id == hit.chunk.chunk_id
        ]
        for c in window:
            claimed.add(c.chunk_id)
        merged_text = "\n".join(c.text for c in window)
        out.append(dataclasses.replace(hit, chunk=dataclasses.replace(hit.chunk, text=merged_text)))
    return out
