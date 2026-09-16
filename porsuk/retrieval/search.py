"""Discrete retrieval (ayrık aramalar): keyword (bge-m3 sparse), semantic
(bge-m3 dense), and hybrid (RRF over the two) as three separate searches, not
fused by default, so the agent can see whether a keyword match failed to
exist or existed and lost in fusion.
"""

from __future__ import annotations

from porsuk.core.config import RetrievalConfig
from porsuk.core.models import ChunkHit, DocumentHit, Filters
from porsuk.core.ports import Embedder, Reranker, VectorStore
from porsuk.retrieval import bm25
from porsuk.retrieval.fusion import reciprocal_rank_fusion
from porsuk.retrieval.neighbors import expand_neighbours
from porsuk.retrieval.stemmer import stem_terms


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None,
        cfg: RetrievalConfig,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._cfg = cfg

    def semantic_search(
        self, query: str, *, k: int, filters: Filters, doc_ids: list[str] | None = None
    ) -> list[ChunkHit]:
        vec = self._embedder.embed_query(query)
        return self._store.search_chunks(vec, k=k, filters=filters, doc_ids=doc_ids)

    def keyword_search(
        self, query: str, *, k: int, filters: Filters, doc_ids: list[str] | None = None
    ) -> list[ChunkHit]:
        """Lexical search (anahtar kelime araması).

        The engine is `keyword_backend`: `text` (plain word
        membership via Qdrant `MatchText`, the default; rank is how many
        distinct query words the chunk contains, no scoring model), `bm25`
        (Qdrant `Modifier.IDF` over per-chunk term frequencies), or `sparse`
        (bge-m3's learned lexical weights, needs the sparse endpoint).
        """
        if self._cfg.keyword_backend == "text":
            terms = bm25.tokenize_terms(query, stem=self._cfg.stemmer_enabled)
            return self._store.search_chunks_text(terms, k=k, filters=filters, doc_ids=doc_ids)
        if self._cfg.keyword_backend == "bm25":
            terms = bm25.query_terms(query, stem=self._cfg.stemmer_enabled)
            return self._store.search_chunks_bm25(terms, k=k, filters=filters, doc_ids=doc_ids)
        text = " ".join(stem_terms(query)) if self._cfg.stemmer_enabled else query
        vec = self._embedder.embed_query(text)
        if vec.sparse is None:
            raise ValueError(
                "keyword_backend='sparse' needs sparse (lexical) vectors, but the embedder "
                "returned none. Set embedder.sparse_base_url, use keyword_backend='text' or "
                "'bm25', or strategy='semantic'."
            )
        return self._store.search_chunks_sparse(vec, k=k, filters=filters, doc_ids=doc_ids)

    def hybrid_search(
        self, query: str, *, k: int, filters: Filters, doc_ids: list[str] | None = None
    ) -> list[ChunkHit]:
        sem = self.semantic_search(query, k=k, filters=filters, doc_ids=doc_ids)
        kw = self.keyword_search(query, k=k, filters=filters, doc_ids=doc_ids)
        by_id = {h.chunk.chunk_id: h for h in (*sem, *kw)}
        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in sem], [h.chunk.chunk_id for h in kw]]
        )
        out: list[ChunkHit] = []
        for cid, score in fused[:k]:
            base = by_id[cid]
            out.append(
                ChunkHit(
                    chunk=base.chunk,
                    score=score,
                    retrieval_method="hybrid",
                    rerank_score=base.rerank_score,
                )
            )
        return out

    def search(
        self,
        query: str,
        *,
        k: int,
        filters: Filters,
        doc_ids: list[str] | None = None,
        strategy: str | None = None,
    ) -> list[ChunkHit]:
        strategy = strategy or self._cfg.strategy
        try:
            fn = {
                "semantic": self.semantic_search,
                "keyword": self.keyword_search,
                "hybrid": self.hybrid_search,
            }[strategy]
        except KeyError:
            raise ValueError(
                f"unknown strategy {strategy!r}; expected 'semantic', 'keyword', or 'hybrid'"
            ) from None
        hits = fn(query, k=max(k, self._cfg.rerank_candidates), filters=filters, doc_ids=doc_ids)

        if self._cfg.rerank_enabled and self._reranker is not None:
            hits = self._reranker.rerank(query, hits, top_n=max(k, self._cfg.rerank_candidates))

        hits = hits[:k]

        if self._cfg.neighbor_expansion > 0:
            hits = expand_neighbours(hits, self._store, width=self._cfg.neighbor_expansion)

        return hits

    def search_profiles(self, query: str, *, k: int, filters: Filters) -> list[DocumentHit]:
        vec = self._embedder.embed_query(query)
        return self._store.search_profiles(vec, k=k, filters=filters)
