"""Tests for the Retriever: semantic, keyword, and hybrid search."""

import pytest

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.rerank.fake import FakeReranker
from porsuk.adapters.store.inmemory import InMemoryStore
from porsuk.core.config import RerankConfig, RetrievalConfig
from porsuk.core.models import Chunk, EmbedResult, Filters
from porsuk.retrieval.search import Retriever


def _store():
    s = InMemoryStore()
    s.ensure_collections(embedding_fingerprint="x")
    chunks = [
        Chunk("c1", "d1", "ödeme koşulları ve gecikme faizi", 1, "tr", None, None, (0, 30)),
        Chunk("c2", "d2", "fesih ve tazminat hükümleri", 1, "tr", None, None, (0, 27)),
        Chunk("c3", "d3", "payment schedule and late fees", 1, "en", None, None, (0, 30)),
    ]
    # give each a real dense + a sparse map keyed by a stable token id per word
    dense = tuple(FakeEmbedder(dim=8).embed_documents([c.text for c in chunks]).dense)
    sparse = ({1: 1.0, 2: 1.0}, {3: 1.0}, {4: 1.0})
    s.upsert_chunks(chunks, EmbedResult(dense=dense, sparse=sparse))
    return s, chunks


class _QSparseEmbedder(FakeEmbedder):
    """Query 'ödeme' -> sparse token 1; everything else -> token 99."""

    def embed_query(self, text):
        base = super().embed_query(text)
        tok = {1: 1.0} if "ödeme" in text else {99: 1.0}
        return EmbedResult(dense=base.dense, sparse=(tok,))


def _retriever(store, *, reranker=None, cfg=None):
    return Retriever(
        embedder=_QSparseEmbedder(dim=8),
        store=store,
        reranker=reranker,
        cfg=cfg or RetrievalConfig(),
    )


def test_semantic_search_returns_chunk_hits():
    store, _ = _store()
    hits = _retriever(store).semantic_search("ödeme", k=3, filters=Filters())
    assert hits
    assert all(h.retrieval_method == "semantic" for h in hits)


def test_keyword_search_uses_plain_text_by_default():
    store, _ = _store()
    # keyword_backend defaults to "text" (plain word membership,
    # no scoring model). "ödeme" is a word in c1's body.
    hits = _retriever(store).keyword_search("ödeme", k=3, filters=Filters())
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].retrieval_method == "keyword"


def test_keyword_search_bm25_backend_when_configured():
    store, _ = _store()
    cfg = RetrievalConfig(keyword_backend="bm25")
    hits = _retriever(store, cfg=cfg).keyword_search("ödeme", k=3, filters=Filters())
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].retrieval_method == "keyword"


def test_keyword_search_sparse_backend_when_configured():
    store, _ = _store()
    cfg = RetrievalConfig(keyword_backend="sparse")
    hits = _retriever(store, cfg=cfg).keyword_search("ödeme", k=3, filters=Filters())
    # _QSparseEmbedder maps "ödeme" -> sparse token 1, which c1 carries.
    assert hits[0].chunk.chunk_id == "c1"


def test_text_and_bm25_backends_need_no_sparse_embedder():
    store, _ = _store()
    for backend in ("text", "bm25"):
        r = Retriever(
            embedder=FakeEmbedder(dim=8),  # returns sparse=None
            store=store,
            reranker=None,
            cfg=RetrievalConfig(keyword_backend=backend),
        )
        hits = r.keyword_search("ödeme", k=3, filters=Filters())  # no ValueError
        assert hits[0].retrieval_method == "keyword"


def test_language_filter_excludes_english():
    store, _ = _store()
    hits = _retriever(store).semantic_search("payment", k=5, filters=Filters(languages=("tr",)))
    assert all(h.chunk.language == "tr" for h in hits)


def test_hybrid_merges_both():
    store, _ = _store()
    hits = _retriever(store).hybrid_search("ödeme", k=3, filters=Filters())
    assert hits
    assert hits[0].retrieval_method == "hybrid"


def test_search_applies_reranker_when_enabled():
    store, _ = _store()
    cfg = RetrievalConfig(
        rerank_enabled=True, rerank_candidates=3, rerank=RerankConfig(provider="fake")
    )
    r = _retriever(store, reranker=FakeReranker(), cfg=cfg)
    hits = r.search("ödeme", k=3, filters=Filters(), strategy="semantic")
    assert all(h.rerank_score is not None for h in hits)


def test_search_returns_k_hits_even_when_k_exceeds_rerank_candidates():
    s = InMemoryStore()
    s.ensure_collections(embedding_fingerprint="x")
    chunks = [
        Chunk(f"c{i}", f"d{i}", f"ödeme koşulları {i}", 1, "tr", None, None, (0, 20))
        for i in range(6)
    ]
    dense = tuple(FakeEmbedder(dim=8).embed_documents([c.text for c in chunks]).dense)
    s.upsert_chunks(chunks, EmbedResult(dense=dense))
    cfg = RetrievalConfig(
        rerank_enabled=True, rerank_candidates=2, rerank=RerankConfig(provider="fake")
    )
    r = _retriever(s, reranker=FakeReranker(), cfg=cfg)
    hits = r.search("ödeme", k=5, filters=Filters(), strategy="semantic")
    assert len(hits) == 5


def test_search_expands_neighbours_when_configured():
    store, chunks = _store()
    # add two more chunks to d1 so there is a neighbour
    more = [
        Chunk("c1a", "d1", "önceki madde", 1, "tr", None, None, (-20, -11)),
        Chunk("c1b", "d1", "sonraki madde", 1, "tr", None, None, (40, 52)),
    ]
    store.upsert_chunks(more, EmbedResult(dense=tuple((0.0,) * 8 for _ in more)))
    cfg = RetrievalConfig(neighbor_expansion=1)
    hits = _retriever(store, cfg=cfg).search("ödeme", k=1, filters=Filters(), strategy="keyword")
    assert "önceki madde" in hits[0].chunk.text or "sonraki madde" in hits[0].chunk.text


def test_search_rejects_unknown_strategy():
    store, _ = _store()
    with pytest.raises(ValueError, match="unknown strategy"):
        _retriever(store).search("q", k=3, filters=Filters(), strategy="dense")


def test_keyword_search_sparse_backend_without_sparse_embedder_raises():
    store, _ = _store()
    r = Retriever(
        embedder=FakeEmbedder(dim=8),  # sparse=None
        store=store,
        reranker=None,
        cfg=RetrievalConfig(keyword_backend="sparse"),
    )
    with pytest.raises(ValueError, match="sparse"):
        r.keyword_search("ödeme", k=3, filters=Filters())
