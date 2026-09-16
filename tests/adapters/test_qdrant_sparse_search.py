import pytest

from porsuk.adapters.store.qdrant import QdrantStore
from porsuk.core.models import Chunk, EmbedResult, Filters

_FP = "test:bge-m3:1024:norm=True:sparse=True"


@pytest.fixture
def store():
    s = QdrantStore(url=":memory:", dim=4)
    s.ensure_collections(embedding_fingerprint=_FP)
    return s


def _chunk(cid, doc_id, text):
    return Chunk(cid, doc_id, text, 1, "tr", "Madde 1", "Madde 1", (0, len(text)))


def test_sparse_search_ranks_by_lexical_overlap(store):
    chunks = [
        _chunk("c1", "d1", "ödeme koşulları"),
        _chunk("c2", "d1", "fesih hükümleri"),
    ]
    vectors = EmbedResult(
        dense=((1.0, 0, 0, 0), (0, 1.0, 0, 0)),
        sparse=({7: 0.9, 42: 0.5}, {13: 0.8}),  # c1: tokens 7,42  c2: token 13
    )
    store.upsert_chunks(chunks, vectors)

    # query sparse hits token 7 -> c1 should rank first
    hits = store.search_chunks_sparse(
        EmbedResult(dense=(), sparse=({7: 1.0},)), k=5, filters=Filters()
    )
    assert hits[0].chunk.chunk_id == "c1"
    assert hits[0].retrieval_method == "keyword"


def test_sparse_search_respects_language_filter(store):
    store.upsert_chunks(
        [
            Chunk("tr1", "d1", "sözleşme", 1, "tr", None, None, (0, 8)),
            Chunk("en1", "d2", "contract", 1, "en", None, None, (0, 8)),
        ],
        EmbedResult(dense=((1.0, 0, 0, 0), (1.0, 0, 0, 0)), sparse=({7: 1.0}, {7: 1.0})),
    )
    hits = store.search_chunks_sparse(
        EmbedResult(dense=(), sparse=({7: 1.0},)), k=5, filters=Filters(languages=("tr",))
    )
    assert [h.chunk.chunk_id for h in hits] == ["tr1"]
