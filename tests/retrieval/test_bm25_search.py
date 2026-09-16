"""BM25 keyword search: inmemory reference and Qdrant adapter. The named
regression: `keyword_search("doktor")` must not drop the chunk that
contains the "Doktor Öğretim Üyesi" definition, which bge-m3's learned sparse
did (token weight 0.0 in that chunk).
"""

from __future__ import annotations

import pytest

from porsuk.adapters.store.inmemory import InMemoryStore
from porsuk.core.models import Chunk, EmbedResult, Filters
from porsuk.retrieval import bm25


def _chunk(cid: str, text: str, document_id: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=document_id,
        text=text,
        page_no=1,
        language="tr",
        section_title=None,
        section_path=None,
        char_span=(0, len(text)),
    )


_DEF = "Doktor Öğretim Üyesi: doktora çalışmalarını başarı ile tamamlamış akademik unvana sahip kişidir"
_KADRO = "Yükseköğretim kurumlarında açık bulunan doktor öğretim üyesi kadroları rektörlükçe ilan edilir"
_TAYIN = "tayın bedeli her yıl bütçe kanununda belirlenen miktarın altında olacak şekilde saptanır"


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore(dim=3)
    s.ensure_collections(embedding_fingerprint="test")
    chunks = [_chunk("d1:00000", _DEF), _chunk("d1:00001", _KADRO), _chunk("d1:00002", _TAYIN)]
    s.upsert_chunks(chunks, EmbedResult(dense=((0.1, 0.2, 0.3),) * 3, sparse=None))
    return s


def test_bm25_finds_definition_chunk_for_bare_term(store):
    hits = store.search_chunks_bm25(bm25.query_terms("doktor"), k=3, filters=Filters())
    assert hits, "no hits for 'doktor'"
    ids = [h.chunk.chunk_id for h in hits]
    assert "d1:00000" in ids  # the definition chunk is not dropped
    assert "d1:00002" not in ids  # the tayın-bedeli chunk has no "doktor"


def test_bm25_method_and_score(store):
    hits = store.search_chunks_bm25(bm25.query_terms("tayın bedeli"), k=1, filters=Filters())
    assert hits[0].chunk.chunk_id == "d1:00002"
    assert hits[0].retrieval_method == "keyword"
    assert hits[0].score > 0


def test_bm25_empty_query_returns_nothing(store):
    assert store.search_chunks_bm25({}, k=3, filters=Filters()) == []


def test_bm25_respects_doc_ids_filter(store):
    hits = store.search_chunks_bm25(
        bm25.query_terms("doktor"), k=3, filters=Filters(), doc_ids=["other"]
    )
    assert hits == []


# ----- Qdrant adapter: same behaviour, real IDF sparse vector -----

from porsuk.adapters.store.qdrant import QdrantStore  # noqa: E402


@pytest.fixture
def qdrant_store() -> QdrantStore:
    s = QdrantStore(url=":memory:", dim=3)
    s.ensure_collections(embedding_fingerprint="test-bm25")
    chunks = [
        _chunk("d1:00000", _DEF),
        _chunk("d1:00001", _KADRO),
        _chunk("d1:00002", "Bu kanunda geçen kavramların tanımları aşağıda belirtilmiştir"),
    ]
    s.upsert_chunks(chunks, EmbedResult(dense=((0.1, 0.2, 0.3),) * 3, sparse=None))
    return s


def test_qdrant_bm25_surfaces_definition_chunk_for_definition_query(qdrant_store):
    # bge-m3 sparse dropped the definition chunk entirely (token weight 0.0);
    # BM25 must at least surface it in the top results for a query whose terms
    # it contains. Exact rank-0 is the reranker's job, not the lexical stage.
    hits = qdrant_store.search_chunks_bm25(
        bm25.query_terms("Doktor Öğretim Üyesi tanımı"), k=3, filters=Filters()
    )
    ids = [h.chunk.chunk_id for h in hits]
    assert "d1:00000" in ids[:2]


def test_qdrant_bm25_bare_term_does_not_drop_the_chunk_that_contains_it(qdrant_store):
    hits = qdrant_store.search_chunks_bm25(bm25.query_terms("doktor"), k=3, filters=Filters())
    ids = [h.chunk.chunk_id for h in hits]
    assert "d1:00000" in ids


def test_qdrant_bm25_empty_query(qdrant_store):
    assert qdrant_store.search_chunks_bm25({}, k=3, filters=Filters()) == []


def test_qdrant_bm25_method(qdrant_store):
    hits = qdrant_store.search_chunks_bm25(bm25.query_terms("doktor"), k=1, filters=Filters())
    assert hits[0].retrieval_method == "keyword"


# ----- plain word-membership backend (keyword_backend=text) -----


def test_inmemory_text_ranks_by_distinct_word_count(store):
    # "doktor öğretim üyesi": all three words are in _DEF and _KADRO; _TAYIN
    # has none. Rank is distinct-word-count, so both doktor chunks come first.
    hits = store.search_chunks_text(["doktor", "öğretim", "üyesi"], k=3, filters=Filters())
    ids = [h.chunk.chunk_id for h in hits]
    assert set(ids[:2]) == {"d1:00000", "d1:00001"}
    assert "d1:00002" not in ids
    assert hits[0].retrieval_method == "keyword"


def test_inmemory_text_bare_term_keeps_the_chunk_that_contains_it(store):
    hits = store.search_chunks_text(["doktor"], k=3, filters=Filters())
    assert "d1:00000" in [h.chunk.chunk_id for h in hits]


def test_inmemory_text_empty_and_no_match(store):
    assert store.search_chunks_text([], k=3, filters=Filters()) == []
    assert store.search_chunks_text(["zürafa"], k=3, filters=Filters()) == []


def test_qdrant_text_matches_word_membership(qdrant_store):
    hits = qdrant_store.search_chunks_text(["doktor", "tanımları"], k=3, filters=Filters())
    ids = [h.chunk.chunk_id for h in hits]
    # d1:00000 has "doktor" (and "doktora"); d1:00002 has "tanımları"
    assert "d1:00000" in ids
    assert hits[0].retrieval_method == "keyword"


def test_qdrant_text_empty(qdrant_store):
    assert qdrant_store.search_chunks_text([], k=3, filters=Filters()) == []
