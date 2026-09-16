"""Tests for the Qdrant vector store adapter."""

from dataclasses import replace

import pytest

from porsuk.adapters.store.qdrant import QdrantStore
from porsuk.core.models import Chunk, DocumentProfile, EmbedResult, Filters
from porsuk.core.ports import VectorStore

_FP = "test:bge-m3:4:norm=True"


@pytest.fixture
def store():
    s = QdrantStore(url=":memory:", dim=4)
    s.ensure_collections(embedding_fingerprint=_FP)
    return s


def _profile(doc_id, *, doc_type="pdf", language="tr", quality=0.9):
    return DocumentProfile(
        document_id=doc_id,
        path=f"/corpus/{doc_id}.pdf",
        filename=f"{doc_id}.pdf",
        doc_type=doc_type,
        language=language,
        created_at=None,
        modified_at=None,
        size=10,
        page_count=1,
        summary="tedarik sözleşmesi",
        topics=("tedarik", "ödeme"),
        entities=("ABC Lojistik",),
        profile_level="cheap",
        profile_source=frozenset({"has_intro"}),
        parse_quality=quality,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )


def _chunk(chunk_id, doc_id, text, *, language="tr"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        page_no=1,
        language=language,
        section_title="Madde 1",
        section_path="Madde 1",
        char_span=(0, len(text)),
    )


def test_satisfies_the_port(store):
    assert isinstance(store, VectorStore)


def _upsert_two_profiles(store):
    a = replace(_profile("a", language="tr"), topics=("mevzuat",))
    b = replace(_profile("b", language="en"), topics=("contract",))
    store.upsert_profiles(
        [a, b],
        EmbedResult(dense=((1.0, 0, 0, 0), (1.0, 0, 0, 0))),
    )


def test_list_documents_applies_filters(store):
    _upsert_two_profiles(store)
    rows = store.list_documents(Filters(languages=("tr",)), limit=50)
    assert {r.document_id for r in rows} == {"a"}
    assert rows[0].topics == ("mevzuat",)


def test_list_documents_caps_after_sorting_by_filename(store):
    # Insertion order z, a, m; filenames sort a < m < z. A scroll(limit=2)
    # would cap before sorting and could drop "a.pdf"; sort-then-cap must not.
    store.upsert_profiles(
        [_profile("z"), _profile("a"), _profile("m")],
        EmbedResult(dense=((1.0, 0, 0, 0), (1.0, 0, 0, 0), (1.0, 0, 0, 0))),
    )
    rows = store.list_documents(Filters(), limit=2)
    assert [r.filename for r in rows] == ["a.pdf", "m.pdf"]


def test_fingerprint_roundtrip(store):
    assert store.stored_fingerprint() == _FP


def test_fingerprint_mismatch_raises():
    s = QdrantStore(url=":memory:", dim=4)
    s.ensure_collections(embedding_fingerprint="one")
    with pytest.raises(ValueError, match="index was built"):
        s.ensure_collections(embedding_fingerprint="two")


def test_ensure_collections_is_idempotent(store):
    store.ensure_collections(embedding_fingerprint=_FP)  # again, same fp
    assert store.stored_fingerprint() == _FP


def test_profile_upsert_and_search(store):
    store.upsert_profiles([_profile("d1")], EmbedResult(dense=((1.0, 0.0, 0.0, 0.0),)))
    hits = store.search_profiles(EmbedResult(dense=((1.0, 0.0, 0.0, 0.0),)), k=5, filters=Filters())
    assert hits[0].profile.document_id == "d1"
    assert hits[0].profile.topics == ("tedarik", "ödeme")
    assert hits[0].retrieval_method == "semantic"


def test_profile_search_filtered_by_language(store):
    store.upsert_profiles(
        [_profile("tr1", language="tr"), _profile("en1", language="en")],
        EmbedResult(dense=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
    )
    hits = store.search_profiles(
        EmbedResult(dense=((1.0, 0.0, 0.0, 0.0),)),
        k=5,
        filters=Filters(languages=("en",)),
    )
    assert [h.profile.document_id for h in hits] == ["en1"]


def test_chunk_search_filtered_by_doc_ids(store):
    chunks = [
        _chunk("c1", "d1", "ödeme koşulları"),
        _chunk("c2", "d2", "fesih hükümleri"),
    ]
    store.upsert_chunks(chunks, EmbedResult(dense=((1.0, 0, 0, 0), (0, 1.0, 0, 0))))
    hits = store.search_chunks(
        EmbedResult(dense=((1.0, 0, 0, 0),)),
        k=5,
        filters=Filters(),
        doc_ids=["d1"],
    )
    assert [h.chunk.chunk_id for h in hits] == ["c1"]
    assert hits[0].chunk.section_path == "Madde 1"


def test_min_parse_quality_filter(store):
    store.upsert_profiles(
        [_profile("good", quality=0.9), _profile("bad", quality=0.3)],
        EmbedResult(dense=((1.0, 0, 0, 0), (1.0, 0, 0, 0))),
    )
    hits = store.search_profiles(
        EmbedResult(dense=((1.0, 0, 0, 0),)),
        k=5,
        filters=Filters(min_parse_quality=0.5),
    )
    assert [h.profile.document_id for h in hits] == ["good"]


def test_chunks_with_sparse_vectors_are_stored(store):
    chunks = [_chunk("c1", "d1", "ödeme koşulları"), _chunk("c2", "d1", "fesih hükümleri")]
    vectors = EmbedResult(
        dense=((1.0, 0, 0, 0), (0, 1.0, 0, 0)),
        sparse=({7: 0.5, 42: 0.9}, {13: 0.3}),
    )
    store.upsert_chunks(chunks, vectors)
    # dense search still works alongside the stored sparse vectors
    hits = store.search_chunks(EmbedResult(dense=((1.0, 0, 0, 0),)), k=5, filters=Filters())
    assert {h.chunk.chunk_id for h in hits} == {"c1", "c2"}
    # the sparse vector is actually persisted on the point
    stored = store.sparse_vector_for("c1")
    assert stored == {7: 0.5, 42: 0.9}


def test_chunks_without_sparse_still_work(store):
    chunks = [_chunk("c1", "d1", "ödeme")]
    store.upsert_chunks(chunks, EmbedResult(dense=((1.0, 0, 0, 0),), sparse=None))
    hits = store.search_chunks(EmbedResult(dense=((1.0, 0, 0, 0),)), k=5, filters=Filters())
    assert hits[0].chunk.chunk_id == "c1"


def _spanned_chunk(chunk_id, doc_id, span):
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=f"{chunk_id} text",
        page_no=1,
        language="tr",
        section_title="Madde 1",
        section_path="Madde 1",
        char_span=span,
    )


def test_chunks_for_document_returns_them_in_char_span_order(store):
    chunks = [
        _spanned_chunk("c3", "d1", (20, 29)),
        _spanned_chunk("c1", "d1", (0, 9)),
        _spanned_chunk("c2", "d1", (10, 19)),
        _spanned_chunk("other", "d2", (0, 9)),
    ]
    dense = tuple((0.0, 0.0, 0.0, 0.0) for _ in chunks)
    store.upsert_chunks(chunks, EmbedResult(dense=dense))

    got = store.chunks_for_document("d1")
    assert [c.chunk_id for c in got] == ["c1", "c2", "c3"]
    assert store.chunks_for_document("d_none") == []


def test_qdrant_scopes_chunks_by_job_id(store):
    chunks = [
        replace(_chunk("c1", "d1", "text one"), job_id="j1"),
        replace(_chunk("c2", "d2", "text two"), job_id="j2"),
        replace(_chunk("c3", "d3", "text three"), job_id=None),
    ]
    store.upsert_chunks(chunks, EmbedResult(dense=((1.0, 0, 0, 0),) * 3))
    query = EmbedResult(dense=((1.0, 0, 0, 0),))

    all_hits = store.search_chunks(query, k=10, filters=Filters())
    assert {h.chunk.chunk_id for h in all_hits} == {"c1", "c2", "c3"}
    assert {h.chunk.job_id for h in all_hits} == {"j1", "j2", None}

    j1_hits = store.search_chunks(query, k=10, filters=Filters(job_ids=("j1",)))
    assert {h.chunk.chunk_id for h in j1_hits} == {"c1"}  # not c2, not the None-job c3

    both = store.search_chunks(query, k=10, filters=Filters(job_ids=("j1", "j2")))
    assert {h.chunk.chunk_id for h in both} == {"c1", "c2"}


def test_qdrant_scopes_profiles_by_job_id(store):
    profiles = [
        replace(_profile("a"), job_id="j1"),
        replace(_profile("b"), job_id="j2"),
        replace(_profile("c"), job_id=None),
    ]
    store.upsert_profiles(profiles, EmbedResult(dense=((1.0, 0, 0, 0),) * 3))
    query = EmbedResult(dense=((1.0, 0, 0, 0),))

    all_hits = store.search_profiles(query, k=10, filters=Filters())
    assert {h.profile.document_id for h in all_hits} == {"a", "b", "c"}
    assert {r.document_id for r in store.list_documents(Filters(), limit=10)} == {"a", "b", "c"}

    j1_hits = store.search_profiles(query, k=10, filters=Filters(job_ids=("j1",)))
    assert {h.profile.document_id for h in j1_hits} == {"a"}
    j1_rows = store.list_documents(Filters(job_ids=("j1",)), limit=10)
    assert {r.document_id for r in j1_rows} == {"a"}

    both = store.search_profiles(query, k=10, filters=Filters(job_ids=("j1", "j2")))
    assert {h.profile.document_id for h in both} == {"a", "b"}


def test_qdrant_round_trips_modified_at(store):
    from datetime import UTC, datetime

    when = datetime(2022, 3, 15, 10, 0, 0, tzinfo=UTC)
    store.upsert_profiles([replace(_profile("a"), modified_at=when)], EmbedResult(dense=((1.0, 0, 0, 0),)))

    query = EmbedResult(dense=((1.0, 0, 0, 0),))
    (hit,) = store.search_profiles(query, k=10, filters=Filters())
    assert hit.profile.modified_at == when


def test_qdrant_filters_profiles_by_date_range(store):
    from datetime import UTC, datetime

    old = replace(_profile("old"), modified_at=datetime(2020, 1, 1, tzinfo=UTC))
    mid = replace(_profile("mid"), modified_at=datetime(2022, 6, 1, tzinfo=UTC))
    new = replace(_profile("new"), modified_at=datetime(2024, 1, 1, tzinfo=UTC))
    store.upsert_profiles([old, mid, new], EmbedResult(dense=((1.0, 0, 0, 0),) * 3))
    query = EmbedResult(dense=((1.0, 0, 0, 0),))

    since_2021 = store.search_profiles(
        query, k=10, filters=Filters(date_from=datetime(2021, 1, 1, tzinfo=UTC))
    )
    assert {h.profile.document_id for h in since_2021} == {"mid", "new"}

    before_2023 = store.search_profiles(
        query, k=10, filters=Filters(date_to=datetime(2023, 1, 1, tzinfo=UTC))
    )
    assert {h.profile.document_id for h in before_2023} == {"old", "mid"}

    between = store.search_profiles(
        query,
        k=10,
        filters=Filters(
            date_from=datetime(2021, 1, 1, tzinfo=UTC), date_to=datetime(2023, 1, 1, tzinfo=UTC)
        ),
    )
    assert {h.profile.document_id for h in between} == {"mid"}

    listed = store.list_documents(Filters(date_from=datetime(2021, 1, 1, tzinfo=UTC)), limit=10)
    assert {r.document_id for r in listed} == {"mid", "new"}


def test_qdrant_date_filter_excludes_profiles_with_no_modified_at(store):
    """A profile with modified_at=None (no filesystem date, or from before
    this feature existed) must not match a date-range filter either way:
    excluded rather than treated as "always in range"."""
    from datetime import UTC, datetime

    store.upsert_profiles(
        [replace(_profile("dated"), modified_at=datetime(2022, 1, 1, tzinfo=UTC)), _profile("undated")],
        EmbedResult(dense=((1.0, 0, 0, 0),) * 2),
    )
    query = EmbedResult(dense=((1.0, 0, 0, 0),))

    hits = store.search_profiles(
        query, k=10, filters=Filters(date_from=datetime(2020, 1, 1, tzinfo=UTC))
    )
    assert {h.profile.document_id for h in hits} == {"dated"}


def test_qdrant_get_profile_returns_the_one_document(store):
    store.upsert_profiles(
        [_profile("d1"), _profile("d2")], EmbedResult(dense=((1.0, 0, 0, 0),) * 2)
    )
    profile = store.get_profile("d1")
    assert profile is not None
    assert profile.document_id == "d1"
    assert profile.path == "/corpus/d1.pdf"


def test_qdrant_get_profile_returns_none_when_not_indexed(store):
    assert store.get_profile("nope") is None


def test_qdrant_set_job_id_re_stamps_chunks_and_profile(store):
    chunks = [
        replace(_chunk("c1", "d1", "birinci parça"), job_id="j1"),
        replace(_chunk("c2", "d1", "ikinci parça"), job_id="j1"),
        replace(_chunk("c3", "d2", "başka belge"), job_id="j1"),
    ]
    store.upsert_chunks(chunks, EmbedResult(dense=((1.0, 0, 0, 0),) * 3))
    profiles = [replace(_profile("d1"), job_id="j1"), replace(_profile("d2"), job_id="j1")]
    store.upsert_profiles(profiles, EmbedResult(dense=((1.0, 0, 0, 0),) * 2))

    store.set_job_id("d1", None)

    d1_chunks = store.chunks_for_document("d1")
    assert len(d1_chunks) == 2
    assert all(c.job_id is None for c in d1_chunks)
    d2_chunks = store.chunks_for_document("d2")
    assert all(c.job_id == "j1" for c in d2_chunks)  # untouched

    unscoped_rows = store.list_documents(Filters(job_ids=()), limit=10)
    d1_row_job_id = next(
        h.profile.job_id
        for h in store.search_profiles(
            EmbedResult(dense=((1.0, 0, 0, 0),)), k=10, filters=Filters()
        )
        if h.profile.document_id == "d1"
    )
    assert d1_row_job_id is None
    assert "d1" in {r.document_id for r in unscoped_rows}


def test_qdrant_promoted_document_is_visible_to_an_unscoped_chat(store):
    """"Kalıcı yap": after set_job_id(doc, None), a chat with NO scope of
    its own (unscoped_excludes_other_jobs=True, the real path a fresh chat
    with no upload takes) must see the promoted document. Regression: Qdrant
    `set_payload` with a None value DELETES the payload key rather than
    storing a JSON null; `IsNullCondition` (which only matches a key that
    exists AND is null) then never matches a promoted document, silently
    hiding it from every unscoped chat forever. `IsEmptyCondition` matches
    both "missing" and "null" and must be used instead."""
    store.upsert_profiles(
        [replace(_profile("d1"), job_id="j1")], EmbedResult(dense=((1.0, 0, 0, 0),))
    )
    store.set_job_id("d1", None)

    query = EmbedResult(dense=((1.0, 0, 0, 0),))
    unscoped_hits = store.search_profiles(
        query, k=10, filters=Filters(unscoped_excludes_other_jobs=True)
    )
    assert {h.profile.document_id for h in unscoped_hits} == {"d1"}


def test_qdrant_unscoped_excludes_other_jobs_flag(store):
    """A chat with no scope of its own must see the base corpus
    (job_id unset) but not another chat's uploaded documents (job_id set):
    `unscoped_excludes_other_jobs=True` is a separate flag from `job_ids`,
    which keeps meaning 'no restriction' when unset (CLI/eval callers)."""
    chunks = [
        replace(_chunk("c1", "d1", "text one"), job_id="j1"),
        replace(_chunk("c2", "d2", "text two"), job_id="j2"),
        replace(_chunk("c3", "d3", "text three"), job_id=None),
    ]
    store.upsert_chunks(chunks, EmbedResult(dense=((1.0, 0, 0, 0),) * 3))
    query = EmbedResult(dense=((1.0, 0, 0, 0),))

    unscoped = store.search_chunks(
        query, k=10, filters=Filters(unscoped_excludes_other_jobs=True)
    )
    assert {h.chunk.chunk_id for h in unscoped} == {"c3"}  # base corpus only

    # job_ids=() alone (the flag unset) is still "no restriction": CLI/eval
    # callers that never pass the flag see everything, unchanged.
    all_hits = store.search_chunks(query, k=10, filters=Filters())
    assert {h.chunk.chunk_id for h in all_hits} == {"c1", "c2", "c3"}
