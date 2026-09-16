"""Tests for the in-memory vector store adapter."""

from datetime import datetime

import pytest

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.store.inmemory import FingerprintMismatchError, InMemoryStore
from porsuk.core.models import Chunk, DocumentProfile, EmbedResult, Filters
from porsuk.core.ports import VectorStore
from porsuk.core.registry import build


def _profile(doc_id: str, language: str = "tr", quality: float = 0.9):
    return DocumentProfile(
        document_id=doc_id,
        path=f"/corpus/{doc_id}.pdf",
        filename=f"{doc_id}.pdf",
        doc_type="pdf",
        language=language,
        created_at=datetime(2024, 1, 1),
        modified_at=datetime(2024, 1, 1),
        size=100,
        page_count=3,
        summary=f"summary of {doc_id}",
        topics=("tedarik",),
        entities=(),
        profile_level="cheap",
        profile_source=frozenset({"has_outline"}),
        parse_quality=quality,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )


def _chunk(chunk_id: str, doc_id: str, text: str, language: str = "tr"):
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        page_no=1,
        language=language,
        section_title=None,
        section_path=None,
        char_span=(0, len(text)),
    )


@pytest.fixture()
def store():
    s = InMemoryStore()
    s.ensure_collections(embedding_fingerprint="fake-8")
    return s


def test_satisfies_the_port():
    assert isinstance(InMemoryStore(), VectorStore)


def _store_with_two_profiles():
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="fake-8")
    embedder = FakeEmbedder(dim=8)
    profiles = [_profile("a", language="tr"), _profile("b", language="en")]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    return store


def test_list_documents_filters_and_caps():
    store = _store_with_two_profiles()
    rows = store.list_documents(Filters(languages=("tr",)), limit=10)
    assert [r.document_id for r in rows] == ["a"]
    assert rows[0].filename  # populated, not empty

    all_rows = store.list_documents(Filters(), limit=1)
    assert len(all_rows) == 1  # limit respected


def test_list_documents_caps_after_sorting_by_filename():
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="fake-8")
    embedder = FakeEmbedder(dim=8)
    # Insertion order z, a, m; filenames sort a < m < z.
    profiles = [_profile("z"), _profile("a"), _profile("m")]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    rows = store.list_documents(Filters(), limit=2)
    assert [r.filename for r in rows] == ["a.pdf", "m.pdf"]


def test_get_profile_returns_the_one_document(store):
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles(
        [_profile("d1", quality=0.9), _profile("d2", quality=0.8)],
        embedder.embed_documents(["summary of d1", "summary of d2"]),
    )
    profile = store.get_profile("d1")
    assert profile is not None
    assert profile.document_id == "d1"
    assert profile.path == "/corpus/d1.pdf"


def test_get_profile_returns_none_when_not_indexed(store):
    assert store.get_profile("nope") is None


def test_is_registered():
    assert isinstance(build("store", {"provider": "inmemory"}), InMemoryStore)


def test_fingerprint_is_stored_on_first_use(store):
    assert store.stored_fingerprint() == "fake-8"


def test_reopening_with_a_different_fingerprint_raises(store):
    with pytest.raises(FingerprintMismatchError) as exc:
        store.ensure_collections(embedding_fingerprint="e5-1024")
    message = str(exc.value)
    assert "fake-8" in message
    assert "e5-1024" in message


def test_search_returns_the_nearest_profile_first(store):
    embedder = FakeEmbedder(dim=8)
    profiles = [_profile("d1"), _profile("d2")]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    query = embedder.embed_query("summary of d2")
    hits = store.search_profiles(query, k=2, filters=Filters())
    assert hits[0].profile.document_id == "d2"
    assert hits[0].retrieval_method == "semantic"


def test_language_filter_excludes_other_languages(store):
    embedder = FakeEmbedder(dim=8)
    profiles = [_profile("tr1", language="tr"), _profile("en1", language="en")]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    hits = store.search_profiles(
        embedder.embed_query("summary"), k=10, filters=Filters(languages=("en",))
    )
    assert [h.profile.document_id for h in hits] == ["en1"]


def test_min_parse_quality_filter(store):
    embedder = FakeEmbedder(dim=8)
    profiles = [_profile("good", quality=0.9), _profile("bad", quality=0.2)]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    hits = store.search_profiles(
        embedder.embed_query("summary"), k=10, filters=Filters(min_parse_quality=0.5)
    )
    assert [h.profile.document_id for h in hits] == ["good"]


def test_chunk_search_can_be_restricted_to_document_ids(store):
    embedder = FakeEmbedder(dim=8)
    chunks = [
        _chunk("c1", "d1", "odeme kosullari"),
        _chunk("c2", "d2", "odeme kosullari"),
    ]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    hits = store.search_chunks(
        embedder.embed_query("odeme"), k=10, filters=Filters(), doc_ids=["d2"]
    )
    assert [h.chunk.chunk_id for h in hits] == ["c2"]


def test_k_limits_the_result_count(store):
    embedder = FakeEmbedder(dim=8)
    chunks = [_chunk(f"c{i}", "d1", f"text {i}") for i in range(10)]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    hits = store.search_chunks(embedder.embed_query("text"), k=3, filters=Filters())
    assert len(hits) == 3


def test_upsert_replaces_a_document_by_id(store):
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["v1"]))
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["v2"]))
    hits = store.search_profiles(embedder.embed_query("v2"), k=10, filters=Filters())
    assert len(hits) == 1


def test_doc_type_filter_excludes_other_types(store):
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["summary of d1"]))
    query = embedder.embed_query("summary")
    assert store.search_profiles(query, k=10, filters=Filters(doc_types=("docx",))) == []
    hits = store.search_profiles(query, k=10, filters=Filters(doc_types=("pdf",)))
    assert [h.profile.document_id for h in hits] == ["d1"]


def test_topic_filter_matches_any_overlap_not_a_subset(store):
    """The fixture profile's topics are ("tedarik",); a filter naming more still matches."""
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["summary of d1"]))
    query = embedder.embed_query("summary")
    hits = store.search_profiles(query, k=10, filters=Filters(topics=("tedarik", "odeme")))
    assert [h.profile.document_id for h in hits] == ["d1"]
    assert store.search_profiles(query, k=10, filters=Filters(topics=("odeme",))) == []


def test_path_prefix_filter_excludes_other_trees(store):
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["summary of d1"]))
    query = embedder.embed_query("summary")
    hits = store.search_profiles(query, k=10, filters=Filters(path_prefix="/corpus/"))
    assert [h.profile.document_id for h in hits] == ["d1"]
    assert store.search_profiles(query, k=10, filters=Filters(path_prefix="/archive/")) == []


def test_date_range_filters_on_modified_at(store):
    """The fixture profile was modified 2024-01-01."""
    embedder = FakeEmbedder(dim=8)
    store.upsert_profiles([_profile("d1")], embedder.embed_documents(["summary of d1"]))
    query = embedder.embed_query("summary")
    assert store.search_profiles(query, k=10, filters=Filters(date_from=datetime(2024, 6, 1))) == []
    assert store.search_profiles(query, k=10, filters=Filters(date_to=datetime(2023, 1, 1))) == []
    hits = store.search_profiles(
        query,
        k=10,
        filters=Filters(date_from=datetime(2023, 1, 1), date_to=datetime(2024, 6, 1)),
    )
    assert [h.profile.document_id for h in hits] == ["d1"]


def test_date_range_excludes_a_profile_with_no_modified_at(store):
    """A profile with modified_at=None (no filesystem date captured) must
    not match a date-range filter: excluded, not treated as always in
    range. Matches QdrantStore's behaviour for the same case."""
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    dated = _profile("dated")
    undated = replace(_profile("undated"), modified_at=None)
    store.upsert_profiles(
        [dated, undated], embedder.embed_documents(["summary of dated", "summary of undated"])
    )
    query = embedder.embed_query("summary")

    hits = store.search_profiles(query, k=10, filters=Filters(date_from=datetime(2020, 1, 1)))
    assert [h.profile.document_id for h in hits] == ["dated"]


def test_chunk_language_filter_excludes_other_languages(store):
    embedder = FakeEmbedder(dim=8)
    chunks = [
        _chunk("c1", "d1", "odeme kosullari", language="tr"),
        _chunk("c2", "d1", "payment terms", language="en"),
    ]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    hits = store.search_chunks(
        embedder.embed_query("odeme"), k=10, filters=Filters(languages=("en",))
    )
    assert [h.chunk.chunk_id for h in hits] == ["c2"]


def test_cosine_scoring_is_scale_invariant(store):
    """A real embedder need not return unit vectors; ranking must not depend on magnitude."""
    embedder = FakeEmbedder(dim=8)
    profiles = [_profile("d1"), _profile("d2")]
    unit = embedder.embed_documents([p.summary for p in profiles])
    scaled = EmbedResult(dense=tuple(tuple(x * 7.0 for x in v) for v in unit.dense))
    store.upsert_profiles(profiles, scaled)
    hits = store.search_profiles(embedder.embed_query("summary of d2"), k=2, filters=Filters())
    assert hits[0].profile.document_id == "d2"
    assert hits[0].score == pytest.approx(1.0)


def test_sparse_search_ranks_by_dot_product(store):
    chunks = [_chunk("c1", "d1", "ödeme"), _chunk("c2", "d1", "fesih")]
    store.upsert_chunks(
        chunks,
        EmbedResult(
            dense=((1.0, 0, 0, 0, 0, 0, 0, 0), (0, 1.0, 0, 0, 0, 0, 0, 0)),
            sparse=({7: 0.9, 42: 0.5}, {13: 0.8}),
        ),
    )
    hits = store.search_chunks_sparse(
        EmbedResult(dense=(), sparse=({7: 1.0},)), k=5, filters=Filters()
    )
    assert [h.chunk.chunk_id for h in hits] == ["c1"]
    assert hits[0].retrieval_method == "keyword"


def test_sparse_search_respects_language_and_doc_id_filters(store):
    store.upsert_chunks(
        [
            _chunk("tr1", "d1", "sözleşme", language="tr"),
            _chunk("en1", "d2", "contract", language="en"),
            _chunk("tr2", "d3", "sözleşme", language="tr"),
        ],
        EmbedResult(
            dense=((1.0,) + (0,) * 7,) * 3,
            sparse=({7: 1.0}, {7: 1.0}, {7: 1.0}),
        ),
    )
    by_lang = store.search_chunks_sparse(
        EmbedResult(dense=(), sparse=({7: 1.0},)), k=5, filters=Filters(languages=("tr",))
    )
    assert {h.chunk.chunk_id for h in by_lang} == {"tr1", "tr2"}
    by_doc = store.search_chunks_sparse(
        EmbedResult(dense=(), sparse=({7: 1.0},)), k=5, filters=Filters(), doc_ids=["d1"]
    )
    assert [h.chunk.chunk_id for h in by_doc] == ["tr1"]


def test_an_empty_doc_ids_list_matches_nothing(store):
    """`doc_ids=None` means no restriction; `doc_ids=[]` means nothing matches."""
    embedder = FakeEmbedder(dim=8)
    chunks = [_chunk("c1", "d1", "odeme kosullari")]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    query = embedder.embed_query("odeme")
    assert store.search_chunks(query, k=10, filters=Filters(), doc_ids=[]) == []
    hits = store.search_chunks(query, k=10, filters=Filters(), doc_ids=None)
    assert [h.chunk.chunk_id for h in hits] == ["c1"]


def test_inmemory_scopes_chunks_by_job_id(store):
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    chunks = [
        replace(_chunk("c1", "d1", "text one"), job_id="j1"),
        replace(_chunk("c2", "d2", "text two"), job_id="j2"),
        replace(_chunk("c3", "d3", "text three"), job_id=None),
    ]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    query = embedder.embed_query("text")

    all_hits = store.search_chunks(query, k=10, filters=Filters())
    assert {h.chunk.chunk_id for h in all_hits} == {"c1", "c2", "c3"}

    j1_hits = store.search_chunks(query, k=10, filters=Filters(job_ids=("j1",)))
    assert {h.chunk.chunk_id for h in j1_hits} == {"c1"}  # not c2, not the None-job c3

    both = store.search_chunks(query, k=10, filters=Filters(job_ids=("j1", "j2")))
    assert {h.chunk.chunk_id for h in both} == {"c1", "c2"}


def test_inmemory_scopes_profiles_by_job_id(store):
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    profiles = [
        replace(_profile("a"), job_id="j1"),
        replace(_profile("b"), job_id="j2"),
        replace(_profile("c"), job_id=None),
    ]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))
    query = embedder.embed_query("summary")

    all_hits = store.search_profiles(query, k=10, filters=Filters())
    assert {h.profile.document_id for h in all_hits} == {"a", "b", "c"}
    assert {r.document_id for r in store.list_documents(Filters(), limit=10)} == {"a", "b", "c"}

    j1_hits = store.search_profiles(query, k=10, filters=Filters(job_ids=("j1",)))
    assert {h.profile.document_id for h in j1_hits} == {"a"}
    j1_rows = store.list_documents(Filters(job_ids=("j1",)), limit=10)
    assert {r.document_id for r in j1_rows} == {"a"}

    both = store.search_profiles(query, k=10, filters=Filters(job_ids=("j1", "j2")))
    assert {h.profile.document_id for h in both} == {"a", "b"}


def test_inmemory_unscoped_excludes_other_jobs_flag(store):
    """unscoped_excludes_other_jobs=True sees only job_id=None rows: a chat
    with no scope of its own must not see another chat's uploads."""
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    chunks = [
        replace(_chunk("c1", "d1", "text one"), job_id="j1"),
        replace(_chunk("c2", "d2", "text two"), job_id="j2"),
        replace(_chunk("c3", "d3", "text three"), job_id=None),
    ]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    query = embedder.embed_query("text")

    unscoped = store.search_chunks(
        query, k=10, filters=Filters(unscoped_excludes_other_jobs=True)
    )
    assert {h.chunk.chunk_id for h in unscoped} == {"c3"}

    # the flag unset is unchanged: job_ids=() alone still means no restriction.
    all_hits = store.search_chunks(query, k=10, filters=Filters())
    assert {h.chunk.chunk_id for h in all_hits} == {"c1", "c2", "c3"}


def test_set_job_id_re_stamps_every_chunk_and_the_profile_for_a_document(store):
    """"Kalıcı yap": promote one chat-scoped upload into the base corpus,
    visible from every chat afterward (job_id=None)."""
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    chunks = [
        replace(_chunk("c1", "d1", "birinci parça"), job_id="j1"),
        replace(_chunk("c2", "d1", "ikinci parça"), job_id="j1"),
        replace(_chunk("c3", "d2", "başka belge"), job_id="j1"),  # different doc, same job
    ]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    profiles = [replace(_profile("d1"), job_id="j1"), replace(_profile("d2"), job_id="j1")]
    store.upsert_profiles(profiles, embedder.embed_documents([p.summary for p in profiles]))

    store.set_job_id("d1", None)

    d1_chunks = store.chunks_for_document("d1")
    assert len(d1_chunks) == 2
    assert all(c.job_id is None for c in d1_chunks)
    # d2's chunks are untouched: set_job_id is scoped to the one document_id.
    d2_chunks = store.chunks_for_document("d2")
    assert all(c.job_id == "j1" for c in d2_chunks)

    unscoped_docs = {
        r.document_id for r in store.list_documents(Filters(job_ids=()), limit=10)
    }
    d1_profile_job_id = next(
        h.profile.job_id
        for h in store.search_profiles(
            embedder.embed_query("summary"), k=10, filters=Filters()
        )
        if h.profile.document_id == "d1"
    )
    assert d1_profile_job_id is None
    assert "d1" in unscoped_docs


def test_set_job_id_can_move_to_a_different_job_not_just_none(store):
    from dataclasses import replace

    embedder = FakeEmbedder(dim=8)
    chunks = [replace(_chunk("c1", "d1", "metin"), job_id="j1")]
    store.upsert_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))

    store.set_job_id("d1", "j2")

    (chunk,) = store.chunks_for_document("d1")
    assert chunk.job_id == "j2"
