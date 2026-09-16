"""Tests for `build_tools`. All seven tools, network-free."""

from __future__ import annotations

from porsuk.agent.tools import build_tools
from porsuk.core.config import AgentConfig
from porsuk.core.models import Chunk, Filters
from tests.agent.conftest import _chunk, _hit, _StubRetriever, _StubStore


def _by_name(retriever=None, store=None, *, cfg=None, lang=None):
    return {
        t.name: t
        for t in build_tools(
            retriever or _StubRetriever(),
            store or _StubStore(),
            cfg=cfg or AgentConfig(),
            lang=lang,
        )
    }


def test_build_tools_returns_the_seven_in_order():
    names = [t.name for t in build_tools(_StubRetriever(), _StubStore(), cfg=AgentConfig())]
    assert names == [
        "search_documents",
        "semantic_search",
        "keyword_search",
        "get_document",
        "get_document_outline",
        "list_documents",
        "expand_context",
    ]


def test_search_documents_emits_file_and_is_profile_level():
    tools = _by_name()
    out = tools["search_documents"].invoke({"query": "ödeme koşulları"})
    assert out[0]["file"] == "Sozlesme.pdf"
    assert out[0]["doc_id"] == "d1"
    assert out[0]["score"] == 0.9123
    assert "topics" in out[0]


def test_search_documents_accepts_iso_date_range_and_forwards_it():
    from datetime import UTC, datetime

    retriever = _StubRetriever()
    tools = _by_name(retriever=retriever)
    tools["search_documents"].invoke(
        {"query": "sözleşme", "date_from": "2022-01-01", "date_to": "2023-06-15"}
    )
    assert retriever.last_filters.date_from == datetime(2022, 1, 1, tzinfo=UTC)
    assert retriever.last_filters.date_to == datetime(2023, 6, 15, tzinfo=UTC)


def test_search_documents_with_no_dates_leaves_filters_unset():
    retriever = _StubRetriever()
    tools = _by_name(retriever=retriever)
    tools["search_documents"].invoke({"query": "sözleşme"})
    assert retriever.last_filters.date_from is None
    assert retriever.last_filters.date_to is None


def test_search_documents_rejects_a_malformed_date():
    tools = _by_name()
    out = tools["search_documents"].invoke({"query": "x", "date_from": "not-a-date"})
    assert "error" in out[0]
    assert "date_from" in out[0]["error"]


def test_list_documents_accepts_iso_date_range_and_forwards_it():
    from datetime import UTC, datetime

    store = _StubStore()
    tools = _by_name(store=store)
    tools["list_documents"].invoke({"date_from": "2022-01-01"})
    assert store.last_filters.date_from == datetime(2022, 1, 1, tzinfo=UTC)


def test_search_documents_does_not_populate_seen():
    tools = _by_name()
    tools["search_documents"].invoke({"query": "x"})
    out = tools["expand_context"].invoke({"chunk_id": "c1"})
    assert "error" in out[0]


def test_chunk_search_tools_emit_doc_id_and_chunk_id_and_populate_seen():
    for name in ("semantic_search", "keyword_search"):
        tools = _by_name()
        out = tools[name].invoke({"query": "ödeme"})
        assert out[0]["doc_id"] == "d1"
        assert out[0]["chunk_id"] == "c1"
        assert out[0]["section"] == "3. Mali > 3.2 Ödeme"
        assert out[0]["page"] == 12
        # side effect: expand_context now knows the chunk
        widened = tools["expand_context"].invoke({"chunk_id": "c1"})
        assert "text" in widened[0]


def test_hit_payload_prefers_rerank_score():
    class _R(_StubRetriever):
        def semantic_search(self, query, *, k, filters, doc_ids=None):
            return [_hit(_chunk("c9", "d1", "text", span=(0, 4)), score=0.1, rerank_score=0.77)]

    tools = _by_name(retriever=_R())
    out = tools["semantic_search"].invoke({"query": "x"})
    assert out[0]["score"] == 0.77


def test_lang_becomes_a_language_filter_on_search_tools():
    r = _StubRetriever()
    tools = _by_name(retriever=r, lang="tr")
    tools["semantic_search"].invoke({"query": "x"})
    assert r.last_filters == Filters(languages=("tr",), unscoped_excludes_other_jobs=True)

    r2 = _StubRetriever()
    _by_name(retriever=r2)["semantic_search"].invoke({"query": "x"})
    assert r2.last_filters == Filters(unscoped_excludes_other_jobs=True)


def test_scope_becomes_a_job_id_filter_on_search_tools():
    r = _StubRetriever()
    tools = {t.name: t for t in build_tools(r, _StubStore(), cfg=AgentConfig(), scope="j1")}
    tools["semantic_search"].invoke({"query": "x"})
    assert r.last_filters == Filters(job_ids=("j1",))

    # scope=None is NOT "every job's uploads", it's the base
    # corpus (job_id unset), so unscoped_excludes_other_jobs is set.
    r2 = _StubRetriever()
    tools2 = {t.name: t for t in build_tools(r2, _StubStore(), cfg=AgentConfig(), scope=None)}
    tools2["keyword_search"].invoke({"query": "x"})
    assert r2.last_filters == Filters(unscoped_excludes_other_jobs=True)


def test_scope_and_lang_merge_into_one_filter():
    r = _StubRetriever()
    tools = {
        t.name: t for t in build_tools(r, _StubStore(), cfg=AgentConfig(), lang="tr", scope="j1")
    }
    tools["search_documents"].invoke({"query": "x"})
    assert r.last_filters == Filters(languages=("tr",), job_ids=("j1",))


def _two_job_app():
    """A real InMemoryStore + Retriever holding doc `d1` under job `j1` and
    doc `d2` under job `j2`. Returns (cfg, app, retriever). `_chunks` / `_profile`
    are borrowed from `test_run.py`: hoisting them to conftest collides with
    conftest's own `_chunk` / `_profile`, so the cross-import stays."""
    import dataclasses

    from porsuk.adapters.embedding.fake import FakeEmbedder
    from porsuk.core.config import load_config
    from porsuk.core.container import build_app, build_retriever
    from tests.agent.test_run import _chunks, _profile

    cfg = load_config("config/local.yaml")
    app = build_app(cfg)
    emb = FakeEmbedder(dim=cfg.embedder.dim or 8)

    j1_chunks = [dataclasses.replace(c, job_id="j1") for c in _chunks()]
    j2_chunks = [
        dataclasses.replace(c, chunk_id=c.chunk_id + "_j2", document_id="d2", job_id="j2")
        for c in _chunks()
    ]
    app.store.upsert_profiles(
        [
            dataclasses.replace(_profile("d1", "One.pdf"), job_id="j1"),
            dataclasses.replace(_profile("d2", "Two.pdf"), job_id="j2"),
        ],
        emb.embed_documents(["One.pdf özeti", "Two.pdf özeti"]),
    )
    app.store.upsert_chunks(
        j1_chunks + j2_chunks,
        emb.embed_documents([c.text for c in j1_chunks + j2_chunks]),
    )
    return cfg, app, build_retriever(cfg, app=app)


def test_build_tools_scope_restricts_searches_to_the_job():
    """build_tools(scope='j1') filters every search tool to job j1's rows.
    scope=None does NOT mean every job's rows: both d1 and d2 here
    carry a job_id (j1/j2), so an unscoped chat sees neither; only documents
    indexed outside any chat (job_id=None) would show up unscoped."""
    cfg, app, retriever = _two_job_app()

    scoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope="j1")}
    rows = scoped["semantic_search"].invoke({"query": "ödeme"})
    assert rows
    assert {r["doc_id"] for r in rows} == {"d1"}
    docs = scoped["search_documents"].invoke({"query": "ödeme"})
    assert {r["doc_id"] for r in docs} == {"d1"}

    unscoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope=None)}
    all_docs = unscoped["search_documents"].invoke({"query": "ödeme"})
    assert all_docs == []  # neither d1 nor d2: both belong to a chat's job


def test_build_tools_scope_hides_other_jobs_from_list_documents():
    """Same rule for list_documents: unscoped sees neither job's docs."""
    cfg, app, retriever = _two_job_app()

    scoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope="j1")}
    listed = scoped["list_documents"].invoke({})
    assert {r["doc_id"] for r in listed} == {"d1"}

    unscoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope=None)}
    assert unscoped["list_documents"].invoke({}) == []


def test_build_tools_unscoped_sees_the_base_corpus_not_other_jobs():
    """The base-corpus case this scoping is actually for: a document indexed
    via `porsuk index` (job_id=None) is visible unscoped, alongside, and
    distinct from, chat-uploaded documents which are not."""
    import dataclasses

    from porsuk.adapters.embedding.fake import FakeEmbedder
    from porsuk.core.config import load_config
    from porsuk.core.container import build_app, build_retriever
    from tests.agent.test_run import _chunks, _profile

    cfg = load_config("config/local.yaml")
    app = build_app(cfg)
    emb = FakeEmbedder(dim=cfg.embedder.dim or 8)

    base_chunks = [dataclasses.replace(c, job_id=None) for c in _chunks()]
    chat_chunks = [
        dataclasses.replace(c, chunk_id=c.chunk_id + "_j1", document_id="d2", job_id="j1")
        for c in _chunks()
    ]
    app.store.upsert_profiles(
        [
            dataclasses.replace(_profile("d1", "Base.pdf"), job_id=None),
            dataclasses.replace(_profile("d2", "ChatUpload.pdf"), job_id="j1"),
        ],
        emb.embed_documents(["Base.pdf özeti", "ChatUpload.pdf özeti"]),
    )
    app.store.upsert_chunks(
        base_chunks + chat_chunks,
        emb.embed_documents([c.text for c in base_chunks + chat_chunks]),
    )
    retriever = build_retriever(cfg, app=app)

    unscoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope=None)}
    docs = unscoped["search_documents"].invoke({"query": "ödeme"})
    assert {r["doc_id"] for r in docs} == {"d1"}  # base corpus only, not d2


def test_build_tools_scope_blocks_get_document_for_another_job():
    cfg, app, retriever = _two_job_app()
    scoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope="j1")}

    # d2 belongs to j2: a j1-scoped chat must not read it
    blocked = scoped["get_document"].invoke({"doc_id": "d2"})
    assert blocked == [{"error": "doc_id not found"}]

    # d1 is in scope, still readable
    ok = scoped["get_document"].invoke({"doc_id": "d1"})
    assert "text" in ok[0]
    assert ok[0]["text"]


def test_build_tools_scope_blocks_get_document_outline_for_another_job():
    cfg, app, retriever = _two_job_app()
    scoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope="j1")}

    assert scoped["get_document_outline"].invoke({"doc_id": "d2"}) == []
    assert scoped["get_document_outline"].invoke({"doc_id": "d1"})  # non-empty for the in-scope doc


def test_build_tools_scope_keeps_search_documents_within_the_job():
    cfg, app, retriever = _two_job_app()
    scoped = {t.name: t for t in build_tools(retriever, app.store, cfg=cfg.agent, scope="j1")}
    docs = scoped["search_documents"].invoke({"query": "ödeme"})
    assert docs
    assert all(r["doc_id"] == "d1" for r in docs)
    assert "d2" not in {r["doc_id"] for r in docs}


def test_semantic_search_passes_doc_ids_through():
    r = _StubRetriever()
    tools = _by_name(retriever=r)
    tools["semantic_search"].invoke({"query": "x", "doc_ids": ["d1", "d2"]})
    assert r.last_doc_ids == ["d1", "d2"]


def test_get_document_section_prefix_match():
    tools = _by_name()
    out = tools["get_document"].invoke({"doc_id": "d1", "section": "3. Mali"})
    assert out[0]["truncated"] is False
    # keeps the three "3. Mali > ..." chunks, drops "4. Sonuç"
    assert "Son bölüm" not in out[0]["text"]
    assert "Birinci bölüm" in out[0]["text"]
    assert "Üçüncü bölüm" in out[0]["text"]


def test_get_document_truncates_over_max_context_chars():
    big = "x" * 5000
    chunks = [
        _chunk("c1", "d1", big, span=(0, 5000), section_path="A"),
        _chunk("c2", "d1", big, span=(5000, 10000), section_path="A"),
        _chunk("c3", "d1", big, span=(10000, 15000), section_path="A"),
    ]
    cfg = AgentConfig()
    tools = _by_name(store=_StubStore(chunks=chunks), cfg=cfg)
    out = tools["get_document"].invoke({"doc_id": "d1"})
    assert out[0]["truncated"] is True
    assert out[0]["text"].endswith("[doküman devam ediyor — get_document_outline ile bölüm seç]")
    assert len(out[0]["text"]) == cfg.max_context_chars + len(
        "\n[doküman devam ediyor — get_document_outline ile bölüm seç]"
    )


def test_get_document_unknown_doc_id_errors():
    tools = _by_name(store=_StubStore(chunks=[]))
    out = tools["get_document"].invoke({"doc_id": "nope"})
    assert out == [{"error": "doc_id not found"}]


def test_get_document_section_prefix_matching_nothing_errors():
    tools = _by_name()
    out = tools["get_document"].invoke({"doc_id": "d1", "section": "9. Yok"})
    assert out[0]["error"].startswith("no chunks match section prefix '9. Yok'")
    assert "get_document_outline" in out[0]["error"]


def test_get_document_populates_seen():
    tools = _by_name()
    tools["get_document"].invoke({"doc_id": "d1"})
    out = tools["expand_context"].invoke({"chunk_id": "c2"})
    assert "text" in out[0]


def test_get_document_outline_distinct_ordered_sections():
    tools = _by_name()
    out = tools["get_document_outline"].invoke({"doc_id": "d1"})
    assert [r["section_path"] for r in out] == [
        "3. Mali > 3.1 Genel",
        "3. Mali > 3.2 Ödeme",
        "4. Sonuç",
    ]


def test_get_document_outline_skips_none_sections():
    chunks = [
        Chunk("c1", "d1", "a", 1, "tr", None, None, (0, 1)),
        _chunk("c2", "d1", "b", span=(1, 2), section_path="Bölüm 1"),
    ]
    tools = _by_name(store=_StubStore(chunks=chunks))
    out = tools["get_document_outline"].invoke({"doc_id": "d1"})
    assert [r["section_path"] for r in out] == ["Bölüm 1"]


def test_list_documents_shape_and_filters():
    tools = _by_name()
    out = tools["list_documents"].invoke({})
    assert {r["file"] for r in out} == {"Sozlesme.pdf", "Rapor.docx"}
    assert out[0]["pages"] == 3
    assert isinstance(out[0]["topics"], list)

    only_tr = tools["list_documents"].invoke({"lang": "tr"})
    assert [r["doc_id"] for r in only_tr] == ["d1"]

    only_docx = tools["list_documents"].invoke({"doc_type": "docx"})
    assert [r["doc_id"] for r in only_docx] == ["d2"]


def test_expand_context_errors_before_any_search():
    tools = _by_name()
    out = tools["expand_context"].invoke({"chunk_id": "c2"})
    assert out == [{"error": "chunk_id not seen yet — run a search first"}]


def test_expand_context_widens_after_a_search():
    chunks = [
        _chunk("a", "d1", "AAAA ", span=(0, 5), section_path="S"),
        _chunk("b", "d1", "BBBB ", span=(5, 10), section_path="S"),
        _chunk("c", "d1", "TARGET", span=(10, 16), section_path="S"),
        _chunk("d", "d1", " DDDD", span=(16, 21), section_path="S"),
        _chunk("e", "d1", " EEEE", span=(21, 26), section_path="S"),
    ]

    class _R(_StubRetriever):
        def semantic_search(self, query, *, k, filters, doc_ids=None):
            return [_hit(chunks[2])]

    tools = _by_name(retriever=_R(), store=_StubStore(chunks=chunks))
    tools["semantic_search"].invoke({"query": "x"})
    out = tools["expand_context"].invoke({"chunk_id": "c", "before_chars": 6, "after_chars": 6})
    assert out[0]["before_chars"] == 6
    assert out[0]["after_chars"] == 6
    # "BBBB " is 5 chars (< 6) so the left walk takes one more chunk ("AAAA ");
    # "TARGET" then sits between the accumulated left and right.
    assert out[0]["text"] == "AAAA BBBB TARGET DDDD EEEE"
    # a wider target chunk means the walk stops sooner
    out2 = tools["expand_context"].invoke({"chunk_id": "c", "before_chars": 5, "after_chars": 5})
    assert out2[0]["text"] == "BBBB TARGET DDDD"


def test_expand_context_return_carries_doc_id_page_section():
    """F-C: the widened dict must include the target chunk's doc_id / page /
    section, so a `get_document`-only chunk still yields Source(file=<real>)."""
    chunks = [
        _chunk("a", "d1", "AAAA ", span=(0, 5), section_path="S", page_no=7),
        _chunk("b", "d1", "TARGET", span=(5, 11), section_path="S", page_no=7),
        _chunk("c", "d1", " CCCC", span=(11, 16), section_path="S", page_no=7),
    ]

    class _R(_StubRetriever):
        def semantic_search(self, query, *, k, filters, doc_ids=None):
            return [_hit(chunks[1])]

    tools = _by_name(retriever=_R(), store=_StubStore(chunks=chunks))
    tools["semantic_search"].invoke({"query": "x"})
    out = tools["expand_context"].invoke({"chunk_id": "b"})
    assert out[0]["doc_id"] == "d1"
    assert out[0]["page"] == 7
    assert out[0]["section"] == "S"


def test_expand_context_errors_when_seen_chunk_absent_from_store():
    seen_chunk = _chunk("ghost", "d1", "GHOST", span=(0, 5), section_path="S")

    class _R(_StubRetriever):
        def semantic_search(self, query, *, k, filters, doc_ids=None):
            return [_hit(seen_chunk)]

    # the store for d1 has entirely different chunk_ids: the index moved
    store = _StubStore(chunks=[_chunk("other", "d1", "X", span=(0, 1), section_path="S")])
    tools = _by_name(retriever=_R(), store=store)
    tools["semantic_search"].invoke({"query": "x"})
    out = tools["expand_context"].invoke({"chunk_id": "ghost"})
    assert out == [{"error": "chunk not in its document — index may have changed"}]


def test_expand_context_defaults_to_cfg_window():
    chunks = [_chunk("only", "d1", "SOLO", span=(0, 4), section_path="S")]

    class _R(_StubRetriever):
        def semantic_search(self, query, *, k, filters, doc_ids=None):
            return [_hit(chunks[0])]

    cfg = AgentConfig()
    tools = _by_name(retriever=_R(), store=_StubStore(chunks=chunks), cfg=cfg)
    tools["semantic_search"].invoke({"query": "x"})
    out = tools["expand_context"].invoke({"chunk_id": "only"})
    assert out[0]["before_chars"] == cfg.expand_before_chars
    assert out[0]["after_chars"] == cfg.expand_after_chars
    assert out[0]["text"] == "SOLO"
