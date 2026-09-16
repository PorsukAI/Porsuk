"""`run_agent` end to end over a real in-memory app.

Network-free and GPU-free: real `InMemoryStore` + `FakeEmbedder`, a couple
of chunks indexed directly, and `build_chat_model` monkeypatched to a
scripted `FakeChatModel`.
"""

from __future__ import annotations

import pytest

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.llm.langgraph_chat import FakeChatModel
from porsuk.core.config import load_config
from porsuk.core.container import build_app
from porsuk.core.models import Chunk, DocumentProfile

# A realistically-shaped chunk_id: <uuid5>:<5 digits>, the format
# porsuk/ingestion/chunking/splitter.py actually produces.
REAL_CHUNK_ID = "6310c32d-dca8-5ee8-8ea7-ea62a9ff29ac:00007"


def _profile(doc_id: str, filename: str) -> DocumentProfile:
    return DocumentProfile(
        document_id=doc_id,
        path=f"/corpus/{filename}",
        filename=filename,
        doc_type="pdf",
        language="tr",
        created_at=None,
        modified_at=None,
        size=100,
        page_count=3,
        summary=f"{filename} özeti",
        topics=("ödeme", "tedarik"),
        entities=(),
        profile_level="cheap",
        profile_source=frozenset(),
        parse_quality=0.9,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            "c1",
            "d1",
            "Ödemeler fatura tarihinden itibaren 30 gün içinde yapılır.",
            12,
            "tr",
            "3.2 Ödeme",
            "3. Mali > 3.2 Ödeme",
            (0, 56),
        ),
        Chunk(
            "c2",
            "d1",
            "Gecikme halinde aylık yüzde bir faiz uygulanır.",
            12,
            "tr",
            "3.2 Ödeme",
            "3. Mali > 3.2 Ödeme",
            (56, 102),
        ),
        Chunk(
            "c3",
            "d1",
            "Ödeme banka havalesi ile tek seferde yapılır.",
            12,
            "tr",
            "3.2 Ödeme",
            "3. Mali > 3.2 Ödeme",
            (102, 147),
        ),
        # A realistically-shaped chunk_id: <uuid5>:<5 digits>, exactly what
        # porsuk/ingestion/chunking/splitter.py builds. `c1`/`c2`/`c3` above
        # are synthetic and never exercise the colon in the citation regex.
        Chunk(
            REAL_CHUNK_ID,
            "d1",
            "Cezai şart, sözleşme bedelinin yüzde onu ile sınırlıdır.",
            13,
            "tr",
            "4.1 Cezalar",
            "4. Yaptırımlar > 4.1 Cezalar",
            (147, 200),
        ),
    ]


@pytest.fixture()
def indexed_app(tmp_path):
    cfg = load_config("config/local.yaml")
    app = build_app(cfg)
    emb = FakeEmbedder(dim=cfg.embedder.dim or 8)
    chunks = _chunks()
    app.store.upsert_profiles(
        [_profile("d1", "Sozlesme.pdf")],
        emb.embed_documents(["Sozlesme.pdf özeti"]),
    )
    app.store.upsert_chunks(chunks, emb.embed_documents([c.text for c in chunks]))
    return cfg, app


def _patch_chat(monkeypatch, script):
    fake = FakeChatModel(script=script)
    monkeypatch.setattr("porsuk.core.container.build_chat_model", lambda cfg: fake)
    return fake


def _patch_chat_recording(monkeypatch, script):
    """Like `_patch_chat`, but also records the message list `_generate`
    (and therefore `graph.stream`'s model node) was called with each turn,
    for asserting `history` actually reaches the model, not just that
    `run_agent` doesn't crash when it's passed.
    """
    fake = FakeChatModel(script=script)
    calls: list[list] = []
    original = type(fake.chat)._generate

    def recording_generate(self, messages, *a, **kw):
        calls.append(list(messages))
        return original(self, messages, *a, **kw)

    monkeypatch.setattr(type(fake.chat), "_generate", recording_generate)
    monkeypatch.setattr("porsuk.core.container.build_chat_model", lambda cfg: fake)
    return fake, calls


def test_run_agent_extracts_sources_from_the_kullanilan_line(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödemeler fatura tarihinden itibaren 30 gün içinde yapılır.\nKULLANILAN: c1",
        ],
    )
    ans = run_agent("ödeme koşulu ne?", cfg=cfg, app=app)

    assert "KULLANILAN" not in ans.text
    assert ans.text.strip().endswith("içinde yapılır.")
    assert [s.file for s in ans.sources] == ["Sozlesme.pdf"]
    assert ans.sources[0].page == 12
    assert ans.sources[0].section == "3. Mali > 3.2 Ödeme"
    assert ans.sources[0].quote.startswith("Ödemeler fatura")
    assert ans.sources[0].chunk_id == "c1"
    assert ans.sources[0].doc_id == "d1"
    assert ans.tool_calls == 1
    assert ans.truncated is False
    assert [s.file for s in ans.documents_seen] == ["Sozlesme.pdf"]
    assert ans.documents_seen[0].doc_id == "d1"


def test_run_agent_falls_back_when_no_kullanilan_line(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert ans.sources  # non-empty, from the touched pool
    assert len(ans.sources) <= 3
    assert {s.file for s in ans.sources} == {"Sozlesme.pdf"}


def test_run_agent_falls_back_when_kullanilan_ids_are_all_unresolvable(monkeypatch, indexed_app):
    """F-A: `KULLANILAN:` names ids absent from the pool (a 4B failure mode):
    resolution is empty, so the top-3 fallback must still run."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.\nKULLANILAN: nope1, nope2",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert "KULLANILAN" not in ans.text
    assert ans.sources  # top-3 fallback, not empty
    assert {s.file for s in ans.sources} == {"Sozlesme.pdf"}


def test_run_agent_documents_seen_includes_search_documents_rows(monkeypatch, indexed_app):
    """F-B: the document-native flow (`search_documents` then a final answer)
    has no chunk pool, but `BAKILAN DOKÜMANLAR` must still list the doc."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "search_documents", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert ans.documents_seen
    assert ans.documents_seen[0].file == "Sozlesme.pdf"


def test_run_agent_strips_inline_kullanilan(monkeypatch, indexed_app):
    """F4: `KULLANILAN:` written mid-line must not leak verbatim to the user."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır. KULLANILAN: c1",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert "KULLANILAN" not in ans.text
    assert ans.text.strip() == "Ödeme 30 gün içinde yapılır."
    assert [s.file for s in ans.sources] == ["Sozlesme.pdf"]


def test_run_agent_falls_back_when_kullanilan_line_is_empty(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.\nKULLANILAN: ",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert "KULLANILAN" not in ans.text  # empty line still stripped
    assert ans.sources  # top-3 fallback, not empty
    assert {s.file for s in ans.sources} == {"Sozlesme.pdf"}


def test_run_agent_collects_inline_chunk_id_marks(monkeypatch, indexed_app):
    """The agent's ⟦chunk_id⟧ marks in the answer text are collected into
    answer.sources, in first-appearance order, and left in answer.text.
    The pool holds c1, c2, c3; the answer marks a subset in a
    non-pool order, so this cannot be the first-3-touched fallback."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Havale ile ödenir ⟦c3⟧. 30 gün içinde yapılır ⟦c1⟧.",
        ],
    )
    ans = run_agent("ödeme ve faiz?", cfg=cfg, app=app)

    assert [s.chunk_id for s in ans.sources] == ["c3", "c1"]  # not fallback [c1, c2, c3]
    assert "⟦c3⟧" in ans.text and "⟦c1⟧" in ans.text  # marks kept
    assert "KULLANILAN" not in ans.text


def test_run_agent_resolves_a_realistic_chunk_id_mark(monkeypatch, indexed_app):
    """The ⟦…⟧ mark must match a real chunk_id: a uuid5, a colon, five digits
    (splitter.py `f"{document_id}:{index:05d}"`). This FAILS with the old
    colon-less _MARK regex: the mark matches nothing and answer.sources
    silently reverts to the first-3-touched fallback (c1, c2, c3)."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ceza"}},
            f"Cezai şart bedelin yüzde onudur ⟦{REAL_CHUNK_ID}⟧.",
        ],
    )
    ans = run_agent("cezai şart nedir?", cfg=cfg, app=app)

    assert [s.chunk_id for s in ans.sources] == [REAL_CHUNK_ID]  # resolved, not fallback
    assert f"⟦{REAL_CHUNK_ID}⟧" in ans.text  # mark kept in text
    assert "c1" not in [s.chunk_id for s in ans.sources]  # not the c1/c2/c3 fallback


def test_run_agent_dedupes_repeated_marks_in_order(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "x"}},
            "A ⟦c2⟧. B ⟦c1⟧. C ⟦c2⟧.",  # c2, c1: c2 not repeated
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert [s.chunk_id for s in ans.sources] == ["c2", "c1"]


def test_run_agent_multi_source_sentence_marks(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "x"}},
            "Hem ödeme hem faiz aynı maddededir ⟦c1⟧⟦c2⟧.",
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    # Adjacent marks resolve in order; c3 is in the pool but unmarked, so this
    # is not the first-3-touched fallback ([c1, c2, c3]) either.
    assert [s.chunk_id for s in ans.sources] == ["c1", "c2"]


def test_run_agent_falls_back_when_no_marks(monkeypatch, indexed_app):
    """No ⟦…⟧ marks and no KULLANILAN: line → first-3-touched fallback,
    text untouched (mark fallback matches the KULLANILAN fallback)."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "x"}},
            "Ödeme 30 gün.",  # no marks at all
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert len(ans.sources) >= 1  # fallback kicked in
    assert ans.text == "Ödeme 30 gün."


def test_run_agent_abstains_with_no_fallback_sources(monkeypatch, indexed_app):
    """'BULAMADIM' means the model judged nothing it found to be
    relevant. Citing the top-3-touched pool anyway would show the user
    sources the model itself rejected, worse than showing none."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "alakasız"}},
            "Bu konuda belgelerde bilgi yok. BULAMADIM.",
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert ans.sources == []
    assert "BULAMADIM" in ans.text


def test_run_agent_abstain_keeps_real_citations_if_present(monkeypatch, indexed_app):
    """An abstain-worded answer that DOES carry a resolvable mark still cites
    it: only the empty-citation fallback is suppressed, not real citations."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme koşulu belirtilmiş ⟦c1⟧ ama vade süresi net değil. BULAMADIM.",
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert [s.chunk_id for s in ans.sources] == ["c1"]


def test_run_agent_still_tolerates_a_trailing_kullanilan_line(monkeypatch, indexed_app):
    """A model that emits the old KULLANILAN: line still works: the line is
    stripped, its ids resolve (backward tolerance)."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "x"}},
            "Ödeme 30 gün.\nKULLANILAN: c1",
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert [s.chunk_id for s in ans.sources] == ["c1"]
    assert "KULLANILAN" not in ans.text
    assert ans.text.strip() == "Ödeme 30 gün."


def test_run_agent_marks_win_over_a_trailing_kullanilan_line(monkeypatch, indexed_app):
    """A model that emits both an inline mark and the old KULLANILAN: line:
    the marks are the source of truth for ids, but the line is still stripped
    from the text."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "x"}},
            "X ⟦c2⟧.\nKULLANILAN: c1",
        ],
    )
    ans = run_agent("soru?", cfg=cfg, app=app)
    assert [s.chunk_id for s in ans.sources] == ["c2"]  # mark wins, not c1
    assert "KULLANILAN" not in ans.text
    assert ans.text.strip() == "X ⟦c2⟧."


def test_run_agent_calls_on_event_for_each_tool_step(monkeypatch, indexed_app):
    cfg, app = indexed_app
    from porsuk.adapters.llm.langgraph_chat import FakeChatModel

    monkeypatch.setattr(
        "porsuk.core.container.build_chat_model",
        lambda _cfg: type(
            "M",
            (),
            {
                "chat": FakeChatModel(
                    script=[
                        {"tool": "semantic_search", "args": {"query": "ödeme"}},
                        "Ödeme 30 gün.\nKULLANILAN: c1",
                    ]
                ).chat
            },
        )(),
    )
    events: list[dict] = []
    from porsuk.agent.run import run_agent

    ans = run_agent("ödeme koşulu?", cfg=cfg, on_event=events.append, app=app)
    kinds = [e["type"] for e in events]
    assert "tool_call" in kinds and "tool_result" in kinds
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["name"] == "semantic_search"
    assert call["args"] == {"query": "ödeme"}
    assert ans.text.startswith("Ödeme 30 gün")


def test_tool_summary_is_human_readable_not_raw_json(monkeypatch, indexed_app):
    """The tool_result event's `summary` must read as an answer
    ('3 results: Sozlesme.pdf Madde 3'), not a raw-JSON prefix, reported from
    the live agent-trace panel as unreadable noise. The trace panel is kept
    in English by choice (unlike the rest of the UI, which is Turkish)."""
    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün.\nKULLANILAN: c1",
        ],
    )
    events: list[dict] = []
    from porsuk.agent.run import run_agent

    run_agent("ödeme koşulu?", cfg=cfg, on_event=events.append, app=app)
    result = next(e for e in events if e["type"] == "tool_result")
    # chunk-level tool rows carry doc_id, not a filename (file is
    # resolved once at the end via store.list_documents); the trace summary
    # is still a readable sentence, just not the raw JSON prefix.
    assert not result["summary"].startswith("[{")
    assert "result" in result["summary"]
    assert "3.2 Ödeme" in result["summary"]


def test_tool_summary_reports_zero_results():
    from porsuk.agent.run import _tool_summary

    assert _tool_summary("[]", {}) == "0 results"


def test_tool_summary_surfaces_a_tool_error():
    from porsuk.agent.run import _tool_summary

    assert _tool_summary('[{"error": "doc_id not found"}]', {}) == "doc_id not found"


def test_tool_summary_falls_back_to_raw_text_for_non_json():
    from porsuk.agent.run import _tool_summary

    assert _tool_summary("plain text result", {}) == "plain text result"


def test_tool_summary_resolves_doc_id_to_filename_for_chunk_rows():
    """keyword_search/semantic_search rows carry doc_id, not a filename:
    the summary must still show the readable name, like search_documents
    rows already do (which carry `file` directly)."""
    from porsuk.agent.run import _tool_summary

    rows = '[{"doc_id": "d1", "chunk_id": "d1:00001", "section": "Madde 3"}]'
    assert _tool_summary(rows, {"d1": "mevzuat_2547.pdf"}) == "1 result: mevzuat_2547.pdf Madde 3"


def test_tool_summary_lists_every_row_without_truncation():
    """The trace panel should show every result, not '... +N diğer':
    the user wants to see the full list, not a guess at what's hidden."""
    from porsuk.agent.run import _tool_summary

    rows = "[" + ", ".join(f'{{"doc_id": "d{i}", "section": "Madde {i}"}}' for i in range(1, 6)) + "]"
    summary = _tool_summary(rows, {})
    assert "diğer" not in summary
    for i in range(1, 6):
        assert f"d{i} Madde {i}" in summary


def test_tool_summary_includes_a_snippet_of_the_chunk_text():
    """semantic_search/keyword_search rows carry the chunk's own `text`: the
    trace panel showed only "file, section", the user wants an actual
    preview of what the tool found, not just where."""
    from porsuk.agent.run import _tool_summary

    rows = (
        '[{"doc_id": "d1", "chunk_id": "d1:00001", "section": "Madde 3", '
        '"text": "Doktor Öğretim Üyesi, doktora çalışmalarını başarıyla tamamlamış kişidir."}]'
    )
    summary = _tool_summary(rows, {"d1": "mevzuat_2547.pdf"})
    assert "mevzuat_2547.pdf Madde 3" in summary
    assert "Doktor Öğretim Üyesi" in summary


def test_tool_summary_snippet_is_truncated_for_a_long_chunk():
    from porsuk.agent.run import _tool_summary

    long_text = "kelime " * 100
    rows = f'[{{"doc_id": "d1", "text": {long_text.strip()!r}}}]'.replace("'", '"')
    summary = _tool_summary(rows, {})
    assert len(summary) < len(long_text)


def test_tool_summary_row_without_text_is_unaffected():
    """search_documents / list_documents rows carry no `text` field: the
    summary must stay exactly as before for those, no dangling separator."""
    from porsuk.agent.run import _tool_summary

    rows = '[{"file": "gaap.pdf", "doc_id": "d1"}]'
    assert _tool_summary(rows, {}) == "1 result: gaap.pdf"


def test_run_agent_emits_sources_update_after_each_tool_result(monkeypatch, indexed_app):
    """A ⟦chunk_id⟧ mark streamed inside the final answer must have
    something to resolve against before the run finishes, or the UI drops
    the badge until the terminal `answer` event. So
    `sources_update` carries the growing chunk pool right after every
    `tool_result`, not just once at the end."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün.\nKULLANILAN: c1",
        ],
    )
    events: list[dict] = []
    run_agent("ödeme koşulu?", cfg=cfg, on_event=events.append, app=app)

    kinds = [e["type"] for e in events]
    assert "sources_update" in kinds
    # arrives right after the tool_result it was built from, not at the end
    assert kinds.index("sources_update") == kinds.index("tool_result") + 1
    update = next(e for e in events if e["type"] == "sources_update")
    assert any(s["chunk_id"] == "c1" for s in update["sources"])


def test_run_agent_emits_chunk_event_for_the_final_answer(monkeypatch, indexed_app):
    """The final model turn (no tool calls) is streamed to on_event as a
    `chunk` event carrying the answer text."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.\nKULLANILAN: c1",
        ],
    )
    events: list[dict] = []
    ans = run_agent("ödeme koşulu?", cfg=cfg, on_event=events.append, app=app)

    chunks = [e for e in events if e["type"] == "chunk"]
    assert chunks, "expected at least one chunk event"
    streamed = "".join(e["text"] for e in chunks)
    assert "Ödeme 30 gün içinde yapılır." in streamed
    assert events[-1]["type"] == "chunk"
    assert ans.text.startswith("Ödeme 30 gün içinde yapılır.")
    assert "KULLANILAN" not in ans.text


def test_run_agent_streams_prose_from_a_turn_that_also_calls_a_tool(monkeypatch, indexed_app):
    """A model turn that produces prose AND a tool call streams that prose as
    `chunk` events as it arrives, not held back until the turn resolves;
    only the tool call's own arguments are withheld until it completes."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app

    # A turn with both content and a tool call, then a final answer.
    def _patch_with_reasoning():
        from langchain_core.messages import AIMessage

        msgs = [
            AIMessage(
                content="Önce sözleşmeyi aramam gerek.",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "args": {"query": "ödeme"},
                        "id": "call_0",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Ödeme 30 gün.\nKULLANILAN: c1"),
        ]
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        class _Fake(GenericFakeChatModel):
            def bind_tools(self, tools, **kwargs):  # noqa: ARG002
                return self

        monkeypatch.setattr(
            "porsuk.core.container.build_chat_model",
            lambda _cfg: type("M", (), {"chat": _Fake(messages=iter(msgs))})(),
        )

    _patch_with_reasoning()
    events: list[dict] = []
    ans = run_agent("ödeme?", cfg=cfg, on_event=events.append, app=app)

    kinds = [e["type"] for e in events]
    assert "thought" not in kinds
    chunk_texts = [e["text"] for e in events if e["type"] == "chunk"]
    assert any("sözleşme" in t.lower() for t in chunk_texts)
    # the reasoning-turn's prose precedes the tool_call it reasoned about
    first_prose_idx = next(i for i, e in enumerate(events) if e["type"] == "chunk")
    assert first_prose_idx < kinds.index("tool_call")
    assert ans.text.startswith("Ödeme 30 gün")


def test_run_agent_marks_truncation_at_the_step_limit(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    cfg.agent.max_tool_calls = 1  # recursion_limit -> 3, trips after one tool call
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            {"tool": "semantic_search", "args": {"query": "faiz"}},
            "Eldeki bilgiyle: ödeme 30 gün.\nKULLANILAN: c1",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)

    assert ans.truncated is True
    assert ans.cancelled is False
    assert ans.text.startswith("[adım limiti aşıldı] ")
    assert "[adım limiti aşıldı]" in ans.text
    assert "KULLANILAN" not in ans.text


def test_run_agent_stops_when_the_event_is_set(monkeypatch, indexed_app):
    """A client disconnect sets the `stop` Event; `run_agent` must break out of
    the graph loop instead of running to `max_tool_calls`."""
    import threading

    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    cfg.agent.max_tool_calls = 20  # loop would run 20 times if never stopped

    def _forever():
        i = 0
        while True:
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "semantic_search",
                        "args": {"query": f"q{i}"},
                        "id": f"call_{i}",
                        "type": "tool_call",
                    }
                ],
            )
            i += 1

    class _Looping(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    looping = type("M", (), {"chat": _Looping(messages=_forever())})()
    monkeypatch.setattr("porsuk.core.container.build_chat_model", lambda _cfg: looping)

    stop = threading.Event()

    def on_event(ev):
        if ev["type"] == "tool_result":
            stop.set()  # first tool round-trip done, cancel

    ans = run_agent("ödeme?", cfg=cfg, on_event=on_event, stop=stop, app=app)

    assert ans.truncated is True
    assert ans.cancelled is True
    assert not ans.text.startswith("[adım limiti aşıldı]")  # cancelled != step-limit
    assert ans.tool_calls == 1  # broke right after the first tool result, not 20


def test_run_agent_verbose_prints_tool_steps(monkeypatch, indexed_app, capsys):
    """`--verbose` with no `on_event` falls back to `_print_event`, which writes
    each tool call and result to stdout (`sink = on_event or (_print_event ...)`)."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün içinde yapılır.\nKULLANILAN: c1",
        ],
    )
    run_agent("ödeme koşulu?", cfg=cfg, verbose=True, app=app)

    out = capsys.readouterr().out
    assert "·" in out  # tool_call line
    assert "semantic_search" in out
    assert "→" in out  # tool_result line


def test_truncation_recovery_input_carries_the_question_and_system_prompt(monkeypatch, indexed_app):
    """The recovery `chat.invoke` after GraphRecursionError must see the
    original question and the system prompt (the ⟦chunk_id⟧ citation
    instruction); the streamed transcript alone carries neither."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from porsuk.agent.run import run_agent

    captured: list = []

    class _RecordingChat(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

        def invoke(self, input, config=None, **kwargs):  # noqa: A002
            if isinstance(input, list):
                captured.append(input)
            return super().invoke(input, config, **kwargs)

    class _Wrapper:
        model = "recording"

        def __init__(self):
            self.chat = _RecordingChat(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "semantic_search",
                                    "args": {"query": "ödeme"},
                                    "id": "call_0",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "semantic_search",
                                    "args": {"query": "faiz"},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(content="Recovered.\nKULLANILAN: c1"),
                    ]
                )
            )

    cfg, app = indexed_app
    cfg.agent.max_tool_calls = 1
    monkeypatch.setattr("porsuk.core.container.build_chat_model", lambda cfg: _Wrapper())

    ans = run_agent("ödeme koşulu ne?", cfg=cfg, app=app)

    assert ans.truncated is True
    assert captured, "recovery chat.invoke was never called with a message list"
    recovery = captured[-1]
    assert any(isinstance(m, SystemMessage) and "⟦chunk_id⟧" in m.content for m in recovery)
    assert any(isinstance(m, HumanMessage) and "ödeme koşulu ne?" in m.content for m in recovery)


def test_run_agent_threads_scope_to_the_tools(monkeypatch, indexed_app):
    """run_agent(scope='j1') hands scope='j1' straight to build_tools, which is
    where retrieval gets restricted to the job."""
    from porsuk.agent import run as run_mod
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(monkeypatch, ["Cevap.\nKULLANILAN: c1"])

    captured = {}
    real_build_tools = run_mod.build_tools

    def spy_build_tools(retriever, store, *, cfg, lang=None, scope=None):
        captured["scope"] = scope
        return real_build_tools(retriever, store, cfg=cfg, lang=lang, scope=scope)

    monkeypatch.setattr(run_mod, "build_tools", spy_build_tools)
    run_agent("soru?", scope="j1", cfg=cfg, app=app)
    assert captured["scope"] == "j1"


def test_run_agent_without_sink_is_unchanged_by_streaming(monkeypatch, indexed_app):
    """With no on_event and no verbose, the multi-mode stream must not change
    the answer or raise: the `messages` tuples are simply ignored."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Ödeme 30 gün.\nKULLANILAN: c1",
        ],
    )
    ans = run_agent("ödeme?", cfg=cfg, app=app)  # no on_event, no verbose
    assert ans.text.startswith("Ödeme 30 gün")
    assert ans.tool_calls == 1
    assert "KULLANILAN" not in ans.text


def test_run_agent_no_sink_uses_single_mode_stream(monkeypatch, indexed_app):
    """With no sink, run_agent must use stream_mode='updates', not the
    multi-mode form, so POST /ask and `porsuk ask --json` keep the exact
    prior model transport (no-sink path byte-identical)."""
    from porsuk.agent import run as run_mod
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(monkeypatch, ["Cevap.\nKULLANILAN: c1"])

    seen = {}
    real_build_graph = run_mod.build_graph

    def spy_build_graph(chat, tools, **kwargs):
        graph = real_build_graph(chat, tools, **kwargs)
        real_stream = graph.stream

        def spy_stream(*a, **kw):
            seen["stream_mode"] = kw.get("stream_mode")
            return real_stream(*a, **kw)

        graph.stream = spy_stream
        return graph

    monkeypatch.setattr(run_mod, "build_graph", spy_build_graph)
    run_agent("soru?", cfg=cfg, app=app)  # no on_event, no verbose
    assert seen["stream_mode"] == "updates"


def test_run_agent_with_sink_uses_multi_mode_stream(monkeypatch, indexed_app):
    """With a sink, run_agent opts into stream_mode=['updates', 'messages'],
    the streaming chat-completions transport that token streaming needs."""
    from porsuk.agent import run as run_mod
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "ödeme"}},
            "Cevap.\nKULLANILAN: c1",
        ],
    )

    seen = {}
    real_build_graph = run_mod.build_graph

    def spy_build_graph(chat, tools, **kwargs):
        graph = real_build_graph(chat, tools, **kwargs)
        real_stream = graph.stream

        def spy_stream(*a, **kw):
            seen["stream_mode"] = kw.get("stream_mode")
            return real_stream(*a, **kw)

        graph.stream = spy_stream
        return graph

    monkeypatch.setattr(run_mod, "build_graph", spy_build_graph)
    events: list[dict] = []
    run_agent("soru?", cfg=cfg, on_event=events.append, app=app)
    assert seen["stream_mode"] == ["updates", "messages"]


def test_run_agent_verbose_prints_thought_and_chunk(monkeypatch, indexed_app, capsys):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    msgs = [
        AIMessage(
            content="Sözleşmeye bakayım.",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "args": {"query": "x"},
                    "id": "c0",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="Cevap burada.\nKULLANILAN: c1"),
    ]

    class _Fake(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ARG002
            return self

    monkeypatch.setattr(
        "porsuk.core.container.build_chat_model",
        lambda _cfg: type("M", (), {"chat": _Fake(messages=iter(msgs))})(),
    )
    run_agent("soru?", cfg=cfg, verbose=True, app=app)
    out = capsys.readouterr().out
    assert "Sözleşmeye bakayım." in out  # thought printed
    assert "Cevap burada." in out  # chunk printed


def test_run_agent_source_quote_is_not_truncated(monkeypatch, indexed_app):
    """The quote is the chunk's full text, not a 200-char cut:
    a mid-sentence cutoff was reported from a real chunk in the source modal
    (mevzuat_2547 Madde 3's tanımlar list, >1200 chars)."""
    from porsuk.agent.run import run_agent
    from porsuk.core.models import Chunk

    cfg, app = indexed_app
    long_text = (
        "(1) Profesör: En yüksek düzeydeki akademik unvana sahip kişidir. "
        "(2) Doçent: Üniversitelerarası Kurul tarafından verilen doçentlik "
        "akademik unvanına sahip kişidir. (3) Doktor Öğretim Üyesi: Doktora "
        "çalışmalarını başarı ile tamamlamış, tıpta, diş hekimliğinde, "
        "eczacılıkta ve veteriner hekimlikte uzmanlık unvanını veya "
        "Üniversitelerarası Kurulun önerisi üzerine Yükseköğretim Kurulunca "
        "tespit edilen belli sanat dallarının birinde yeterlik kazanmış olan "
        "akademik unvana sahip kişidir."
    )
    assert len(long_text) > 200, "fixture must actually exercise the old cutoff"
    from porsuk.adapters.embedding.fake import FakeEmbedder

    long_chunk = Chunk(
        "u:00007", "d1", long_text, 2, "tr", "Madde 3", "BİRİNCİ BÖLÜM > Madde 3", (0, len(long_text))
    )
    emb = FakeEmbedder(dim=cfg.embedder.dim or 8)
    app.store.upsert_chunks([long_chunk], emb.embed_documents([long_text]))

    _patch_chat(
        monkeypatch,
        [
            {"tool": "semantic_search", "args": {"query": "doktor öğretim üyesi"}},
            "Doktor öğretim üyesi tanımı burada. ⟦u:00007⟧",
        ],
    )
    ans = run_agent("doktor öğretim üyesi kimdir?", cfg=cfg, app=app)

    assert ans.sources[0].quote == long_text
    assert ans.sources[0].quote.endswith("akademik unvana sahip kişidir.")


def test_run_agent_history_reaches_the_model(monkeypatch, indexed_app):
    """Prior turns passed as `history` are in the message list the model sees
    on its first call, the multi-turn case: a follow-up question ("peki ya
    X?") only makes sense to the model with the earlier exchange present.
    """
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    fake, calls = _patch_chat_recording(monkeypatch, ["İkinci cevap.\nKULLANILAN: "])
    history = [
        {"role": "user", "content": "İlk soru?"},
        {"role": "assistant", "content": "İlk cevap."},
    ]

    run_agent("İkinci soru?", cfg=cfg, app=app, history=history)

    assert calls, "the model was never called"
    first_call_texts = [getattr(m, "content", "") for m in calls[0]]
    assert any("İlk soru?" in t for t in first_call_texts)
    assert any("İlk cevap." in t for t in first_call_texts)
    # history precedes the new question in the seed
    idx_hist_q = next(i for i, t in enumerate(first_call_texts) if "İlk soru?" in t)
    idx_new_q = next(i for i, t in enumerate(first_call_texts) if "İkinci soru?" in t)
    assert idx_hist_q < idx_new_q


def test_run_agent_without_history_is_unaffected(monkeypatch, indexed_app):
    """`history=None` (the default) must reproduce the original message seed
    exactly, no empty turns, no stray roles."""
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    fake, calls = _patch_chat_recording(monkeypatch, ["Cevap.\nKULLANILAN: "])

    run_agent("Soru?", cfg=cfg, app=app)

    texts = [getattr(m, "content", "") for m in calls[0]]
    # exactly the system prompt + the one question, nothing extra
    assert sum(1 for t in texts if "Soru?" in t) == 1


def test_run_agent_history_ignores_unknown_roles_and_empty_content(monkeypatch, indexed_app):
    from porsuk.agent.run import run_agent

    cfg, app = indexed_app
    fake, calls = _patch_chat_recording(monkeypatch, ["Cevap.\nKULLANILAN: "])
    history = [
        {"role": "system", "content": "gizli talimat"},  # dropped: not user/assistant
        {"role": "user", "content": ""},  # dropped: empty content
        {"role": "user", "content": "Gerçek önceki soru."},
    ]

    run_agent("Yeni soru?", cfg=cfg, app=app, history=history)

    texts = [getattr(m, "content", "") for m in calls[0]]
    assert any("Gerçek önceki soru." in t for t in texts)
    assert not any("gizli talimat" in t for t in texts)
