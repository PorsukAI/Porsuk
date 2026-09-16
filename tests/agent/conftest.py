"""Shared test doubles for the agent tests (network-free, GPU-free).

`_StubRetriever` and `_StubStore` model the two ports `build_tools` consumes.
`test_graph_loop.py` reuses `stub_tools` to drive the loop without a
real retriever, so the fixtures live here rather than in one test file.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from porsuk.core.config import AgentConfig
from porsuk.core.models import (
    Chunk,
    ChunkHit,
    DocumentHit,
    DocumentProfile,
    DocumentSummary,
)


def _profile(doc_id: str, filename: str, language: str = "tr") -> DocumentProfile:
    return DocumentProfile(
        document_id=doc_id,
        path=f"/corpus/{filename}",
        filename=filename,
        doc_type="pdf",
        language=language,
        created_at=None,
        modified_at=None,
        size=100,
        page_count=3,
        summary=f"summary of {doc_id}",
        topics=("tedarik", "odeme"),
        entities=(),
        profile_level="cheap",
        profile_source=frozenset(),
        parse_quality=0.9,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    span: tuple[int, int],
    section_path: str | None = "3. Mali > 3.2 Ödeme",
    page_no: int | None = 12,
    language: str | None = "tr",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        page_no=page_no,
        language=language,
        section_title=section_path.split(" > ")[-1] if section_path else None,
        section_path=section_path,
        char_span=span,
    )


def _hit(chunk: Chunk, *, score: float = 0.8, rerank_score: float | None = None) -> ChunkHit:
    return ChunkHit(
        chunk=chunk,
        score=score,
        retrieval_method="semantic",
        rerank_score=rerank_score,
    )


class _StubRetriever:
    """Returns fixed hits; records the filters it was handed for assertions."""

    def __init__(self) -> None:
        self.last_filters = None
        self.last_doc_ids = None

    def _one(self, text: str) -> list[ChunkHit]:
        return [_hit(_chunk("c1", "d1", text, span=(100, 100 + len(text))))]

    def semantic_search(self, query, *, k, filters, doc_ids=None):
        self.last_filters, self.last_doc_ids = filters, doc_ids
        return self._one("ödeme 30 gün içinde yapılır")

    def keyword_search(self, query, *, k, filters, doc_ids=None):
        self.last_filters, self.last_doc_ids = filters, doc_ids
        return self._one("ödeme 30 gün")

    def hybrid_search(self, query, *, k, filters, doc_ids=None):
        self.last_filters, self.last_doc_ids = filters, doc_ids
        return self._one("ödeme")

    def search(self, query, *, k, filters, doc_ids=None, strategy=None):
        return self.semantic_search(query, k=k, filters=filters, doc_ids=doc_ids)

    def search_profiles(self, query, *, k, filters):
        self.last_filters = filters
        return [
            DocumentHit(
                profile=_profile("d1", "Sozlesme.pdf"),
                score=0.912345,
                retrieval_method="semantic",
            )
        ]


class _StubStore:
    """`chunks_for_document` returns a fixed ordered chunk list; `list_documents`
    a fixed row set. Chunks are already sorted by `char_span[0]`."""

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._chunks = chunks if chunks is not None else _default_chunks()
        self.last_filters = None

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        return [c for c in self._chunks if c.document_id == document_id]

    def list_documents(self, filters, *, limit):
        self.last_filters = filters
        rows = [
            DocumentSummary(
                document_id="d1",
                path="/corpus/Sozlesme.pdf",
                filename="Sozlesme.pdf",
                doc_type="pdf",
                language="tr",
                page_count=3,
                parse_quality=0.9,
                topics=("tedarik", "odeme"),
            ),
            DocumentSummary(
                document_id="d2",
                path="/corpus/Rapor.docx",
                filename="Rapor.docx",
                doc_type="docx",
                language="en",
                page_count=8,
                parse_quality=0.8,
                topics=("finans",),
            ),
        ]
        if filters.languages:
            rows = [r for r in rows if r.language in filters.languages]
        if filters.doc_types:
            rows = [r for r in rows if r.doc_type in filters.doc_types]
        return rows[:limit]


def _default_chunks() -> list[Chunk]:
    mali_1 = "3. Mali > 3.1 Genel"
    mali_2 = "3. Mali > 3.2 Ödeme"
    return [
        _chunk("c1", "d1", "Birinci bölüm metni. ", span=(0, 21), section_path=mali_1),
        _chunk("c2", "d1", "İkinci bölüm metni. ", span=(21, 41), section_path=mali_2),
        _chunk("c3", "d1", "Üçüncü bölüm metni. ", span=(41, 61), section_path=mali_2),
        _chunk("c4", "d1", "Son bölüm metni.", span=(61, 77), section_path="4. Sonuç"),
    ]


@pytest.fixture()
def cfg() -> AgentConfig:
    return AgentConfig()


@pytest.fixture()
def retriever() -> _StubRetriever:
    return _StubRetriever()


@pytest.fixture()
def store() -> _StubStore:
    return _StubStore()


@pytest.fixture()
def stub_tools(retriever, store, cfg):
    from porsuk.agent.tools import build_tools

    return build_tools(retriever, store, cfg=cfg)


# --------------------------------------------------------------------------- #
# GenericFakeChatModel: make a tool-call-only turn streamable.
#
# `run_agent` drives the graph with `stream_mode=["updates", "messages"]`
# whenever there is an event sink (token streaming). LangGraph's
# `messages` mode installs a streaming callback handler, which flips
# `_should_stream` on for any model that implements `_stream`, so the model
# node calls `.stream()`, not `.invoke()`.
#
# Two things the stock `GenericFakeChatModel._stream` gets wrong under that
# path:
#   1. It never emits `.tool_calls` at all: they live on the message's
#      `.tool_calls`, not `additional_kwargs.function_call`, which is the only
#      thing the stock `_stream` looks at. So WITHOUT this shim, ANY
#      tool-calling turn loses its tool calls under `.stream()`, not just the
#      empty-content ones, and the loop would then never call a tool.
#   2. For a message with empty `.content` it yields **zero** chunks, and
#      `generate_from_stream` then raises "No generations found in stream."
#      A real model always yields at least one chunk.
# This autouse shim restores both invariants for every agent test:
# token-split non-empty content as the stock does, and always emit a final
# chunk carrying the tool calls. Behaviour for plain-text answers is unchanged.
#
# Mirrors langchain-core 1.6.2's GenericFakeChatModel._stream; re-check on upgrade.
#
# `autouse` across all of `tests/agent/`: inert for tests that build no chat
# model (`test_tools.py`, `test_prompts.py`), and undone per-test by
# `monkeypatch` teardown.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _streamable_fake_chat_model(monkeypatch):
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Pull the scripted message exactly once (as the stock `_stream` does
        # via `_generate`).
        message = (
            self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            .generations[0]
            .message
        )
        content = message.content
        emitted = False

        if content and isinstance(content, str):
            for token in re.split(r"(\s)", content):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=token, id=message.id))
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                emitted = True
                yield chunk

        tool_calls = list(getattr(message, "tool_calls", None) or [])
        if not emitted or tool_calls:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="" if emitted else content,
                    id=message.id,
                    tool_calls=tool_calls,
                    tool_call_chunks=list(getattr(message, "tool_call_chunks", None) or []),
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

    monkeypatch.setattr(GenericFakeChatModel, "_stream", _stream, raising=True)
