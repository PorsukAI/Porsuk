"""Tests for the Porsuk HTTP API (API katmanı). The handlers take the runner /
retriever through the `get_runner` / `get_retriever` FastAPI dependencies, so
these tests stub the agent and the retriever with `app.dependency_overrides`
and never touch `lifespan`'s real `build_agent_runner` / `build_retriever`.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from porsuk.agent.models import AgentAnswer, Source
from porsuk.core.models import Chunk, ChunkHit


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PORSUK_API_KEY", "test-key")
    monkeypatch.setenv("PORSUK_CONFIG", "config/local.yaml")
    from porsuk.api.server import app

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _override(client, dep, value):
    client.app.dependency_overrides[dep] = lambda: value


def test_health_needs_no_key(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["store"] == "inmemory"


def test_ask_requires_key(client):
    r = client.post("/ask", json={"question": "x"})
    assert r.status_code in (401, 403, 422)


def test_ask_wrong_key_is_401(client):
    r = client.post("/ask", json={"question": "x"}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_ask_returns_agent_answer_shape(client):
    from porsuk.api.server import get_runner

    def fake_run(q, *, lang=None, verbose=False, on_event=None, scope=None, history=None):
        return AgentAnswer(
            text="cevap",
            sources=[Source("f.pdf", 1, "3.2", "q", "c1")],
            tool_calls=1,
        )

    _override(client, get_runner, fake_run)
    r = client.post("/ask", json={"question": "soru"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "cevap"
    assert body["sources"][0]["file"] == "f.pdf"
    assert body["sources"][0]["chunk_id"] == "c1"


def test_ask_empty_question_is_422(client):
    r = client.post("/ask", json={"question": ""}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_ask_handler_error_is_500(client):
    from porsuk.api.server import get_runner

    def boom(q, *, lang=None, verbose=False, on_event=None, scope=None):
        raise RuntimeError("boom")

    _override(client, get_runner, boom)
    r = client.post("/ask", json={"question": "soru"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 500
    assert "boom" in r.json()["detail"]


def test_ask_stream_emits_sse_events(client):
    from porsuk.api.server import get_runner

    def fake_run(q, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        if on_event is not None:
            on_event({"type": "tool_call", "name": "search", "args": {"query": q}})
            on_event({"type": "tool_result", "name": "search", "summary": "1 hit"})
        return AgentAnswer(text="akan cevap", sources=[], tool_calls=1)

    _override(client, get_runner, fake_run)
    r = client.post(
        "/ask/stream", json={"question": "soru"}, headers={"X-API-Key": "test-key"}
    )
    assert r.status_code == 200
    text = r.text
    assert "event: tool_call" in text
    assert "event: answer" in text
    assert "akan cevap" in text

    # The answer event's payload carries the full AgentAnswer, and the tool
    # step is streamed before it.
    answer_data = next(
        line[len("data: ") :]
        for block in text.split("\r\n\r\n")
        if "event: answer" in block
        for line in block.splitlines()
        if line.startswith("data: ")
    )
    parsed = json.loads(answer_data)
    assert parsed["answer"]["text"] == "akan cevap"
    assert text.index("event: tool_call") < text.index("event: answer")


def test_ask_stream_emits_thought_and_chunk_events(client):
    from porsuk.api.server import get_runner

    def fake_run(q, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        if on_event is not None:
            on_event({"type": "thought", "text": "önce arama yapayım"})
            on_event({"type": "tool_call", "name": "search", "args": {"query": q}})
            on_event({"type": "tool_result", "name": "search", "summary": "1 hit"})
            on_event({"type": "chunk", "text": "Akan "})
            on_event({"type": "chunk", "text": "cevap."})
        return AgentAnswer(text="Akan cevap.", sources=[], tool_calls=1)

    _override(client, get_runner, fake_run)
    r = client.post(
        "/ask/stream", json={"question": "soru"}, headers={"X-API-Key": "test-key"}
    )
    assert r.status_code == 200
    text = r.text
    assert "event: thought" in text
    assert "event: chunk" in text
    assert "event: answer" in text
    # order: thought → chunk → answer
    assert text.index("event: thought") < text.index("event: chunk") < text.index("event: answer")
    # the chunk payloads carry the streamed text
    assert "Akan " in text and "cevap." in text


def test_ask_stream_passes_scope_to_the_runner(client):
    from porsuk.api.server import get_runner

    seen = {}

    def fake_run(q, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        seen["scope"] = scope
        return AgentAnswer(text="x")

    _override(client, get_runner, fake_run)
    client.post(
        "/ask/stream",
        json={"question": "soru", "scope": "job-42"},
        headers={"X-API-Key": "test-key"},
    )
    assert seen["scope"] == "job-42"


def test_ask_post_passes_scope(client):
    from porsuk.api.server import get_runner

    seen = {}

    def fake_run(q, *, lang=None, verbose=False, on_event=None, scope=None, history=None):
        seen["scope"] = scope
        return AgentAnswer(text="x")

    _override(client, get_runner, fake_run)
    r = client.post(
        "/ask",
        json={"question": "soru", "scope": "job-42"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200


def test_ask_post_passes_history_as_plain_dicts(client):
    """`history` is stateless: the client supplies prior turns
    on every call; nothing is stored server-side. Pydantic models must reach
    `run` as plain `{"role":..., "content":...}` dicts (what `run_agent`
    expects), not `HistoryTurn` objects."""
    from porsuk.api.server import get_runner

    seen = {}

    def fake_run(q, *, lang=None, verbose=False, on_event=None, scope=None, history=None):
        seen["history"] = history
        return AgentAnswer(text="x")

    _override(client, get_runner, fake_run)
    r = client.post(
        "/ask",
        json={
            "question": "peki ya temsilcilik?",
            "history": [
                {"role": "user", "content": "komisyonculuk yapabilir mi?"},
                {"role": "assistant", "content": "hayır, yapamaz."},
            ],
        },
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert seen["history"] == [
        {"role": "user", "content": "komisyonculuk yapabilir mi?"},
        {"role": "assistant", "content": "hayır, yapamaz."},
    ]


def test_ask_post_without_history_passes_none(client):
    from porsuk.api.server import get_runner

    seen = {}

    def fake_run(q, *, lang=None, verbose=False, on_event=None, scope=None, history=None):
        seen["history"] = history
        return AgentAnswer(text="x")

    _override(client, get_runner, fake_run)
    r = client.post("/ask", json={"question": "soru"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert seen["history"] is None


def test_ask_post_rejects_a_bad_history_role(client):
    r = client.post(
        "/ask",
        json={"question": "soru", "history": [{"role": "system", "content": "x"}]},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 422


def test_ask_stream_passes_history_as_json_body(client):
    from porsuk.api.server import get_runner

    seen = {}

    def fake_run(q, *, lang=None, verbose=False, on_event=None, stop=None, scope=None, history=None):
        seen["history"] = history
        return AgentAnswer(text="x")

    _override(client, get_runner, fake_run)
    client.post(
        "/ask/stream",
        json={
            "question": "soru",
            "history": [
                {"role": "user", "content": "ilk soru"},
                {"role": "assistant", "content": "ilk cevap"},
            ],
        },
        headers={"X-API-Key": "test-key"},
    )
    assert seen["history"] == [
        {"role": "user", "content": "ilk soru"},
        {"role": "assistant", "content": "ilk cevap"},
    ]


def test_ask_stream_rejects_a_bad_history_role(client):
    r = client.post(
        "/ask/stream",
        json={"question": "soru", "history": [{"role": "system", "content": "x"}]},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 422


def test_ask_stream_cancels_the_run_on_client_disconnect(client):
    """On disconnect the agent run is cancelled, not left to burn
    LLM calls. Driven at the ASGI level: Starlette's TestClient does not send
    `http.disconnect` on an early close."""
    import time

    import anyio

    from porsuk.api.server import get_runner

    calls: list[int] = []

    def slow_run(q, *, lang=None, verbose=False, on_event=None, stop=None, scope=None):
        for i in range(50):
            calls.append(i)
            if on_event is not None:
                on_event({"type": "tool_call", "name": "search", "args": {}})
            if stop is not None and stop.is_set():
                break
            time.sleep(0.05)
        return AgentAnswer(text="done", tool_calls=len(calls))

    _override(client, get_runner, slow_run)
    app = client.app

    body = json.dumps({"question": "x"}).encode()

    async def _drive() -> None:
        sent_body = {"v": False}
        disconnected = {"v": False}

        async def receive() -> dict:
            if not sent_body["v"]:
                sent_body["v"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            if not disconnected["v"]:
                await anyio.sleep(0.3)
                disconnected["v"] = True
                return {"type": "http.disconnect"}
            await anyio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(_message: dict) -> None:
            pass

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/ask/stream",
            "raw_path": b"/ask/stream",
            "query_string": b"",
            "headers": [
                (b"x-api-key", b"test-key"),
                (b"host", b"t"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "scheme": "http",
            "server": ("t", 80),
            "client": ("c", 1),
            "asgi": {"version": "3.0"},
            "root_path": "",
        }
        with anyio.move_on_after(8):
            await app(scope, receive, send)

    anyio.run(_drive)
    time.sleep(0.5)  # let the cooperative break land
    settled = len(calls)
    time.sleep(0.5)
    assert len(calls) == settled, "run kept going after disconnect"
    assert settled < 50, "run was not cancelled: it ran to the tool-call limit"


def test_search_returns_hits(client):
    from porsuk.api.server import get_retriever

    hit = ChunkHit(
        chunk=Chunk(
            "c1",
            "d1",
            "ödeme 30 gün",
            1,
            "tr",
            "3.2",
            "3. Mali > 3.2",
            (0, 12),
        ),
        score=0.9,
        retrieval_method="semantic",
    )

    class FakeRetriever:
        def search(self, query, *, k, filters, doc_ids=None, strategy=None):
            return [hit]

    _override(client, get_retriever, FakeRetriever())
    r = client.post("/search", json={"query": "ödeme"}, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["chunk_id"] == "c1"
    assert body["hits"][0]["text"] == "ödeme 30 gün"


def test_search_passes_scope_as_job_ids_filter(client):
    from porsuk.api.server import get_retriever

    seen = {}

    class FakeRetriever:
        def search(self, query, *, k, filters, doc_ids=None, strategy=None):
            seen["job_ids"] = filters.job_ids
            return []

    _override(client, get_retriever, FakeRetriever())
    r = client.post(
        "/search",
        json={"query": "ödeme", "scope": "job-42"},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    assert seen["job_ids"] == ("job-42",)


def test_documents_needs_a_key(client):
    r = client.get("/documents")
    assert r.status_code in (401, 403, 422)


def test_documents_returns_the_list(client):
    from porsuk.api.server import get_store
    from porsuk.core.models import DocumentSummary

    row = DocumentSummary(
        document_id="d1",
        path="mevzuat/mevzuat_2547.pdf",
        filename="mevzuat_2547.pdf",
        doc_type="pdf",
        language="tr",
        page_count=12,
        parse_quality=0.97,
        topics=("yükseköğretim", "unvan"),
    )

    class FakeStore:
        def list_documents(self, filters, *, limit):
            return [row]

    _override(client, get_store, FakeStore())
    r = client.get("/documents", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == [
        {
            "document_id": "d1",
            "file": "mevzuat_2547.pdf",
            "doc_type": "pdf",
            "language": "tr",
            "pages": 12,
            "topics": ["yükseköğretim", "unvan"],
        }
    ]


def test_documents_passes_scope_as_job_ids_filter(client):
    from porsuk.api.server import get_store

    seen = {}

    class FakeStore:
        def list_documents(self, filters, *, limit):
            seen["job_ids"] = filters.job_ids
            seen["unscoped_excludes_other_jobs"] = filters.unscoped_excludes_other_jobs
            return []

    _override(client, get_store, FakeStore())
    r = client.get("/documents?scope=job-42", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert seen["job_ids"] == ("job-42",)
    assert seen["unscoped_excludes_other_jobs"] is False


def test_documents_without_scope_excludes_other_jobs(client):
    from porsuk.api.server import get_store

    seen = {}

    class FakeStore:
        def list_documents(self, filters, *, limit):
            seen["unscoped_excludes_other_jobs"] = filters.unscoped_excludes_other_jobs
            return []

    _override(client, get_store, FakeStore())
    r = client.get("/documents", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert seen["unscoped_excludes_other_jobs"] is True


def test_promote_document_needs_a_key(client):
    r = client.post("/documents/d1/promote")
    assert r.status_code in (401, 403, 422)


def test_promote_document_calls_set_job_id_with_none(client):
    from porsuk.api.server import get_store

    seen = {}

    class FakeStore:
        def set_job_id(self, document_id, job_id):
            seen["document_id"] = document_id
            seen["job_id"] = job_id

    _override(client, get_store, FakeStore())
    r = client.post("/documents/d1/promote", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert seen == {"document_id": "d1", "job_id": None}
    assert r.json() == {"document_id": "d1", "job_id": None}


def test_document_file_needs_a_key(client):
    r = client.get("/documents/d1/file")
    assert r.status_code in (401, 403, 422)


def test_document_file_streams_the_file_with_the_right_media_type(client, tmp_path):
    from porsuk.api.server import get_store
    from porsuk.core.models import DocumentProfile

    f = tmp_path / "sozlesme.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf bytes")

    profile = DocumentProfile(
        document_id="d1",
        path=str(f),
        filename="sozlesme.pdf",
        doc_type="pdf",
        language="tr",
        created_at=None,
        modified_at=None,
        size=f.stat().st_size,
        page_count=1,
        summary="",
        topics=(),
        entities=(),
        profile_level="cheap",
        profile_source=frozenset(),
        parse_quality=0.9,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )

    class FakeStore:
        def get_profile(self, document_id):
            return profile if document_id == "d1" else None

    _override(client, get_store, FakeStore())
    r = client.get("/documents/d1/file", headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content == b"%PDF-1.4 fake pdf bytes"


def test_document_file_404s_when_document_id_is_unknown(client):
    from porsuk.api.server import get_store

    class FakeStore:
        def get_profile(self, document_id):
            return None

    _override(client, get_store, FakeStore())
    r = client.get("/documents/nope/file", headers={"X-API-Key": "test-key"})
    assert r.status_code == 404


def test_document_file_404s_when_the_file_is_gone_from_disk(client, tmp_path):
    from porsuk.api.server import get_store
    from porsuk.core.models import DocumentProfile

    profile = DocumentProfile(
        document_id="d1",
        path=str(tmp_path / "deleted.pdf"),  # never written
        filename="deleted.pdf",
        doc_type="pdf",
        language="tr",
        created_at=None,
        modified_at=None,
        size=0,
        page_count=1,
        summary="",
        topics=(),
        entities=(),
        profile_level="cheap",
        profile_source=frozenset(),
        parse_quality=0.9,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="h",
    )

    class FakeStore:
        def get_profile(self, document_id):
            return profile

    _override(client, get_store, FakeStore())
    r = client.get("/documents/d1/file", headers={"X-API-Key": "test-key"})
    assert r.status_code == 404


def test_missing_api_key_env_raises_at_startup(monkeypatch):
    monkeypatch.delenv("PORSUK_API_KEY", raising=False)
    monkeypatch.setenv("PORSUK_CONFIG", "config/local.yaml")

    import porsuk.api.server as srv

    importlib.reload(srv)
    with pytest.raises(RuntimeError):
        with TestClient(srv.app):
            pass


def test_empty_api_key_env_raises_at_startup(monkeypatch):
    """An empty-string value (`PORSUK_API_KEY=$UNSET_VAR`) is as unusable as an
    absent one: startup must reject it too."""
    monkeypatch.setenv("PORSUK_API_KEY", "")
    monkeypatch.setenv("PORSUK_CONFIG", "config/local.yaml")

    import porsuk.api.server as srv

    importlib.reload(srv)
    with pytest.raises(RuntimeError):
        with TestClient(srv.app):
            pass
