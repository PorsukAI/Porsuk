"""The Porsuk HTTP API (API katmanı).

Presentation only: one `App` built at startup, thin handlers that run the
blocking agent / retriever work in a threadpool and serialise the same
dataclasses `porsuk ask --json` / `porsuk search --json` print.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from porsuk.api._deps import get_retriever, get_runner, get_store, require_key
from porsuk.api._events import agent_events
from porsuk.api._serialize import _PERSISTENT_STORES, _hit_dict
from porsuk.api.index_api import router as index_router
from porsuk.core.config import load_config
from porsuk.core.container import build_agent_runner, build_app, build_retriever
from porsuk.core.models import Filters

_log = logging.getLogger("porsuk.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("PORSUK_API_KEY"):
        raise RuntimeError("PORSUK_API_KEY is required to run the API")
    cfg = load_config(os.environ.get("PORSUK_CONFIG", "config/vllm.yaml"))
    shared = build_app(cfg)
    if cfg.store.provider not in _PERSISTENT_STORES:
        _log.warning(
            "store.provider %r does not persist between `porsuk index` and the API, "
            "so the agent sees an empty index; use a persistent qdrant store",
            cfg.store.provider,
        )
    app.state.cfg = cfg
    app.state.run = build_agent_runner(cfg, app=shared)
    app.state.retriever = build_retriever(cfg, app=shared)
    app.state.store = shared.store

    from porsuk.api.jobs import IndexJobManager

    app.state.jobs = IndexJobManager(cfg, app=shared)
    yield


app = FastAPI(lifespan=lifespan, title="Porsuk", version="0.1.0")
app.include_router(index_router)

# The UI runs on a different origin than the API (local Vite dev server on
# its own port, or the deployed frontend on Netlify) - the browser blocks
# the request without this. PORSUK_CORS_ORIGINS is a comma-separated list
# so a deploy can name its real frontend origin(s) without a code change;
# unset falls back to the two local-dev origins Porsuk-UI's README documents
# (direct Vite, and the vite.config.ts proxy path bypasses CORS entirely so
# it isn't needed there, but a direct-origin dev setup still wants it).
_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "PORSUK_CORS_ORIGINS", "http://localhost:9001,http://127.0.0.1:9001"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    lang: str | None = None
    scope: str | None = None
    history: list[HistoryTurn] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: Literal["semantic", "keyword", "hybrid"] | None = None
    k: int | None = Field(default=None, gt=0)
    lang: str | None = None
    scope: str | None = None


@app.get("/health")
async def health(request: Request) -> dict:
    cfg = request.app.state.cfg
    return {"status": "ok", "store": cfg.store.provider, "agent_model": cfg.agent.model}


@app.post("/ask", dependencies=[Depends(require_key)])
async def ask(req: AskRequest, run=Depends(get_runner)) -> dict:
    history = [t.model_dump() for t in req.history] if req.history else None
    try:
        answer = await run_in_threadpool(
            run, req.question, lang=req.lang, scope=req.scope, history=history
        )
    except Exception as exc:  # noqa: BLE001 - any agent failure becomes a 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return dataclasses.asdict(answer)


@app.post("/ask/stream", dependencies=[Depends(require_key)])
async def ask_stream(req: AskRequest, run=Depends(get_runner)):
    # `sse()` is async on purpose: sse-starlette runs an *async* body
    # generator's `finally` on client disconnect but leaves a sync one
    # suspended in its thread-pool wrapper, `finally` unrun (verified against
    # sse-starlette 3.3+). So the cancel Event is owned
    # here and set from `sse()`'s `finally`; `agent_events` threads it through
    # to the agent loop, which breaks at its next step ("bağlantı
    # koparsa agent task iptal edilir"). The blocking queue pull inside
    # `agent_events` is off-loaded with `iterate_in_threadpool` so it never
    # blocks the event loop.
    stop = threading.Event()
    # POST + JSON body, not GET + query string: a long conversation history
    # pushed the encoded query string past
    # nginx/uvicorn's URI length limit (414), which the browser reports as an
    # opaque CORS failure since the rejected response carries no CORS headers.
    parsed_history = [t.model_dump() for t in req.history] if req.history else None

    async def sse() -> AsyncIterator[dict]:
        events = agent_events(
            run, req.question, lang=req.lang, scope=req.scope, stop=stop, history=parsed_history
        )
        try:
            async for event in iterate_in_threadpool(events):
                yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}
        finally:
            stop.set()

    return EventSourceResponse(sse())


@app.get("/documents", dependencies=[Depends(require_key)])
async def documents(scope: str | None = None, store=Depends(get_store)) -> dict:
    """List indexed documents, the UI's "indexed
    documents" panel after an /index run completes. Same scope semantics as
    /ask and /search: omit `scope` for the unscoped base corpus, pass a
    job_id to see only that chat's uploads."""
    filters = Filters(
        job_ids=(scope,) if scope else (),
        unscoped_excludes_other_jobs=scope is None,
    )
    rows = await run_in_threadpool(store.list_documents, filters, limit=500)
    return {
        "documents": [
            {
                "document_id": r.document_id,
                "file": r.filename,
                "doc_type": r.doc_type,
                "language": r.language,
                "pages": r.page_count,
                "topics": list(r.topics),
            }
            for r in rows
        ]
    }


@app.post("/documents/{document_id}/promote", dependencies=[Depends(require_key)])
async def promote_document(document_id: str, store=Depends(get_store)) -> dict:
    """"Kalıcı yap": move one chat-scoped document into the unscoped base
    corpus (job_id=None), so every chat can see it afterward, not just the
    one it was uploaded into."""
    await run_in_threadpool(store.set_job_id, document_id, None)
    return {"document_id": document_id, "job_id": None}


_DOC_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "txt": "text/plain",
}


@app.get("/documents/{document_id}/file", dependencies=[Depends(require_key)])
async def document_file(document_id: str, store=Depends(get_store)):
    """Streams the original file for `document_id` so the UI can render it
    in place ("Dokümanda göster") instead of only showing the cited snippet.
    `document_id` is opaque and resolved server-side through the store, so a
    caller cannot pass an arbitrary filesystem path, only what was actually
    indexed. 404 covers both "never indexed" and "indexed but the file has
    since moved/been deleted from disk"."""
    profile = await run_in_threadpool(store.get_profile, document_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="document not found")
    path = Path(profile.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file no longer on disk")
    media_type = _DOC_MEDIA_TYPES.get(profile.doc_type, "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=profile.filename)


@app.post("/search", dependencies=[Depends(require_key)])
async def search(req: SearchRequest, request: Request, retriever=Depends(get_retriever)) -> dict:
    k = req.k or request.app.state.cfg.retrieval.chunk_k
    filters = Filters(
        languages=(req.lang,) if req.lang else (),
        job_ids=(req.scope,) if req.scope else (),
        unscoped_excludes_other_jobs=req.scope is None,
    )
    try:
        hits = await run_in_threadpool(
            retriever.search, req.query, k=k, filters=filters, strategy=req.strategy
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"hits": [_hit_dict(h) for h in hits]}
