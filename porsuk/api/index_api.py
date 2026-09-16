"""HTTP index API: upload a folder of documents and index it.

Thin translation between HTTP and `IndexJobManager`. The job lifecycle, the
one-at-a-time rule and the pipeline thread all live in `jobs.py`; this module
only maps its exceptions to status codes and streams its events as SSE.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from porsuk.api._deps import get_jobs, require_key
from porsuk.api._serialize import _PERSISTENT_STORES
from porsuk.api.jobs import IndexJobManager, JobConflictError, JobStateError

_log = logging.getLogger("porsuk.api")

router = APIRouter(dependencies=[Depends(require_key)])


@router.post("/index", status_code=202)
async def open_index_job(jobs: IndexJobManager = Depends(get_jobs)) -> dict:
    return {"job_id": jobs.open_job()}


@router.post("/index/{job_id}/files")
async def upload_files(
    job_id: str,
    files: list[UploadFile] = File(...),
    jobs: IndexJobManager = Depends(get_jobs),
) -> dict:
    for upload in files:
        data = await upload.read()
        name = upload.filename or "unnamed"
        try:
            jobs.save_file(job_id, name, data)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id}") from None
        except JobStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            # A pathological filename (e.g. "." or "sub/") can still reach the
            # filesystem: belt-and-braces so it becomes a 400, never a 500.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"received": len(files), "total": jobs.file_count(job_id)}


@router.post("/index/{job_id}/start", status_code=202)
async def start_index_job(
    job_id: str,
    request: Request,
    jobs: IndexJobManager = Depends(get_jobs),
) -> dict:
    provider = request.app.state.cfg.store.provider
    if provider not in _PERSISTENT_STORES:
        _log.warning(
            "store.provider %r does not persist; the index will be empty after "
            "restart. Use qdrant for real use.",
            provider,
        )
    try:
        jobs.start(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}") from None
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"state": "running"}


@router.get("/index/{job_id}")
async def index_job_status(job_id: str, jobs: IndexJobManager = Depends(get_jobs)) -> dict:
    try:
        return jobs.status(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}") from None


@router.get("/index/{job_id}/stream")
async def index_job_stream(job_id: str, jobs: IndexJobManager = Depends(get_jobs)):
    try:
        state = jobs.status(job_id)["state"]  # 404 before we open the stream
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown job {job_id}") from None
    if state == "open":
        # `jobs.events()` polls forever for a job that was never started: it
        # would yield nothing and hang the SSE response. Reject it up front.
        raise HTTPException(status_code=409, detail="job not started")

    # No `finally` / cancel Event here, unlike `server.ask_stream`: the index
    # job runs on its own thread inside `IndexJobManager`, independent of this
    # stream, so there is nothing per-request to cancel. `events()` now yields
    # once per ~0.25s tick (a bare `None` when nothing changed), so each
    # `next()` returns promptly and a client disconnect is picked up within a
    # tick instead of pinning a threadpool thread through a quiet stretch.
    async def sse() -> AsyncIterator[dict]:
        async for event in iterate_in_threadpool(jobs.events(job_id)):
            if event is None:
                continue
            yield {"event": event["type"], "data": json.dumps(event, ensure_ascii=False)}

    return EventSourceResponse(sse())
