"""FastAPI dependencies shared by the API modules; a leaf so `server` and
`index_api` don't import each other.

`require_key` guards every non-`/health` route. `get_runner` / `get_retriever`
/ `get_jobs` pull the objects `lifespan` stashed on `app.state`.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, Request


def require_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != os.environ["PORSUK_API_KEY"]:
        raise HTTPException(status_code=401, detail="invalid api key")


def get_runner(request: Request):
    return request.app.state.run


def get_retriever(request: Request):
    return request.app.state.retriever


def get_store(request: Request):
    return request.app.state.store


def get_jobs(request: Request):
    return request.app.state.jobs
