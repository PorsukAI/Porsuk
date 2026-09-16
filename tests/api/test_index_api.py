"""Tests for the /index upload-and-index-job HTTP API."""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("PORSUK_API_KEY", "test-key")
    monkeypatch.setenv("PORSUK_CONFIG", "config/local.yaml")
    # Redirect the corpus into tmp by patching load_config's result.
    import porsuk.core.config as config_mod

    real_load = config_mod.load_config

    def load_into_tmp(path):
        c = real_load(path)
        c.ingest.corpus_dir = str(tmp_path / "corpus")
        return c

    monkeypatch.setattr(config_mod, "load_config", load_into_tmp)
    monkeypatch.setattr("porsuk.api.server.load_config", load_into_tmp)

    from porsuk.api.server import app

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _upload(client, job_id, name, content):
    return client.post(
        f"/index/{job_id}/files",
        files=[("files", (name, io.BytesIO(content), "text/plain"))],
        headers=KEY,
    )


def test_open_requires_key(client):
    assert client.post("/index").status_code in (401, 403, 422)


def test_full_flow_open_upload_start_status(client):
    r = client.post("/index", headers=KEY)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    r = _upload(client, job_id, "a.txt", "Türkçe belge. ".encode() * 30)
    assert r.status_code == 200
    assert r.json() == {"received": 1, "total": 1}

    r = _upload(client, job_id, "sub/b.txt", b"Another document body. " * 30)
    assert r.json() == {"received": 1, "total": 2}

    r = client.post(f"/index/{job_id}/start", headers=KEY)
    assert r.status_code == 202
    assert r.json()["state"] == "running"

    deadline = time.monotonic() + 30
    state = "running"
    while state == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        state = client.get(f"/index/{job_id}", headers=KEY).json()["state"]
    body = client.get(f"/index/{job_id}", headers=KEY).json()
    assert body["state"] == "done", body
    assert body["total"] == 2
    assert body["failed"] == 0


def test_upload_to_unknown_job_is_404(client):
    r = _upload(client, "deadbeef", "a.txt", b"x")
    assert r.status_code == 404


def test_upload_after_start_is_409(client):
    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "a.txt", b"text body here " * 20)
    client.post(f"/index/{job_id}/start", headers=KEY)
    r = _upload(client, job_id, "b.txt", b"late")
    assert r.status_code == 409


def test_path_traversal_upload_is_400(client):
    job_id = client.post("/index", headers=KEY).json()["job_id"]
    r = client.post(
        f"/index/{job_id}/files",
        files=[("files", ("../evil.txt", io.BytesIO(b"x"), "text/plain"))],
        headers=KEY,
    )
    assert r.status_code == 400


@pytest.mark.parametrize("name", [".", "sub/"])
def test_upload_with_no_filename_component_is_400(client, name):
    """A multipart part whose filename resolves to the job root (".") or ends
    in a separator ("sub/") must be a 400, never an unhandled 500."""
    job_id = client.post("/index", headers=KEY).json()["job_id"]
    r = client.post(
        f"/index/{job_id}/files",
        files=[("files", (name, io.BytesIO(b"x"), "text/plain"))],
        headers=KEY,
    )
    assert r.status_code == 400


def test_second_start_while_running_is_409(client, monkeypatch):
    import threading

    gate = threading.Event()

    class Hanging:
        def run(self, folder):
            gate.wait(timeout=5)
            return iter(())

    from porsuk.api.server import app

    monkeypatch.setattr(app.state.jobs, "_build_pipeline", lambda *a, **kw: Hanging())

    try:
        a = client.post("/index", headers=KEY).json()["job_id"]
        _upload(client, a, "a.txt", b"body body body " * 10)
        client.post(f"/index/{a}/start", headers=KEY)

        b = client.post("/index", headers=KEY).json()["job_id"]
        _upload(client, b, "b.txt", b"body body body " * 10)
        r = client.post(f"/index/{b}/start", headers=KEY)
        assert r.status_code == 409
    finally:
        gate.set()


def test_status_unknown_job_is_404(client):
    assert client.get("/index/nope", headers=KEY).status_code == 404


def test_stream_emits_progress_then_done(client):
    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "d1.txt", b"Belge bir. " * 30)
    _upload(client, job_id, "d2.txt", b"Belge iki. " * 30)
    client.post(f"/index/{job_id}/start", headers=KEY)

    r = client.get(f"/index/{job_id}/stream", headers=KEY)
    assert r.status_code == 200
    assert "event: done" in r.text


def test_index_job_failure_is_reported(client, monkeypatch):
    """The `_run` `except` branch and `event: failed` in the stream: a
    pipeline that raises must land the job in `state=failed` with the error
    surfaced."""
    from porsuk.api.server import app

    class Boom:
        def run(self, folder):
            raise RuntimeError("boom")

    monkeypatch.setattr(app.state.jobs, "_build_pipeline", lambda *a, **kw: Boom())

    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "a.txt", b"body body body " * 10)
    client.post(f"/index/{job_id}/start", headers=KEY)

    deadline = time.monotonic() + 10
    body = client.get(f"/index/{job_id}", headers=KEY).json()
    while body["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
        body = client.get(f"/index/{job_id}", headers=KEY).json()
    assert body["state"] == "failed", body
    assert "boom" in body["error"]

    r = client.get(f"/index/{job_id}/stream", headers=KEY)
    assert r.status_code == 200
    assert "event: failed" in r.text


def test_stream_yields_through_a_quiet_pipeline(client, monkeypatch):
    """A pipeline that emits one event, then blocks ~1s, then finishes: the SSE
    response must still complete (each `next()` returns within a tick, so the
    threadpool thread is not pinned through the quiet stretch)."""
    import threading

    from porsuk.api.server import app

    released = threading.Event()

    class Quiet:
        def run(self, folder):
            from porsuk.ingestion.pipeline import ProgressEvent

            yield ProgressEvent(parsed=1, embedded=0, profiled=0, failed=0, total=1)
            released.wait(timeout=3)

    monkeypatch.setattr(app.state.jobs, "_build_pipeline", lambda *a, **kw: Quiet())
    threading.Timer(1.0, released.set).start()

    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "a.txt", b"body body body " * 10)
    client.post(f"/index/{job_id}/start", headers=KEY)

    r = client.get(f"/index/{job_id}/stream", headers=KEY)
    assert r.status_code == 200
    assert "event: done" in r.text


def test_stream_before_start_is_409(client):
    """`IndexJobManager.events()` never returns for a job still at "open"
    (it polls forever, yielding nothing). The stream handler must reject an
    unstarted job with a 409 rather than open a hung SSE response."""
    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "a.txt", b"body body body " * 10)
    r = client.get(f"/index/{job_id}/stream", headers=KEY)
    assert r.status_code == 409
    assert r.json()["detail"] == "job not started"


def test_start_on_non_persistent_store_still_works(client, caplog):
    """A non-persistent store (local.yaml is `inmemory`) is a warning,
    not a failure: the pipeline still runs and the job reaches `done`."""
    import logging

    job_id = client.post("/index", headers=KEY).json()["job_id"]
    _upload(client, job_id, "a.txt", b"body body body " * 20)
    with caplog.at_level(logging.WARNING, logger="porsuk.api"):
        r = client.post(f"/index/{job_id}/start", headers=KEY)
    assert r.status_code == 202
    assert any("does not persist" in m for m in caplog.messages)

    deadline = time.monotonic() + 30
    state = "running"
    while state == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        state = client.get(f"/index/{job_id}", headers=KEY).json()["state"]
    assert client.get(f"/index/{job_id}", headers=KEY).json()["state"] == "done"


def test_modules_import_without_cycle():
    """No import cycle: either module can be imported first in a clean interpreter."""
    import subprocess
    import sys

    for first in ("porsuk.api.server", "porsuk.api.index_api"):
        r = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import {first}; import porsuk.api.server; import porsuk.api.index_api",
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"importing {first} first failed:\n{r.stderr}"
