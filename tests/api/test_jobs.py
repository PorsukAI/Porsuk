"""Tests for `IndexJobManager`, the per-job upload-and-index state machine."""

from __future__ import annotations

import time

import pytest

from porsuk.core.config import load_config
from porsuk.core.container import build_app


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config/local.yaml")
    c.ingest.corpus_dir = str(tmp_path / "corpus")
    return c


@pytest.fixture
def manager(cfg):
    from porsuk.api.jobs import IndexJobManager

    return IndexJobManager(cfg, app=build_app(cfg))


def test_open_job_makes_a_directory(manager, cfg):
    from pathlib import Path

    job_id = manager.open_job()
    assert (Path(cfg.ingest.corpus_dir) / job_id).is_dir()
    assert manager.status(job_id)["state"] == "open"


def test_save_file_writes_under_the_job_dir_and_is_idempotent(manager, cfg):
    from pathlib import Path

    job_id = manager.open_job()
    manager.save_file(job_id, "a/b.txt", b"hello")
    manager.save_file(job_id, "a/b.txt", b"hello again")
    p = Path(cfg.ingest.corpus_dir) / job_id / "a" / "b.txt"
    assert p.read_bytes() == b"hello again"
    assert manager.file_count(job_id) == 1


def test_save_file_rejects_path_traversal(manager):
    job_id = manager.open_job()
    with pytest.raises(ValueError):
        manager.save_file(job_id, "../escape.txt", b"x")


@pytest.mark.parametrize("bad", [".", "sub/", ""])
def test_save_file_rejects_names_with_no_file_component(manager, bad):
    job_id = manager.open_job()
    with pytest.raises(ValueError):
        manager.save_file(job_id, bad, b"x")


def test_save_file_rejects_unknown_job(manager):
    with pytest.raises(KeyError):
        manager.save_file("nope", "a.txt", b"x")


def test_start_runs_the_pipeline_and_reaches_done(manager):
    job_id = manager.open_job()
    manager.save_file(job_id, "doc1.txt", "Türkçe bir belge. " * 20)  # noqa: E501
    manager.save_file(job_id, "doc2.txt", "Another short document. " * 20)
    manager.start(job_id)

    deadline = time.monotonic() + 30
    while manager.status(job_id)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.1)

    st = manager.status(job_id)
    assert st["state"] == "done", st
    assert st["total"] == 2
    assert st["failed"] == 0


def test_index_job_passes_job_id_to_the_pipeline(manager, monkeypatch):
    """IndexJobManager._run builds the pipeline with job_id=<the job's id> so
    ingestion stamps every chunk/profile with it."""

    class _FakePipeline:
        def run(self, folder):
            return iter(())

    captured = {}
    monkeypatch.setattr(
        manager,
        "_build_pipeline",
        lambda cfg, sp, app=None, job_id=None: captured.update(job_id=job_id) or _FakePipeline(),
    )
    job_id = manager.open_job()
    manager.save_file(job_id, "a.txt", "content " * 20)
    manager.start(job_id)

    deadline = time.monotonic() + 10
    while manager.status(job_id)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)

    assert manager.status(job_id)["state"] == "done"
    assert captured["job_id"] == job_id


def test_start_rejects_a_second_running_job(manager, monkeypatch):
    # Make the pipeline hang so the first job stays running.
    import threading

    gate = threading.Event()

    class HangingPipeline:
        def run(self, folder):
            gate.wait(timeout=5)
            return iter(())

    monkeypatch.setattr(
        manager,
        "_build_pipeline",
        lambda cfg, state_path, app=None, job_id=None: HangingPipeline(),
    )
    a = manager.open_job()
    manager.save_file(a, "x.txt", b"content here")
    manager.start(a)

    b = manager.open_job()
    manager.save_file(b, "y.txt", b"content here")
    from porsuk.api.jobs import JobConflictError

    with pytest.raises(JobConflictError):
        manager.start(b)
    gate.set()


def test_start_rejects_a_job_that_is_not_open(manager):
    from porsuk.api.jobs import JobStateError

    job_id = manager.open_job()
    manager.save_file(job_id, "d.txt", "text " * 30)
    manager.start(job_id)
    with pytest.raises(JobStateError):
        manager.start(job_id)


def test_events_streams_progress_then_a_terminal_event(manager):
    job_id = manager.open_job()
    manager.save_file(job_id, "d1.txt", "Belge bir. " * 30)
    manager.save_file(job_id, "d2.txt", "Belge iki. " * 30)
    manager.start(job_id)

    kinds = [ev["type"] for ev in manager.events(job_id) if ev is not None]
    assert kinds[-1] in ("done", "failed")
    assert kinds[-1] == "done"


def test_events_yields_none_through_a_quiet_pipeline(manager, monkeypatch):
    """Between ProgressEvents `events()` must still return each tick (yielding
    None) so the SSE threadpool thread is released and a disconnect is seen
    promptly, not spin silently through a minutes-long parse."""
    import threading

    gate = threading.Event()

    class QuietPipeline:
        def run(self, folder):
            gate.wait(timeout=5)
            return iter(())

    monkeypatch.setattr(
        manager,
        "_build_pipeline",
        lambda cfg, state_path, app=None, job_id=None: QuietPipeline(),
    )
    job_id = manager.open_job()
    manager.save_file(job_id, "x.txt", b"content here")
    manager.start(job_id)

    events = manager.events(job_id)
    try:
        # First tick yields the running-state snapshot; the next tick has
        # nothing new, and must still return (a None) within ~0.25s rather
        # than spin silently until the pipeline finishes.
        assert next(events) == {
            "type": "progress",
            "state": "running",
            "error": None,
            "parsed": 0,
            "embedded": 0,
            "profiled": 0,
            "failed": 0,
            "total": 0,
        }
        start = time.monotonic()
        assert next(events) is None
        assert time.monotonic() - start < 1.0
    finally:
        gate.set()
        events.close()


def test_start_failure_sets_failed_state(manager, monkeypatch):
    class BoomPipeline:
        def run(self, folder):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        manager,
        "_build_pipeline",
        lambda cfg, state_path, app=None, job_id=None: BoomPipeline(),
    )
    job_id = manager.open_job()
    manager.save_file(job_id, "x.txt", b"content here")
    manager.start(job_id)

    deadline = time.monotonic() + 10
    while manager.status(job_id)["state"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)

    st = manager.status(job_id)
    assert st["state"] == "failed", st
    assert "boom" in st["error"]

    kinds = [ev["type"] for ev in manager.events(job_id) if ev is not None]
    assert kinds[-1] == "failed"
