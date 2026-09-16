"""One-at-a-time index jobs for the HTTP index API: `IndexJobManager` tracks a
job open → running → done|failed and runs it on a daemon thread. Not a queue:
a second `start` while one job runs is a 409, not a wait.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

from porsuk.core.config import Config
from porsuk.core.container import build_pipeline

_ZERO = {"parsed": 0, "embedded": 0, "profiled": 0, "failed": 0, "total": 0}


class JobStateError(RuntimeError):
    """The job is not in a state that allows the requested transition."""


class JobConflictError(RuntimeError):
    """Another index job is already running."""


@dataclasses.dataclass
class _Job:
    job_id: str
    directory: Path
    state: str = "open"  # open | running | done | failed
    counts: dict = dataclasses.field(default_factory=lambda: dict(_ZERO))
    error: str | None = None


class IndexJobManager:
    def __init__(self, cfg: Config, *, app, build_pipeline=build_pipeline) -> None:
        self._cfg = cfg
        self._app = app
        self._build_pipeline = build_pipeline
        self._root = Path(cfg.ingest.corpus_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        # `corpus_dir` is the single source of truth for where a job's files and
        # its dedup state DB live (ruling from Task 1/2 review): the state DB
        # sits at `<corpus_dir>/state.sqlite`, derived here rather than a
        # separate config key that a deploy could leave pointing elsewhere.
        self._state_path = self._root / "state.sqlite"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._tick = threading.Condition(self._lock)  # notified on any counts/state change

    # ----- lifecycle ------------------------------------------------

    def open_job(self) -> str:
        job_id = uuid.uuid4().hex
        directory = self._root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        with self._lock:
            self._jobs[job_id] = _Job(job_id=job_id, directory=directory)
        return job_id

    def save_file(self, job_id: str, relative_path: str, data: str | bytes) -> None:
        with self._lock:
            job = self._jobs[job_id]  # KeyError propagates
            if job.state != "open":
                raise JobStateError(f"job {job_id} is {job.state}, not open")
            directory = job.directory
        # Reject a path with no real final component: "." / "" (empty `name`)
        # and "sub/" (a trailing separator that `Path` would silently strip,
        # turning an intended directory into a file write).
        if Path(relative_path).name == "" or relative_path != relative_path.rstrip("/"):
            raise ValueError(f"{relative_path!r} has no filename")
        target = (directory / relative_path).resolve()
        root = directory.resolve()
        # A target equal to the root is never a valid file destination, so the
        # containment test is strict: `target` must lie *inside* `root`.
        if target == root or not target.is_relative_to(root):
            raise ValueError(f"{relative_path!r} escapes the job directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        target.write_bytes(data)

    def file_count(self, job_id: str) -> int:
        with self._lock:
            directory = self._jobs[job_id].directory
        return sum(1 for p in directory.rglob("*") if p.is_file())

    def start(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]  # KeyError propagates
            if job.state != "open":
                raise JobStateError(f"job {job_id} is {job.state}, not open")
            if any(j.state == "running" for j in self._jobs.values()):
                raise JobConflictError("another index job is already running")
            job.state = "running"
            self._tick.notify_all()
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()

    # ----- reads --------------------------------------------------

    def status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]  # KeyError propagates
            return {"state": job.state, "error": job.error, **job.counts}

    def events(self, job_id: str) -> Iterator[dict | None]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
        last: dict | None = None
        while True:
            with self._tick:
                self._tick.wait(timeout=0.25)
                job = self._jobs[job_id]
                snap = {"state": job.state, "error": job.error, **job.counts}
            if snap["state"] in ("done", "failed"):
                yield {"type": snap["state"], **snap}
                return
            if snap != last:
                last = snap
                yield {"type": "progress", **snap}
            else:
                # Nothing changed this tick. Yield anyway so the consumer's
                # `next()` returns promptly: `iterate_in_threadpool` drives each
                # `next()` on a non-cancellable pool thread, so a silent spin
                # here would pin that thread through a minutes-long PDF parse and
                # past a client disconnect. The endpoint drops falsy events.
                yield None

    # ----- worker -------------------------------------------------

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        try:
            pipeline = self._build_pipeline(
                self._cfg, str(self._state_path), app=self._app, job_id=job.job_id
            )
            for event in pipeline.run(str(job.directory)):
                with self._tick:
                    job.counts = dataclasses.asdict(event)
                    self._tick.notify_all()
            with self._tick:
                job.state = "done"
                self._tick.notify_all()
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as job.error
            with self._tick:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                self._tick.notify_all()
