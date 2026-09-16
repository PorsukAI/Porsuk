"""Ingestion state in SQLite so an interrupted run resumes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

IngestStatus = Literal["pending", "parsed", "embedded", "profiled", "failed"]
_VALID_STATUSES = frozenset(get_args(IngestStatus))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    path          TEXT NOT NULL UNIQUE,
    content_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    error         TEXT,
    updated_at    TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    profile_level TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
"""


@dataclass(frozen=True)
class DocumentState:
    document_id: str
    path: str
    content_hash: str
    status: str
    error: str | None
    updated_at: str
    chunk_count: int = 0
    profile_level: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns an older database is missing. Cheap and idempotent."""
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(documents)")}
        if "chunk_count" not in have:
            self._conn.execute(
                "ALTER TABLE documents ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0"
            )
        if "profile_level" not in have:
            self._conn.execute("ALTER TABLE documents ADD COLUMN profile_level TEXT")

    def close(self) -> None:
        self._conn.close()

    def upsert_document(self, document_id: str, path: str, content_hash: str) -> bool:
        """Insert or update. Returns True when the file must be processed.

        `document_id` must be a deterministic function of `path`: the caller
        derives one from the other, so the two are in bijection. The schema's
        `path UNIQUE` constraint enforces it: reusing a path under a different
        document_id raises sqlite3.IntegrityError rather than storing two rows
        for one file.
        """
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is not None and row["content_hash"] == content_hash:
            return False
        self._conn.execute(
            """
            INSERT INTO documents
                (document_id, path, content_hash, status, error, updated_at)
            VALUES (?, ?, ?, 'pending', NULL, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                path = excluded.path,
                content_hash = excluded.content_hash,
                status = 'pending',
                error = NULL,
                updated_at = excluded.updated_at,
                chunk_count = 0,
                profile_level = NULL
            """,
            (document_id, path, content_hash, _now()),
        )
        self._conn.commit()
        return True

    def set_status(
        self,
        document_id: str,
        status: str,
        error: str | None = None,
        *,
        chunk_count: int | None = None,
        profile_level: str | None = None,
    ) -> None:
        if status not in _VALID_STATUSES:
            raise ValueError(f"unknown status {status!r}; valid: {sorted(_VALID_STATUSES)}")
        sets = ["status = ?", "error = ?", "updated_at = ?"]
        params: list[object] = [status, error, _now()]
        if chunk_count is not None:
            sets.append("chunk_count = ?")
            params.append(chunk_count)
        if profile_level is not None:
            sets.append("profile_level = ?")
            params.append(profile_level)
        params.append(document_id)
        self._conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE document_id = ?", params)
        self._conn.commit()

    def documents_with_status(self, status: str, limit: int | None = None) -> list[DocumentState]:
        sql = "SELECT * FROM documents WHERE status = ? ORDER BY document_id"
        params: tuple[object, ...] = (status,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        return [DocumentState(**dict(r)) for r in self._conn.execute(sql, params)]

    def counts(self, *, under: Path | None = None) -> dict[str, int]:
        """Status counts, optionally restricted to documents whose path
        resolves under the folder `under`. The state DB is shared across
        every index job (`<corpus_dir>/state.sqlite`), so an
        unfiltered call mixes in every other job's documents: the progress
        `run()` streams back must stay scoped to its own folder.

        Filtered in Python, not SQL: `documents.path` is stored exactly as
        the caller passed it to `run()` (often relative), so a `LIKE`
        against a resolved prefix would silently match nothing. `resolve()`
        gives both sides the same absolute form, the same comparison
        `run()`'s resume sweep already does against `folder_root`."""
        rows = self._conn.execute("SELECT status, path FROM documents")
        counts: dict[str, int] = {}
        for r in rows:
            if under is not None and not Path(r["path"]).resolve().is_relative_to(under):
                continue
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return counts

    def needs_processing(self, path: str, content_hash: str) -> bool:
        row = self._conn.execute(
            "SELECT content_hash FROM documents WHERE path = ?", (path,)
        ).fetchone()
        return row is None or row["content_hash"] != content_hash
