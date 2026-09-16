"""The ingestion queue: parse -> embed -> cheap profile -> (background) LLM
profile. State lives in SQLite so an interrupted run resumes, unchanged files
are skipped by content_hash, and one bad file is recorded as `failed` without
stopping the run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import uuid
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from os import cpu_count
from pathlib import Path

from porsuk.core.config import ChunkingConfig, ProfileConfig, QualityConfig
from porsuk.core.models import Chunk, ParseOutcome
from porsuk.core.state import StateStore
from porsuk.ingestion.chunking import chunk as chunk_document
from porsuk.ingestion.profiler import build_cheap_profile, build_llm_profile
from porsuk.ingestion.router import ParserRouter

_SKIP_SUFFIXES = frozenset(
    {".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3", ".sqlite-wal", ".sqlite-shm"}
)
_SKIP_NAMES = frozenset({".DS_Store", "Thumbs.db"})


def content_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def document_id_for(path: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(path).resolve())))


@dataclass(frozen=True)
class ProgressEvent:
    parsed: int
    embedded: int
    profiled: int
    failed: int
    total: int


def _parse_one(args: tuple[str, tuple[str, ...]]) -> tuple[str, ParseOutcome]:
    """Runs in a worker process: rebuild the router from parser names.

    A fresh worker has an empty registry, so import the adapters here for
    their @register side effects before resolving any parser.
    """
    from porsuk.core.container import _load_adapters
    from porsuk.core.registry import build

    _load_adapters()
    path, parser_names = args
    router = ParserRouter(
        [build("parser", {"provider": name}) for name in parser_names], QualityConfig()
    )
    return path, router.parse(path)


class Pipeline:
    def __init__(
        self,
        *,
        router: ParserRouter,
        embedder,
        store,
        llm,
        state: StateStore,
        chunking_cfg: ChunkingConfig,
        profile_cfg: ProfileConfig,
        parse_workers: int = 0,
        embed_batch_size: int = 32,
        job_id: str | None = None,
    ) -> None:
        self._router = router
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._state = state
        self._chunking_cfg = chunking_cfg
        self._profile_cfg = profile_cfg
        self._workers = parse_workers or (cpu_count() or 1)
        self._batch = embed_batch_size
        self._job_id = job_id
        self._folder_root: Path | None = None

    # ----- discovery / progress -------------------------------------

    def _discover(self, folder: str) -> list[str]:
        return [
            str(p)
            for p in sorted(Path(folder).rglob("*"))
            if p.is_file() and p.suffix.lower() not in _SKIP_SUFFIXES and p.name not in _SKIP_NAMES
        ]

    def _event(self, total: int) -> ProgressEvent:
        # Scoped to this run's own folder: the state DB is shared across
        # every index job, so an unscoped count would mix in
        # other jobs' documents and misreport this job's progress.
        c = self._state.counts(under=self._folder_root)
        return ProgressEvent(
            parsed=c.get("parsed", 0),
            embedded=c.get("embedded", 0),
            profiled=c.get("profiled", 0),
            failed=c.get("failed", 0),
            total=total,
        )

    # ----- run ------------------------------------------------------

    def run(self, folder: str) -> Iterator[ProgressEvent]:
        self._folder_root = Path(folder).resolve()
        files = self._discover(folder)
        hashes = {path: content_hash(path) for path in files}
        todo = [
            path
            for path in files
            if self._state.upsert_document(document_id_for(path), path, hashes[path])
        ]
        total = len(files)

        # Resume: a previous run may have crashed after parse/embed but
        # before the profile stage, leaving documents stranded at `parsed`
        # or `embedded`. Their hash is unchanged so stage 1 skips them; drive
        # them through the rest before starting fresh work.
        #
        # The state DB is shared by every index job (API path:
        # `<corpus_dir>/state.sqlite`), so `documents_with_status` returns
        # stranded docs from *other* jobs too. Skip anything not under the
        # folder being indexed: it is not ours to finish, and re-driving it
        # would re-stamp its chunks/profile with this job's job_id, leaking
        # another chat's upload into this scoped chat. For
        # `porsuk index <folder>` every stranded doc IS under the folder, so
        # this is a no-op there; a job's files all live under
        # `<corpus_dir>/<job_id>/`, so it makes the API path job-local.
        folder_root = self._folder_root
        for status in ("parsed", "embedded"):
            for st in list(self._state.documents_with_status(status)):
                if not Path(st.path).resolve().is_relative_to(folder_root):
                    continue
                try:
                    self._finish_document(st.document_id, st.path)
                except Exception as exc:  # noqa: BLE001
                    self._state.set_status(
                        st.document_id, "failed", error=f"{type(exc).__name__}: {exc}"
                    )
                yield self._event(total)

        if not todo:
            return

        parser_names = self._router.parser_names
        work = ((path, parser_names) for path in todo)

        if self._workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(max_workers=self._workers) as pool:
                results = pool.map(_parse_one, work)
                yield from self._consume_parses(results, hashes, total)
        else:
            yield from self._consume_parses((_parse_one(w) for w in work), hashes, total)

        if self._profile_cfg.llm_enabled:
            # Same containment as the resume sweep above: the shared state DB
            # can hold another job's doc stranded at `embedded` (that job
            # crashed mid-profile); skip it, it is not ours to finish, and
            # counting it here would misreport this job's own progress.
            for st in list(self._state.documents_with_status("embedded")):
                if not Path(st.path).resolve().is_relative_to(folder_root):
                    continue
                # A bad LLM response (or a transient endpoint error) on one
                # document must not abort the run for the rest, same
                # partial-failure contract as parse/embed above.
                try:
                    self._llm_profile(st.document_id, st.path)
                except Exception as exc:  # noqa: BLE001
                    self._state.set_status(
                        st.document_id, "failed", error=f"{type(exc).__name__}: {exc}"
                    )
                yield self._event(total)

    def _consume_parses(
        self, results, hashes: dict[str, str], total: int
    ) -> Iterator[ProgressEvent]:
        for path, outcome in results:
            doc_id = document_id_for(path)
            if outcome.status == "failed" or outcome.document is None:
                self._state.set_status(doc_id, "failed", error=outcome.error or "parse failed")
                yield self._event(total)
                continue
            # Embed and profile can fail too - a flaky endpoint, an OOM, a
            # chunk the store rejects. Partial failure is normal
            # at 5000 files; a failure here records the document as `failed`
            # and the run continues, exactly as a parse failure does.
            try:
                chunks = list(chunk_document(outcome.document, doc_id, self._chunking_cfg))
                self._state.set_status(doc_id, "parsed", chunk_count=len(chunks))
                self._embed_and_store_chunks(chunks)
                self._state.set_status(doc_id, "embedded")
                self._cheap_profile(outcome, doc_id, path, hashes[path])
            except Exception as exc:  # noqa: BLE001
                self._state.set_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
            yield self._event(total)

    def _finish_document(self, doc_id: str, path: str) -> None:
        """Re-drive a document stranded mid-pipeline by an earlier crash.

        Cheaper than re-indexing from scratch would be, but not free: it
        re-parses, because the stage-1 parse result was not persisted.
        """
        outcome = self._router.parse(path)
        if outcome.document is None:
            self._state.set_status(doc_id, "failed", error=outcome.error or "re-parse failed")
            return
        chunks = list(chunk_document(outcome.document, doc_id, self._chunking_cfg))
        self._state.set_status(doc_id, "parsed", chunk_count=len(chunks))
        self._embed_and_store_chunks(chunks)
        self._state.set_status(doc_id, "embedded")
        self._cheap_profile(outcome, doc_id, path, content_hash(path))
        if self._profile_cfg.llm_enabled:
            self._llm_profile(doc_id, path)

    # ----- stages -------------------------------------------------

    def _embed_and_store_chunks(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), self._batch):
            batch = [
                dataclasses.replace(c, job_id=self._job_id) for c in chunks[i : i + self._batch]
            ]
            vectors = self._embedder.embed_documents([c.text for c in batch])
            self._store.upsert_chunks(batch, vectors)

    def _cheap_profile(
        self, outcome: ParseOutcome, doc_id: str, path: str, content_hash_: str
    ) -> None:
        stat = Path(path).stat()
        profile, text = build_cheap_profile(
            outcome.document,
            outcome,
            document_id=doc_id,
            content_hash=content_hash_,
            size=stat.st_size,
            cfg=self._profile_cfg,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )
        profile = dataclasses.replace(profile, job_id=self._job_id)
        self._store.upsert_profiles([profile], self._embedder.embed_documents([text.text]))
        if not self._profile_cfg.llm_enabled:
            self._state.set_status(doc_id, "profiled", profile_level="cheap")

    def _llm_profile(self, doc_id: str, path: str) -> None:
        # Re-parses: the stage-1 blocks were not persisted. Cheap for text,
        # but for a scanned PDF that escalated to OCR/Docling this pays that
        # cost a second time - the trade-off for not carrying ParsedDocument
        # through the SQLite state (kept intentionally minimal).
        outcome = self._router.parse(path)
        if outcome.document is None:
            self._state.set_status(doc_id, "profiled", profile_level="cheap")
            return
        stat = Path(path).stat()
        base, _ = build_cheap_profile(
            outcome.document,
            outcome,
            document_id=doc_id,
            content_hash=content_hash(path),
            size=stat.st_size,
            cfg=self._profile_cfg,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )
        profile, text = build_llm_profile(
            outcome.document, base, llm=self._llm, cfg=self._profile_cfg
        )
        profile = dataclasses.replace(profile, job_id=self._job_id)
        self._store.upsert_profiles([profile], self._embedder.embed_documents([text.text]))
        self._state.set_status(doc_id, "profiled", profile_level=profile.profile_level)
