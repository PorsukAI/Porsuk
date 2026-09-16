"""Tests for `StateStore`, the SQLite ingestion-state tracker."""

import sqlite3

import pytest

from porsuk.core.state import StateStore


@pytest.fixture()
def store(tmp_path):
    s = StateStore(tmp_path / "state.db")
    yield s
    s.close()


def test_new_document_needs_processing(store):
    assert store.upsert_document("d1", "/corpus/a.pdf", "hash1") is True


def test_unchanged_document_does_not_need_reprocessing(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    assert store.upsert_document("d1", "/corpus/a.pdf", "hash1") is False


def test_changed_hash_needs_reprocessing(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    assert store.upsert_document("d1", "/corpus/a.pdf", "hash2") is True


def test_new_document_starts_pending(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    assert store.counts() == {"pending": 1}


def test_status_transitions_are_recorded(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    store.set_status("d1", "parsed")
    store.set_status("d1", "embedded")
    store.set_status("d1", "profiled")
    assert store.counts() == {"profiled": 1}


def test_counts_scoped_under_a_folder_excludes_other_folders(store, tmp_path):
    job_a = tmp_path / "job-a"
    job_b = tmp_path / "job-b"
    job_a.mkdir()
    job_b.mkdir()
    store.upsert_document("d1", str(job_a / "a.pdf"), "hash1")
    store.upsert_document("d2", str(job_b / "b.pdf"), "hash2")
    store.set_status("d1", "profiled")
    store.set_status("d2", "profiled")
    assert store.counts(under=job_a) == {"profiled": 1}


def test_counts_scoped_under_a_folder_does_not_match_a_sibling_with_shared_prefix(
    store, tmp_path
):
    # "job-a" must not match "job-ab/x.pdf": a naive string-prefix
    # comparison (no separator) would; `is_relative_to` does not.
    job_a = tmp_path / "job-a"
    job_ab = tmp_path / "job-ab"
    job_a.mkdir()
    job_ab.mkdir()
    store.upsert_document("d1", str(job_ab / "x.pdf"), "hash1")
    store.set_status("d1", "profiled")
    assert store.counts(under=job_a) == {}


def test_counts_scoped_under_a_folder_resolves_relative_paths(store, tmp_path, monkeypatch):
    # documents.path is stored exactly as passed to run() (often relative)
    # so counts(under=...) must resolve before comparing, not string-match.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus").mkdir()
    store.upsert_document("d1", "corpus/a.pdf", "hash1")
    store.set_status("d1", "profiled")
    assert store.counts(under=(tmp_path / "corpus").resolve()) == {"profiled": 1}


def test_set_status_records_chunk_count_and_profile_level(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    store.set_status("d1", "parsed", chunk_count=7)
    store.set_status("d1", "profiled", profile_level="cheap")
    (d,) = store.documents_with_status("profiled")
    assert d.chunk_count == 7
    assert d.profile_level == "cheap"


def test_chunk_count_and_profile_level_default_to_zero_and_none(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    (d,) = store.documents_with_status("pending")
    assert d.chunk_count == 0
    assert d.profile_level is None


def test_changed_hash_resets_chunk_count_and_profile_level(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    store.set_status("d1", "parsed", chunk_count=5)
    store.set_status("d1", "profiled", profile_level="llm")
    store.upsert_document("d1", "/corpus/a.pdf", "hash2")
    (d,) = store.documents_with_status("pending")
    assert d.chunk_count == 0
    assert d.profile_level is None


def test_failed_status_stores_the_error(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    store.set_status("d1", "failed", error="password protected")
    failed = store.documents_with_status("failed")
    assert len(failed) == 1
    assert failed[0].error == "password protected"


def test_documents_with_status_supports_a_limit(store):
    for i in range(5):
        store.upsert_document(f"d{i}", f"/corpus/{i}.pdf", "h")
    assert len(store.documents_with_status("pending", limit=2)) == 2


def test_state_survives_reopening(tmp_path):
    path = tmp_path / "state.db"
    first = StateStore(path)
    first.upsert_document("d1", "/corpus/a.pdf", "hash1")
    first.set_status("d1", "parsed")
    first.close()

    second = StateStore(path)
    assert second.counts() == {"parsed": 1}
    (restored,) = second.documents_with_status("parsed")
    assert restored.document_id == "d1"
    assert restored.path == "/corpus/a.pdf"
    assert restored.content_hash == "hash1"
    second.close()


def test_rejects_an_unknown_status(store):
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    with pytest.raises(ValueError):
        store.set_status("d1", "banana")


def test_needs_processing_is_true_for_an_unseen_path(store):
    assert store.needs_processing("/corpus/new.pdf", "hash1") is True


def test_reusing_a_path_under_a_different_id_is_rejected(store):
    """document_id is a function of path; the UNIQUE constraint enforces it."""
    store.upsert_document("d1", "/corpus/a.pdf", "hash1")
    with pytest.raises(sqlite3.IntegrityError):
        store.upsert_document("d2", "/corpus/a.pdf", "hash1")
