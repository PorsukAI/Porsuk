"""Tests for the ingestion `Pipeline`: indexing, resume, and job scoping."""

from porsuk.adapters.embedding.fake import FakeEmbedder
from porsuk.adapters.parsers.text import TextParser
from porsuk.adapters.store.inmemory import InMemoryStore
from porsuk.core.config import ChunkingConfig, ProfileConfig, QualityConfig
from porsuk.core.state import StateStore
from porsuk.ingestion.chunking import chunk as chunk_document
from porsuk.ingestion.pipeline import Pipeline, content_hash, document_id_for
from porsuk.ingestion.router import ParserRouter


class _FakeLLM:
    model = "fake"

    def complete(self, prompt, *, max_tokens=512, enable_thinking=True):
        return "Özet cümlesi.\nKONULAR: a, b\nTİP: rapor"


def _pipeline(tmp_path, *, llm_enabled=False, job_id=None):
    router = ParserRouter([TextParser()], QualityConfig())
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="fake:fake:8:norm=True")
    state = StateStore(tmp_path / "s.db")
    return (
        Pipeline(
            router=router,
            embedder=FakeEmbedder(dim=8),
            store=store,
            llm=_FakeLLM(),
            state=state,
            chunking_cfg=ChunkingConfig(),
            profile_cfg=ProfileConfig(llm_enabled=llm_enabled),
            parse_workers=1,
            embed_batch_size=4,
            job_id=job_id,
        ),
        store,
        state,
    )


def test_content_hash_and_id_are_deterministic(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    assert content_hash(str(f)) == content_hash(str(f))
    assert document_id_for(str(f)) == document_id_for(str(f))
    g = tmp_path / "y.txt"
    g.write_text("hello")
    assert content_hash(str(f)) == content_hash(str(g))  # same bytes
    assert document_id_for(str(f)) != document_id_for(str(g))  # different path


def test_indexes_a_folder_of_text_files(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Bu bir tedarik sözleşmesi metnidir. " * 12)
    (corpus / "b.txt").write_text("This is a supply agreement document body. " * 12)

    pipe, store, state = _pipeline(tmp_path)
    events = list(pipe.run(str(corpus)))

    assert events[-1].total == 2
    assert events[-1].profiled == 2
    assert events[-1].failed == 0
    assert state.counts().get("profiled") == 2
    assert len(store.all_chunk_ids()) > 0
    assert len(store.all_profile_ids()) == 2


def test_indexed_document_carries_the_files_mtime(tmp_path):
    """The profile's modified_at comes from the file's own filesystem mtime
    (a document with no embedded metadata date still gets a usable one for
    date-range filtering)."""
    import os
    from datetime import UTC, datetime

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.txt"
    f.write_text("Bu bir tedarik sözleşmesi metnidir. " * 12)
    # Force a known mtime rather than trusting "just now": makes the
    # assertion exact instead of a loose recency check.
    known = datetime(2022, 3, 15, 10, 0, 0, tzinfo=UTC)
    os.utime(f, (known.timestamp(), known.timestamp()))

    pipe, store, _state = _pipeline(tmp_path)
    list(pipe.run(str(corpus)))

    (doc_id,) = store.all_profile_ids()
    profile, _vector = store._profiles[doc_id]
    assert profile.modified_at == known


def test_pipeline_tags_chunks_and_profiles_with_job_id(tmp_path):
    """Built with a job_id, every stored chunk and profile carries it;
    default job_id=None leaves them untagged."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Bu bir tedarik sözleşmesi metnidir. " * 12)
    (corpus / "b.txt").write_text("This is a supply agreement document body. " * 12)

    pipe, store, _state = _pipeline(tmp_path, job_id="job-xyz")
    list(pipe.run(str(corpus)))

    assert len(store.all_chunk_ids()) > 0
    assert len(store.all_profile_ids()) == 2
    assert all(c.job_id == "job-xyz" for c, _ in store._chunks.values())
    assert all(p.job_id == "job-xyz" for p, _ in store._profiles.values())

    corpus2 = tmp_path / "corpus2"
    corpus2.mkdir()
    (corpus2 / "c.txt").write_text("Ayrı bir belge metni burada. " * 12)
    sub = tmp_path / "sub"
    sub.mkdir()

    pipe2, store2, _state2 = _pipeline(sub, job_id=None)
    list(pipe2.run(str(corpus2)))

    assert len(store2.all_chunk_ids()) > 0
    assert len(store2.all_profile_ids()) == 1
    assert all(c.job_id is None for c, _ in store2._chunks.values())
    assert all(p.job_id is None for p, _ in store2._profiles.values())


def test_second_run_skips_unchanged_files(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Sabit içerik burada duruyor. " * 15)

    pipe, _store, state = _pipeline(tmp_path)
    list(pipe.run(str(corpus)))
    assert state.counts().get("profiled") == 1

    pipe2, _store2, state2 = _pipeline(tmp_path)  # same state db
    events = list(pipe2.run(str(corpus)))
    assert events == []  # nothing to do


def test_a_broken_file_does_not_stop_the_run(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.txt").write_text("Geçerli Türkçe metin burada. " * 15)
    (corpus / "bad.bin").write_bytes(b"\x00\x01\x02\x00")

    pipe, _store, state = _pipeline(tmp_path)
    list(pipe.run(str(corpus)))
    counts = state.counts()
    assert counts.get("profiled") == 1
    assert counts.get("failed") == 1


def test_llm_profile_pass_runs_when_enabled(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Bir rapor metni burada. " * 20)

    pipe, _store, state = _pipeline(tmp_path, llm_enabled=True)
    list(pipe.run(str(corpus)))
    (d,) = state.documents_with_status("profiled")
    assert d.profile_level == "llm"


def test_chunk_count_is_recorded(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Uzunca bir metin. " * 200)

    pipe, _store, state = _pipeline(tmp_path)
    list(pipe.run(str(corpus)))
    (d,) = state.documents_with_status("profiled")
    assert d.chunk_count >= 1


def test_an_embedder_failure_marks_the_doc_failed_without_stopping(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Bir metin. " * 30)
    (corpus / "b.txt").write_text("Başka metin. " * 30)

    class _BrokenOnSecond:
        dim = 8
        _n = 0

        def embed_documents(self, texts):
            _BrokenOnSecond._n += 1
            if _BrokenOnSecond._n == 2:
                raise RuntimeError("endpoint down")
            from porsuk.core.models import EmbedResult

            return EmbedResult(dense=tuple((0.1,) * 8 for _ in texts))

        def embed_query(self, text):
            return self.embed_documents([text])

    from porsuk.adapters.store.inmemory import InMemoryStore
    from porsuk.core.config import ChunkingConfig, ProfileConfig, QualityConfig
    from porsuk.core.state import StateStore
    from porsuk.ingestion.pipeline import Pipeline
    from porsuk.ingestion.router import ParserRouter

    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="x")
    state = StateStore(tmp_path / "s.db")
    pipe = Pipeline(
        router=ParserRouter([TextParser()], QualityConfig()),
        embedder=_BrokenOnSecond(),
        store=store,
        llm=_FakeLLM(),
        state=state,
        chunking_cfg=ChunkingConfig(),
        profile_cfg=ProfileConfig(llm_enabled=False),
        parse_workers=1,
        embed_batch_size=4,
    )
    list(pipe.run(str(corpus)))
    counts = state.counts()
    assert counts.get("failed") == 1
    assert counts.get("profiled") == 1


def test_llm_profile_pass_does_not_pick_up_another_jobs_stranded_doc(tmp_path):
    """The API path shares one state DB across every index job
    (`<corpus_dir>/state.sqlite`). If job A's LLM-profile pass is interrupted
    (crash, exception) leaving one of its docs stranded at `embedded`, a
    later job B run against a *different* folder must not pick that doc up:
    same containment rule the resume sweep above already applies to
    `parsed`/`embedded` docs, but the LLM-profile loop skipped it entirely,
    so job B's progress counts included doc counts that were never its own."""
    from porsuk.ingestion.pipeline import content_hash, document_id_for

    shared_state = tmp_path / "shared.db"

    job_a_dir = tmp_path / "job-a"
    job_a_dir.mkdir()
    stranded = job_a_dir / "stranded.txt"
    stranded.write_text("Job A'nın yarım kalan belgesi burada. " * 15)

    router = ParserRouter([TextParser()], QualityConfig())
    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="fake:fake:8:norm=True")
    state = StateStore(shared_state)
    doc_id = document_id_for(str(stranded))
    state.upsert_document(doc_id, str(stranded), content_hash(str(stranded)))
    state.set_status(doc_id, "parsed", chunk_count=1)
    state.set_status(doc_id, "embedded")  # simulates job A crashing mid-profile

    job_b_dir = tmp_path / "job-b"
    job_b_dir.mkdir()
    (job_b_dir / "own.txt").write_text("Job B'nin kendi belgesi burada. " * 15)

    pipe_b = Pipeline(
        router=router,
        embedder=FakeEmbedder(dim=8),
        store=store,
        llm=_FakeLLM(),
        state=state,  # same state DB as job A
        chunking_cfg=ChunkingConfig(),
        profile_cfg=ProfileConfig(llm_enabled=True),
        parse_workers=1,
        embed_batch_size=4,
        job_id="job-b",
    )
    events = list(pipe_b.run(str(job_b_dir)))

    assert events[-1].total == 1  # only job B's own file
    assert events[-1].profiled == 1  # not 2: job A's stranded doc excluded
    # job A's doc is untouched, still stranded at `embedded`, not silently
    # profiled under job B's run.
    (a_status,) = [d for d in state.documents_with_status("embedded") if d.document_id == doc_id]
    assert a_status.document_id == doc_id


def test_an_llm_profile_failure_marks_only_that_doc_failed_without_stopping(tmp_path):
    """A bad LLM response (or a transient endpoint error) on one document must
    not abort the whole index run: the pipeline's own docstring calls
    partial failure "normal at scale", and the earlier
    parse/embed stages already honour that. The LLM-profiling pass must too:
    other documents still get profiled, and this one is recorded as
    `failed`, not left to crash `pipeline.run()` for everyone."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("İyi bir belge metni burada. " * 20)
    (corpus / "b_bad.txt").write_text("Bozuk olacak belge metni burada. " * 20)

    class _BrokenOnBadFile:
        model = "fake"

        def complete(self, prompt, *, max_tokens=512, enable_thinking=True):
            if "Bozuk" in prompt:
                raise RuntimeError("endpoint returned garbage")
            return "Özet cümlesi.\nKONULAR: a, b\nTİP: rapor"

    pipe, _store, state = _pipeline(tmp_path, llm_enabled=True)
    pipe._llm = _BrokenOnBadFile()  # same seam _FakeLLM is injected through
    list(pipe.run(str(corpus)))

    counts = state.counts()
    assert counts.get("profiled") == 1
    assert counts.get("failed") == 1


def test_a_doc_stranded_at_embedded_is_finished_on_resume(tmp_path):
    """A crash (process killed, not an exception) after set_status('embedded')
    leaves a doc with no profile. With llm_enabled=False stage 3 does not run,
    so only the resume sweep can finish it."""
    from porsuk.ingestion.pipeline import content_hash, document_id_for

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.txt"
    f.write_text("Bir sözleşme metni burada duruyor. " * 20)

    # simulate a killed run: the doc reached `embedded` and stopped there.
    _pipe, _store, state = _pipeline(tmp_path, llm_enabled=False)
    doc_id = document_id_for(str(f))
    state.upsert_document(doc_id, str(f), content_hash(str(f)))
    state.set_status(doc_id, "parsed", chunk_count=2)
    state.set_status(doc_id, "embedded")
    assert state.counts().get("embedded") == 1

    # a resumed run reuses that state db and must finish the document.
    pipe2, store2, state2 = _pipeline(tmp_path, llm_enabled=False)
    list(pipe2.run(str(corpus)))
    assert state2.counts().get("profiled") == 1
    assert state2.counts().get("embedded", 0) == 0
    assert len(store2.all_profile_ids()) == 1


def test_resume_loop_does_not_touch_another_jobs_stranded_document(tmp_path):
    """Two jobs share one state DB (the API path: `<corpus_dir>/state.sqlite`).
    Job B strands a doc at `embedded`. Job A then indexes a *different* folder.
    Job A's resume sweep must skip Job B's stranded doc: not re-parse it and
    not re-stamp its chunks with job_id='A' (cross-chat leak)."""
    from porsuk.ingestion.pipeline import content_hash, document_id_for

    shared_state = tmp_path / "state.sqlite"

    corpus_b = tmp_path / "corpus_b"
    corpus_b.mkdir()
    fb = corpus_b / "b.txt"
    fb.write_text("Job B'ye ait bir sözleşme metni. " * 20)

    # Job B: strand its doc at `embedded` (killed run, no profile).
    router_b = ParserRouter([TextParser()], QualityConfig())
    store_b = InMemoryStore()
    store_b.ensure_collections(embedding_fingerprint="fake:fake:8:norm=True")
    state_b = StateStore(shared_state)
    pipe_b = Pipeline(
        router=router_b,
        embedder=FakeEmbedder(dim=8),
        store=store_b,
        llm=_FakeLLM(),
        state=state_b,
        chunking_cfg=ChunkingConfig(),
        profile_cfg=ProfileConfig(llm_enabled=False),
        parse_workers=1,
        embed_batch_size=4,
        job_id="B",
    )
    doc_b = document_id_for(str(fb))
    state_b.upsert_document(doc_b, str(fb), content_hash(str(fb)))
    chunks_b = list(chunk_document(pipe_b._router.parse(str(fb)).document, doc_b, ChunkingConfig()))
    pipe_b._embed_and_store_chunks(chunks_b)
    state_b.set_status(doc_b, "embedded")
    assert all(c.job_id == "B" for c, _ in store_b._chunks.values())

    # Job A: a different folder, same shared state DB.
    corpus_a = tmp_path / "corpus_a"
    corpus_a.mkdir()
    (corpus_a / "a.txt").write_text("Job A'ya ait ayrı bir belge. " * 20)
    router_a = ParserRouter([TextParser()], QualityConfig())
    store_a = InMemoryStore()
    store_a.ensure_collections(embedding_fingerprint="fake:fake:8:norm=True")
    state_a = StateStore(shared_state)
    pipe_a = Pipeline(
        router=router_a,
        embedder=FakeEmbedder(dim=8),
        store=store_a,
        llm=_FakeLLM(),
        state=state_a,
        chunking_cfg=ChunkingConfig(),
        profile_cfg=ProfileConfig(llm_enabled=False),
        parse_workers=1,
        embed_batch_size=4,
        job_id="A",
    )
    list(pipe_a.run(str(corpus_a)))

    # Job B's stranded doc must still be `embedded`, Job A did not finish it,
    # and Job A's store holds no chunk for it.
    assert state_a.counts().get("embedded", 0) == 1
    assert not any(c.document_id == doc_b for c, _ in store_a._chunks.values()), (
        "Job A re-stamped Job B's stranded document"
    )
