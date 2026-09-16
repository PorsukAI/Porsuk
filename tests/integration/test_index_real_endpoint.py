"""End-to-end index against a live bge-m3 endpoint.

Skipped unless EMBEDDING_BASE_URL is set (infra/vast_provision.py up writes
it). Uses the in-memory Qdrant, so no Docker is needed - only the embedder
is real. Run it once after bringing the GPU up to confirm the whole chain
works against a genuine model.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.needs_network

_BASE_URL = os.environ.get("EMBEDDING_BASE_URL")


_SPARSE_URL = os.environ.get("EMBEDDING_SPARSE_BASE_URL")


def _config(tmp_path, *, with_sparse: bool):
    sparse_line = f"  sparse_base_url: {_SPARSE_URL}\n" if with_sparse and _SPARSE_URL else ""
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder:\n"
        "  provider: openai_compatible\n"
        f"  model: {os.environ.get('EMBEDDING_MODEL', 'bge-m3')}\n"
        "  dim: 1024\n"
        f"  base_url: {_BASE_URL}\n"
        f"{sparse_line}"
        "store: {provider: qdrant, url: ':memory:'}\n"
        "parsing: {parsers: [pymupdf, docx, xlsx, pptx, text]}\n"
        "profile: {llm_enabled: false}\n"
    )
    return cfg_file


@pytest.mark.skipif(_BASE_URL is None, reason="EMBEDDING_BASE_URL not set (no live embedder)")
def test_index_generated_fixtures_against_real_bge_m3(tmp_path):
    from porsuk.core.config import load_config
    from porsuk.core.container import build_pipeline

    cfg = load_config(_config(tmp_path, with_sparse=False))
    pipeline = build_pipeline(cfg, str(tmp_path / "s.db"))
    events = list(pipeline.run("tests/fixtures/generated"))

    assert events, "no progress events - nothing was indexed"
    final = events[-1]
    assert final.total >= 15
    assert final.profiled >= 8  # the clean fixtures
    assert final.failed <= 6  # broken.* and some damaged fixtures


@pytest.mark.skipif(
    _BASE_URL is None or _SPARSE_URL is None,
    reason="EMBEDDING_BASE_URL / EMBEDDING_SPARSE_BASE_URL not set",
)
def test_real_sparse_vectors_are_embedded_and_stored(tmp_path):
    from porsuk.core.config import load_config
    from porsuk.core.container import build_app
    from porsuk.ingestion.chunking import chunk as chunk_document
    from porsuk.ingestion.pipeline import document_id_for

    app = build_app(load_config(_config(tmp_path, with_sparse=True)))
    path = "tests/fixtures/generated/text_pdf/simple_tr.pdf"
    outcome = app.router.parse(path)
    doc_id = document_id_for(path)
    chunks = list(chunk_document(outcome.document, doc_id, app.config.chunking))

    vectors = app.embedder.embed_documents([c.text for c in chunks])
    assert vectors.sparse is not None
    assert len(vectors.sparse) == len(chunks)

    app.store.upsert_chunks(chunks, vectors)
    stored = app.store.sparse_vector_for(chunks[0].chunk_id)
    assert stored, "sparse vector was not stored"
    assert all(isinstance(k, int) and v > 0 for k, v in stored.items())
