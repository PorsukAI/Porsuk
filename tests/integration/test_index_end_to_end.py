"""Full pipeline against the generated fixtures, with fake providers.

No network, no GPU, no Docker: the fake embedder and in-memory Qdrant, but
real parsers, real chunking, real profiler over the actual fixture files.
This is the always-on end-to-end check; the live-endpoint version is in
test_index_real_endpoint.py.
"""

from __future__ import annotations

from porsuk.core.config import load_config
from porsuk.core.container import build_pipeline


def test_index_generated_fixtures(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: qdrant, url: ':memory:'}\n"
        "parsing: {parsers: [pymupdf, docx, xlsx, pptx, text]}\n"
        "profile: {llm_enabled: false}\n"
        "pipeline: {parse_workers: 1}\n"
    )
    pipeline = build_pipeline(load_config(cfg_file), str(tmp_path / "s.db"))
    events = list(pipeline.run("tests/fixtures/generated"))

    assert events, "nothing was indexed"
    final = events[-1]
    assert final.total >= 15
    # the clean fixtures profile; the broken.* ones fail; neither stops the run
    assert final.profiled >= 8
    assert final.failed >= 1
    assert final.profiled + final.failed == final.total


def test_index_is_resumable_and_hash_cached(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: qdrant, url: ':memory:'}\n"
        "parsing: {parsers: [pymupdf, docx, xlsx, pptx, text]}\n"
        "profile: {llm_enabled: false}\n"
        "pipeline: {parse_workers: 1}\n"
    )
    state = str(tmp_path / "s.db")
    first = list(build_pipeline(load_config(cfg_file), state).run("tests/fixtures/generated"))
    second = list(build_pipeline(load_config(cfg_file), state).run("tests/fixtures/generated"))

    assert first[-1].profiled >= 8
    assert second == []  # every file already current
