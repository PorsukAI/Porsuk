"""Tests for the `porsuk` CLI: parse, index, and search commands."""

import json
from pathlib import Path

import pytest

from porsuk.api.cli import main


def test_parses_a_clean_pdf_and_reports_the_gate(fixtures_dir, capsys):
    code = main(["parse", str(fixtures_dir / "text_pdf" / "simple_tr.pdf")])
    out = capsys.readouterr().out
    assert code == 0
    assert "pymupdf" in out
    assert "ok" in out
    assert "text_plausibility" in out, "the components must be visible, not just the score"


def test_json_output_is_machine_readable(fixtures_dir, capsys):
    code = main(["parse", str(fixtures_dir / "text_pdf" / "simple_tr.pdf"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["parser_used"] == "pymupdf"
    assert payload["status"] == "ok"
    assert 0.0 <= payload["quality"] <= 1.0
    assert set(payload["components"]) == {
        "chars_per_page",
        "bad_char_ratio",
        "text_plausibility",
        "empty_table_blocks",
    }
    assert payload["chunk_count"] > 0
    assert payload["outline_source"] in {"toc", "pattern", "none"}


def test_json_reports_an_escalation_that_was_fulfilled(fixtures_dir, capsys):
    """A scanned PDF now climbs pymupdf -> ocr and lands on real text.

    The escalation block reports a request the chain could *not* satisfy, so
    a fulfilled climb shows up as `parser_used`, not there.
    """
    ocr = pytest.importorskip("porsuk.adapters.parsers.ocr", reason="needs the parsers extra")
    if not hasattr(ocr.OCRParser, "_recognise"):
        pytest.skip("the ocr slot is the bare-install stub")

    main(["parse", str(fixtures_dir / "scanned_pdf" / "scan_tr.pdf"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["parser_used"] == "ocr"
    assert payload["status"] == "ok"
    assert payload["escalation"] is None


def test_json_names_an_escalation_it_could_not_fulfil(fixtures_dir, tmp_path, capsys):
    """A score says a document parsed badly, never why.

    Uses a chain with no `ocr` in it, which is also the shape of a bare
    install: the gate still asks, and the answer has to say what it asked for
    and what triggered it.
    """
    # A profile is validated whole, not merged over the defaults, so this
    # starts from the real one and only takes `ocr` out of the chain.
    base = Path("config/local.yaml").read_text(encoding="utf-8")
    profile = tmp_path / "no_ocr.yaml"
    profile.write_text(
        base + "\nparsing:\n  parsers: [pymupdf, docx, xlsx, pptx, text]\n", encoding="utf-8"
    )
    main(
        [
            "parse",
            str(fixtures_dir / "scanned_pdf" / "scan_tr.pdf"),
            "--config",
            str(profile),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "degraded"
    assert payload["escalation"]["requested"] == "ocr"
    assert payload["escalation"]["trigger"] == "chars_per_page"
    assert payload["escalation"]["reason"]


def test_a_broken_file_exits_non_zero_without_a_traceback(fixtures_dir, capsys):
    code = main(["parse", str(fixtures_dir / "broken" / "zero_byte.pdf")])
    out = capsys.readouterr().out
    assert code == 1
    assert "failed" in out
    assert "Traceback" not in out


def test_a_missing_file_reports_cleanly(tmp_path, capsys):
    code = main(["parse", str(tmp_path / "nope.pdf")])
    assert code == 1
    assert "Traceback" not in capsys.readouterr().out


def test_an_outlined_pdf_reports_its_outline_source(fixtures_dir, capsys):
    main(["parse", str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"), "--json"])
    assert json.loads(capsys.readouterr().out)["outline_source"] == "toc"


def test_config_flag_selects_the_profile(fixtures_dir, capsys):
    code = main(
        [
            "parse",
            str(fixtures_dir / "text_pdf" / "simple_tr.pdf"),
            "--config",
            "config/local.yaml",
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["parser_used"] == "pymupdf"


def test_chunks_flag_prints_the_chunks(fixtures_dir, capsys):
    main(["parse", str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"), "--chunks"])
    out = capsys.readouterr().out
    assert "Mali Hükümler" in out


def test_no_command_exits_non_zero(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_malformed_config_exits_non_zero_without_a_traceback(fixtures_dir, tmp_path, capsys):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("llm: {provider: fake\n  broken: [unclosed\n", encoding="utf-8")
    code = main(
        [
            "parse",
            str(fixtures_dir / "text_pdf" / "simple_tr.pdf"),
            "--config",
            str(bad_config),
        ]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_json_output_includes_the_chunks_when_asked(capsys, fixtures_dir):
    """`--chunks --json` silently ignored --chunks: the payload carried
    chunk_count and never the chunks, so the machine-readable form could not
    answer the question the human-readable one could."""
    main(["parse", str(fixtures_dir / "pattern_pdf" / "madde_numbering.pdf"), "--json", "--chunks"])
    payload = json.loads(capsys.readouterr().out)
    assert "chunks" in payload
    assert len(payload["chunks"]) == payload["chunk_count"]
    assert payload["chunks"][0]["text"]
    assert "section_path" in payload["chunks"][0]


def test_json_omits_chunks_unless_asked(capsys, fixtures_dir):
    main(["parse", str(fixtures_dir / "pattern_pdf" / "madde_numbering.pdf"), "--json"])
    assert "chunks" not in json.loads(capsys.readouterr().out)


def test_a_failed_parse_still_reports_the_path_it_was_given(capsys, tmp_path):
    """`path` came from the document, which a failed parse does not have, so
    it printed None, exactly when a batch run most needs to know which file."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"")
    main(["parse", str(broken), "--json"])
    assert json.loads(capsys.readouterr().out)["path"] == str(broken)


def test_index_command_reports_progress(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Bir belge metni burada duruyor. " * 30)
    (corpus / "b.txt").write_text("Another document body goes here. " * 30)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "parsing: {parsers: [text]}\n"
        "profile: {llm_enabled: false}\n"
    )
    rc = main(["index", str(corpus), "--config", str(cfg), "--state", str(tmp_path / "s.db")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2/2" in out or "profiled" in out


def test_index_command_nonzero_when_all_fail(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.bin").write_bytes(b"\x00\x00\x00")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "parsing: {parsers: [text]}\n"
        "profile: {llm_enabled: false}\n"
    )
    rc = main(["index", str(corpus), "--config", str(cfg), "--state", str(tmp_path / "s.db")])
    assert rc != 0


_SEARCH_CFG = (
    "llm: {provider: fake, model: fake}\n"
    "embedder: {provider: fake, dim: 8}\n"
    "store: {provider: inmemory}\n"
    "parsing: {parsers: [text]}\n"
    "profile: {llm_enabled: false}\n"
    "retrieval: {strategy: semantic}\n"
)


def _populated_retriever():
    from porsuk.adapters.embedding.fake import FakeEmbedder
    from porsuk.adapters.store.inmemory import InMemoryStore
    from porsuk.core.config import RetrievalConfig
    from porsuk.core.models import Chunk, EmbedResult
    from porsuk.retrieval.search import Retriever

    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="x")
    chunks = [
        Chunk(
            "c1", "d1", "ödeme koşulları ve gecikme faizi", 1, "tr", "Madde 1", "Madde 1", (0, 30)
        ),
        Chunk("c2", "d2", "fesih ve tazminat hükümleri", 1, "tr", "Madde 5", "Madde 5", (0, 27)),
    ]
    emb = FakeEmbedder(dim=8)
    store.upsert_chunks(
        chunks,
        EmbedResult(dense=tuple(emb.embed_documents([c.text for c in chunks]).dense)),
    )
    return Retriever(
        embedder=emb, store=store, reranker=None, cfg=RetrievalConfig(strategy="semantic")
    )


def _empty_retriever():
    from porsuk.adapters.embedding.fake import FakeEmbedder
    from porsuk.adapters.store.inmemory import InMemoryStore
    from porsuk.core.config import RetrievalConfig
    from porsuk.retrieval.search import Retriever

    store = InMemoryStore()
    store.ensure_collections(embedding_fingerprint="x")
    return Retriever(
        embedder=FakeEmbedder(dim=8),
        store=store,
        reranker=None,
        cfg=RetrievalConfig(strategy="semantic"),
    )


def test_search_command_human_output(monkeypatch, tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(_SEARCH_CFG, encoding="utf-8")
    monkeypatch.setattr("porsuk.api.cli.build_retriever", lambda cfg: _populated_retriever())

    rc = main(["search", "ödeme", "--config", str(cfg_file), "--k", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Madde 1" in out or "ödeme" in out
    assert "[1]" in out


def test_search_command_json(monkeypatch, tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(_SEARCH_CFG, encoding="utf-8")
    monkeypatch.setattr("porsuk.api.cli.build_retriever", lambda cfg: _populated_retriever())

    rc = main(["search", "ödeme", "--config", str(cfg_file), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert isinstance(payload["hits"], list)
    assert payload["hits"]
    hit = payload["hits"][0]
    for key in ("chunk_id", "score", "section_path", "text", "retrieval_method"):
        assert key in hit


def test_search_k_defaults_to_config_chunk_k(monkeypatch, tmp_path, capsys):
    from porsuk.core.config import RetrievalConfig
    from porsuk.retrieval.search import Retriever

    seen = {}

    class _RecordingRetriever(Retriever):
        def search(self, query, *, k, filters, doc_ids=None, strategy=None):
            seen["k"] = k
            return []

    def _make(cfg):
        return _RecordingRetriever(embedder=None, store=None, reranker=None, cfg=RetrievalConfig())

    monkeypatch.setattr("porsuk.api.cli.build_retriever", _make)

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "retrieval: {strategy: semantic, chunk_k: 7}\n",
        encoding="utf-8",
    )

    assert main(["search", "q", "--config", str(cfg_file)]) == 0
    assert seen["k"] == 7

    assert main(["search", "q", "--config", str(cfg_file), "--k", "3"]) == 0
    assert seen["k"] == 3


def test_search_command_no_results(monkeypatch, tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(_SEARCH_CFG, encoding="utf-8")
    monkeypatch.setattr("porsuk.api.cli.build_retriever", lambda cfg: _empty_retriever())

    rc = main(["search", "ödeme", "--config", str(cfg_file)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no results" in out


def test_search_warns_when_store_is_inmemory(tmp_path, capsys):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(_SEARCH_CFG, encoding="utf-8")

    rc = main(["search", "x", "--config", str(cfg_file)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "inmemory" in captured.err
    assert "persist" in captured.err


def test_search_config_error_exits_nonzero(tmp_path, capsys):
    rc = main(["search", "x", "--config", str(tmp_path / "nonexistent.yaml")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_index_command_nothing_to_do(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Sabit metin. " * 20)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "llm: {provider: fake, model: fake}\n"
        "embedder: {provider: fake, dim: 8}\n"
        "store: {provider: inmemory}\n"
        "parsing: {parsers: [text]}\n"
        "profile: {llm_enabled: false}\n"
    )
    state = tmp_path / "s.db"
    main(["index", str(corpus), "--config", str(cfg), "--state", str(state)])
    capsys.readouterr()
    rc = main(["index", str(corpus), "--config", str(cfg), "--state", str(state)])
    assert rc == 0
    assert "nothing to index" in capsys.readouterr().out
