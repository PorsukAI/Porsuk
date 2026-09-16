"""Tests for the PyMuPDF PDF parser adapter."""

import pytest

from porsuk.adapters.parsers.pymupdf import PyMuPDFParser
from porsuk.core.ports import Parser


@pytest.fixture
def parser():
    return PyMuPDFParser()


def test_conforms_to_the_parser_protocol(parser):
    assert isinstance(parser, Parser)


def test_handles_pdfs_and_nothing_else(parser):
    assert parser.can_handle("/tmp/a.pdf")
    assert parser.can_handle("/tmp/A.PDF")
    assert not parser.can_handle("/tmp/a.docx")
    assert not parser.can_handle("/tmp/a")


def test_parses_a_text_pdf_into_blocks(parser, fixtures_dir):
    outcome = parser.parse(str(fixtures_dir / "text_pdf" / "simple_tr.pdf"))
    assert outcome.status == "ok"
    assert outcome.parser_used == "pymupdf"
    assert outcome.document is not None
    assert outcome.document.page_count == 2
    assert outcome.document.blocks
    text = "\n".join(b.text for b in outcome.document.blocks)
    assert "sözleşme" in text


def test_turkish_orthography_survives(parser, fixtures_dir):
    """Guards the font fix from the parser side: if the parser
    mangles these, text_plausibility scores clean documents as damaged."""
    outcome = parser.parse(str(fixtures_dir / "text_pdf" / "simple_tr.pdf"))
    text = "\n".join(b.text for b in outcome.document.blocks)
    for char in "İşğıŞ":
        assert char in text


def test_reads_an_embedded_outline(parser, fixtures_dir):
    """Tier 1: section boundaries come from the embedded TOC."""
    outcome = parser.parse(str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"))
    outline = outcome.document.outline
    assert outline.source == "toc"
    titles = [n.title for n in outline.nodes]
    assert "Taraflar" in titles
    assert "Mali Hükümler" in titles


def test_outline_nests_by_level(parser, fixtures_dir):
    """The fixture's third entry is level 2 under 'Mali Hükümler'."""
    outcome = parser.parse(str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"))
    by_title = {n.title: n for n in outcome.document.outline.nodes}
    assert [c.title for c in by_title["Mali Hükümler"].children] == ["Ödeme Koşulları"]


def test_a_pdf_without_a_toc_reports_source_none(parser, fixtures_dir):
    """An absent outline must not be fabricated: tier 2 needs to know."""
    outcome = parser.parse(str(fixtures_dir / "pattern_pdf" / "madde_numbering.pdf"))
    assert outcome.document.outline.source == "none"
    assert outcome.document.outline.nodes == ()


def test_scanned_pdf_parses_but_yields_almost_no_text(parser, fixtures_dir):
    """It must succeed structurally: the gate, not the parser, decides this
    needs OCR."""
    outcome = parser.parse(str(fixtures_dir / "scanned_pdf" / "scan_tr.pdf"))
    assert outcome.document is not None
    text = "".join(b.text for b in outcome.document.blocks)
    assert len(text.strip()) < 20


def test_counts_images(parser, fixtures_dir):
    """image_heavy needs this; a rasterised scan is all image."""
    outcome = parser.parse(str(fixtures_dir / "scanned_pdf" / "scan_tr.pdf"))
    assert outcome.document.image_count > 0


@pytest.mark.parametrize("name", ["zero_byte.pdf", "truncated.pdf", "password_protected.pdf"])
def test_broken_files_fail_without_raising(parser, fixtures_dir, name):
    """Partial failure is normal at scale, so it is a Result."""
    outcome = parser.parse(str(fixtures_dir / "broken" / name))
    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.error
    assert outcome.parser_used == "pymupdf"


def test_a_missing_file_fails_without_raising(parser, tmp_path):
    outcome = parser.parse(str(tmp_path / "nope.pdf"))
    assert outcome.status == "failed"
    assert outcome.error


def test_tags_table_like_blocks(parser, fixtures_dir):
    """The gate's empty_table_blocks component is meaningless unless something
    is tagged as a table."""
    outcome = parser.parse(str(fixtures_dir / "table_pdf" / "table_like.pdf"))
    assert any(b.kind == "table" for b in outcome.document.blocks)
