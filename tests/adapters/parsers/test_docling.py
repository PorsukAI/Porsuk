"""The real Docling parser against the generated fixtures.

Skipped without the `parsers` extra. Slow: each test parses a PDF through
layout and table-structure models, so the parser is module-scoped and the
fixtures are deliberately tiny.
"""

import pytest

from porsuk.adapters.parsers.docling import DoclingParser
from porsuk.adapters.parsers.unavailable import UnavailableParser

pytestmark = pytest.mark.skipif(
    issubclass(DoclingParser, UnavailableParser),
    reason="docling not installed; the docling slot is the bare-install stub",
)


@pytest.fixture(scope="module")
def parser() -> DoclingParser:
    return DoclingParser()


def test_can_handle_pdf_only(parser):
    assert parser.can_handle("/x/report.pdf") is True
    assert parser.can_handle("/x/sheet.xlsx") is False


def test_a_ruled_table_becomes_a_table_block_carrying_its_cells(parser, fixtures_dir):
    """The capability Docling is in the chain for.

    Asserts the values *and* that they survive as flattened rows: a table
    block whose text lost the column a value sat under would still count as a
    table block while being useless to the embedder.
    """
    outcome = parser.parse(str(fixtures_dir / "table_pdf" / "ruled_table.pdf"))
    assert outcome.parser_used == "docling"
    assert outcome.document is not None, outcome.error

    tables = [b for b in outcome.document.blocks if b.kind == "table"]
    assert len(tables) == 1, [b.kind for b in outcome.document.blocks]
    text = tables[0].text
    for value in ("Fatura No", "Tutar", "Vade", "F-001", "15000", "2024-04-01"):
        assert value in text, f"{value!r} missing from {text!r}"
    assert "F-001,15000,2024-04-01" in text, f"cells lost their row: {text!r}"


def test_pipe_separated_text_is_not_called_a_table(parser, fixtures_dir):
    """The negative control, and a real difference from PyMuPDF.

    table_like.pdf is "a | b | c" lines. PyMuPDF's separator regex labels
    those tabular; Docling looks for visual structure, finds none, and calls
    it prose. Docling is right, and this records that the two parsers
    disagree here on purpose.
    """
    outcome = parser.parse(str(fixtures_dir / "table_pdf" / "table_like.pdf"))
    assert outcome.document is not None, outcome.error
    assert [b.kind for b in outcome.document.blocks] == ["text"]


def test_outlined_pdf_gets_a_heading_hierarchy(parser, fixtures_dir):
    outcome = parser.parse(str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"))
    assert outcome.document is not None, outcome.error
    # "pattern", not "toc": the hierarchy is inferred from layout, not read
    # from an authored table of contents.
    assert outcome.document.outline.source == "pattern"
    assert len(outcome.document.outline.nodes) >= 1
    assert outcome.document.outline.nodes[0].title == "Taraflar"


def test_a_file_that_is_not_a_pdf_fails_without_raising(tmp_path, parser):
    """A Result, not an exception, or one bad file stops the whole run."""
    junk = tmp_path / "not_really.pdf"
    junk.write_bytes(b"this is not a PDF")
    outcome = parser.parse(str(junk))
    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.error


def test_the_converter_is_not_built_until_something_is_parsed():
    """Docling loads torch-backed models. Building the chain must not pay for
    that on a run where nothing escalates."""
    assert DoclingParser()._converter is None
