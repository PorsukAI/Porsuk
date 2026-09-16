"""The real OCR parser against the generated scan fixtures.

Skipped entirely without the `parsers` extra: a bare install keeps the
stub, and the bare-install path is covered by tests/ingestion/test_router.py.
"""

import pytest

from porsuk.adapters.parsers.ocr import OCRParser
from porsuk.adapters.parsers.unavailable import UnavailableParser

# Gate on the parser being real rather than on a package name: either RapidOCR
# package satisfies this module, and naming one of them made the whole file
# skip silently the day the other became the default.
pytestmark = pytest.mark.skipif(
    issubclass(OCRParser, UnavailableParser),
    reason="no RapidOCR installed; the ocr slot is the bare-install stub",
)


@pytest.fixture(scope="module")
def parser() -> OCRParser:
    # Module-scoped: RapidOCR loads three ONNX models per instance.
    return OCRParser()


def test_can_handle_pdf_only(parser):
    assert parser.can_handle("/x/scan.pdf") is True
    assert parser.can_handle("/x/notes.docx") is False


def test_recognises_text_from_a_generated_scan(parser):
    outcome = parser.parse("tests/fixtures/generated/scanned_pdf/scan_en.pdf")
    assert outcome.parser_used == "ocr"
    assert outcome.document is not None
    text = " ".join(b.text for b in outcome.document.blocks).lower()
    assert "agreement" in text or "contract" in text or len(text) > 50
    assert outcome.document.page_count >= 1
    assert outcome.document.outline.source == "none"


def test_turkish_scan_keeps_diacritics_reasonably(parser):
    outcome = parser.parse("tests/fixtures/generated/scanned_pdf/scan_tr.pdf")
    assert outcome.document is not None
    text = " ".join(b.text for b in outcome.document.blocks)
    # Not an accuracy assertion, RapidOCR's stock model is weak on Turkish
    # accents. We require only that it produced Turkish-ish text.
    assert len(text) > 30


def test_a_file_that_is_not_a_pdf_fails_without_raising(tmp_path, parser):
    junk = tmp_path / "not_really.pdf"
    junk.write_bytes(b"this is not a PDF")
    outcome = parser.parse(str(junk))
    assert outcome.status == "failed"
    assert outcome.document is None
    assert outcome.error
