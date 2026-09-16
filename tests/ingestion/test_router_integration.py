"""The router against the real parsers and the real fixture corpus."""

import pytest

from porsuk.core.config import load_config
from porsuk.core.container import build_router


def _ocr_is_stubbed() -> bool:
    from porsuk.adapters.parsers.unavailable import UnavailableParser
    from porsuk.core.registry import build

    return isinstance(build("parser", {"provider": "ocr"}), UnavailableParser)


@pytest.fixture(scope="module")
def router():
    return build_router(load_config("config/local.yaml", env={}))


def test_a_clean_turkish_pdf_parses_ok(router, fixtures_dir):
    outcome = router.parse(str(fixtures_dir / "text_pdf" / "simple_tr.pdf"))
    assert outcome.status == "ok"
    assert outcome.parser_used == "pymupdf"
    assert outcome.quality > 0.5


def test_a_scanned_pdf_climbs_from_pymupdf_to_ocr(router, fixtures_dir):
    """The escalation the whole chain exists for.

    PyMuPDF gets almost nothing off a raster scan, the gate fires on
    chars_per_page and names `ocr`, and the OCR result outscores the cheap one
    so keep-best takes it. Previously this same fixture came back `degraded`
    from `pymupdf` because the target was a stub.
    """
    if _ocr_is_stubbed():
        pytest.skip("ocr is the bare-install stub")
    outcome = router.parse(str(fixtures_dir / "scanned_pdf" / "scan_tr.pdf"))
    assert outcome.status == "ok"
    assert outcome.parser_used == "ocr"
    assert outcome.document is not None
    assert outcome.document.blocks
    assert outcome.error is None


def test_a_clean_pdf_does_not_climb(router, fixtures_dir):
    """The negative control for the test above. Without this, a router that
    escalated everything would still pass."""
    outcome = router.parse(str(fixtures_dir / "text_pdf" / "simple_en.pdf"))
    assert outcome.parser_used == "pymupdf"
    assert outcome.status == "ok"


def test_mojibake_is_reported_rather_than_laundered_through_ocr(router, fixtures_dir):
    """Encoding damage is not a rasterisation problem.

    The page renders the wrong glyphs, so an OCR pass reads the wrong glyphs
    back, and because those are individually valid characters it would clear
    every threshold and come back `ok`. The gate flags it and keeps the cheap
    parser's text, which at least still carries the evidence.
    """
    outcome = router.parse(str(fixtures_dir / "damaged_text_pdf" / "mojibake.pdf"))
    assert outcome.status == "degraded"
    assert outcome.parser_used == "pymupdf"
    assert "script_validity" in (outcome.error or "")


@pytest.mark.parametrize("name", ["zero_byte.pdf", "truncated.pdf", "password_protected.pdf"])
def test_broken_files_fail_cleanly(router, fixtures_dir, name):
    outcome = router.parse(str(fixtures_dir / "broken" / name))
    assert outcome.status == "failed"
    assert outcome.document is None


@pytest.mark.parametrize(
    ("category", "filename", "expected_parser"),
    [
        ("docx", "styled_headings.docx", "docx"),
        ("xlsx", "two_sheets.xlsx", "xlsx"),
        ("pptx", "two_slides.pptx", "pptx"),
    ],
)
def test_office_formats_route_to_their_parser(
    router, fixtures_dir, category, filename, expected_parser
):
    outcome = router.parse(str(fixtures_dir / category / filename))
    assert outcome.parser_used == expected_parser
    assert outcome.status in {"ok", "degraded"}
