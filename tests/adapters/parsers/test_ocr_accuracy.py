"""How well OCR actually reads Turkish, measured rather than asserted.

The fixtures are rasterised from text this repository generates, so the
ground truth is known exactly, which makes a real accuracy number possible
here and nowhere else in the suite. This is that measurement, kept as a test
so a model or configuration change cannot quietly undo it. The thresholds
are floors with margin, not the measured values.
"""

import difflib

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

# Copied from scripts/fixtures/generate.py, which rasterises exactly this text
# into scanned_pdf/. Duplicated deliberately: importing it would make the test
# pass even if the generator changed underneath the fixtures.
TURKISH_BODY = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
)
ENGLISH_BODY = (
    "This agreement is entered into between the parties for the provision of "
    "supply services. Payments shall be made within 30 days of the invoice date."
)


@pytest.fixture(scope="module")
def parser() -> OCRParser:
    return OCRParser()


def _first_page_text(parser: OCRParser, path: str) -> str:
    outcome = parser.parse(path)
    assert outcome.document is not None, outcome.error
    return " ".join(b.text for b in outcome.document.blocks if b.page_no == 1)


def _similarity(got: str, truth: str) -> float:
    return difflib.SequenceMatcher(None, " ".join(got.split()), truth).ratio()


def test_turkish_scan_is_read_accurately(parser):
    """0.973 measured 2026-09-07 with the Latin PP-OCRv5 recogniser. The
    legacy Chinese/English recogniser managed 0.912 and could not produce a
    single Turkish letter."""
    text = _first_page_text(parser, "tests/fixtures/generated/scanned_pdf/scan_tr.pdf")
    assert _similarity(text, TURKISH_BODY) > 0.93


def test_english_scan_is_read_accurately(parser):
    """1.000 measured 2026-09-07."""
    text = _first_page_text(parser, "tests/fixtures/generated/scanned_pdf/scan_en.pdf")
    assert _similarity(text, ENGLISH_BODY) > 0.95


def test_turkish_letters_survive_the_round_trip(parser):
    """The specific thing the legacy recogniser could not do at all: its
    dictionary contained no ş ğ ı İ ö ü ç, so "içinde" came back "iginde"."""
    text = _first_page_text(parser, "tests/fixtures/generated/scanned_pdf/scan_tr.pdf")
    for word in ("sözleşme", "akdedilmiştir", "Ödemeler", "gün", "içinde"):
        assert word in text, f"{word!r} missing from {text!r}"


@pytest.mark.xfail(
    reason="dotless ı is absent from the Latin dictionary; needs a Turkish "
    "fine-tune, not configuration. Recorded so the day a model "
    "fixes it, this test says so.",
    strict=False,
)
def test_dotless_i_is_still_wrong(parser):
    text = _first_page_text(parser, "tests/fixtures/generated/scanned_pdf/scan_tr.pdf")
    assert "arasında" in text and "yapılır" in text
