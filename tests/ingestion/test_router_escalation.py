"""Escalation with the real escalation targets, not stubs. `test_router.py`
proves the mechanism against StubParsers (ordering, keep-best, never-raise);
this file proves the same rules hold once the targets are RapidOCR and
Docling, the parsers that actually cost something. The scanned-PDF climb and
the mojibake refusal live in `test_router_integration.py`, next to the rest
of the real-corpus runs; what is here is what only becomes testable with
real targets: keep-best when the expensive parser is genuinely worse, the
Docling path end to end, and the reason that path never fires on its own.
"""

import pytest

from porsuk.adapters.parsers.docling import DoclingParser
from porsuk.adapters.parsers.ocr import OCRParser
from porsuk.adapters.parsers.pymupdf import PyMuPDFParser
from porsuk.adapters.parsers.unavailable import UnavailableParser
from porsuk.core.config import QualityConfig
from porsuk.core.models import Block, Outline, ParsedDocument, ParseOutcome
from porsuk.ingestion.quality import decide, measure
from porsuk.ingestion.router import ParserRouter

_CLEAN_TURKISH = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
) * 4

needs_ocr = pytest.mark.skipif(
    issubclass(OCRParser, UnavailableParser), reason="no RapidOCR installed"
)
needs_docling = pytest.mark.skipif(
    issubclass(DoclingParser, UnavailableParser), reason="docling not installed"
)


@needs_ocr
def test_keep_best_holds_when_the_expensive_parser_is_worse(fixtures_dir):
    """An escalation is not automatically an improvement.

    Forcing the gate to escalate a *clean* text PDF is the only way to see
    this with real parsers: PyMuPDF reads it perfectly, OCR re-reads a
    rasterisation of it and does slightly worse. The router must keep the
    cheap result it already had.

    With stubs this rule is asserted against numbers a test invented. Here
    both scores come from the gate measuring two real parses of one real
    document.
    """
    force_escalation = QualityConfig(ocr_min_chars_per_page=100_000.0)
    router = ParserRouter([PyMuPDFParser(), OCRParser()], force_escalation)

    outcome = router.parse(str(fixtures_dir / "text_pdf" / "simple_tr.pdf"))

    assert outcome.parser_used == "pymupdf"
    assert outcome.document is not None
    # The gate still wanted OCR and OCR still ran; the result was kept only
    # because it scored higher, and the outcome says so rather than claiming
    # a clean parse.
    assert outcome.status == "degraded"
    assert "ocr" in (outcome.error or "")


@needs_docling
def test_the_gate_can_reach_docling_when_something_reports_an_empty_table(fixtures_dir):
    """The Docling path end to end, with the trigger supplied by hand.

    No parser in the tree emits an empty table block today (see the test
    below), so the cheap parser here is a stub that does. Everything after
    that point is real: the gate counts the empty block, names `docling`, the
    router resolves it out of the chain and Docling parses the file.

    What is asserted is that Docling was *reached*, not that it won. It does
    not win here and should not: the stub hands the gate 700 characters of
    clean prose (0.60) while Docling reads a small real table (0.568,
    measured 2026-09-07), so keep-best correctly prefers the cheap result and
    the outcome is downgraded to say the escalation did not improve on it.
    Asserting a winner would mean sizing the stub's text until the numbers
    came out right, which tests the arithmetic of a fixture rather than the
    chain.

    Without this, the entire Docling branch would be untested: reachable
    only through a signal that never fires.
    """
    parsed: list[str] = []

    class SpyDocling(DoclingParser):
        def parse(self, path):
            parsed.append(path)
            return super().parse(path)

    class ReportsAnEmptyTable:
        name = "stub-with-empty-table"
        cost = 1

        def can_handle(self, path):
            return True

        def parse(self, path):
            document = ParsedDocument(
                path=path,
                doc_type="pdf",
                blocks=(
                    # Long enough, and plausible enough, to clear every
                    # earlier trigger: `decide` checks text starvation and
                    # damage before it looks at tables, because Docling on a
                    # page with no text layer has nothing to work with.
                    Block(text=_CLEAN_TURKISH, page_no=1),
                    Block(text="", page_no=1, kind="table"),
                ),
                outline=Outline(),
                page_count=1,
            )
            return ParseOutcome(document=document, status="ok", quality=0.0, parser_used=self.name)

    target = str(fixtures_dir / "table_pdf" / "ruled_table.pdf")
    router = ParserRouter([ReportsAnEmptyTable(), SpyDocling()], QualityConfig())
    outcome = router.parse(target)

    assert parsed == [target], "the gate named docling but the router never ran it"
    assert outcome.status == "degraded"
    assert "docling" in (outcome.error or "")
    assert "empty" in (outcome.error or "")


def test_the_docling_trigger_cannot_fire_on_any_real_document(fixtures_dir):
    """Recorded, not tolerated: the Docling threshold is left open and this is
    the measurement that says a threshold is not what is missing.
    `empty_table_blocks` counts table blocks whose text came back empty. No
    parser can produce one: PyMuPDF emits no block at all where it found no
    text, so the count is structurally 0 rather than merely 0 today. Worse,
    the two parsers disagree about tables in opposite directions, asserted
    below. Measured 2026-09-07 across the generated corpus; the replacement
    signal (PyMuPDF reports 12 vector rectangles for ruled_table.pdf and 0
    for every other fixture) is calibrated against real documents separately,
    not here.
    """
    parser = PyMuPDFParser()

    for pdf in sorted(fixtures_dir.rglob("*.pdf")):
        outcome = parser.parse(str(pdf))
        if outcome.document is None:
            continue
        assert measure(outcome.document).empty_table_blocks == 0, (
            f"{pdf.name} produced an empty table block: the Docling trigger is "
            "reachable after all, and this test should become a real escalation test"
        )

    # A real ruled table is not recognised as tabular at all...
    ruled = parser.parse(str(fixtures_dir / "table_pdf" / "ruled_table.pdf"))
    assert {b.kind for b in ruled.document.blocks} == {"text"}
    assert decide(measure(ruled.document), QualityConfig()).escalate_to is None

    # ...while pipe-separated prose is.
    piped = parser.parse(str(fixtures_dir / "table_pdf" / "table_like.pdf"))
    assert {b.kind for b in piped.document.blocks} == {"table"}
