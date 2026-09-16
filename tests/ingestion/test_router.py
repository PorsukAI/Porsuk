"""Tests for the parser router: cost ordering, escalation, and never-raise."""

from porsuk.core.config import QualityConfig
from porsuk.core.models import Block, Outline, ParsedDocument, ParseOutcome
from porsuk.ingestion.router import ParserRouter

CLEAN = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
) * 4


def _document(text, page_count=1, image_count=0, kind="text"):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=(Block(text=text, page_no=1, kind=kind),),
        outline=Outline(),
        page_count=page_count,
        image_count=image_count,
    )


class StubParser:
    """A parser whose behaviour each test dictates."""

    def __init__(self, name, cost, outcome=None, raises=None, handles=True, can_handle_raises=None):
        self.name = name
        self.cost = cost
        self._outcome = outcome
        self._raises = raises
        self._handles = handles
        self._can_handle_raises = can_handle_raises
        self.calls = 0

    def can_handle(self, path):
        if self._can_handle_raises is not None:
            raise self._can_handle_raises
        return self._handles

    def parse(self, path):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._outcome


def _ok(name, text, **kw):
    return ParseOutcome(
        document=_document(text, **kw), status="ok", quality=0.0, parser_used=name, error=None
    )


def _failed(name, error="boom"):
    return ParseOutcome(document=None, status="failed", quality=0.0, parser_used=name, error=error)


def test_uses_the_cheapest_parser_that_can_handle_the_file():
    cheap = StubParser("cheap", 5, _ok("cheap", CLEAN))
    dear = StubParser("dear", 100, _ok("dear", CLEAN))
    router = ParserRouter([dear, cheap], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.parser_used == "cheap"
    assert dear.calls == 0


def test_skips_parsers_that_cannot_handle_the_file():
    wrong = StubParser("wrong", 1, _ok("wrong", CLEAN), handles=False)
    right = StubParser("right", 5, _ok("right", CLEAN))
    router = ParserRouter([wrong, right], QualityConfig())
    assert router.parse("/tmp/x.pdf").parser_used == "right"
    assert wrong.calls == 0


def test_no_parser_can_handle_the_file():
    router = ParserRouter([StubParser("a", 1, _ok("a", CLEAN), handles=False)], QualityConfig())
    outcome = router.parse("/tmp/x.zip")
    assert outcome.status == "failed"
    assert "no parser" in outcome.error


def test_scores_the_outcome_it_returns():
    """Parsers report quality 0.0; the router is what runs the gate."""
    router = ParserRouter([StubParser("cheap", 5, _ok("cheap", CLEAN))], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.quality > 0.0
    assert outcome.components is not None
    assert outcome.components.text_plausibility > 0.0


def test_escalates_when_the_gate_asks_and_a_higher_cost_parser_exists():
    cheap = StubParser("cheap", 5, _ok("cheap", "", page_count=10))
    ocr = StubParser("ocr", 100, _ok("ocr", CLEAN))
    router = ParserRouter([cheap, ocr], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.parser_used == "ocr"
    assert outcome.status == "ok"


def test_keeps_the_better_result_when_escalation_makes_things_worse():
    """The rule this behavior gained in 31e383b: ocr and docling are stubs
    that always fail, so without it every scanned PDF would come back failed
    instead of coming back as the low-quality result the gate correctly
    flagged."""
    cheap = StubParser("cheap", 5, _ok("cheap", "kısa metin", page_count=50))
    ocr = StubParser("ocr", 100, _failed("ocr", "not installed"))
    router = ParserRouter([cheap, ocr], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.parser_used == "cheap"
    assert outcome.document is not None
    assert ocr.calls == 1, "the escalation must actually be attempted"


def test_an_unfulfillable_escalation_downgrades_the_status():
    """The document parsed, but the gate wanted a parser we do not have. That
    is not the same as a clean parse."""
    cheap = StubParser("cheap", 5, _ok("cheap", "kısa", page_count=50))
    ocr = StubParser("ocr", 100, _failed("ocr", "not installed"))
    router = ParserRouter([cheap, ocr], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "degraded"
    assert "ocr" in outcome.error


def test_status_is_degraded_when_no_escalation_target_exists_at_all():
    cheap = StubParser("cheap", 5, _ok("cheap", "kısa", page_count=50))
    router = ParserRouter([cheap], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "degraded"
    assert outcome.document is not None


def test_a_parser_that_raises_becomes_a_failed_outcome():
    """One bad file must not stop 5000."""
    boom = StubParser("boom", 5, raises=RuntimeError("segfault-ish"))
    router = ParserRouter([boom], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "failed"
    assert "RuntimeError" in outcome.error


def test_falls_through_to_the_next_parser_when_the_cheapest_fails():
    broken = StubParser("broken", 5, _failed("broken"))
    working = StubParser("working", 10, _ok("working", CLEAN))
    router = ParserRouter([broken, working], QualityConfig())
    assert router.parse("/tmp/x.pdf").parser_used == "working"


def test_all_parsers_failing_returns_the_first_failure():
    a = StubParser("a", 5, _failed("a", "first reason"))
    b = StubParser("b", 10, _failed("b", "second reason"))
    outcome = ParserRouter([a, b], QualityConfig()).parse("/tmp/x.pdf")
    assert outcome.status == "failed"
    assert "first reason" in outcome.error


def test_no_parser_is_tried_twice():
    """Guards the escalation loop against re-entering a parser and spinning."""
    cheap = StubParser("cheap", 5, _ok("cheap", "kısa", page_count=50))
    ocr = StubParser("ocr", 100, _ok("ocr", "hâlâ kısa", page_count=50))
    router = ParserRouter([cheap, ocr], QualityConfig())
    router.parse("/tmp/x.pdf")
    assert cheap.calls == 1
    assert ocr.calls == 1


def test_image_heavy_reaches_the_outcome():
    cheap = StubParser("cheap", 5, _ok("cheap", "", page_count=3, image_count=9))
    router = ParserRouter([cheap], QualityConfig())
    assert router.parse("/tmp/x.pdf").image_heavy is True


def test_a_parser_whose_can_handle_raises_is_skipped_and_a_later_parser_still_handles_the_file():
    """can_handle is part of the Parser protocol, so it is part of the surface
    the never-raise guarantee has to cover, not just parse()."""
    hostile = StubParser("hostile", 5, can_handle_raises=RuntimeError("can_handle blew up"))
    right = StubParser("right", 10, _ok("right", CLEAN))
    router = ParserRouter([hostile, right], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.parser_used == "right"
    assert outcome.status == "ok"


def test_when_every_can_handle_raises_parse_returns_a_failed_outcome_not_an_exception():
    a = StubParser("a", 5, can_handle_raises=RuntimeError("boom-a"))
    b = StubParser("b", 10, can_handle_raises=ValueError("boom-b"))
    router = ParserRouter([a, b], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "failed"
    assert outcome.document is None
    # Recorded, not silent: a can_handle that always throws should not look
    # identical to a parser that legitimately never matches.
    assert "boom-a" in outcome.error
    assert "boom-b" in outcome.error


def test_can_handle_failures_stay_bounded_by_parser_count_not_file_count():
    """Round 2 fix: the router outlives one file (build_router hands back a
    single instance reused for the whole corpus), so a systematically broken
    parser must not retain one entry per file parsed. The bound is the number
    of registered parsers, not the number of times parse() was called."""
    hostile = StubParser("hostile", 5, can_handle_raises=RuntimeError("always throws"))
    right = StubParser("right", 10, _ok("right", CLEAN))
    router = ParserRouter([hostile, right], QualityConfig())

    for _ in range(5):
        outcome = router.parse("/tmp/x.pdf")
        assert outcome.parser_used == "right"

    assert len(router.can_handle_failures) == 1
    failure = router.can_handle_failures["hostile"]
    assert failure.count == 5
    assert "always throws" in failure.first_error


def test_a_parser_returning_a_non_outcome_does_not_escape():
    """`_safe_parse` caught exceptions but never validated the return value.
    The never-raise guarantee exists to survive the parser nobody has written
    yet, and the OCR and Docling wrappers are exactly that parser.
    `StubParser` even defaults `outcome=None`, so the hole was one line away."""
    router = ParserRouter([StubParser("broken", 5, outcome=None)], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "failed"
    assert "broken" in (outcome.error or "")


def test_a_failed_escalation_does_not_fall_through_to_another_parser():
    """`_escalation_target`'s own docstring calls this unacceptable: falling
    back to 'any pricier parser' silently routes a document the gate wanted
    OCR'd into Docling. Free while both are stubs; once real, it is a
    torch-backed pass nobody asked for, on every scanned page."""
    scanned = StubParser("pymupdf", 10, _ok("pymupdf", "", page_count=2))
    ocr = StubParser("ocr", 100, _failed("ocr", "not implemented"))
    docling = StubParser("docling", 200, _ok("docling", CLEAN))
    router = ParserRouter([scanned, ocr, docling], QualityConfig())
    router.parse("/tmp/x.pdf")
    assert ocr.calls == 1, "the gate asked for OCR"
    assert docling.calls == 0, "docling ran without the gate ever asking for it"


def test_the_reported_decision_belongs_to_the_outcome_that_was_kept():
    """`decision` was rebound on every parse while `best` was chosen by
    quality, so when they diverged the router reported the LAST gate decision
    against a DIFFERENT document, announcing `ok` over a parse the gate had
    flagged. Unreachable while the escalation targets are stubs; live the
    moment one returns a document."""
    rich_but_flagged = ParseOutcome(
        document=ParsedDocument(
            path="/tmp/x.pdf",
            doc_type="pdf",
            # Rich text, plus a table whose cells came back empty: the gate
            # asks for Docling while the score stays high.
            blocks=(
                Block(text=CLEAN, page_no=1, kind="text"),
                Block(text="", page_no=1, kind="table"),
            ),
            outline=Outline(),
            page_count=1,
        ),
        status="ok",
        quality=0.0,
        parser_used="pymupdf",
        error=None,
    )
    first = StubParser("pymupdf", 10, rich_but_flagged)
    second = StubParser("docling", 200, _ok("docling", "kısa", page_count=8))
    router = ParserRouter([first, second], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")

    assert outcome.parser_used == "pymupdf", "keep-best should have kept the richer parse"
    assert outcome.status == "degraded", "reported clean while its own gate asked to escalate"
    # The reason must name the escalation THIS document's gate asked for.
    # Reporting the last parser's verdict happened to produce "degraded" too,
    # by a different route and with the wrong target named: a diagnosis
    # pointing at the wrong symptom is exactly what this guards against.
    assert "docling" in (outcome.error or "")
    assert "ocr" not in (outcome.error or "")


def test_a_failed_escalations_own_reason_reaches_the_user():
    """`unavailable.py` writes a careful reason naming what is missing and when
    it lands, and it was thrown away. The user pointing the CLI at a scanned
    PDF saw only that an escalation 'did not produce a better result', which
    never says OCR is unimplemented rather than broken."""
    scanned = StubParser("pymupdf", 10, _ok("pymupdf", "", page_count=2))
    reason = "ocr parser is not available: OCR is not yet implemented"
    ocr = StubParser("ocr", 100, _failed("ocr", reason))
    router = ParserRouter([scanned, ocr], QualityConfig())
    outcome = router.parse("/tmp/x.pdf")
    assert outcome.status == "degraded"
    assert "not yet implemented" in (outcome.error or ""), outcome.error
