"""Tests for the parse quality gate: scoring, normalisation, and escalation decisions."""

import pytest

from porsuk.core.config import QualityConfig
from porsuk.core.models import Block, Outline, ParsedDocument, ParseQualityComponents
from porsuk.ingestion.quality import GateDecision, decide, measure, normalise, score
from porsuk.ingestion.text_stats import bad_char_ratio, script_validity, text_plausibility

CLEAN_TR = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında "
    "tedarik hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura "
    "tarihinden itibaren 30 gün içinde yapılır."
)


def _document(blocks, page_count=1, image_count=0):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=tuple(blocks),
        outline=Outline(),
        page_count=page_count,
        image_count=image_count,
    )


def _components(**overrides):
    base = {
        "chars_per_page": 900.0,
        "bad_char_ratio": 0.0,
        "text_plausibility": 0.8,
        "empty_table_blocks": 0,
    }
    base.update(overrides)
    return ParseQualityComponents(**base)


def test_measure_divides_characters_by_pages():
    document = _document([Block(text="x" * 600, page_no=1)], page_count=3)
    assert measure(document).chars_per_page == pytest.approx(200.0)


def test_measure_survives_a_zero_page_document():
    """A parser can return a document it could not paginate; dividing by
    page_count must not take down the pipeline."""
    document = _document([Block(text="abc", page_no=1)], page_count=0)
    assert measure(document).chars_per_page == 0.0


def test_measure_counts_table_blocks_that_came_back_empty():
    """A table block detected but empty is the Docling trigger."""
    blocks = [
        Block(text="", page_no=1, kind="table"),
        Block(text="   \n ", page_no=1, kind="table"),
        Block(text="A | B", page_no=1, kind="table"),
        Block(text="", page_no=1, kind="text"),
    ]
    assert measure(_document(blocks)).empty_table_blocks == 2


def test_normalise_ramps_linearly_and_clamps_at_both_ends():
    assert normalise(800.0, bad=50.0, good=800.0) == pytest.approx(1.0)
    assert normalise(50.0, bad=50.0, good=800.0) == pytest.approx(0.0)
    assert normalise(425.0, bad=50.0, good=800.0) == pytest.approx(0.5)
    assert normalise(5000.0, bad=50.0, good=800.0) == pytest.approx(1.0)
    assert normalise(0.0, bad=50.0, good=800.0) == pytest.approx(0.0)


def test_normalise_handles_a_descending_pair():
    """bad_char_ratio and empty_table_blocks are costs: less is better, so
    `good` is numerically below `bad`."""
    assert normalise(0.0, bad=0.10, good=0.0) == pytest.approx(1.0)
    assert normalise(0.10, bad=0.10, good=0.0) == pytest.approx(0.0)
    assert normalise(0.05, bad=0.10, good=0.0) == pytest.approx(0.5)


def test_normalise_does_not_divide_by_zero_on_a_degenerate_pair():
    assert normalise(1.0, bad=0.5, good=0.5) == 0.0


def test_score_stays_inside_the_unit_interval():
    cfg = QualityConfig()
    assert 0.0 <= score(_components(), cfg) <= 1.0
    worst = _components(
        chars_per_page=0.0, bad_char_ratio=1.0, text_plausibility=0.0, empty_table_blocks=10
    )
    assert score(worst, cfg) == pytest.approx(0.0)


def test_score_is_not_dominated_by_the_largest_unit_component():
    """The reason normalisation is required.

    chars_per_page is measured in hundreds and the others in 0-1. Unnormalised,
    a document with excellent chars_per_page and everything else broken would
    outscore a clean one.
    """
    cfg = QualityConfig()
    huge_but_broken = _components(
        chars_per_page=100_000.0, bad_char_ratio=1.0, text_plausibility=0.0
    )
    modest_but_clean = _components(chars_per_page=800.0)
    assert score(modest_but_clean, cfg) > score(huge_but_broken, cfg)


def test_low_characters_per_page_escalates_to_ocr_and_names_its_trigger():
    cfg = QualityConfig()
    decision = decide(_components(chars_per_page=5.0), cfg)
    assert decision.escalate_to == "ocr"
    assert decision.trigger == "chars_per_page"
    assert "chars_per_page" in decision.reason


def test_damaged_text_escalates_to_ocr():
    cfg = QualityConfig()
    assert decide(_components(bad_char_ratio=0.9), cfg).trigger == "bad_char_ratio"
    assert decide(_components(text_plausibility=0.05), cfg).trigger == "text_plausibility"


def test_empty_table_blocks_escalate_to_docling():
    cfg = QualityConfig()
    decision = decide(_components(empty_table_blocks=2), cfg)
    assert decision.escalate_to == "docling"
    assert decision.trigger == "empty_table_blocks"


def test_a_clean_document_is_not_escalated():
    decision = decide(_components(), QualityConfig())
    assert decision == GateDecision(
        escalate_to=None, trigger=None, reason="no escalation trigger fired", image_heavy=False
    )


def test_image_heavy_needs_images_as_well_as_little_text():
    """A scanned page and an image-heavy page both have little
    text. Only the image count separates them, which is why the flag is not
    derived from chars_per_page alone."""
    cfg = QualityConfig()
    empty = measure(_document([Block(text="", page_no=1)]))
    assert decide(empty, cfg, image_count=0).image_heavy is False
    assert decide(empty, cfg, image_count=12).image_heavy is True


def test_measure_uses_real_text_statistics():
    """measure must delegate to text_stats rather than reimplement it."""
    clean = measure(_document([Block(text=CLEAN_TR, page_no=1)]))
    garbage = measure(_document([Block(text="Iıl ııı tlıc rn1 vvhh zzz", page_no=1)]))
    assert clean.text_plausibility > garbage.text_plausibility


def test_thresholds_come_from_config_not_from_literals():
    """A threshold change in config must change the decision."""
    permissive = QualityConfig(ocr_min_chars_per_page=1.0)
    strict = QualityConfig(ocr_min_chars_per_page=10_000.0)
    components = _components(chars_per_page=500.0)
    assert decide(components, permissive).escalate_to is None
    assert decide(components, strict).escalate_to == "ocr"


def _mojibake(text: str) -> str:
    """The real-world damage: UTF-8 bytes read back as Latin-1."""
    return text.encode("utf-8").decode("latin-1")


REAL_TURKISH = (
    "İşbu sözleşme, ABC Lojistik A.Ş. ile XYZ Sanayi Ltd. Şti. arasında tedarik "
    "hizmetlerinin sağlanması amacıyla akdedilmiştir. Ödemeler fatura tarihinden "
    "itibaren 30 gün içinde yapılır. Taraflar, uyuşmazlıklarda İstanbul "
    "mahkemelerinin yetkili olduğunu kabul eder."
)


def test_realistic_turkish_mojibake_is_flagged_unrecoverable():
    """`text_plausibility` cannot catch this. `script_validity` is the only one
    of its three signals that sees mojibake at all, and averaging dilutes it to
    a third: realistic Turkish mojibake measures ~0.65 against a 0.57
    threshold, so it passed. The fixture escalated only because its source
    sentence is ~40% diacritics; ordinary Turkish is 5-10%.

    Mojibake is the failure `script_validity` was added for, and the single
    most likely Turkish-specific corruption in a real 5000-file folder.

    It is flagged rather than escalated: the page renders the wrong glyphs, so
    OCR reads the wrong glyphs back (measured 2026-09-07, see `decide`)."""
    damaged = _mojibake(REAL_TURKISH)
    assert text_plausibility(damaged) > QualityConfig().ocr_min_text_plausibility, (
        "precondition: the blended score does not catch this"
    )
    components = ParseQualityComponents(
        chars_per_page=float(len(damaged)),
        bad_char_ratio=bad_char_ratio(damaged),
        text_plausibility=text_plausibility(damaged),
        empty_table_blocks=0,
    )
    gate = decide(components, QualityConfig(), script_validity=script_validity(damaged))
    assert gate.trigger == "script_validity"
    assert gate.unrecoverable is True
    assert gate.escalate_to is None, "OCR cannot undo an encoding bug"


def test_clean_turkish_is_not_escalated_by_the_script_check():
    """Negative control. Real mevzuat measures 0.998-0.999 and the lowest clean
    document in either corpus is hyphenated.pdf at 0.989."""
    components = ParseQualityComponents(
        chars_per_page=float(len(REAL_TURKISH)),
        bad_char_ratio=0.0,
        text_plausibility=text_plausibility(REAL_TURKISH),
        empty_table_blocks=0,
    )
    gate = decide(components, QualityConfig(), script_validity=script_validity(REAL_TURKISH))
    assert gate.escalate_to is None
    assert gate.unrecoverable is False
