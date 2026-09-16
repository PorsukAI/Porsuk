"""Tests that the generated fixture corpus matches its declared manifest."""

import yaml

MANIFEST = "tests/fixtures/manifest.yaml"


def _manifest():
    with open(MANIFEST, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["categories"]


def test_every_category_meets_its_minimum_count(fixtures_dir):
    for name, spec in _manifest().items():
        found = list((fixtures_dir / name).glob("*"))
        assert len(found) >= spec["min_count"], (
            f"category {name!r} has {len(found)} files, needs {spec['min_count']}"
        )


def test_non_broken_fixtures_are_non_empty(fixtures_dir):
    for name in _manifest():
        if name == "broken":
            continue
        for path in (fixtures_dir / name).glob("*"):
            assert path.stat().st_size > 0, path


# scan_with_garbage_layer.pdf is a scan that deliberately *does* carry a text
# layer, so it is excluded from the no-text assertion below rather than
# weakening that assertion for every scan.
_SCANS_WITHOUT_A_TEXT_LAYER = ("scan_tr.pdf", "scan_en.pdf")


def test_scanned_pdfs_have_almost_no_extractable_text(fixtures_dir):
    """The whole point of a synthetic scan: it must look scanned."""
    import pymupdf as fitz

    for name in _SCANS_WITHOUT_A_TEXT_LAYER:
        path = fixtures_dir / "scanned_pdf" / name
        with fitz.open(path) as doc:
            text = "".join(page.get_text() for page in doc)
        assert len(text.strip()) < 20, f"{path} still has a text layer"


def test_outlined_pdf_has_an_embedded_toc(fixtures_dir):
    import pymupdf as fitz

    path = next((fixtures_dir / "outlined_pdf").glob("*.pdf"))
    with fitz.open(path) as doc:
        assert len(doc.get_toc()) > 0


def test_pattern_pdf_has_no_embedded_toc_but_has_numbering(fixtures_dir):
    import pymupdf as fitz

    pattern_dir = fixtures_dir / "pattern_pdf"
    for path in pattern_dir.glob("*.pdf"):
        with fitz.open(path) as doc:
            assert doc.get_toc() == [], f"{path} has an embedded TOC"

    with fitz.open(pattern_dir / "madde_numbering.pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    assert "MADDE" in text


def test_broken_set_covers_the_three_failure_shapes(fixtures_dir):
    names = {p.name for p in (fixtures_dir / "broken").glob("*")}
    assert "zero_byte.pdf" in names
    assert "truncated.pdf" in names
    assert "password_protected.pdf" in names


def test_generation_is_idempotent(fixtures_dir):
    from scripts.fixtures.generate import generate_fixtures

    before = sorted(p.name for p in fixtures_dir.rglob("*") if p.is_file())
    generate_fixtures(fixtures_dir)
    after = sorted(p.name for p in fixtures_dir.rglob("*") if p.is_file())
    assert before == after


def test_turkish_text_survives_the_pdf_round_trip(fixtures_dir):
    """The parse quality gate scores `text_plausibility` on Turkish + English
    function words and alphabet share, deliberately not a dictionary, which
    agglutination makes useless.

    The `text_pdf` category is the "PyMuPDF happy path", so it has to contain
    real Turkish orthography. ASCII-fied text ("Isbu", "arasinda") would score
    as damaged under the very metric that is supposed to treat it as clean.
    """
    import pymupdf as fitz

    path = fixtures_dir / "text_pdf" / "simple_tr.pdf"
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    for char in "İşğıŞ":
        assert char in text, f"{char!r} did not survive the PDF round-trip"
    assert "?" not in text, "a glyph was substituted with '?': font lacks Turkish coverage"


def test_damaged_text_fixture_carries_control_characters(fixtures_dir):
    """`bad_char_ratio` counts U+FFFD and control characters.

    This fixture is the only thing in the corpus that gives that component a
    signal. It uses control characters rather than U+FFFD because PyMuPDF's
    text API cannot round-trip U+FFFD, see the note in generate.py.
    """
    import pymupdf as fitz

    path = fixtures_dir / "damaged_text_pdf" / "control_chars.pdf"
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    control = [c for c in text if ord(c) < 32 and c not in "\n\r\t"]
    assert control, "no control characters survived; bad_char_ratio has no fixture signal"


def test_a_scan_carries_a_garbage_text_layer(fixtures_dir):
    """The rasterised scans have exactly zero
    characters by construction, so any threshold above zero passes them. Real
    scans usually carry a garbage text layer, and that is the case a threshold
    has to be calibrated for."""
    import pymupdf as fitz

    path = fixtures_dir / "scanned_pdf" / "scan_with_garbage_layer.pdf"
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    assert len(text.strip()) > 40, "there must be enough text to look like a text layer"

    from porsuk.ingestion.text_stats import text_plausibility

    assert text_plausibility(text) < 0.4, "but it must not read as language"


def test_a_mojibake_fixture_exists(fixtures_dir):
    """Closes the other half of the KNOWN GAP: PyMuPDF cannot round-trip
    U+FFFD, but mojibake exercises damage that bad_char_ratio cannot see at
    all, because every character in it is individually valid."""
    import pymupdf as fitz

    from porsuk.ingestion.text_stats import bad_char_ratio, script_validity

    path = fixtures_dir / "damaged_text_pdf" / "mojibake.pdf"
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    assert "Ã" in text
    assert bad_char_ratio(text) < 0.01, "mojibake is invisible to bad_char_ratio"
    assert script_validity(text) < 0.95, "script_validity is what catches it"


def test_a_hyphenated_fixture_exists(fixtures_dir):
    """Real Turkish PDFs hyphenate across line breaks while being parsed
    perfectly correctly. If that alone trips the gate, clean mevzuat gets
    escalated to OCR."""
    import pymupdf as fitz

    path = fixtures_dir / "damaged_text_pdf" / "hyphenated.pdf"
    with fitz.open(path) as doc:
        text = "".join(page.get_text() for page in doc)
    assert "-\n" in text or "-" in text
