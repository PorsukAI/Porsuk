"""Tests for tier-2 pattern-based chunking (MADDE/BOLUM/EK numbering)."""

import pytest

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Outline, ParsedDocument
from porsuk.ingestion.chunking import chunk
from porsuk.ingestion.chunking.pattern import (
    _MAX_HEADING_CHARS,
    heading_level,
    sections_from_patterns,
)

BODY = "Ödemeler fatura tarihinden itibaren otuz gün içinde yapılır."


def _document(blocks, outline=None):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=tuple(blocks),
        outline=outline or Outline(source="none"),
        page_count=2,
    )


@pytest.mark.parametrize(
    ("text", "level"),
    [
        ("BİRİNCİ BÖLÜM", 1),
        ("İKİNCİ BÖLÜM", 1),
        ("BIRINCI BOLUM", 1),
        ("MADDE 1", 2),
        ("MADDE 12 -", 2),
        ("Madde 3 —", 2),
        ("EK-1", 1),
        ("EK-2 Teslimat Programı", 1),
        ("1. Genel Hükümler", 1),
        ("1.1 Kapsam", 2),
        ("3.2.1 Ödeme Koşulları", 3),
    ],
)
def test_recognised_heading_patterns(text, level):
    assert heading_level(text) == level


@pytest.mark.parametrize(
    "text",
    [
        BODY,
        "Ödemeler 30 gün içinde yapılır ve fatura kesilir.",
        "2024 yılında imzalanmıştır.",
        "",
        "   ",
    ],
)
def test_body_text_is_not_a_heading(text):
    assert heading_level(text) is None


def test_a_numbered_line_with_a_long_tail_is_not_a_heading():
    """'1. ' opening a long paragraph is a list item, not a section heading."""
    assert heading_level("1. " + BODY * 3) is None


def test_diacritic_stripped_bolum_is_recognised():
    """PDF extraction sometimes loses the diacritic. A pattern that only
    matched the correct spelling would miss exactly the damaged documents that
    most need structure."""
    assert heading_level("ÜÇÜNCÜ BOLUM") == 1


def test_madde_documents_get_one_section_per_article():
    blocks = [
        Block(text="BİRİNCİ BÖLÜM", page_no=1),
        Block(text="MADDE 1 -", page_no=1),
        Block(text=BODY, page_no=1),
        Block(text="MADDE 2 -", page_no=2),
        Block(text=BODY, page_no=2),
    ]
    sections = sections_from_patterns(_document(blocks))
    assert [s.title for s in sections] == ["BİRİNCİ BÖLÜM", "MADDE 1 -", "MADDE 2 -"]


def test_section_path_nests_articles_under_their_chapter():
    blocks = [
        Block(text="BİRİNCİ BÖLÜM", page_no=1),
        Block(text="MADDE 1 -", page_no=1),
        Block(text=BODY, page_no=1),
    ]
    sections = {s.title: s.path for s in sections_from_patterns(_document(blocks))}
    assert sections["MADDE 1 -"] == "BİRİNCİ BÖLÜM > MADDE 1 -"


def test_decimal_numbering_nests_by_depth():
    blocks = [
        Block(text="1. Genel Hükümler", page_no=1),
        Block(text=BODY, page_no=1),
        Block(text="1.1 Kapsam", page_no=1),
        Block(text=BODY, page_no=1),
    ]
    sections = {s.title: s.path for s in sections_from_patterns(_document(blocks))}
    assert sections["1.1 Kapsam"] == "1. Genel Hükümler > 1.1 Kapsam"


def test_a_short_bold_block_is_promoted_to_a_heading():
    """The weaker signal: font weight only promotes an already short and
    standalone block."""
    blocks = [
        Block(text="Mali Hükümler", page_no=1, bold=True, font_size=16.0),
        Block(text=BODY, page_no=1, font_size=11.0),
    ]
    sections = sections_from_patterns(_document(blocks))
    assert sections is not None
    assert sections[0].title == "Mali Hükümler"


def test_a_long_bold_block_is_not_a_heading():
    """A bold run mid-paragraph is emphasis, not structure."""
    blocks = [Block(text=BODY * 3, page_no=1, bold=True, font_size=16.0)]
    assert sections_from_patterns(_document(blocks)) is None


def test_returns_none_when_no_pattern_is_found():
    """Tier 2 must decline cleanly so tier 3 gets its turn."""
    blocks = [Block(text=BODY, page_no=1), Block(text=BODY, page_no=2)]
    assert sections_from_patterns(_document(blocks)) is None


def test_blocks_before_the_first_heading_are_kept():
    blocks = [
        Block(text="KAPAK", page_no=1),
        Block(text="MADDE 1 -", page_no=1),
        Block(text=BODY, page_no=1),
    ]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title is None
    assert sections[0].blocks[0].text == "KAPAK"


def test_every_block_ends_up_in_exactly_one_section():
    blocks = [
        Block(text="KAPAK", page_no=1),
        Block(text="BİRİNCİ BÖLÜM", page_no=1),
        Block(text="MADDE 1 -", page_no=1),
        Block(text=BODY, page_no=1),
    ]
    document = _document(blocks)
    claimed = [id(b) for s in sections_from_patterns(document) for b in s.blocks]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == {id(b) for b in document.blocks}


def test_tier_two_is_skipped_when_a_real_outline_exists():
    """Tier order matters: a real TOC beats an inferred one."""
    from porsuk.core.models import OutlineNode

    blocks = [
        Block(text="MADDE 1 -", page_no=1),
        Block(text=BODY, page_no=1),
    ]
    outline = Outline(nodes=(OutlineNode(title="MADDE 1 -", level=1, page_no=1),), source="toc")
    chunks = chunk(_document(blocks, outline), "doc1", ChunkingConfig())
    assert all(c.section_path == "MADDE 1 -" for c in chunks)


def test_tier_two_applies_to_the_real_mevzuat_fixture(fixtures_dir):
    """The corpus category tier 2 exists for."""
    from porsuk.adapters.parsers.pymupdf import PyMuPDFParser

    outcome = PyMuPDFParser().parse(str(fixtures_dir / "pattern_pdf" / "madde_numbering.pdf"))
    chunks = chunk(outcome.document, "doc1", ChunkingConfig())
    assert chunks
    assert any(c.section_path and "MADDE" in c.section_path for c in chunks)


def test_numeric_sections_fixture_is_chunked_by_pattern(fixtures_dir):
    from porsuk.adapters.parsers.pymupdf import PyMuPDFParser

    outcome = PyMuPDFParser().parse(str(fixtures_dir / "pattern_pdf" / "numeric_sections.pdf"))
    chunks = chunk(outcome.document, "doc1", ChunkingConfig())
    assert any(c.section_path and "Genel Hükümler" in c.section_path for c in chunks)


def test_a_marker_fused_to_its_article_body_titles_as_the_bare_marker():
    """PyMuPDF routinely extracts 'MADDE 1 - <the whole article>' as one
    block with no line break in between (how mevzuat PDFs are laid out).
    Using the whole block as the title would leak ~200 characters of
    duplicated paragraph text into every chunk's section_path."""
    blocks = [Block(text="MADDE 1 - " + BODY * 3, page_no=1)]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title == "MADDE 1"


def test_a_short_caption_after_a_marker_is_kept_whole():
    """The opposite direction of the fused-marker case: a real short caption
    must not be truncated down to the bare marker."""
    blocks = [Block(text="EK-2 Teslimat Programı", page_no=1), Block(text=BODY, page_no=1)]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title == "EK-2 Teslimat Programı"


def test_pattern_pdf_fixture_section_paths_stay_short(fixtures_dir):
    """Regression guard: a marker fused to its body must not leak paragraph
    text into section_path. 100 chars comfortably covers a real nested path
    of trimmed titles ('BİRİNCİ BÖLÜM > MADDE 12') while still catching the
    ~200-char paragraph leak this pins against."""
    from porsuk.adapters.parsers.pymupdf import PyMuPDFParser

    for name in ("madde_numbering.pdf", "numeric_sections.pdf"):
        outcome = PyMuPDFParser().parse(str(fixtures_dir / "pattern_pdf" / name))
        chunks = chunk(outcome.document, "doc1", ChunkingConfig())
        for c in chunks:
            if c.section_path:
                assert len(c.section_path) <= 100, (name, c.section_path)


# --- Fix round 2: real mevzuat.gov.tr PDFs exposed two further defects. ---


def test_a_fused_marker_trims_even_when_well_under_the_heading_cap():
    """Finding (a): trimming must fire on MARKER PRESENCE, not on total
    length. This real example from mevzuat.gov.tr's 6098 (TBK) is only 66
    characters, comfortably under the 80-char heading cap, yet its
    remainder is a sentence fragment, not a caption, cut off mid-clause by a
    PDF line wrap ('...bir kimsenin diğerine karşı' trails into the next
    block). A length-gated trim never engages here; a marker-gated one must."""
    text = "MADDE 607- Ömür boyu gelir sözleşmesi, bir kimsenin diğerine karşı"
    assert len(text) < _MAX_HEADING_CHARS
    blocks = [Block(text=text, page_no=1)]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title == "MADDE 607"


def test_a_short_dashed_caption_survives_the_marker_gated_trim():
    """The opposite direction of the previous test: a marker followed by a
    genuinely short caption, joined by a dash, as real captions are, must
    not be trimmed just because trimming is now marker-gated rather than
    length-gated."""
    blocks = [Block(text="MADDE 5 - Fesih ve Tazminat", page_no=1), Block(text=BODY, page_no=1)]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title == "MADDE 5 - Fesih ve Tazminat"


@pytest.mark.parametrize(
    "text",
    [
        "1. Asıl borç ile borçlunun kusur veya temerrüdünün yasal sonuçları.",
        "2. Alacaklı, dava veya def'i yoluyla mahkemeye veya hakeme başvurmuşsa, icra",
    ],
)
def test_a_numbered_sentence_inside_an_article_is_not_a_heading(text):
    """Finding (b): both examples are real numbered list items from
    mevzuat.gov.tr's 6098 (TBK), sitting inside an article body, not
    section headings. Both are under the 80-char heading cap (67 and 76
    chars), so only a caption-shaped check on the remainder, not the
    existing total-length cap, catches them."""
    assert len(text) <= _MAX_HEADING_CHARS
    assert heading_level(text) is None


def test_a_short_numbered_caption_is_still_a_heading():
    """The opposite direction of the previous test: a bare 'N.' followed by
    a genuine short caption (a real sub-heading style used throughout TBK)
    must still be recognised, not swept up by the new stricter check."""
    assert heading_level("4. Yanılmada kusur") == 1


def test_a_numbered_list_item_does_not_become_a_spurious_section_ancestor():
    """End-to-end version of finding (b): a numbered list item preceding a
    real MADDE marker in the same document must not itself open a section,
    otherwise it becomes the article's ancestor in section_path, which is
    the defect this pins against."""
    blocks = [
        Block(
            text="1. Asıl borç ile borçlunun kusur veya temerrüdünün yasal sonuçları.",
            page_no=1,
        ),
        Block(text="MADDE 590- " + BODY, page_no=1),
    ]
    sections = sections_from_patterns(_document(blocks))
    assert sections[0].title is None
    assert sections[1].title == "MADDE 590"
    assert sections[1].path == "MADDE 590"


@pytest.mark.parametrize(
    "line",
    [
        "01.01.2024 tarihinden itibaren yürürlüğe girer.",
        "12.03.2024 tarihli Resmî Gazete'de yayımlanmıştır.",
        "1.000.000 TL tutarındaki teminat iade edilir.",
        "1.500 TL ödenir.",
    ],
)
def test_turkish_dates_and_amounts_are_not_headings(line):
    """A dotted marker was treated as strong evidence on its own, exempt from
    the caption check. But `dd.mm.yyyy` dates and `.`-grouped amounts open
    lines far more often in Turkish legal and commercial prose than `1.1`
    sub-headings do, and each one started a spurious section."""
    assert heading_level(line) is None


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("GEÇİCİ MADDE 1 - Bu kanunun yürürlüğü.", 2),
        ("GECICI MADDE 2 - Diacritic dropped by extraction.", 2),
        ("EK MADDE 1 - İlave hükümler.", 2),
        ("ON BİRİNCİ BÖLÜM", 1),
        ("ONBİRİNCİ BÖLÜM", 1),
        ("ON İKİNCİ BOLUM", 1),
        ("EK 1 - Teslimat Programı", 1),
    ],
)
def test_the_other_turkish_heading_forms_are_recognised(line, level):
    """GEÇİCİ MADDE and EK MADDE appear in essentially every Turkish kanun,
    and `^MADDE` excludes both. Ordinals past ONUNCU and an EK without its
    hyphen were equally invisible."""
    assert heading_level(line) == level


def test_a_decimal_subsection_is_still_a_heading():
    """Negative control for the date fix: real dotted numbering must survive."""
    assert heading_level("1.1 Kapsam") == 2
    assert heading_level("3.2.1 Teslimat Koşulları") == 3


def test_a_heading_fused_into_the_middle_of_a_block_still_opens_a_section():
    """`heading_level` only ever inspected a block's first line, but PyMuPDF
    routinely returns one block holding a caption, the article that follows it,
    and the next caption. Measured on the real corpus, tier 2 saw 16% of the
    articles actually present, and the misses are not merely coarse: the body
    of Madde 28 ends up filed under Madde 23, which is a wrong citation rather
    than an absent one."""
    fused = Block(
        text=(
            "Madde 23 - İşçinin haklı nedenle derhal fesih hakkı saklıdır.\n"
            "Çalışma belgesi\n"
            "Madde 28 - İşten ayrılan işçiye çalışma belgesi verilir."
        ),
        page_no=1,
    )
    chunks = chunk(_document([fused]), "doc1", ChunkingConfig())
    paths = [c.section_path for c in chunks]
    assert any("28" in (p or "") for p in paths), f"Madde 28 never opened a section: {paths}"
    article = next(c for c in chunks if "28" in (c.section_path or ""))
    assert "çalışma belgesi verilir" in article.text.lower()
    assert "23" not in (article.section_path or ""), "filed under the previous article"


def test_splitting_a_block_keeps_char_spans_indexing_the_original_text():
    """The line-level split must not move the offsets: `char_span` indexes the
    document's full text, and a chunk that points at the wrong characters is
    worse than one that points at none."""
    fused = Block(
        text="MADDE 1 - " + BODY + "\nMADDE 2 - " + BODY,
        page_no=1,
    )
    other = Block(text="MADDE 3 - " + BODY, page_no=2)
    document = _document([fused, other])
    full_text = "\n\n".join(b.text for b in document.blocks)
    chunks = chunk(document, "doc1", ChunkingConfig())
    assert len(chunks) >= 3, "the fused block did not split"
    for c in chunks:
        start, end = c.char_span
        assert full_text[start:end] == c.text
