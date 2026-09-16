"""Tests for tier-1 TOC-based chunking (embedded PDF outline)."""

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Outline, OutlineNode, ParsedDocument
from porsuk.ingestion.chunking import chunk
from porsuk.ingestion.chunking.toc import sections_from_outline

BODY = "Ödemeler fatura tarihinden itibaren otuz gün içinde yapılır."


def _document(blocks, outline):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=tuple(blocks),
        outline=outline,
        page_count=3,
    )


def _outlined():
    outline = Outline(
        nodes=(
            OutlineNode(title="1. Taraflar", level=1, page_no=1),
            OutlineNode(
                title="2. Mali Hükümler",
                level=1,
                page_no=2,
                children=(OutlineNode(title="2.1 Ödeme", level=2, page_no=2),),
            ),
        ),
        source="toc",
    )
    blocks = [
        Block(text="1. Taraflar", page_no=1, kind="heading"),
        Block(text=BODY, page_no=1),
        Block(text="2. Mali Hükümler", page_no=2, kind="heading"),
        Block(text=BODY, page_no=2),
        Block(text="2.1 Ödeme", page_no=2, kind="heading"),
        Block(text=BODY, page_no=2),
    ]
    return _document(blocks, outline)


def test_returns_none_without_a_real_outline():
    """Tier 1 must decline cleanly so tier 2 gets its turn."""
    document = _document([Block(text=BODY, page_no=1)], Outline(source="none"))
    assert sections_from_outline(document) is None


def test_returns_none_when_the_outline_source_is_pattern():
    document = _document([Block(text=BODY, page_no=1)], Outline(source="pattern"))
    assert sections_from_outline(document) is None


def test_one_section_per_outline_entry():
    sections = sections_from_outline(_outlined())
    assert [s.title for s in sections] == ["1. Taraflar", "2. Mali Hükümler", "2.1 Ödeme"]


def test_section_path_reflects_the_outline_nesting():
    """section_path is what the contextual prefix is built from."""
    sections = sections_from_outline(_outlined())
    paths = {s.title: s.path for s in sections}
    assert paths["1. Taraflar"] == "1. Taraflar"
    assert paths["2.1 Ödeme"] == "2. Mali Hükümler > 2.1 Ödeme"


def test_sections_claim_the_blocks_that_follow_their_heading():
    sections = {s.title: s for s in sections_from_outline(_outlined())}
    assert len(sections["1. Taraflar"].blocks) == 2
    assert sections["2.1 Ödeme"].blocks[-1].text == BODY


def test_blocks_before_the_first_entry_are_not_dropped():
    """A cover page precedes the first outline entry in most real PDFs."""
    outline = Outline(nodes=(OutlineNode(title="1. Taraflar", level=1, page_no=2),), source="toc")
    blocks = [
        Block(text="KAPAK SAYFASI", page_no=1),
        Block(text="1. Taraflar", page_no=2, kind="heading"),
        Block(text=BODY, page_no=2),
    ]
    sections = sections_from_outline(_document(blocks, outline))
    all_text = [b.text for s in sections for b in s.blocks]
    assert "KAPAK SAYFASI" in all_text
    assert sections[0].title is None, "preamble has no heading of its own"


def test_an_entry_with_no_matching_block_still_contributes_to_the_path():
    """PDF outlines routinely name sections whose heading is an image or is
    styled past recognition. The entry must not vanish from descendants'
    paths."""
    outline = Outline(
        nodes=(
            OutlineNode(
                title="2. Mali Hükümler",
                level=1,
                page_no=1,
                children=(OutlineNode(title="2.1 Ödeme", level=2, page_no=1),),
            ),
        ),
        source="toc",
    )
    blocks = [Block(text="2.1 Ödeme", page_no=1, kind="heading"), Block(text=BODY, page_no=1)]
    sections = sections_from_outline(_document(blocks, outline))
    assert [s.path for s in sections if s.title == "2.1 Ödeme"] == ["2. Mali Hükümler > 2.1 Ödeme"]


def test_returns_none_when_nothing_matched_at_all():
    """An outline naming sections that appear nowhere is not usable structure;
    tier 2 deserves the chance."""
    outline = Outline(nodes=(OutlineNode(title="Bölüm X", level=1, page_no=1),), source="toc")
    document = _document([Block(text=BODY, page_no=1)], outline)
    assert sections_from_outline(document) is None


def test_no_block_is_claimed_twice():
    sections = sections_from_outline(_outlined())
    claimed = [id(b) for s in sections for b in s.blocks]
    assert len(claimed) == len(set(claimed))


def test_every_block_ends_up_in_exactly_one_section():
    document = _outlined()
    sections = sections_from_outline(document)
    claimed = {id(b) for s in sections for b in s.blocks}
    assert claimed == {id(b) for b in document.blocks}


def test_chunk_prefers_tier_one_when_an_outline_exists():
    chunks = chunk(_outlined(), "doc1", ChunkingConfig())
    assert chunks
    assert any(c.section_path == "2. Mali Hükümler > 2.1 Ödeme" for c in chunks)


def test_tier_one_applies_to_a_real_outlined_pdf(fixtures_dir):
    from porsuk.adapters.parsers.pymupdf import PyMuPDFParser

    outcome = PyMuPDFParser().parse(str(fixtures_dir / "outlined_pdf" / "with_toc.pdf"))
    chunks = chunk(outcome.document, "doc1", ChunkingConfig())
    assert chunks
    assert any(c.section_path and "Mali Hükümler" in c.section_path for c in chunks)


def test_a_one_entry_outline_does_not_claim_the_whole_document():
    """A PDF whose outline holds a single bookmark naming the cover would
    otherwise file every article in the document under 'Kapak'.

    `semantic.py` states the governing rule: a fabricated section path is worse
    than an absent one, because retrieval shows the user a heading that is not
    in their document. A single-section result means the outline found no
    internal structure, so the tier below it deserves its turn.
    """
    outline = Outline(nodes=(OutlineNode(title="Kapak", level=1, page_no=1),), source="toc")
    blocks = [
        Block(text="Kapak", page_no=1, kind="heading"),
        Block(text="MADDE 1 - " + BODY, page_no=1),
        Block(text="MADDE 2 - " + BODY, page_no=2),
        Block(text="MADDE 3 - " + BODY, page_no=2),
    ]
    chunks = chunk(_document(blocks, outline), "doc1", ChunkingConfig())
    paths = {c.section_path for c in chunks}
    assert paths != {"Kapak"}, "tier 1 filed the whole document under the cover"
    assert any("MADDE" in (p or "") for p in paths)
