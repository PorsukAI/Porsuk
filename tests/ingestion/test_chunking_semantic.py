"""Tests for tier-3 semantic (structure-free) chunking."""

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Outline, ParsedDocument
from porsuk.ingestion.chunking import chunk
from porsuk.ingestion.chunking.semantic import sections_semantic

PARAGRAPH = "Ödemeler fatura tarihinden itibaren otuz gün içinde yapılır. " * 3


def _document(blocks, outline=None):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=tuple(blocks),
        outline=outline or Outline(),
        page_count=1,
    )


def test_semantic_produces_a_single_untitled_section():
    """Tier 3 has no structure to go on, so it does not invent any."""
    document = _document([Block(text=PARAGRAPH, page_no=1)])
    sections = sections_semantic(document)
    assert len(sections) == 1
    assert sections[0].title is None
    assert sections[0].path is None


def test_semantic_keeps_every_block():
    document = _document([Block(text=f"Blok {i}.", page_no=1) for i in range(5)])
    sections = sections_semantic(document)
    assert sum(len(s.blocks) for s in sections) == 5


def test_chunk_falls_through_to_tier_three_with_no_structure():
    document = _document([Block(text=PARAGRAPH * 4, page_no=1)])
    chunks = chunk(document, "doc1", ChunkingConfig(target_chars=300))
    assert chunks
    assert all(c.section_path is None for c in chunks)


def test_chunk_returns_nothing_for_an_empty_document():
    assert chunk(_document([]), "doc1", ChunkingConfig()) == ()


def test_chunk_is_deterministic():
    document = _document([Block(text=PARAGRAPH * 4, page_no=1)])
    cfg = ChunkingConfig(target_chars=300)
    assert chunk(document, "doc1", cfg) == chunk(document, "doc1", cfg)


def test_size_is_measured_in_characters_not_tokens():
    """The unit is characters, because Turkish agglutination makes
    token counts drift between documents."""
    document = _document([Block(text=PARAGRAPH * 8, page_no=1)])
    small = chunk(document, "doc1", ChunkingConfig(target_chars=300))
    large = chunk(document, "doc1", ChunkingConfig(target_chars=2000))
    assert len(small) > len(large)
