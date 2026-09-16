"""TOC-first chunking: three tiers tried in order (real outline, then heading
patterns like MADDE/BÖLÜM/EK-, then semantic paragraph splitting as the
fallback that always succeeds), each returning None when it has nothing to
work with so the chain degrades rather than guesses.
"""

from __future__ import annotations

from dataclasses import replace

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Chunk, ParsedDocument
from porsuk.ingestion.chunking.pattern import heading_level, sections_from_patterns
from porsuk.ingestion.chunking.semantic import sections_semantic
from porsuk.ingestion.chunking.splitter import Section, sections_to_chunks
from porsuk.ingestion.chunking.toc import sections_from_outline

_BLOCK_GAP = 2  # len(splitter._BLOCK_JOIN)

__all__ = ["Section", "chunk"]


def chunk(document: ParsedDocument, document_id: str, cfg: ChunkingConfig) -> tuple[Chunk, ...]:
    working, offsets = _split_at_headings(document)
    sections = _choose_sections(working)
    if not sections:
        return ()
    return sections_to_chunks(working, document_id, sections, cfg, offsets=offsets)


def _split_at_headings(document: ParsedDocument) -> tuple[ParsedDocument, dict[int, int]]:
    """Cut blocks open where a heading begins on an interior line.

    A parser hands back the blocks its library found, and PyMuPDF's are laid
    out by the PDF's content stream: one block routinely holds a caption, the
    article beneath it, and the next caption. Tier 2 only ever inspected a
    block's first line, so every heading after the first was invisible:
    measured against the real mevzuat corpus, it found 16% of the articles
    present, and the bodies of the missed ones were filed under the previous
    article's number. That is a wrong citation, not a coarse one.

    Splitting here rather than in the tier keeps one rule in one place: tier 1
    matches outline entries to blocks and benefits from the same cut, and the
    offsets stay correct for both because they are computed once, over the
    original text, before anything is sectioned.

    Table blocks are never split.
    """
    blocks: list[Block] = []
    offsets: dict[int, int] = {}
    cursor = 0
    for block in document.blocks:
        for start, text in _heading_pieces(block.text):
            piece = replace(block, text=text)
            blocks.append(piece)
            offsets[id(piece)] = cursor + start
        cursor += len(block.text) + _BLOCK_GAP
    return replace(document, blocks=tuple(blocks)), offsets


def _heading_pieces(text: str) -> list[tuple[int, str]]:
    """`(offset within the block, text)` for each piece the block splits into.

    Offsets are exact positions in the block's own text, so a caller can turn
    them into positions in the document's text by adding the block's base. The
    newline that separated two pieces belongs to neither, which is what keeps
    `char_span` indexing the original characters.
    """
    lines = text.split("\n")
    if len(lines) == 1:
        return [(0, text)]
    starts: list[int] = []
    position = 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    cuts = [0] + [starts[i] for i in range(1, len(lines)) if heading_level(lines[i]) is not None]
    if len(cuts) == 1:
        return [(0, text)]
    pieces: list[tuple[int, str]] = []
    for index, start in enumerate(cuts):
        end = cuts[index + 1] - 1 if index + 1 < len(cuts) else len(text)
        pieces.append((start, text[start:end]))
    return pieces


def _choose_sections(document: ParsedDocument) -> list[Section] | None:
    """The first tier that actually divides the document.

    A tier returning a single section has found a title, not a structure, and
    a title stretched over a whole document is the failure `semantic.py` names:
    a fabricated section path is worse than an absent one, because retrieval
    shows the user a heading their document does not contain. One-bookmark PDFs
    (an outline naming only the cover) are common, and under a plain
    first-non-empty chain they starved tier 2 of its turn.

    So a single-section result is held rather than accepted, and the next tier
    is tried. It is still returned if nothing below it does better, because one
    real title beats none.
    """
    fallback: list[Section] | None = None
    for tier in (sections_from_outline, sections_from_patterns, sections_semantic):
        sections = tier(document)
        if not sections:
            continue
        if len(sections) > 1:
            return sections
        if fallback is None:
            fallback = sections
    return fallback
