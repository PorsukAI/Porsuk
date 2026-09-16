"""Turning sections into chunks: the rules every chunking tier shares (tables
are never split, every chunk carries `section_path`, sizes are measured in
characters). Two invariants worth knowing before editing this module:
`char_span` always points at the chunk's raw SOURCE REGION in the document's
full text, even for table chunks where `chunk.text` is a markdown rendering
that does not equal that slice (pointing at generated markdown would break
provenance). And chunking works on whole sections, not per-block: consecutive
non-table blocks within a section are concatenated (joined by `_BLOCK_JOIN`)
before splitting, so `target_chars` is honoured per section rather than
yielding one tiny chunk per paragraph; a table block flushes the accumulated
text and is emitted as its own chunk.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Chunk, ParsedDocument
from porsuk.ingestion.chunking.tables import table_to_markdown
from porsuk.ingestion.text_stats import detect_language

_BLOCK_JOIN = "\n\n"
# The separator every tier joins a section path with. Defined here, with
# `Section`, because a chunk's path has to mean the same thing whichever tier
# built it.
PATH_JOIN = " > "
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


@dataclass(frozen=True)
class Section:
    """A run of blocks under one heading. What a chunking tier produces."""

    title: str | None
    path: str | None
    blocks: tuple[Block, ...]


def _pieces(text: str, target_chars: int) -> list[tuple[str, int, int]]:
    """Greedy accumulation over paragraphs, then sentences, then characters.

    Each fallback is more destructive than the last, so each is only reached
    when the previous one cannot fit the target.
    """
    units: list[tuple[str, int]] = []
    cursor = 0
    for part in _PARAGRAPH_SPLIT.split(text):
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(part)
        if not part.strip():
            continue
        if len(part) <= target_chars:
            units.append((part, start))
            continue
        inner = 0
        for sentence in _SENTENCE_END.split(part):
            at = part.find(sentence, inner)
            if at < 0:
                at = inner
            inner = at + len(sentence)
            if not sentence.strip():
                continue
            if len(sentence) <= target_chars:
                units.append((sentence, start + at))
            else:
                # Nothing left to split on. Hard-cut rather than emit a unit
                # that can never fit, which would loop forever.
                for offset in range(0, len(sentence), target_chars):
                    units.append((sentence[offset : offset + target_chars], start + at + offset))

    out: list[tuple[str, int, int]] = []
    current: list[tuple[str, int]] = []
    length = 0
    for unit, start in units:
        if current and length + len(unit) > target_chars:
            out.append(_join(current, text))
            current, length = [], 0
        current.append((unit, start))
        length += len(unit)
    if current:
        out.append(_join(current, text))
    return out


def _join(units: list[tuple[str, int]], text: str) -> tuple[str, int, int]:
    start = units[0][1]
    end = units[-1][1] + len(units[-1][0])
    return text[start:end], start, end


def split_text(text: str, *, target_chars: int, overlap_ratio: float) -> list[tuple[str, int, int]]:
    """Split into `(text, start, end)` triples, offsets indexing `text`.

    The target is measured in characters, never tokens. Turkish is
    agglutinative, so the same information packs into fewer whitespace-separated
    tokens and a token budget drifts from document to document.
    """
    if not text.strip():
        return []
    pieces = _pieces(text, target_chars)
    if overlap_ratio <= 0 or len(pieces) < 2:
        return pieces

    overlap = int(target_chars * overlap_ratio)
    out = [pieces[0]]
    for _body, start, end in pieces[1:]:
        widened = _snap_to_sentence(text, max(0, start - overlap), start)
        out.append((text[widened:end], widened, end))
    return out


def _snap_to_sentence(text: str, lo: int, hi: int) -> int:
    """The first sentence start in `[lo, hi)`, or `lo` if there is none.

    The overlap window used to widen a piece's start by a raw
    character count, so an overlapped chunk began mid-word ("s Üstü:") or
    mid-sentence ("üyeleridir."). Snapping the widened start forward to the
    character just after the nearest `_SENTENCE_END` match keeps every chunk
    starting at a sentence boundary. `hi` is the piece's own start: the
    window never crosses it, because moving past it would drop text rather
    than overlap it. A run with no sentence break in the window (a single
    unpunctuated block) falls back to `lo`, exactly as before.
    """
    window = text[lo:hi]
    m = _SENTENCE_END.search(window)
    return lo + m.end() if m is not None else lo


def block_offsets(blocks: Sequence[Block]) -> dict[int, int]:
    """Where each block begins in the document's full text.

    Keyed by identity rather than by content: two blocks can hold the same
    text, and a content key would give them the same span.

    Takes blocks rather than a document because the chunker may split a block
    at a heading line before sectioning (see `chunking.__init__`), and the
    offsets that matter are the split blocks' positions in the ORIGINAL text.
    """
    offsets: dict[int, int] = {}
    cursor = 0
    for block in blocks:
        offsets[id(block)] = cursor
        cursor += len(block.text) + len(_BLOCK_JOIN)
    return offsets


def _page_for_offset(bounds: list[tuple[int, int, int]], offset: int) -> int:
    """The page of the block whose span the offset falls into.

    `bounds` is ascending by local start, so the last block whose start is at
    or before `offset` is the one the split piece begins in, including when
    overlap widens a piece's start back into a still-earlier block.
    """
    page = bounds[0][2]
    for local_start, _local_end, block_page in bounds:
        if local_start <= offset:
            page = block_page
        else:
            break
    return page


def _run_chunks(
    document_id: str,
    start_index: int,
    run: list[Block],
    offsets: dict[int, int],
    section: Section,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """Concatenate a run of consecutive non-table blocks, then split once.

    This is the section-level concatenation ruling: `target_chars` is honoured
    per run of prose, not per block, or a document made of one-paragraph
    blocks would produce dozens of tiny chunks regardless of `target_chars`.
    """
    base = offsets[id(run[0])]
    bounds: list[tuple[int, int, int]] = []
    run_text = ""
    previous_end: int | None = None
    for block in run:
        start = offsets[id(block)]
        if previous_end is not None:
            # The real gap, not a fixed separator. Blocks produced by splitting
            # a parent at a line boundary are one newline apart, not two, and
            # rebuilding them with `_BLOCK_JOIN` would shift every span after
            # the split by a character.
            run_text += "\n" * (start - previous_end)
        local = len(run_text)
        run_text += block.text
        bounds.append((local, local + len(block.text), block.page_no))
        previous_end = start + len(block.text)

    chunks: list[Chunk] = []
    for i, (body, start, end) in enumerate(
        split_text(run_text, target_chars=cfg.target_chars, overlap_ratio=cfg.overlap_ratio)
    ):
        page = _page_for_offset(bounds, start)
        chunks.append(
            _chunk(document_id, start_index + i, body, page, section, (base + start, base + end))
        )
    return chunks


def _table_chunk(
    document_id: str, index: int, block: Block, offsets: dict[int, int], section: Section
) -> Chunk | None:
    """None when the table rendered to nothing.

    `quality.py` counts a table block whose cells came back empty as a real,
    expected condition, so the splitter needs an answer for it that is not an
    empty chunk. The invariant "no chunk is ever empty" belongs here rather
    than in each parser's discipline.
    """
    base = offsets[id(block)]
    text = table_to_markdown(block.text)
    if not text.strip():
        return None
    return _chunk(document_id, index, text, block.page_no, section, (base, base + len(block.text)))


def sections_to_chunks(
    document: ParsedDocument,
    document_id: str,
    sections: Sequence[Section],
    cfg: ChunkingConfig,
    offsets: dict[int, int] | None = None,
) -> tuple[Chunk, ...]:
    """Apply the shared rules and emit chunks in document order.

    Within each section, consecutive non-table blocks are concatenated before
    splitting (see the module docstring); a table block flushes the run and is
    emitted as its own single, never-split chunk.
    """
    if offsets is None:
        offsets = block_offsets(document.blocks)
    chunks: list[Chunk] = []
    for section in sections:
        run: list[Block] = []
        for block in section.blocks:
            if block.kind == "table":
                if run:
                    chunks.extend(_run_chunks(document_id, len(chunks), run, offsets, section, cfg))
                    run = []
                table = _table_chunk(document_id, len(chunks), block, offsets, section)
                if table is not None:
                    chunks.append(table)
            else:
                run.append(block)
        if run:
            chunks.extend(_run_chunks(document_id, len(chunks), run, offsets, section, cfg))
    return _renumber(document_id, _drop_redundant_headings(chunks))


def _is_heading_only(chunk: Chunk) -> bool:
    """The chunk's whole body is its own section title."""
    title = (chunk.section_title or "").strip()
    return bool(title) and chunk.text.strip() == title


def _drop_redundant_headings(chunks: list[Chunk]) -> list[Chunk]:
    """Remove chunks that carry nothing their `section_path` does not.

    Tier 2 opens a section at every heading, so a BÖLÜM whose articles each
    open their own child section is left holding only its own heading. That
    chunk is embedded and can be retrieved and cited while saying nothing.

    It is only safe to drop when the heading survives elsewhere: some other
    chunk with real content must be filed under this exact path, or beneath it.
    A trailing heading with nothing under it (`EK-1 Teslimat Programı` at the
    end of a contract) is kept, because dropping it would delete those words
    from the document entirely.
    """
    substantive = [c for c in chunks if not _is_heading_only(c)]
    keep: list[Chunk] = []
    for chunk in chunks:
        if not _is_heading_only(chunk):
            keep.append(chunk)
            continue
        path = chunk.section_path or ""
        covered = any(
            other.section_path == path or (other.section_path or "").startswith(path + PATH_JOIN)
            for other in substantive
        )
        if not covered:
            keep.append(chunk)
    return keep


def _renumber(document_id: str, chunks: list[Chunk]) -> tuple[Chunk, ...]:
    """`chunk_id` is index-based, so a dropped chunk must not leave a gap."""
    return tuple(
        replace(chunk, chunk_id=f"{document_id}:{index:05d}") for index, chunk in enumerate(chunks)
    )


def _chunk(
    document_id: str,
    index: int,
    text: str,
    page_no: int | None,
    section: Section,
    span: tuple[int, int],
) -> Chunk:
    return Chunk(
        chunk_id=f"{document_id}:{index:05d}",
        document_id=document_id,
        text=text,
        page_no=page_no,
        language=detect_language(text),
        section_title=_flatten(section.title),
        section_path=_flatten(section.path),
        char_span=span,
    )


def _flatten(title: str | None) -> str | None:
    """Collapse internal whitespace in a section title.

    PDF headings routinely arrive split across lines (`BİRİNCİ BÖLÜM \nBorç
    İlişkisinin Kaynakları`). A newline inside a ' > '-joined path corrupts the
    contextual prefix derived from it at embed time, and makes any
    citation ragged. Normalised here rather than in each tier so it holds for
    every chunk, whichever tier produced the sectioning.
    """
    if title is None:
        return None
    return " ".join(title.split()) or None
