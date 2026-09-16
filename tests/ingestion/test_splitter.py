"""Tests for text splitting into chunks: offsets, overlap, tables, sections."""

from porsuk.core.config import ChunkingConfig
from porsuk.core.models import Block, Outline, ParsedDocument
from porsuk.ingestion.chunking.splitter import Section, sections_to_chunks, split_text
from porsuk.ingestion.chunking.tables import table_to_markdown

PARAGRAPH = "Ödemeler fatura tarihinden itibaren otuz gün içinde yapılır. " * 4


def _document(blocks):
    return ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=tuple(blocks),
        outline=Outline(),
        page_count=1,
    )


def test_short_text_stays_one_piece():
    pieces = split_text("kısa metin", target_chars=1200, overlap_ratio=0.15)
    assert [p[0] for p in pieces] == ["kısa metin"]
    assert pieces[0][1:] == (0, len("kısa metin"))


def test_long_text_splits_near_the_target():
    text = "\n\n".join([PARAGRAPH] * 12)
    pieces = split_text(text, target_chars=600, overlap_ratio=0.0)
    assert len(pieces) > 1
    for body, _, _ in pieces:
        assert len(body) <= 900, "a piece ran far past the target"


def test_paragraph_boundaries_are_preserved():
    """Semantic chunking keeps paragraphs intact."""
    text = "Birinci paragraf.\n\nİkinci paragraf.\n\nÜçüncü paragraf."
    pieces = split_text(text, target_chars=30, overlap_ratio=0.0)
    for body, _, _ in pieces:
        assert not body.startswith("aragraf"), "a paragraph was cut mid-word"


def test_offsets_point_back_into_the_source():
    text = "\n\n".join([PARAGRAPH] * 6)
    for body, start, end in split_text(text, target_chars=500, overlap_ratio=0.0):
        assert text[start:end] == body


def test_overlap_repeats_the_tail_of_the_previous_piece():
    text = "\n\n".join(f"Paragraf {i} metni burada yer alır." for i in range(20))
    pieces = split_text(text, target_chars=200, overlap_ratio=0.2)
    assert len(pieces) > 1
    assert pieces[1][1] < pieces[0][2], "the second piece must start before the first ended"


def test_overlap_starts_at_a_sentence_boundary():
    # an overlapped piece must not begin mid-sentence.
    text = (
        "Birinci cümle burada bitiyor. İkinci cümle de burada sona eriyor. "
        "Üçüncü cümle biraz daha uzun ve devam ediyor sonra biter. "
        "Dördüncü cümle de vardır. Beşinci cümle son cümledir."
    )
    pieces = split_text(text, target_chars=70, overlap_ratio=0.4)
    assert len(pieces) >= 2
    for body, _start, _end in pieces[1:]:
        first = body.lstrip()[0]
        assert first.isupper() or first.isdigit(), f"piece starts mid-sentence: {body[:40]!r}"


def test_overlap_never_widens_past_the_piece_start():
    text = "A cümlesi. " + "x" * 200 + ". B cümlesi. " + "y" * 200 + "."
    for body, start, end in split_text(text, target_chars=120, overlap_ratio=0.25):
        assert text[start:end] == body  # char_span invariant holds


def test_overlap_with_no_sentence_boundary_in_window_falls_back():
    text = "x" * 800  # no punctuation
    pieces = split_text(text, target_chars=150, overlap_ratio=0.2)
    assert len(pieces) >= 2  # still splits, no infinite loop


def test_no_overlap_means_no_repetition():
    text = "\n\n".join(f"Paragraf {i} metni burada yer alır." for i in range(20))
    pieces = split_text(text, target_chars=200, overlap_ratio=0.0)
    for previous, following in zip(pieces, pieces[1:], strict=False):
        assert following[1] >= previous[2]


def test_a_single_oversized_paragraph_is_still_split():
    """A paragraph longer than the target cannot be kept whole; splitting on
    sentences is the least destructive option left."""
    text = "Bu bir cümledir. " * 200
    pieces = split_text(text, target_chars=400, overlap_ratio=0.0)
    assert len(pieces) > 1
    assert all(len(p[0]) < 800 for p in pieces)


def test_text_with_no_sentence_or_paragraph_boundary_is_hard_split():
    """Guards against an infinite loop on pathological input."""
    pieces = split_text("x" * 5000, target_chars=400, overlap_ratio=0.0)
    assert len(pieces) > 1
    assert "".join(p[0] for p in pieces) == "x" * 5000


def test_empty_text_produces_nothing():
    assert split_text("", target_chars=1200, overlap_ratio=0.15) == []
    assert split_text("   \n\n  ", target_chars=1200, overlap_ratio=0.15) == []


def test_tables_become_markdown():
    """Tables stay markdown."""
    markdown = table_to_markdown("Fatura No | Tutar\nF-001 | 15000")
    assert markdown.startswith("| Fatura No | Tutar |")
    assert "| --- | --- |" in markdown
    assert "| F-001 | 15000 |" in markdown


def test_a_table_block_is_never_split():
    """Tables are not split, however long they are."""
    rows = "\n".join(f"F-{i:04d} | {i * 1000} | 2024-01-01" for i in range(400))
    document = _document([Block(text=f"No | Tutar | Vade\n{rows}", page_no=1, kind="table")])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(target_chars=200),
    )
    assert len(chunks) == 1
    assert chunks[0].text.startswith("| No | Tutar | Vade |")


def test_every_chunk_carries_its_section_path():
    """'her chunk section_path taşır'."""
    document = _document([Block(text=PARAGRAPH, page_no=1)])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title="3.2 Ödeme", path="3. Mali Hükümler > 3.2 Ödeme", blocks=document.blocks)],
        ChunkingConfig(),
    )
    assert chunks
    for c in chunks:
        assert c.section_path == "3. Mali Hükümler > 3.2 Ödeme"
        assert c.section_title == "3.2 Ödeme"


def test_the_contextual_prefix_is_not_stored_in_the_text():
    """The contextual prefix is derived at embed time, not baked into text.

    Storing it would put the prefix into every source snippet shown to a user.
    """
    document = _document([Block(text=PARAGRAPH, page_no=1)])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title="3.2 Ödeme", path="3. Mali Hükümler > 3.2 Ödeme", blocks=document.blocks)],
        ChunkingConfig(),
    )
    assert not chunks[0].text.startswith("3. Mali Hükümler")
    assert ">" not in chunks[0].text


def test_chunk_ids_are_unique_deterministic_and_carry_the_document_id():
    document = _document([Block(text=PARAGRAPH * 6, page_no=1)])
    section = [Section(title=None, path=None, blocks=document.blocks)]
    first = sections_to_chunks(document, "doc1", section, ChunkingConfig(target_chars=300))
    second = sections_to_chunks(document, "doc1", section, ChunkingConfig(target_chars=300))
    ids = [c.chunk_id for c in first]
    assert ids == [c.chunk_id for c in second]
    assert len(set(ids)) == len(ids)
    assert all(c.document_id == "doc1" for c in first)
    assert all(c.chunk_id.startswith("doc1") for c in first)


def test_chunks_carry_the_page_they_came_from():
    document = _document([Block(text=PARAGRAPH, page_no=1), Block(text=PARAGRAPH, page_no=7)])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(target_chars=100),
    )
    assert {c.page_no for c in chunks} == {1, 7}


def test_chunk_language_is_detected_per_chunk():
    """The chunk-level language field exists for mixed documents."""
    document = _document(
        [
            Block(text="Ödemeler fatura tarihinden itibaren yapılır ve bu böyledir.", page_no=1),
            Block(text="Payments shall be made within thirty days of the invoice.", page_no=2),
        ]
    )
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(target_chars=60),
    )
    assert {c.language for c in chunks} >= {"tr", "en"}


def test_char_spans_index_the_documents_full_text():
    document = _document(
        [Block(text="Birinci blok.", page_no=1), Block(text="İkinci blok.", page_no=1)]
    )
    full_text = "\n\n".join(b.text for b in document.blocks)
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(target_chars=20),
    )
    for c in chunks:
        start, end = c.char_span
        assert full_text[start:end].strip() in c.text


def test_an_empty_table_block_produces_no_chunk():
    """`quality.py` counts an empty table block as a real, expected condition
    (`empty_table_blocks`), so the splitter must have an answer for it. Emitting
    a chunk with no text at all would be embedded and could be retrieved."""
    document = _document(
        [Block(text="Giriş metni.", page_no=1), Block(text="", page_no=1, kind="table")]
    )
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(),
    )
    assert all(c.text.strip() for c in chunks), "a chunk carrying no text was emitted"


def test_a_heading_only_chunk_is_not_emitted():
    """A chunk whose whole body is its own section title carries nothing the
    `section_path` does not already hold, yet it is embedded, retrievable and
    citable as a source.

    This is the shape tier 2 produces on real documents: a BÖLÜM heading opens
    a section of its own, and the MADDE below it opens a child section, so the
    parent's only block is its own heading. The words are not lost by dropping
    it: they remain in every child's `section_path`.
    """
    heading = Block(text="BİRİNCİ BÖLÜM", page_no=1, kind="heading")
    body = Block(text="MADDE 1 - " + PARAGRAPH, page_no=1)
    document = _document([heading, body])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [
            Section(title="BİRİNCİ BÖLÜM", path="BİRİNCİ BÖLÜM", blocks=(heading,)),
            Section(
                title="MADDE 1",
                path="BİRİNCİ BÖLÜM > MADDE 1",
                blocks=(body,),
            ),
        ],
        ChunkingConfig(),
    )
    assert not any(c.text.strip() == "BİRİNCİ BÖLÜM" for c in chunks)
    assert any("MADDE 1" in c.text for c in chunks), "the body must survive"


def test_chunk_ids_stay_contiguous_when_a_heading_chunk_is_dropped():
    """`chunk_id` is index-based, so dropping a chunk must renumber rather than
    leave a gap: a gap would make ids depend on what was filtered."""
    heading = Block(text="Ödemeler", page_no=1, kind="heading")
    table = Block(text="Fatura No | Tutar\nF-001 | 15000", page_no=1, kind="table")
    document = _document([heading, table])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title="Ödemeler", path="Ödemeler", blocks=document.blocks)],
        ChunkingConfig(),
    )
    assert [c.chunk_id for c in chunks] == [f"doc1:{i:05d}" for i in range(len(chunks))]


def test_a_lone_heading_block_still_reaches_a_chunk():
    """Dropping heading-only chunks must not delete the heading's text from the
    document: a section whose only block is its heading still has to be
    represented, or the words in it become unsearchable."""
    heading = Block(text="EK-1 Teslimat Programı", page_no=1, kind="heading")
    document = _document([heading])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title="EK-1 Teslimat Programı", path="EK-1", blocks=document.blocks)],
        ChunkingConfig(),
    )
    assert chunks, "a document consisting of one heading produced nothing"


def test_char_spans_index_the_full_text_exactly():
    """The strong form of the invariant `splitter.py` documents. The weaker
    `in` form would pass under whitespace drift or a truncating bug."""
    document = _document(
        [Block(text="Birinci blok.", page_no=1), Block(text="İkinci blok.", page_no=1)]
    )
    full_text = "\n\n".join(b.text for b in document.blocks)
    chunks = sections_to_chunks(
        document,
        "doc1",
        [Section(title=None, path=None, blocks=document.blocks)],
        ChunkingConfig(target_chars=20),
    )
    for c in chunks:
        start, end = c.char_span
        assert full_text[start:end] == c.text


def test_section_titles_never_carry_newlines():
    """A newline inside a ' > '-joined path corrupts the contextual prefix
    derived from it, and makes any citation ragged."""
    document = _document([Block(text=PARAGRAPH, page_no=1)])
    chunks = sections_to_chunks(
        document,
        "doc1",
        [
            Section(
                title="BİRİNCİ BÖLÜM \nBorç İlişkisinin Kaynakları",
                path="BİRİNCİ BÖLÜM \nBorç İlişkisinin Kaynakları",
                blocks=document.blocks,
            )
        ],
        ChunkingConfig(),
    )
    for c in chunks:
        assert "\n" not in (c.section_path or "")
        assert "\n" not in (c.section_title or "")
