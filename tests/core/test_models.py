"""Tests for the core domain dataclasses (`porsuk.core.models`)."""

import dataclasses

import pytest

from porsuk.core.models import (
    Block,
    Chunk,
    DocumentProfile,
    Filters,
    Outline,
    OutlineNode,
    ParsedDocument,
    ParseOutcome,
    ParseQualityComponents,
)


def test_parse_outcome_failed_carries_error_and_no_document():
    outcome = ParseOutcome(
        document=None,
        status="failed",
        quality=0.0,
        parser_used="pymupdf",
        error="password protected",
    )
    assert outcome.document is None
    assert outcome.status == "failed"
    assert outcome.error == "password protected"


def test_models_are_frozen():
    chunk = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="hello",
        page_no=1,
        language="tr",
        section_title=None,
        section_path=None,
        char_span=(0, 5),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "changed"


def test_outline_defaults_to_empty_with_source_none():
    outline = Outline()
    assert outline.nodes == ()
    assert outline.source == "none"


def test_outline_nests():
    child = OutlineNode(title="3.2 Odeme", level=2, page_no=12)
    parent = OutlineNode(title="3. Mali Hukumler", level=1, page_no=11, children=(child,))
    outline = Outline(nodes=(parent,), source="toc")
    assert outline.nodes[0].children[0].title == "3.2 Odeme"
    assert outline.source == "toc"


def test_parsed_document_holds_blocks_and_outline():
    doc = ParsedDocument(
        path="/corpus/a.pdf",
        doc_type="pdf",
        blocks=(Block(text="body", page_no=1),),
        outline=Outline(),
        page_count=1,
        metadata={"title": "A"},
    )
    assert doc.blocks[0].kind == "text"
    assert doc.page_count == 1


def test_document_profile_records_profile_source_components():
    profile = DocumentProfile(
        document_id="d1",
        path="/corpus/a.pdf",
        filename="a.pdf",
        doc_type="pdf",
        language="tr",
        created_at=None,
        modified_at=None,
        size=1024,
        page_count=14,
        summary="Tedarik Sozlesmesi",
        topics=("tedarik",),
        entities=("ABC Lojistik",),
        profile_level="cheap",
        profile_source=frozenset({"has_outline", "has_metadata_title"}),
        parse_quality=0.9,
        parse_quality_components=None,
        parser_used="pymupdf",
        image_heavy=False,
        content_hash="abc123",
    )
    assert "has_outline" in profile.profile_source
    assert profile.profile_level == "cheap"


def test_parse_quality_components_are_separable_from_score():
    components = ParseQualityComponents(
        chars_per_page=1200.0,
        bad_char_ratio=0.01,
        text_plausibility=0.82,
        empty_table_blocks=0,
    )
    outcome = ParseOutcome(
        document=None,
        status="degraded",
        quality=0.55,
        parser_used="pymupdf",
        error=None,
        components=components,
    )
    assert outcome.components.text_plausibility == 0.82


def test_filters_default_to_no_filtering():
    f = Filters()
    assert f.doc_types == ()
    assert f.languages == ()
    assert f.min_parse_quality is None


def test_parsed_document_reports_image_count():
    """`image_heavy` needs more than low text.

    A scanned page and an image-heavy page both have little text; only the
    image count separates them.
    """
    from porsuk.core.models import Outline, ParsedDocument

    document = ParsedDocument(
        path="/tmp/x.pdf",
        doc_type="pdf",
        blocks=(),
        outline=Outline(),
        page_count=2,
        image_count=7,
    )
    assert document.image_count == 7


def test_chunk_and_profile_and_filters_carry_job_id():
    c = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="t",
        page_no=1,
        language="tr",
        section_title=None,
        section_path=None,
        char_span=(0, 1),
    )
    assert c.job_id is None
    c2 = Chunk(
        chunk_id="c1",
        document_id="d1",
        text="t",
        page_no=1,
        language="tr",
        section_title=None,
        section_path=None,
        char_span=(0, 1),
        job_id="job-abc",
    )
    assert c2.job_id == "job-abc"
    assert Filters().job_ids == ()
    assert Filters(job_ids=("job-abc",)).job_ids == ("job-abc",)
