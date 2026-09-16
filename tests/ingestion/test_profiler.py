"""Tests for the document profiler (cheap and LLM-backed profile passes)."""

from porsuk.core.config import ProfileConfig
from porsuk.core.models import (
    Block,
    Outline,
    OutlineNode,
    ParsedDocument,
    ParseOutcome,
)
from porsuk.ingestion.profiler import build_cheap_profile, build_llm_profile


class _FakeLLM:
    model = "fake"

    def complete(self, prompt, *, max_tokens=512, enable_thinking=True):
        return (
            "Bu bir tedarik sözleşmesidir. Taraflar ABC ve XYZ şirketleridir. "
            "Ödeme ve fesih koşulları düzenlenmiştir.\n"
            "KONULAR: tedarik, ödeme, fesih\n"
            "TİP: sözleşme"
        )


def _doc():
    return ParsedDocument(
        path="/2024/Sözleşmeler/Tedarik/ABC_Lojistik_v3.pdf",
        doc_type="pdf",
        blocks=(
            Block("TEDARİK SÖZLEŞMESİ", 1, kind="heading", font_size=18.0, bold=True),
            Block(
                "İşbu tedarik sözleşmesi ABC Lojistik ile XYZ Sanayi arasında "
                "akdedilmiştir. Ödeme koşulları ve gecikme faizi düzenlenmiştir. "
                "Teslimat programı ekte yer almaktadır.",
                1,
            ),
        ),
        outline=Outline(
            nodes=(OutlineNode("Taraflar", 1), OutlineNode("Mali Hükümler", 1)),
            source="pattern",
        ),
        page_count=14,
        image_count=0,
        metadata={"title": "Microsoft Word - Belge1"},
    )


def _outcome(doc):
    return ParseOutcome(
        document=doc,
        status="ok",
        quality=0.9,
        parser_used="pymupdf",
        error=None,
        components=None,
        image_heavy=False,
    )


def test_cheap_profile_composes_available_components():
    doc = _doc()
    cfg = ProfileConfig()
    profile, text = build_cheap_profile(
        doc, _outcome(doc), document_id="d1", content_hash="h", size=1000, cfg=cfg
    )
    assert profile.profile_level == "cheap"
    assert profile.language == "tr"
    low = text.text.lower()
    assert "tedarik" in low  # folder path + keywords
    assert "sözleşmeler" in low  # folder path
    assert "Mali Hükümler" in text.text  # outline is the most valuable part
    assert "Belge1" not in text.text  # junk metadata title rejected
    assert "has_outline" in text.source
    assert "has_intro" in text.source
    assert "has_metadata_title" not in text.source


def test_cheap_profile_carries_parse_fields():
    doc = _doc()
    profile, _ = build_cheap_profile(
        doc, _outcome(doc), document_id="d1", content_hash="abc", size=1000, cfg=ProfileConfig()
    )
    assert profile.parse_quality == 0.9
    assert profile.parser_used == "pymupdf"
    assert profile.content_hash == "abc"
    assert profile.document_id == "d1"
    assert profile.filename == "ABC_Lojistik_v3.pdf"


def test_cheap_profile_carries_modified_at_when_given():
    from datetime import UTC, datetime

    doc = _doc()
    when = datetime(2023, 5, 1, tzinfo=UTC)
    profile, _ = build_cheap_profile(
        doc,
        _outcome(doc),
        document_id="d1",
        content_hash="h",
        size=1000,
        cfg=ProfileConfig(),
        modified_at=when,
    )
    assert profile.modified_at == when


def test_cheap_profile_modified_at_defaults_to_none():
    doc = _doc()
    profile, _ = build_cheap_profile(
        doc, _outcome(doc), document_id="d1", content_hash="h", size=1000, cfg=ProfileConfig()
    )
    assert profile.modified_at is None


def test_llm_profile_disabled_returns_cheap_unchanged():
    doc = _doc()
    cfg = ProfileConfig(llm_enabled=False)
    base, _ = build_cheap_profile(
        doc, _outcome(doc), document_id="d1", content_hash="h", size=1000, cfg=cfg
    )
    profile, _ = build_llm_profile(doc, base, llm=_FakeLLM(), cfg=cfg)
    assert profile is base
    assert profile.profile_level == "cheap"


def test_llm_profile_adds_summary_and_topics():
    doc = _doc()
    cfg = ProfileConfig(llm_enabled=True)
    base, _ = build_cheap_profile(
        doc, _outcome(doc), document_id="d1", content_hash="h", size=1000, cfg=cfg
    )
    profile, text = build_llm_profile(doc, base, llm=_FakeLLM(), cfg=cfg)
    assert profile.profile_level == "llm"
    assert "tedarik sözleşmesidir" in profile.summary
    assert "tedarik" in profile.topics
    assert profile.doc_type == "pdf"  # file format unchanged
    # the LLM's content classification leads the topic list
    assert profile.topics[0] == "sözleşme"
    assert "tedarik" in text.text.lower()
