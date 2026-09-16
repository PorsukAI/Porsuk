"""Two-tier document profile (doküman profili): a cheap tier (recall, not
precision, filters documents down to a candidate set the chunk search then
works on) and an LLM tier (summary, topic tags, document-type label) that
never blocks the query path and is skipped when cfg.llm_enabled is False.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePath

from porsuk.core.config import ProfileConfig
from porsuk.core.models import (
    DocumentProfile,
    Outline,
    ParsedDocument,
    ParseOutcome,
    ProfileText,
)
from porsuk.core.ports import LLMProvider
from porsuk.ingestion.entities import extract_entities
from porsuk.ingestion.keywords import keyword_terms
from porsuk.ingestion.language import detect_language

_VERSION_SUFFIX = re.compile(r"[_-](v\d+|final|son|rev\d*|draft|taslak)$", re.IGNORECASE)
_JUNK_TITLE = re.compile(r"^(microsoft word|untitled|document\d*|belge\d*|adsız)", re.IGNORECASE)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-zğüşıöç])(?=[A-ZĞÜŞİÖÇ])")


def _path_terms(path: str) -> str:
    words: list[str] = []
    for part in PurePath(path).parts:
        stem = _VERSION_SUFFIX.sub("", PurePath(part).stem)
        spaced = _CAMEL_BOUNDARY.sub(" ", stem)
        for token in re.split(r"[_\-\s]+", spaced):
            if token and token not in {"/", "\\"}:
                words.append(token)
    return " ".join(words)


def _title(doc: ParsedDocument) -> tuple[str | None, bool]:
    """(title, came_from_metadata). Junk metadata titles are rejected."""
    meta_title = doc.metadata.get("title", "").strip()
    if meta_title and not _JUNK_TITLE.match(meta_title):
        return meta_title, True
    if doc.outline.nodes:
        return doc.outline.nodes[0].title, False
    biggest = max(
        (b for b in doc.blocks if b.font_size),
        key=lambda b: b.font_size or 0.0,
        default=None,
    )
    return (biggest.text if biggest else None), False


def _outline_text(outline: Outline) -> str:
    lines: list[str] = []
    for node in outline.nodes[:20]:
        lines.append(node.title)
        for child in node.children[:5]:
            lines.append(f"  {child.title}")
    return " · ".join(line.strip() for line in lines)


def _intro(doc: ParsedDocument, max_chars: int) -> str:
    for block in doc.blocks:
        if block.kind == "text" and len(block.text.strip()) >= 60:
            return block.text.strip()[:max_chars]
    return ""


def build_cheap_profile(
    doc: ParsedDocument,
    outcome: ParseOutcome,
    *,
    document_id: str,
    content_hash: str,
    size: int,
    cfg: ProfileConfig,
    modified_at: datetime | None = None,
) -> tuple[DocumentProfile, ProfileText]:
    path_terms = _path_terms(doc.path)
    title, title_is_meta = _title(doc)
    outline_text = _outline_text(doc.outline)
    intro = _intro(doc, cfg.first_paragraph_chars)
    all_text = " ".join(b.text for b in doc.blocks)
    lang = detect_language(intro or all_text[:500])
    keywords = keyword_terms(all_text, top_k=cfg.yake_top_n, language=lang)
    entities = extract_entities(" ".join(b.text for b in doc.blocks[:30]))

    source: set[str] = set()
    lines: list[str] = []
    if title:
        lines.append(title)
        if title_is_meta:
            source.add("has_metadata_title")
    if path_terms:
        lines.append(f"Yol: {path_terms}")
    if outline_text:
        lines.append(f"Bölümler: {outline_text}")
        source.add("has_outline")
    if keywords:
        lines.append(f"Anahtar terimler: {', '.join(keywords)}")
        source.add("has_keywords")
    if intro:
        lines.append(f"Giriş: {intro}")
        source.add("has_intro")
    if entities:
        lines.append(f"Varlıklar: {', '.join(entities[:15])}")
        source.add("has_entities")
    lines.append(f"Tip: {doc.doc_type} · {doc.page_count} sayfa")

    text = ProfileText(text="\n".join(lines), source=frozenset(source))
    profile = DocumentProfile(
        document_id=document_id,
        path=doc.path,
        filename=PurePath(doc.path).name,
        doc_type=doc.doc_type,
        language=lang,
        created_at=None,
        modified_at=modified_at,
        size=size,
        page_count=doc.page_count,
        summary=intro[:300],
        topics=tuple(keywords[:10]),
        entities=tuple(entities[:15]),
        profile_level="cheap",
        profile_source=frozenset(source),
        parse_quality=outcome.quality,
        parse_quality_components=outcome.components,
        parser_used=outcome.parser_used,
        image_heavy=outcome.image_heavy,
        content_hash=content_hash,
    )
    return profile, text


_SUMMARY_PROMPT = (
    "Aşağıdaki belgeyi 3-5 cümleyle özetle, sonra 'KONULAR:' satırında "
    "virgülle ayrılmış konu etiketleri, 'TİP:' satırında belge tipini "
    "(sözleşme / rapor / fatura / sunum / yazışma) ver.\n\n{body}"
)


def _document_body(doc: ParsedDocument, max_chars: int) -> str:
    body = "\n".join(b.text for b in doc.blocks)
    if len(body) <= max_chars:
        return body
    half = max_chars // 2
    return f"{body[:half]}\n...\n{body[-half:]}"


def build_llm_profile(
    doc: ParsedDocument,
    base: DocumentProfile,
    *,
    llm: LLMProvider,
    cfg: ProfileConfig,
) -> tuple[DocumentProfile, ProfileText]:
    if not cfg.llm_enabled:
        return base, ProfileText(text=base.summary, source=base.profile_source)

    raw = llm.complete(
        _SUMMARY_PROMPT.format(body=_document_body(doc, cfg.llm_context_budget)),
        max_tokens=512,
        enable_thinking=False,
    )
    summary, topics, content_type = _parse_llm_output(raw)
    # doc_type stays the file format (pdf/docx) so metadata filters and the
    # file-type split keep working; the LLM's content classification
    # (sözleşme / rapor / ...) leads the normalised topic list instead.
    merged_topics = tuple(dict.fromkeys((*([content_type] if content_type else []), *topics)))
    profile = DocumentProfile(
        **{
            **base.__dict__,
            "summary": summary or base.summary,
            "topics": merged_topics or base.topics,
            "profile_level": "llm",
        }
    )
    text = ProfileText(
        text=f"{summary}\nKonular: {', '.join(merged_topics)}",
        source=base.profile_source | {"has_llm_summary"},
    )
    return profile, text


def _parse_llm_output(raw: str) -> tuple[str, tuple[str, ...], str | None]:
    summary_lines: list[str] = []
    topics: tuple[str, ...] = ()
    content_type: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("KONULAR:"):
            topics = tuple(
                t.strip().lower() for t in stripped.split(":", 1)[1].split(",") if t.strip()
            )
        elif upper.startswith(("TİP:", "TIP:")):
            content_type = stripped.split(":", 1)[1].strip().lower() or None
        elif stripped:
            summary_lines.append(stripped)
    return " ".join(summary_lines), topics, content_type
