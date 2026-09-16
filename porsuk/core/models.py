"""Domain models shared by every layer.

These types are the vocabulary of the system. They import nothing from
adapters and nothing from provider libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

BlockKind = Literal["text", "heading", "table", "figure_caption"]
OutlineSource = Literal["toc", "pattern", "none"]
ParseStatus = Literal["ok", "degraded", "failed"]
ProfileLevel = Literal["cheap", "llm"]
RetrievalMethod = Literal["keyword", "semantic", "hybrid"]


@dataclass(frozen=True)
class Block:
    """One extracted region of a document, in reading order."""

    text: str
    page_no: int
    kind: BlockKind = "text"
    font_size: float | None = None
    bold: bool = False


@dataclass(frozen=True)
class OutlineNode:
    title: str
    level: int
    page_no: int | None = None
    children: tuple[OutlineNode, ...] = ()


@dataclass(frozen=True)
class Outline:
    """Heading hierarchy. `source` records how it was obtained."""

    nodes: tuple[OutlineNode, ...] = ()
    source: OutlineSource = "none"


@dataclass(frozen=True)
class ParsedDocument:
    path: str
    doc_type: str
    blocks: tuple[Block, ...]
    outline: Outline
    page_count: int
    image_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    page_no: int | None
    language: str | None
    section_title: str | None
    section_path: str | None
    char_span: tuple[int, int]
    job_id: str | None = None  # index job ID; None for `porsuk index` or an unscoped run


@dataclass(frozen=True)
class ParseQualityComponents:
    """Raw inputs to parse_quality, stored so failures are diagnosable."""

    chars_per_page: float
    bad_char_ratio: float
    text_plausibility: float
    empty_table_blocks: int


@dataclass(frozen=True)
class ParseOutcome:
    """Result type, not an exception. Partial failure is normal."""

    document: ParsedDocument | None
    status: ParseStatus
    quality: float
    parser_used: str
    error: str | None = None
    components: ParseQualityComponents | None = None
    image_heavy: bool = False


@dataclass(frozen=True)
class DocumentProfile:
    """One record per document. profile_vector lives in the store, not here."""

    document_id: str
    path: str
    filename: str
    doc_type: str
    language: str | None
    created_at: datetime | None
    modified_at: datetime | None
    size: int
    page_count: int
    summary: str
    topics: tuple[str, ...]
    entities: tuple[str, ...]
    profile_level: ProfileLevel
    profile_source: frozenset[str]
    parse_quality: float
    parse_quality_components: ParseQualityComponents | None
    parser_used: str
    image_heavy: bool
    content_hash: str
    job_id: str | None = None


@dataclass(frozen=True)
class EmbedResult:
    """Dense and optional sparse vectors from one embedding pass."""

    dense: tuple[tuple[float, ...], ...]
    sparse: tuple[dict[int, float], ...] | None = None


@dataclass(frozen=True)
class ProfileText:
    """The text a document profile embeds, and which components fed it.

    `source` mirrors `DocumentProfile.profile_source`: has_outline,
    has_metadata_title, has_intro, has_keywords, has_entities, turning
    "do our documents have a heading structure?" into a number.
    """

    text: str
    source: frozenset[str]


@dataclass(frozen=True)
class ChunkHit:
    chunk: Chunk
    score: float
    retrieval_method: RetrievalMethod
    rerank_score: float | None = None


@dataclass(frozen=True)
class DocumentSummary:
    """One row of `list_documents`: metadata, no vectors, no text."""

    document_id: str
    path: str
    filename: str
    doc_type: str
    language: str | None
    page_count: int
    parse_quality: float
    topics: tuple[str, ...]


@dataclass(frozen=True)
class DocumentHit:
    profile: DocumentProfile
    score: float
    retrieval_method: RetrievalMethod


@dataclass(frozen=True)
class Filters:
    """Same shape everywhere a search accepts filters."""

    doc_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    date_from: datetime | None = None
    date_to: datetime | None = None
    path_prefix: str | None = None
    topics: tuple[str, ...] = ()
    job_ids: tuple[str, ...] = ()  # index job IDs to filter by; () = no restriction
    min_parse_quality: float | None = None
    # A chat with no scope must not see another chat's uploaded
    # documents (job_id set), only the base corpus (job_id is None). This is
    # a separate flag from job_ids, not a special value of it: job_ids=()
    # keeps meaning "no restriction" everywhere it already did (CLI, eval,
    # any caller that predates per-chat scope), and this flag is additive,
    # opt-in, set only by the agent's scope=None path.
    unscoped_excludes_other_jobs: bool = False
