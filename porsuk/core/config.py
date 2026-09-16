"""YAML profiles validated by Pydantic. Secrets come from the environment."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# ${VAR} is required; ${VAR:-default} falls back when VAR is unset or empty.
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    """Raised for any invalid or unresolvable configuration."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Base):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None


class EmbedderConfig(_Base):
    provider: str
    model: str | None = None
    dim: int | None = None
    normalize: bool = True
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    # Optional bge-m3 sparse (lexical) endpoint. Unset -> dense only, and
    # EmbedResult.sparse stays None. This is the wire so sparse retrieval
    # can be turned on without a schema change when it lands.
    sparse_base_url: str | None = None

    @field_validator("sparse_base_url", "base_url", mode="before")
    @classmethod
    def _blank_string_is_none(cls, value: object) -> object:
        # `${VAR:-}` in a profile yields "" when the var is unset; treat that
        # the same as absent so the embedder does not build a "/embed" URL.
        return None if value == "" else value


class StoreConfig(_Base):
    provider: str
    url: str | None = None
    path: str | None = None


class ChunkingConfig(_Base):
    """Size is measured in characters, not tokens."""

    target_chars: int = Field(default=1200, gt=0)
    overlap_ratio: float = Field(default=0.15, ge=0.0, lt=0.5)


class QualityConfig(_Base):
    """Normalisation bounds, weights and escalation thresholds.

    Calibrated on 2026-09-05 against both corpora: the 21 generated fixtures
    and 5 real mevzuat.gov.tr PDFs. Every number below is a measured row, and
    the row that constrains it from each side is named in
    `docs/calibration/2026-09-05-parse-quality.md`. These are config keys
    rather than literals in the gate on purpose, so they can be recalibrated
    without a code change.

    Normalisation: `<name>_good` is the raw value scoring 1.0 and `<name>_bad`
    the raw value scoring 0.0, with a linear ramp between and clamping outside.
    Which of the pair is numerically larger says whether the component is a
    benefit (more is better) or a cost (less is better): the ramp handles both
    directions without a separate flag.
    """

    # Observed ranges across both corpora, not round numbers. The ceilings are
    # real documents: 2666.5 is mevzuat_5651.pdf and 0.806 is simple_en.pdf.
    # The previous 800.0 ceiling saturated all five real documents at quality
    # 1.0, leaving parse_quality no headroom to rank them at all.
    chars_per_page_bad: float = 0.0
    chars_per_page_good: float = 2666.5
    bad_char_ratio_bad: float = 0.0648
    bad_char_ratio_good: float = 0.0
    text_plausibility_bad: float = 0.0
    text_plausibility_good: float = 0.806
    # NOT calibrated. `empty_table_blocks` measured 0 on all 26 documents and
    # structurally cannot be anything else at this stage: PyMuPDF emits no
    # block where it extracted no text, and the xlsx parser skips empty
    # sheets. Recognising a table whose cells came back empty is a layout
    # analysis capability - i.e. the escalation target itself. Left at a
    # placeholder until that capability lands.
    empty_table_blocks_bad: float = 3.0
    empty_table_blocks_good: float = 0.0

    # empty_table_blocks is identically 0 across every measured document, so
    # as a weighted component it separates nothing - it contributes a constant
    # 1.0 and, because `score` divides by the total weight, pulls every
    # document's score toward 1.0 by exactly its weight. At 0.10 that
    # compressed the whole usable range of parse_quality into [0.10, 1.0] for
    # no information. Its weight is redistributed across the three live
    # components in proportion to what they already had; it returns once
    # Docling makes the component measurable.
    weight_chars_per_page: float = Field(default=0.39, ge=0.0)
    weight_bad_char_ratio: float = Field(default=0.28, ge=0.0)
    weight_text_plausibility: float = Field(default=0.33, ge=0.0)
    weight_empty_table_blocks: float = Field(default=0.0, ge=0.0)

    # Escalation triggers, evaluated on raw component values.
    # Bounded below by two_sheets.xlsx at 50.5 chars/page - a clean spreadsheet
    # that OCR could not help even in principle - and above by the rasterised
    # scans at 0.0.
    ocr_min_chars_per_page: float = 25.0
    # Bounded above by control_chars.pdf at 0.0648 and below by every clean
    # document in both corpora, all of which measured exactly 0.0.
    ocr_max_bad_char_ratio: float = 0.02
    # Bounded above by table_like.pdf and two_sheets.xlsx at 0.667 - NOT by
    # real mevzuat, which measured 0.710-0.718 and was expected to be the
    # binding case. Bounded below by mojibake.pdf at 0.473.
    ocr_min_text_plausibility: float = 0.57
    # Its own trigger rather than a share of text_plausibility. Mojibake is
    # the damage script_validity was added for, but averaging it with
    # vowel_rate and function_word_rate dilutes it to a third: realistic
    # Turkish mojibake blends to ~0.65, above ocr_min_text_plausibility, and
    # passed. Measured 2026-09-05 over both corpora - every clean document
    # scores 0.989-1.000 (real mevzuat 0.998-0.999, lowest clean is
    # hyphenated.pdf at 0.989) while realistic Turkish mojibake scores
    # 0.877-0.893. Bounded below by hyphenated.pdf and above by the worst
    # mojibake sample; 0.95 sits between them with margin on both sides.
    ocr_min_script_validity: float = 0.95
    # NOT calibrated - see empty_table_blocks above. At 0 every document
    # escalates to Docling, clean ones included, so 1 is the floor.
    docling_min_empty_table_blocks: int = Field(default=1, ge=1)
    # Bounded below by scan_with_garbage_layer.pdf, a rasterised page carrying
    # 121.5 chars/page of bad OCR that the previous 100.0 failed to flag as
    # image-heavy, and above by mevzuat_6356.pdf, a real text document with an
    # image at 2506.9 chars/page.
    image_heavy_max_chars_per_page: float = 150.0

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> QualityConfig:
        """`score` divides by the total weight, so a profile whose weights sum
        to something else quietly changes what every calibrated threshold
        means, and all-zero weights score every document 0.0, which is
        indistinguishable from every document being broken. The invariant was
        asserted only against the defaults; a profile overriding them was
        unchecked, while config/local.yaml states it as though enforced."""
        total = (
            self.weight_chars_per_page
            + self.weight_bad_char_ratio
            + self.weight_text_plausibility
            + self.weight_empty_table_blocks
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"quality weights must sum to 1.0, got {total}")
        return self


class ParsingConfig(_Base):
    """The parser chain, in no particular order.

    Cost order is a property of the parsers, not of this list: `ParserRouter`
    sorts by `Parser.cost` at construction, and `text` (cost 5) is cheaper
    than the `pymupdf` named first here.

    Declaring the chain here is what lets `build_router` assemble it through
    the registry without branching on any provider name.
    """

    parsers: tuple[str, ...] = (
        "pymupdf",
        "docx",
        "xlsx",
        "pptx",
        "text",
        "ocr",
        "docling",
    )


class ServingConfig(_Base):
    """One shared VRAM budget. Measured 2026-09-09 with the
    full four-head stack (bge-m3 dense + sparse + reranker + qwen3-4b) on one
    RTX 3090: docs/eval/2026-09-09-agent.md.

      - context_length 16384: 8192 could not hold an agent loop over
        content-heavy Turkish corpora (RAGTurk overflowed at chunk_k=10, and
        also at chunk_k=5).
      - kv_cache_dtype fp8: halves KV memory, which is what makes the 16k
        context fit alongside the embedder/reranker on 24 GB. No visible
        answer-quality loss in the box smoke test.

    These values are RECORDED here and APPLIED by
    `infra/serve.sh`; nothing in `porsuk/` reads them; changing them here does
    not change how the model is served.
    """

    context_length: int = Field(default=16384, gt=0)  # measured
    kv_cache_dtype: Literal["fp16", "fp8"] = "fp8"  # measured
    gpu_memory_utilization: float = Field(default=0.75, ge=0.5, le=0.95)


class RerankConfig(_Base):
    provider: str
    model: str | None = None  # required by providers with more than one rerank model (e.g. openrouter)
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None

    @field_validator("base_url", mode="before")
    @classmethod
    def _blank_string_is_none(cls, value: object) -> object:
        # `${VAR:-}` in a profile yields "" when the var is unset; treat that
        # the same as absent so the reranker adapter does not try to POST to "".
        return None if value == "" else value


class RetrievalConfig(_Base):
    """Retrieval-engine parameters. The rows here were
    measured, evidence in docs/eval/2026-09-08-retrieval.md and
    docs/eval/2026-09-08-parameter-sweep.md (RAGTurk + XQuAD-tr + a manual
    mevzuat set, 2026-09-08).

      - strategy: `semantic`, best or tied on every golden set, and the only
        one that works cross-lingual (Türkçe soru → İngilizce belge).
      - reranking: helps every golden set, hurts none (+2.5-13 pt recall@5 on
        the cross-lingual probe). It is on by default *in the shipped
        config/vllm.yaml*, not here: `rerank_enabled=True` requires a
        `rerank:` block, which a code default cannot carry, and a bare
        `RetrievalConfig()` must stay valid for the tests that construct it.
      - chunk_k / neighbor_expansion / chunk size / stemmer: left at
        early defaults. The golden set at its current size cannot separate
        them: doc-level recall@5 is insensitive to chunk_k and to neighbour
        expansion by construction, RAGTurk is pre-chunked so chunk size is
        unsweepable against it, and the manual set is too small. Marked
        as still open, pending a larger golden set.
    """

    profile_k: int = Field(default=15, gt=0)
    chunk_k: int = Field(default=10, gt=0)  # still open (see docstring)
    strategy: Literal["semantic", "keyword", "hybrid"] = "semantic"  # measured
    rerank_enabled: bool = False  # measured: on by default in vllm.yaml (needs a rerank block)
    rerank_candidates: int = Field(default=30, gt=0)
    rerank: RerankConfig | None = None
    neighbor_expansion: int = Field(default=0, ge=0)  # still open, moved to a production A/B
    hybrid_enabled: bool = True
    stemmer_enabled: bool = False  # measured: both lexical engines got worse with it on
    keyword_backend: Literal["text", "bm25", "sparse"] = "text"  # 'text' = plain word search (MatchText, unscored)

    @model_validator(mode="after")
    def _rerank_enabled_needs_a_reranker(self) -> RetrievalConfig:
        if self.rerank_enabled and self.rerank is None:
            raise ValueError("retrieval.rerank_enabled is true but no rerank: block is configured")
        return self


class PipelineConfig(_Base):
    """The three-stage parse/embed/profile queue.

    `embed_batch_size` caps a single embed call. The pipeline currently
    embeds one document's chunks at a time, so this only bites documents
    with more chunks than the cap; cross-document batching is a future step.
    """

    parse_workers: int = Field(default=0, ge=0)  # 0 = os.cpu_count()
    embed_batch_size: int = Field(default=32, gt=0)


class IngestConfig(_Base):
    """Where the HTTP index API keeps uploaded documents and the
    pipeline's SQLite state, both of which must survive process restarts.

    `corpus_dir` holds one subdirectory per index job (`<corpus_dir>/<job_id>/`);
    the raw files stay after indexing: incremental re-indexing and
    a future `get_figure` both need the originals. It is the single source of
    truth: the ingestion state DB is derived as `<corpus_dir>/state.sqlite`, so
    a second job that re-uploads an unchanged file has its `content_hash`
    recognised and skips it.

    The CLI's `porsuk index` keeps its own `--state` flag and does not read this.
    """

    corpus_dir: str = ".porsuk/corpus"


class ProfileConfig(_Base):
    yake_top_n: int = Field(default=15, gt=0)
    first_paragraph_chars: int = Field(default=500, gt=0)
    llm_context_budget: int = Field(default=8192, gt=0)
    llm_enabled: bool = True
    llm: LLMConfig | None = None


class AgentConfig(_Base):
    """Agent-loop parameters. The model and endpoint are here; the
    LangGraph ReAct loop reads `max_tool_calls` for its step limit,
    `max_context_chars` caps a `get_document` full-text return, and the two
    `expand_*` values are the default window for `expand_context`.

    A bare `AgentConfig()` is valid on purpose: the tests that build `Config`
    construct it with no arguments, and `provider="fake_chat"` keeps that path
    GPU-free.
    """

    provider: str = "fake_chat"
    model: str = "fake-model"
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tool_calls: int = Field(default=8, gt=0)  # step limit
    max_context_chars: int = Field(default=12_000, gt=0)  # get_document cap
    expand_before_chars: int = Field(default=500, ge=0)  # expand_context
    expand_after_chars: int = Field(default=500, ge=0)
    # Approximate-token trigger for clearing old tool results from the
    # LLM's context (a user report: stale search results burned
    # context the model then had to re-search for). None disables trimming.
    context_trim_tokens: int | None = Field(default=8_000, gt=0)

    @field_validator("base_url", mode="before")
    @classmethod
    def _blank_string_is_none(cls, value: object) -> object:
        # `${VAR:-}` in a profile yields "" when unset; treat as absent so the
        # chat adapter does not build a "/chat/completions" URL onto nothing.
        return None if value == "" else value


class Config(_Base):
    llm: LLMConfig
    embedder: EmbedderConfig
    store: StoreConfig
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)


def _interpolate(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if env.get(name):
                return env[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name!r} is not set")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env) for v in value]
    return value


def _resolve_secret(section: dict[str, Any], env: Mapping[str, str]) -> None:
    name = section.get("api_key_env")
    if name is None:
        return
    if name not in env:
        raise ConfigError(f"config names api_key_env {name!r} but that variable is not set")
    section["api_key"] = env[name]


def _resolve_secrets(node: Any, env: Mapping[str, str]) -> None:
    """Resolve api_key_env wherever a provider section appears, nested or not."""
    if isinstance(node, dict):
        if "api_key_env" in node:
            _resolve_secret(node, env)
        for value in node.values():
            _resolve_secrets(value, env)
    elif isinstance(node, list):
        for item in node:
            _resolve_secrets(item, env)


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> Config:
    env = os.environ if env is None else env
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{path}: not valid UTF-8: {exc}") from exc
    try:
        raw = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level of a profile must be a mapping")
    raw = _interpolate(raw, env)
    _resolve_secrets(raw, env)
    try:
        return Config(**raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
