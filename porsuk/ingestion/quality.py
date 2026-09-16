"""The quality gate (kalite kapısı): `measure` (pure, config-free), `score`
(applies configured weights), and `decide` (reads raw component values, not
the blended score, since escalation is a per-symptom rule and a blend cannot
say which symptom occurred) as three separate functions rather than one.
"""

from __future__ import annotations

from dataclasses import dataclass

from porsuk.core.config import QualityConfig
from porsuk.core.models import ParsedDocument, ParseQualityComponents
from porsuk.ingestion.text_stats import bad_char_ratio, script_validity, text_plausibility


@dataclass(frozen=True)
class GateDecision:
    """What the gate concluded, and what made it conclude that.

    `trigger` and `reason` exist so `list_documents --low-quality` and
    `parse --json` can explain themselves.
    """

    escalate_to: str | None
    trigger: str | None
    reason: str
    image_heavy: bool
    # Damage no parser in the chain can undo. Distinct from `escalate_to is
    # None`, which means "good enough as parsed": this means "known bad, and
    # climbing the chain will not help". The router downgrades on it without
    # trying anything else.
    unrecoverable: bool = False


def measure(document: ParsedDocument) -> ParseQualityComponents:
    text = "\n".join(block.text for block in document.blocks)
    pages = document.page_count if document.page_count > 0 else 0
    empty_tables = sum(
        1 for block in document.blocks if block.kind == "table" and not block.text.strip()
    )
    return ParseQualityComponents(
        chars_per_page=len(text) / pages if pages else 0.0,
        bad_char_ratio=bad_char_ratio(text),
        text_plausibility=text_plausibility(text),
        empty_table_blocks=empty_tables,
    )


def document_script_validity(document: ParsedDocument) -> float:
    """`script_validity` over the document's whole text, for `decide`.

    Lives here rather than in each caller so the gate's inputs are all derived
    the same way `measure` derives its components: from the joined block text.
    """
    return script_validity("\n".join(block.text for block in document.blocks))


def normalise(value: float, bad: float, good: float) -> float:
    """Map a raw component onto 0-1 with a linear ramp, clamped outside.

    `good` may be numerically below `bad`: bad_char_ratio and
    empty_table_blocks are costs where less is better. Handling both directions
    here is why no per-component orientation flag is needed.
    """
    span = good - bad
    if span == 0:
        return 0.0
    return max(0.0, min(1.0, (value - bad) / span))


def score(components: ParseQualityComponents, cfg: QualityConfig) -> float:
    """The weighted combination, over normalised components."""
    parts = (
        (
            normalise(components.chars_per_page, cfg.chars_per_page_bad, cfg.chars_per_page_good),
            cfg.weight_chars_per_page,
        ),
        (
            normalise(components.bad_char_ratio, cfg.bad_char_ratio_bad, cfg.bad_char_ratio_good),
            cfg.weight_bad_char_ratio,
        ),
        (
            normalise(
                components.text_plausibility,
                cfg.text_plausibility_bad,
                cfg.text_plausibility_good,
            ),
            cfg.weight_text_plausibility,
        ),
        (
            normalise(
                float(components.empty_table_blocks),
                cfg.empty_table_blocks_bad,
                cfg.empty_table_blocks_good,
            ),
            cfg.weight_empty_table_blocks,
        ),
    )
    total_weight = sum(weight for _, weight in parts)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in parts) / total_weight


def decide(
    components: ParseQualityComponents,
    cfg: QualityConfig,
    *,
    image_count: int = 0,
    script_validity: float | None = None,
) -> GateDecision:
    """Which upgrade, and which component asked for it.

    Order matters: a document that is both text-starved and table-broken is an
    OCR problem first, because Docling on a page with no text layer has nothing
    to work with.

    `image_count` arrives as an argument rather than as a fifth component
    because `ParseQualityComponents` is fixed at four entries. It is the
    only thing separating a scanned page from an illustrated one.

    `script_validity` arrives the same way, and for the same reason. It is
    already one of the three signals inside `text_plausibility`, but averaged
    with the other two it is diluted to a third, and it is the only one of
    them that can see mojibake at all, because mojibaked text is made of
    characters that are each individually valid and inflect normally. Realistic
    Turkish mojibake blends to ~0.65, above `ocr_min_text_plausibility`, and
    passed the gate. Read raw, it separates cleanly. Omitted, the check is
    skipped, so callers that do not have the text keep working.
    """
    image_heavy = components.chars_per_page < cfg.image_heavy_max_chars_per_page and image_count > 0

    if components.chars_per_page < cfg.ocr_min_chars_per_page:
        return GateDecision(
            escalate_to="ocr",
            trigger="chars_per_page",
            reason=(
                f"chars_per_page {components.chars_per_page:.1f} below "
                f"{cfg.ocr_min_chars_per_page:.1f}; likely scanned"
            ),
            image_heavy=image_heavy,
        )
    # After the text-starvation check and before the blended one. A document
    # with no text at all scores 0.0 here, and "likely scanned" is the truer
    # diagnosis for it; but a document that HAS text in the wrong alphabet is
    # damaged in a specific, nameable way, and reporting `text_plausibility`
    # would point the reader at the wrong symptom.
    if script_validity is not None and script_validity < cfg.ocr_min_script_validity:
        # Deliberately does NOT escalate. Mojibake is an encoding bug, and the
        # page itself renders the wrong glyphs: "Ã¼" is what a human sees, so
        # OCR reads "Ã¼" back and no rasterisation can recover "ü". Measured
        # 2026-09-07 on mojibake.pdf: routing it to OCR made things worse than
        # useless. The OCR text is built from individually valid characters,
        # so bad_char_ratio fell 0.065 -> 0.000 and text_plausibility rose
        # 0.473 -> 0.697, clearing every threshold. The document passed as
        # `ok` while still being unreadable, and script_validity could no
        # longer see it either, because the Ã/Â pairs that gave it away were
        # gone. Naming the damage and stopping is the honest answer.
        return GateDecision(
            escalate_to=None,
            trigger="script_validity",
            reason=(
                f"script_validity {round(script_validity, 3)} below "
                f"{cfg.ocr_min_script_validity}; text is not in the expected alphabet "
                "(encoding damage — no parser can recover this, the page renders "
                "the wrong glyphs)"
            ),
            image_heavy=image_heavy,
            unrecoverable=True,
        )

    if components.bad_char_ratio > cfg.ocr_max_bad_char_ratio:
        return GateDecision(
            escalate_to="ocr",
            trigger="bad_char_ratio",
            reason=(
                f"bad_char_ratio {components.bad_char_ratio:.3f} above "
                f"{cfg.ocr_max_bad_char_ratio:.3f}; extracted text is damaged"
            ),
            image_heavy=image_heavy,
        )
    if components.text_plausibility < cfg.ocr_min_text_plausibility:
        return GateDecision(
            escalate_to="ocr",
            trigger="text_plausibility",
            reason=(
                f"text_plausibility {components.text_plausibility:.3f} below "
                f"{cfg.ocr_min_text_plausibility:.3f}; text does not read as language"
            ),
            image_heavy=image_heavy,
        )
    if components.empty_table_blocks >= cfg.docling_min_empty_table_blocks:
        return GateDecision(
            escalate_to="docling",
            trigger="empty_table_blocks",
            reason=(
                f"{components.empty_table_blocks} table block(s) detected but empty; "
                f"needs full layout analysis"
            ),
            image_heavy=image_heavy,
        )
    return GateDecision(
        escalate_to=None,
        trigger=None,
        reason="no escalation trigger fired",
        image_heavy=image_heavy,
    )
