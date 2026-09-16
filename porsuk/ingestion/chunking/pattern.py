"""Tier 2 chunking: section boundaries from heading patterns (MADDE / BÖLÜM /
EK- numbering, and font weight as a weaker signal that only promotes a block
already short and standalone).
"""

from __future__ import annotations

import re

from porsuk.core.models import Block, ParsedDocument
from porsuk.ingestion.chunking.splitter import PATH_JOIN, Section

# Longest heading a numbering pattern is allowed to open. '1. ' followed by a
# paragraph is a list item; '1. Genel Hükümler' is a section.
_MAX_HEADING_CHARS = 80

# Longest a heading's CAPTION (the text after its marker) may be for the
# marker+caption to be trusted as a title in its own right. Measured against
# the real mevzuat.gov.tr corpus, not guessed: genuine article
# captions and Turkish legal numbered sub-headings ("Fesih ve Tazminat",
# "Amaç ve kapsam", "4. Yanılmada kusur") land well under this; a marker
# fused to its article body ("MADDE 607- Ömür boyu gelir sözleşmesi, bir
# kimsenin diğerine karşı...") or a numbered list item inside a body
# ("1. Asıl borç ile borçlunun kusur veya temerrüdünün yasal sonuçları.")
# both land well over it, even though both are under `_MAX_HEADING_CHARS`.
_MAX_CAPTION_CHARS = 40

# Punctuation a marker and its caption are typically joined by, stripped
# before measuring the caption so "MADDE 5 - Fesih" and "MADDE 5- Fesih"
# are judged the same way.
_CAPTION_JOINERS = " \t-–—:"

# Both spellings of BÖLÜM on purpose: PDF extraction sometimes drops the
# diacritic, and matching only the correct spelling would miss exactly the
# damaged documents that most need structure.
_ORDINALS = (
    r"(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU"
    r"|BIRINCI|IKINCI|UCUNCU|DORDUNCU|BESINCI|ALTINCI|YEDINCI|SEKIZINCI|DOKUZUNCU|ONUNCU)"
)
# Compound ordinals: a kanun with more than ten chapters writes 'ON BİRİNCİ
# BÖLÜM', and extraction may or may not keep the space. Without this every
# chapter past the tenth was invisible to the tier.
_TENS = r"(?:(?:ON|YİRMİ|YIRMI|OTUZ)\s*)?"
_BOLUM = re.compile(rf"^{_TENS}{_ORDINALS}\s+B[ÖO]L[ÜU]M\b", re.IGNORECASE)
# GEÇİCİ MADDE and EK MADDE appear in essentially every Turkish kanun, and a
# bare ^MADDE anchor excludes both.
_MADDE = re.compile(r"^(?:(?:GEÇİCİ|GECICI|EK)\s+)?MADDE\s+\d+\b", re.IGNORECASE)
# The hyphen is optional and the number may be a letter: 'EK-1', 'EK 1',
# 'EK-A' are all used. Checked after _MADDE so 'EK MADDE 1' is read as an
# article, not as annex 'M'.
_EK = re.compile(r"^EK\s*[-–—]?\s*(?:\d+|[A-ZÇĞİÖŞÜ])\b", re.IGNORECASE)

# Requires either multi-level dotting ("1.1", "3.2.1") or a trailing period on
# a bare integer ("1."). A bare integer with no trailing period ("2024 yılında
# imzalanmıştır.") is body text, not a heading: matching it too would make
# a stray four-digit year read as a level-1 section.
_DECIMAL = re.compile(r"^(\d+(?:\.\d+)+|\d+\.)\s+\S")


def _is_short_caption(remainder: str) -> bool:
    """Whether text following a marker reads as a caption, not a sentence."""
    return len(remainder.strip(_CAPTION_JOINERS)) <= _MAX_CAPTION_CHARS


def _is_section_numbering(marker: str) -> bool:
    """Whether a dotted marker is section numbering rather than a date or an
    amount.

    Multi-level dotting was treated as strong evidence on its own, exempt from
    the caption check. Measured against the real corpus, that is backwards for
    Turkish: `01.01.2024 tarihinden itibaren...` and `1.000.000 TL...` open
    lines far more often than `1.1` sub-headings do, and the caption check
    cannot save them: "tarihinden itibaren yürürlüğe girer." is a perfectly
    caption-length 36 characters.

    Shape is what separates them. Section numbering counts up in ones and
    twos; a four-digit group is a year, a three-digit group is a thousands
    separator, and a leading zero is a day or a month.
    """
    for group in marker.rstrip(".").split("."):
        if len(group) > 2 or (len(group) > 1 and group.startswith("0")):
            return False
    return True


def heading_level(text: str) -> int | None:
    """Depth of the heading this line opens, or None if it is not one.

    Level is what nests `section_path`: BÖLÜM and EK- sit at 1, MADDE at 2
    beneath them, and decimal numbering takes its depth from its own dots.
    """
    line = text.strip()
    if not line:
        return None
    # BÖLÜM / MADDE / EK- are near-unambiguous regardless of what follows on
    # the same block: mevzuat PDFs routinely glue "MADDE 1 - " directly onto
    # its article body with no line break, so capping these on length would
    # make tier 2 decline on exactly the real documents it exists for. Only
    # the decimal pattern is capped: "1. " opening a long paragraph is a list
    # item, not a section, and length is the only signal that tells them apart.
    if _MADDE.match(line):
        return 2
    if _BOLUM.match(line) or _EK.match(line):
        return 1
    if len(line) > _MAX_HEADING_CHARS:
        return None
    match = _DECIMAL.match(line)
    if not match:
        return None
    marker = match.group(1)
    if not _is_section_numbering(marker):
        return None
    # Multi-level dotted numbering ("1.1", "3.2.1") is strong evidence of a
    # real heading on its own and stays as permissive as before. A bare
    # "N." is not: Turkish legal prose is dense with numbered list items
    # inside articles ("1. Asıl borç ile borçlunun kusur veya temerrüdünün
    # yasal sonuçları.", a clause, not a section), and every one of them
    # would otherwise start a spurious section. The line already passed the
    # 80-char cap above, which is too loose to catch these: measured
    # against the real corpus, list items commonly land in the 60-80 char
    # range. So a bare "N." additionally needs its own caption check: only
    # a short, caption-shaped remainder ("1. Genel Hükümler",
    # "4. Yanılmada kusur") makes it a heading; a full clause does not.
    if marker.endswith(".") and not _is_short_caption(line[match.end(1) :]):
        return None
    return marker.rstrip(".").count(".") + 1


def _heading_title(text: str) -> str:
    """The section title for a heading block.

    Short captions are kept whole: "EK-2 Teslimat Programı" and "MADDE 5 -
    Fesih ve Tazminat" are titles in their entirety, not a marker with the
    rest chopped off. But BÖLÜM/MADDE/EK- bypass the length cap in
    `heading_level` (see there) precisely because mevzuat PDFs routinely
    glue a marker straight onto its article body, "MADDE 1 - <the whole
    article>", with no line break between them. Used as a title verbatim,
    that reaches `section_path` as hundreds of characters of duplicated
    paragraph text: expensive at embed time (it prefixes every chunk),
    unreadable as citation provenance, and a source of noise in retrieval
    numbers that no single golden-set failure would point back to this line.

    The trigger for trimming is the MARKER, not the block's total length:
    measured against the real corpus, a fused marker's caption is very
    often still under `_MAX_HEADING_CHARS` on its own (a mid-block PDF line
    wrap cuts it well before 80 characters), so gating on total length would
    silently skip most real cases. What decides caption-vs-prose is the
    length of what follows the marker (`_is_short_caption`): short means a
    real caption, kept whole; anything longer is prose the tier has no
    reliable way to separate a short caption from a paragraph on (Turkish
    abbreviations like "A.Ş." make sentence-boundary guessing unreliable),
    so the title is trimmed to the matched marker itself, the
    near-unambiguous, trustworthy part of the line, instead of guessed at.
    """
    line = text.strip()
    for pattern in (_BOLUM, _MADDE, _EK):
        match = pattern.match(line)
        if match:
            return line if _is_short_caption(line[match.end() :]) else match.group(0)
    if len(line) <= _MAX_HEADING_CHARS:
        return line
    # Unreachable in practice: decimal and styled-heading detection already
    # reject anything over `_MAX_HEADING_CHARS` before a title is built, so
    # only BÖLÜM/MADDE/EK- (handled above) ever arrive here over the cap.
    # Kept as a defensive floor rather than an assumption a future edit
    # could silently break.
    return line[:_MAX_HEADING_CHARS]


def _is_styled_heading(block: Block, body_size: float | None) -> bool:
    """The weak signal. Only promotes a short, standalone, emphasised block."""
    if len(block.text.strip()) > _MAX_HEADING_CHARS or not block.text.strip():
        return False
    if block.bold:
        return True
    return (
        body_size is not None and block.font_size is not None and block.font_size > body_size * 1.2
    )


def _body_font_size(blocks: tuple[Block, ...]) -> float | None:
    """The most common font size: the document's body text.

    Taking the mode rather than the mean keeps a single huge title from
    dragging the baseline up and hiding every real heading.
    """
    sizes = [b.font_size for b in blocks if b.font_size is not None]
    if not sizes:
        return None
    return max(set(sizes), key=sizes.count)


def sections_from_patterns(document: ParsedDocument) -> list[Section] | None:
    blocks = document.blocks
    if not blocks:
        return None

    body_size = _body_font_size(blocks)
    headings: list[tuple[int, str, int]] = []
    for index, block in enumerate(blocks):
        level = heading_level(block.text)
        if level is None and _is_styled_heading(block, body_size):
            level = 1
        if level is not None:
            headings.append((index, _heading_title(block.text), level))

    if not headings:
        return None

    sections: list[Section] = []
    if headings[0][0] > 0:
        sections.append(Section(title=None, path=None, blocks=blocks[: headings[0][0]]))

    ancestors: list[tuple[int, str]] = []
    for position, (index, title, level) in enumerate(headings):
        while ancestors and ancestors[-1][0] >= level:
            ancestors.pop()
        path = PATH_JOIN.join([*(t for _, t in ancestors), title])
        ancestors.append((level, title))
        end = headings[position + 1][0] if position + 1 < len(headings) else len(blocks)
        sections.append(Section(title=title, path=path, blocks=blocks[index:end]))
    return sections
