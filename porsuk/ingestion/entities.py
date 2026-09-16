"""Pattern-based entity extraction.

Regex, not NER: NER is expensive for a layer whose job is recall, not
precision. The capitalised-phrase rule is crude for company names and
Turkish sentence-initial capitals produce false positives - an accepted
error, because the job is recall.

Patterns run in a fixed order; the first appearance of each distinct value
wins, so the returned list is stable.
"""

from __future__ import annotations

import re

# TCKN before VKN: an 11-digit run must be tried before a 10-digit one so a
# TCKN is not clipped to a VKN. Dates and contract numbers before the bare
# digit runs for the same reason. Money before dates so "1.250.000,00" is
# taken whole rather than leaving "000,00" behind.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d{2})?\s?(?:TL|₺|USD|\$|EUR|€|TRY)"),  # money
    re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"),  # dates
    re.compile(r"\b\d{4}[/-][A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b"),  # contract / invoice no
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # e-mail
    re.compile(r"\b\d{11}\b"),  # TCKN
    re.compile(r"\b\d{10}\b"),  # VKN
    re.compile(r"[A-ZĞÜŞİÖÇ][\wğüşıöç]+(?:\s+[A-ZĞÜŞİÖÇ][\wğüşıöç]+)+"),  # capitalised phrases
)


def extract_entities(text: str) -> list[str]:
    if not text.strip():
        return []
    seen: set[str] = set()
    hits: list[tuple[int, str]] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value and value not in seen:
                seen.add(value)
                hits.append((match.start(), value))
    hits.sort(key=lambda pair: pair[0])
    return [value for _pos, value in hits]
