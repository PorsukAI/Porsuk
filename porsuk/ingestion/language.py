"""Language of a piece of text, at chunk level.

fastText, not a wordlist: it is fast, offline, and needs no Turkish
dictionary. The model ships with `fasttext-langdetect` and loads once on
first call.
"""

from __future__ import annotations

from ftlangdetect import detect

_MIN_CHARS = 20
_MIN_SCORE = 0.5


def detect_language(text: str) -> str | None:
    """ISO 639-1 code (``"tr"``, ``"en"``, …), or ``None`` when the text is
    too short or the detector is not confident."""
    cleaned = " ".join(text.split())
    if len(cleaned) < _MIN_CHARS:
        return None
    result = detect(text=cleaned, low_memory=True)
    if result and result.get("score", 0.0) >= _MIN_SCORE:
        return result["lang"]
    return None
